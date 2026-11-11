// ── быстрый честный захват: snapdom вместо html-to-image ─────────────────
(function installFastCapture() {
  if (window.__r2GlassReady) return;
  const s = document.createElement('script');
  s.src = '/static/vendor/snapdom/snapdom.js';
  s.async = true;
  window.__r2GlassReady = new Promise((resolve) => {
  s.onload = () => {
    const api = window.snapdom;
    if (!api || !api.toCanvas) return;
    window.__r2GlassCapture = true;
    window.__glassToCanvas = async (el, o = {}) => api.toCanvas(el, {
      scale: o.pixelRatio || 1,
      width: o.width,
      height: o.height,
      backgroundColor: o.backgroundColor || 'transparent',
      filter: o.filter,
      fast: true,
    });
    resolve(true);
  };
  s.onerror = () => { console.warn('[lens] snapdom не загрузился'); resolve(false); };
  });
  document.head.appendChild(s);
})();

// Стили линзы и скрытие старого кольца курсора (.cur .c2) — без сборки CSS.
(function injectStyles() {
  if (document.getElementById('r2-lens-style')) return;
  const st = document.createElement('style');
  st.id = 'r2-lens-style';
  st.textContent = `
    .r2-lens { border-radius: 50%; }
    html.has-lens .cur .c2 { opacity: 0 !important; }
  `;
  document.head.appendChild(st);
})();

// Стеклянная линза-курсор для панели R2.
//
// Сцена для шейдера собирается из живого DOM: находим ближайший элемент,
// который полностью покрывает область линзы, и снимаем ТОЛЬКО его через
// snapdom (10-40мс вместо 100-300мс у html-to-image на всю секцию).
// Оптику считает LiquidGlass, поэтому преломление/блюр/фаска — настоящие.
//
// Инварианты:
//  1) фон сцены — только от непрозрачного предка, иначе остаётся белая
//     заливка библиотеки (белый диск над полупрозрачными элементами);
//  2) композер не отдаёт сцену библиотеке никогда — при ошибке заливает фон;
//  3) снимки не делаются во время скролла и во время ввода текста.

const SIZE = 36;
const HALF = SIZE / 2;
const SAMPLE = SIZE + 40;      // 36 + SHADOW_PAD*2 — область сцены
const SKIP = new Set(['CANVAS', 'IMG', 'VIDEO', 'SCRIPT', 'STYLE', 'LINK', 'NOSCRIPT']);

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const coarse = window.matchMedia('(pointer: coarse)');

if (!reduceMotion.matches && !coarse.matches && window.LiquidGlass) {
  start();

  function start() {
    const state = { x: innerWidth / 2, y: innerHeight / 2 };
    const cur = { x: state.x, y: state.y, vx: 0, vy: 0 };

    let lensEl = null;
    let inst = null;
    let building = false;
    let ready = false;

    let deepEl = null;
    let deepCanvas = null;
    let deepBusy = false;
    let deepAt = 0;
    let lastDeepAt = 0;
    let deepQueued = false;

    let scrolling = false;
    let scrollIdle = 0;
    let lastMoveAt = 0;
    let lastInputAt = 0;
    let hoverSig = '';
    let hidden = false;

    const markDirty = () => { try { inst?.markChanged?.(lensEl); } catch { /* ignore */ } };

    // ── цвет и прозрачность ──────────────────────────────────────────────
    function alphaOf(c) {
      if (!c) return 0;
      if (c === 'transparent') return 0;
      const slash = c.match(/\/\s*([\d.]+)\s*\)/);            // color(srgb … / .08)
      if (slash) return parseFloat(slash[1]);
      const nums = c.match(/[\d.]+/g);
      if (/^rgba?\(/i.test(c) || /^hsla?\(/i.test(c)) {
        return nums && nums.length >= 4 ? parseFloat(nums[3]) : 1;
      }
      if (/^#[0-9a-f]{8}$/i.test(c)) return parseInt(c.slice(7, 9), 16) / 255;
      if (/^#[0-9a-f]{4}$/i.test(c)) return parseInt(c.slice(4), 16) / 15;
      return 1;
    }
    const isOpaque = (c) => alphaOf(c) >= 0.95;

    function bgColorFor(el) {
      let n = el;
      while (n && n.nodeType === 1) {
        const c = getComputedStyle(n).backgroundColor;
        if (isOpaque(c)) return c;
        n = n.parentElement;
      }
      const root = getComputedStyle(document.documentElement).backgroundColor;
      if (isOpaque(root)) return root;
      const body = getComputedStyle(document.body).backgroundColor;
      if (isOpaque(body)) return body;
      return '#0d0d0d';
    }

    // ── выбор элемента под линзой ────────────────────────────────────────
    // Кэш результата: в панели Golden Layout цепочка длинная, а
    // getBoundingClientRect на каждый элемент = принудительный layout (300мс в профиле).
    const nodeCount = new WeakMap();
    function nodesIn(el) {
      let n = nodeCount.get(el);
      if (n === undefined) { n = el.querySelectorAll('*').length + 1; nodeCount.set(el, n); }
      return n;
    }

    let findAt = 0;
    let findKey = '';
    let findRes = null;

    function findDeep() {
      const key = Math.round(cur.x / 8) + ':' + Math.round(cur.y / 8);
      if (key === findKey && performance.now() - findAt < 120) return findRes;
      findKey = key;
      findAt = performance.now();

      const chain = document.elementsFromPoint(cur.x, cur.y).reverse(); // снаружи внутрь
      const minArea = (SAMPLE * SAMPLE) / 2;   // слишком мелкий элемент не нужен
      const maxArea = 420 * 420;               // крупный = дорогой снимок (lm_items)
      let best = null;
      let bestArea = Infinity;
      let covers = null;
      const half = SAMPLE / 2;

      for (const el of chain) {
        if (!el.getBoundingClientRect || el === lensEl || SKIP.has(el.tagName)) continue;
        const r = el.getBoundingClientRect();
        if (r.width < 24 || r.height < 24) continue;
        const area = r.width * r.height;
        // «покрывающий» — запасной вариант, если ничего подходящего по размеру
        if (r.left <= cur.x - half + 1 && r.right >= cur.x + half - 1
          && r.top <= cur.y - half + 1 && r.bottom >= cur.y + half - 1) covers = el;
        if (area < minArea || area > maxArea) continue;
        // Тяжёлый узел = дорогой снимок (lm_items давал 557мс). Лучше плоский
        // фон в линзе, чем пропущенный кадр.
        if (nodesIn(el) > 250) continue;
        if (area < bestArea) { bestArea = area; best = el; }
      }
      findRes = best || covers;
      return findRes;
    }

    // Частота пересъёмок: в панели DOM меняется 30 раз в секунду, снимок
    // на каждую мутацию (20-150мс) съедал кадры.
    const MIN_INTERVAL = 250;

    async function updateDeep(force) {
      if (!inst) return;
      const now = performance.now();
      // В движении не снимаем вообще: только фон и живые медиа.
      if (!force && now - lastMoveAt < 140) { deepQueued = true; return; }
      if (now - lastDeepAt < MIN_INTERVAL) { deepQueued = true; return; }
      lastDeepAt = now;
      if (deepBusy) { deepQueued = true; return; }
      const el = findDeep();
      if (!el) return;
      if (!force && el === deepEl && deepCanvas && performance.now() - deepAt < 250) return;
      deepBusy = true;
      deepEl = el;
      try {
        await inst.capture.captureElement(el, true);
        const rec = inst.capture.cache.get(el);
        deepCanvas = rec ? rec.canvas : null;
        deepAt = performance.now();
        markDirty();
      } catch { /* ignore */ }
      deepBusy = false;
      if (deepQueued) { deepQueued = false; setTimeout(() => updateDeep(false), 60); }
    }

    // Канвас/video/img snapdom не рендерит (foreignObject), поэтому такие
    // элементы рисуем напрямую — живые пиксели, без белых дыр.
    const mediaCache = new WeakMap();
    function mediaIn(el) {
      const now = performance.now();
      const rec = mediaCache.get(el);
      if (rec && now - rec.at < 500) return rec.list;
      const list = /^(IMG|VIDEO|CANVAS)$/.test(el.tagName)
        ? [el]
        : [...el.querySelectorAll('img, video, canvas')].slice(0, 12);
      mediaCache.set(el, { at: now, list });
      return list;
    }

    function drawMedia(ctx, sampleRect, rootRect, dpr) {
      if (!deepEl) return false;
      const nodes = mediaIn(deepEl);
      let drew = false;
      for (const m of nodes) {
        const r = m.getBoundingClientRect();
        if (r.width < 2 || r.height < 2) continue;
        const x = Math.round((r.left - rootRect.left) * dpr) - sampleRect.x;
        const y = Math.round((r.top - rootRect.top) * dpr) - sampleRect.y;
        const w = Math.round(r.width * dpr);
        const h = Math.round(r.height * dpr);
        if (x > sampleRect.w || y > sampleRect.h || x + w < 0 || y + h < 0) continue;
        try {
          if (m.tagName === 'CANVAS') {
            if (m.width > 0 && m.height > 0) { ctx.drawImage(m, x, y, w, h); drew = true; }
          } else {
            ctx.drawImage(m, x, y, w, h);
            drew = true;
          }
        } catch { /* ignore */ }
      }
      return drew;
    }

    // ── сборка сцены ─────────────────────────────────────────────────────
    function compose(ctx, sampleRect, rootRect, dpr) {
      try {
        if (!deepCanvas || !deepEl) {
          ctx.save();
          ctx.fillStyle = bgColorFor(deepEl || document.body);
          ctx.fillRect(0, 0, sampleRect.w, sampleRect.h);
          ctx.restore();
          return true;
        }
        ctx.save();
        ctx.fillStyle = bgColorFor(deepEl);
        ctx.fillRect(0, 0, sampleRect.w, sampleRect.h);
        ctx.restore();
        const r = deepEl.getBoundingClientRect();
        const x = Math.round((r.left - rootRect.left) * dpr) - sampleRect.x;
        const y = Math.round((r.top - rootRect.top) * dpr) - sampleRect.y;
        const hasMedia = /^(IMG|VIDEO|CANVAS)$/.test(deepEl.tagName)
          || deepEl.querySelector('img, video, canvas') !== null;
        // Если под линзой медиа — снимок DOM рисуем, но сверху кладём живые
        // пиксели: в foreignObject их нет, и без этого получалось белое пятно.
        ctx.drawImage(deepCanvas, x, y, Math.round(r.width * dpr), Math.round(r.height * dpr));
        if (hasMedia) drawMedia(ctx, sampleRect, rootRect, dpr);
      } catch {
        try {
          ctx.save();
          ctx.fillStyle = bgColorFor(deepEl || document.body);
          ctx.fillRect(0, 0, sampleRect.w, sampleRect.h);
          ctx.restore();
        } catch { /* ignore */ }
      }
      return true;
    }

    // ── сама линза ───────────────────────────────────────────────────────
    const CONFIG = {
      blurAmount: 0.2, refraction: 2, chromAberration: 0, edgeHighlight: 0,
      specular: 0.04, fresnel: 2, distortion: 0.15, cornerRadius: 18, zRadius: 18,
      saturation: -1, shadowOpacity: 0.12, shadowSpread: 6, shadowOffsetY: 2, bevelMode: 1,
    };

    function placeLens(x, y) {
      if (lensEl) lensEl.style.transform = `translate3d(${x - HALF}px,${y - HALF}px,0)`;
    }

    async function build() {
      if (lensEl || building) return;
      building = true;
      try { await (window.__r2GlassReady || Promise.resolve()); } catch { /* ignore */ }
      lensEl = document.createElement('div');
      lensEl.className = 'r2-lens';
      lensEl.id = 'r2-lens';
      lensEl.dataset.config = JSON.stringify(CONFIG);
      lensEl.style.cssText = `position:fixed;left:0;top:0;width:${SIZE}px;height:${SIZE}px;`
        + 'pointer-events:none;z-index:9997;opacity:0;will-change:transform';
      document.body.appendChild(lensEl);
      placeLens(state.x, state.y);
      try {
        inst = await window.LiquidGlass.init({ root: document.body, glassElements: [lensEl] });
        inst.sceneComposer = (ctx, sampleRect, rootRect, dpr) => compose(ctx, sampleRect, rootRect, dpr);
        updateDeep(true);
        lensEl.style.opacity = '1';
        document.documentElement.classList.add('has-lens');
      } catch (e) {
        console.warn('[lens] инициализация не удалась:', e);
        lensEl.remove();
        lensEl = null;
      }
      building = false;
    }

    // ── движение: пружина ────────────────────────────────────────────────
    let lastT = performance.now();
    let lastRenderAt = 0;
    (function tick(now) {
      const dt = Math.min(0.033, Math.max(0.001, (now - lastT) / 1000));
      lastT = now;
      if (!document.hidden) {
        cur.vx += (state.x - cur.x) * 420 * dt;
        cur.vy += (state.y - cur.y) * 420 * dt;
        const decay = Math.exp(-30 * dt);
        cur.vx *= decay;
        cur.vy *= decay;
        cur.x += cur.vx * dt;
        cur.y += cur.vy * dt;
        placeLens(cur.x, cur.y);
        // При быстром движении стекло «растворяется»: рендер 76px-шейдера
        // каждый кадр отбирает кадры у панели (видео + терминал). Как только
        // курсор замедлился — стекло проявляется и обновляется.
        const speed = Math.min(1, dt > 0 ? Math.hypot(cur.vx, cur.vy) / 1000 : 0);
        const visible = speed < 0.35;
        if (lensEl) {
          const want = visible ? '1' : '0';
          if (lensEl.dataset.vis !== want) {
            lensEl.dataset.vis = want;
            lensEl.style.transition = 'opacity 120ms ease';
            lensEl.style.opacity = want;
          }
        }
        if (visible && (Math.abs(state.x - cur.x) > 0.4 || Math.abs(state.y - cur.y) > 0.4)
          && now - lastRenderAt >= 70) { lastRenderAt = now; markDirty(); }
      }
      requestAnimationFrame(tick);
    })(performance.now());

    let moveCapTimer = 0;
    addEventListener('pointermove', (e) => {
      lastMoveAt = performance.now();
      state.x = e.clientX;
      state.y = e.clientY;
      if (!building && !lensEl) build();
      // Снимок — только когда курсор остановился: в движении важнее кадры,
      // а стекло доли миллисекунды показывает предыдущую сцену.
      clearTimeout(moveCapTimer);
      moveCapTimer = setTimeout(() => updateDeep(true), 130);
    }, { passive: true });

    // ── ввод текста: не мешаем ───────────────────────────────────────────
    addEventListener('input', () => {
      lastInputAt = performance.now();
      clearTimeout(hidden);
      hidden = setTimeout(() => updateDeep(true), 420);
    }, true);
    const typing = () => performance.now() - lastInputAt < 400;

    // ── изменения без мутаций DOM (hover, анимации) ──────────────────────
    function styleSig() {
      return document.elementsFromPoint(cur.x, cur.y)
        .filter((e) => e !== lensEl)
        .slice(0, 3)
        .map((e) => {
          const cs = getComputedStyle(e);
          return [e.tagName, e.className, cs.backgroundColor, cs.color, cs.borderTopColor,
            cs.opacity, cs.transform, cs.filter].join('~');
        })
        .join('|');
    }

    setInterval(() => {
      if (!inst || scrolling || document.hidden || typing()) return;
      if (performance.now() - lastMoveAt < 40) return;
      const sig = styleSig();
      if (!sig || sig === hoverSig) return;
      hoverSig = sig;
      updateDeep(false);
    }, 200);

    // мутации DOM (классы, src, текст)
    let domQueued = false;
    new MutationObserver((list) => {
      for (const m of list.slice(0, 3)) {
        const t = m.target.nodeType === 1 ? m.target : m.target.parentElement;
        if (!t || (lensEl && lensEl.contains(t))) continue;
        // Без чтения геометрии: getBoundingClientRect в MutationObserver — это
        // принудительный layout на каждую мутацию (328мс в профиле панели).
        // Частоту всё равно ограничивает троттлинг пересъёмки.
        if (scrolling || typing() || domQueued) continue;
        domQueued = true;
        requestAnimationFrame(() => { domQueued = false; updateDeep(false); });
        return;
      }
    }).observe(document.body, {
      subtree: true, childList: true, characterData: true,
      attributes: true, attributeFilter: ['src', 'srcset', 'style', 'class', 'hidden', 'data-config'],
    });

    // ── скролл ───────────────────────────────────────────────────────────
    let scrollQueued = false;
    addEventListener('scroll', () => {
      scrolling = true;
      clearTimeout(scrollIdle);
      scrollIdle = setTimeout(() => {
        scrolling = false;
        updateDeep(true);
      }, 260);
      if (scrollQueued) return;
      scrollQueued = true;
      requestAnimationFrame(() => { scrollQueued = false; markDirty(); });
    }, { passive: true, capture: true });

    // ── пауза/уборка ─────────────────────────────────────────────────────
    document.addEventListener('visibilitychange', () => {
      if (!inst) return;
      if (document.hidden) inst.pause?.();
      else inst.resume?.();
    });

    addEventListener('pagehide', () => {
      try { inst?.destroy?.(); } catch { /* ignore */ }
      lensEl?.remove();
      document.documentElement.classList.remove('has-lens');
      lensEl = null;
      inst = null;
    }, { once: true });
  }
}

