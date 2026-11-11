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
    function findDeep() {
      const chain = document.elementsFromPoint(cur.x, cur.y).reverse();
      let deep = null;
      for (const el of chain) {
        if (!el.getBoundingClientRect || el === lensEl || SKIP.has(el.tagName)) continue;
        const cs = getComputedStyle(el);
        if (cs.position === 'fixed' || cs.visibility === 'hidden' || cs.display === 'none') continue;
        if (parseFloat(cs.opacity) === 0) continue;
        const r = el.getBoundingClientRect();
        const half = SAMPLE / 2;
        const covers = r.left <= cur.x - half + 1 && r.right >= cur.x + half - 1
          && r.top <= cur.y - half + 1 && r.bottom >= cur.y + half - 1;
        if (covers) deep = el;
      }
      return deep;
    }

    async function updateDeep(force) {
      if (!inst || deepBusy) return;
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
        ctx.drawImage(deepCanvas, x, y, Math.round(r.width * dpr), Math.round(r.height * dpr));
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
        if (Math.abs(state.x - cur.x) > 0.4 || Math.abs(state.y - cur.y) > 0.4) markDirty();
      }
      requestAnimationFrame(tick);
    })(performance.now());

    addEventListener('pointermove', (e) => {
      lastMoveAt = performance.now();
      state.x = e.clientX;
      state.y = e.clientY;
      if (!building && !lensEl) build();
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
      updateDeep(true);
    }, 200);

    // мутации DOM (классы, src, текст)
    let domQueued = false;
    new MutationObserver((list) => {
      for (const m of list) {
        const t = m.target.nodeType === 1 ? m.target : m.target.parentElement;
        if (!t || (lensEl && lensEl.contains(t))) continue;
        const b = t.getBoundingClientRect();
        if (b.right < cur.x - 60 || b.left > cur.x + 60) continue;
        if (b.bottom < cur.y - 60 || b.top > cur.y + 60) continue;
        if (scrolling || typing() || domQueued) continue;
        domQueued = true;
        requestAnimationFrame(() => { domQueued = false; updateDeep(true); });
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

