// R2 UI kit — vanilla DOM helpers and components in the Loginov language.
// Токены и примитивы (.lbl .num .display .tile .seg) живут в static/css/input.css.

const EASE = 'cubic-bezier(0.23,1,0.32,1)';

/** Create an element: h('div', { class: 'x', onclick: fn }, child, ...). */
export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'class' || key === 'className') node.className = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value);
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
    else if (key === 'value') node.value = value;
    else if (key === 'checked' || key === 'disabled' || key === 'selected') node[key] = Boolean(value);
    else node.setAttribute(key, value);
  }
  append(node, children);
  return node;
}

function append(parent, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    parent.appendChild(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return parent;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

// --- Icons (tabler-style, 24x24 stroke) ---
const ICONS = {
  video: '<path d="M15 10l4.55-2.27a1 1 0 0 1 1.45.9v6.74a1 1 0 0 1-1.45.9L15 14"/><rect x="3" y="6" width="12" height="12" rx="2"/>',
  sliders: '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
  arm: '<path d="M12 3v4M9 7h6l1 6H8l1-6zM10 13l-.7 8M14 13l.7 8M7 21h10"/>',
  move: '<path d="M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3M21 12l-3-3M21 12l-3 3"/>',
  terminal: '<path d="M5 7l5 5-5 5M13 17h7"/>',
  list: '<path d="M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01"/>',
  code: '<path d="M7 8l-4 4 4 4M17 8l4 4-4 4M14 4l-4 16"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h3l2 2h9a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  file: '<path d="M14 3v5h5M6 2h8l5 5v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.6"/><path d="M4 18l5-5 3 3 3-4 5 6"/>',
  database: '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="10" y="10" width="4" height="4"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/>',
  sight: '<circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1"/><path d="M12 3v2.5M12 18.5V21M3 12h2.5M18.5 12H21"/>',
  x: '<path d="M6 6l12 12M18 6l-12 12"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-2.7 1.1V21a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-2.7-1.1l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.6 1.6 0 0 0 3 15a2 2 0 1 1 0-4 1.6 1.6 0 0 0 1.1-2.7l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1A1.6 1.6 0 0 0 9 4.6a2 2 0 1 1 4 0 1.6 1.6 0 0 0 2.7 1.1l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1A1.6 1.6 0 0 0 20 11a2 2 0 1 1 0 4z"/>',
  power: '<path d="M12 3v9M6.3 6.3a8 8 0 1 0 11.4 0"/>',
  refresh: '<path d="M21 12a9 9 0 1 1-2.6-6.4M21 4v5h-5"/>',
  logout: '<path d="M15 4h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-3M10 17l-5-5 5-5M5 12h11"/>',
  chevronRight: '<path d="M9 6l6 6-6 6"/>',
  check: '<path d="M5 12l5 5L20 6"/>',
  alert: '<circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/>',
  upload: '<path d="M12 20V8M7 13l5-5 5 5M5 4h14"/>',
  folderPlus: '<path d="M3 7a2 2 0 0 1 2-2h3l2 2h9a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2zM12 11v5M9.5 13.5h5"/>',
  save: '<path d="M5 3h11l3 3v13a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zM8 3v6h7V3M8 21v-7h8v7"/>',
  panelLeft: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9.5 4v16"/>',
  monitor: '<rect x="2" y="4" width="20" height="14" rx="2"/><path d="M8 20h8M12 18v2"/>',
  phone: '<rect x="6" y="2" width="12" height="20" rx="3"/><path d="M11 18h2"/>',
};

export function icon(name, cls = 'size-4', stroke = 1.7) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', String(stroke));
  svg.setAttribute('stroke-linecap', 'square');
  svg.setAttribute('stroke-linejoin', 'miter');
  svg.setAttribute('class', cls);
  svg.innerHTML = ICONS[name] || ICONS.file;
  return svg;
}

// --- Brand mark: monolines "R2", no frame ---
const SVG_NS = 'http://www.w3.org/2000/svg';
const MARK_PATHS = [
  'M8 26V6h9a4.5 4.5 0 0 1 0 9H8M13 15l6.5 11', // R
  'M25 10a4.5 4.5 0 0 1 9 0l-9 16h9',            // 2
];

/**
 * Brand mark. `loading` loops a draw/erase; `intro` draws once on mount.
 * Uses pathLength="1" so the CSS animation is independent of path length.
 */
export function mark({ loading = false, intro = false, className = 'size-6', stroke = 2.4 } = {}) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 42 32');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('class', className);
  svg.setAttribute('aria-hidden', 'true');
  const animate = loading || intro;
  if (animate) svg.classList.add(loading ? 'r2-mark-draw' : 'r2-mark-intro');
  for (const d of MARK_PATHS) {
    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', d);
    path.setAttribute('stroke', 'currentColor');
    path.setAttribute('stroke-width', String(stroke));
    path.setAttribute('stroke-linecap', 'square');
    path.setAttribute('stroke-linejoin', 'miter');
    if (animate) path.setAttribute('pathLength', '1');
    svg.appendChild(path);
  }
  return svg;
}

// --- Buttons ---
const VARIANTS = {
  /* Кнопка = текст. Рамка появляется только у «выбранного» (.tab/.seg). */
  default: '',
  /* Главное действие — подчёркнутый текст (.cta-mail), не белая плашка. */
  primary: '',
  outline: '',
  ghost: 'text-foreground/45',
  /* Аварийное действие: единственное место, где живёт красный. */
  destructive: 'text-destructive/80 hover:text-destructive',
  link: 'text-primary underline-offset-4 hover:underline',
};
const SIZES = {
  default: 'gap-1.5 px-[.8rem] text-[.625rem] leading-none',
  sm: 'gap-1 px-3 text-[10px] leading-none',
  xs: 'gap-1 px-2.5 text-[9.5px] leading-none',
  icon: 'size-8',
  'icon-sm': 'size-7',
  'icon-xs': 'size-6',
};

export function button({ variant = 'default', size = 'default', label, iconName, className = '', onClick, title, disabled } = {}) {
  const pill = !size.startsWith('icon');
  const btn = h('button', {
    type: 'button',
    class: `press r2-bloom r2-btn${pill ? '' : ' r2-btn--icon'} group/button inline-flex shrink-0 items-center justify-center whitespace-nowrap font-normal uppercase tracking-[.09em] outline-none transition-[color,border-color,background-color,text-shadow] duration-300 disabled:pointer-events-none disabled:opacity-50 ${pill ? 'rounded-full' : 'rounded-full'} ${VARIANTS[variant] || VARIANTS.default} ${SIZES[size] || SIZES.default} ${className}`,
    onclick: onClick,
    title,
    disabled,
  });
  if (iconName) btn.appendChild(icon(iconName, size.startsWith('icon') ? 'size-4' : 'size-3.5'));
  if (label) btn.appendChild(h('span', { text: label }));
  return btn;
}

export function iconButton(iconName, { onClick, title, size = 'icon-sm', variant = 'ghost', className = '' } = {}) {
  return button({ variant, size, iconName, onClick, title, className });
}

// --- Micro-label: заменяет подписи-абзацы ---
export function label(text, className = '') {
  return h('span', { class: `lbl ${className}`, text });
}

/** Число в display-шрифте: tabular-nums, тонкий вес. */
export function num(text, className = '') {
  return h('span', { class: `num ${className}`, text });
}

/* Поле ввода — линия, а не коробка (как .cta-mail в ASBDS). */
export const inputClass =
  'h-9 w-full rounded-full border border-ink-12 bg-[color-mix(in_srgb,var(--fg)_5%,transparent)] px-4 text-[13px] text-foreground outline-none transition-colors placeholder:text-muted-foreground hover:border-ink-32 focus-visible:border-foreground disabled:opacity-50';

export function spinner(className = 'size-4') {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('class', `animate-spin ${className}`);
  svg.innerHTML = '<circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="3" opacity="0.2"/><path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>';
  return svg;
}

export function emptyState(text, className = '') {
  return h('div', { class: `flex h-full items-center justify-center p-6 text-center text-[12px] text-muted-foreground ${className}`, text });
}

// --- Toasts ---
let toastHost = null;
export function toast(message, kind = 'default', timeout = 4000) {
  if (!toastHost) {
    toastHost = h('div', { class: 'pointer-events-none fixed bottom-4 right-4 z-[100] flex flex-col gap-2' });
    document.body.appendChild(toastHost);
  }
  const tone = kind === 'error'
    ? 'border-destructive/30 text-destructive'
    : kind === 'success'
      ? 'border-primary/30 text-foreground'
      : 'border-border text-foreground';
  const node = h('div', { class: `anim-in pointer-events-auto flex max-w-sm items-start gap-2 rounded-[18px] border border-ink-12 bg-card px-4 py-2.5 text-[12px] ${tone}` },
    kind === 'error' ? icon('alert', 'size-4 shrink-0') : kind === 'success' ? icon('check', 'size-4 shrink-0') : null,
    h('span', { text: message }));
  toastHost.appendChild(node);
  setTimeout(() => {
    node.style.transition = `opacity 150ms ${EASE}, translate 150ms ${EASE}`;
    node.style.opacity = '0';
    node.style.translate = '0 4px';
    setTimeout(() => node.remove(), 160);
  }, timeout);
}

// --- Modal ---
export function modal({ title, body, footer, width = 'max-w-lg', onClose } = {}) {
  const overlay = h('div', { class: 'anim-fade fixed inset-0 z-[90] flex items-center justify-center bg-black/60 p-4' });
  const close = () => { overlay.remove(); document.removeEventListener('keydown', onKey); onClose?.(); };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  const card = h('div', { class: `anim-zoom flex max-h-[85vh] w-full ${width} flex-col overflow-hidden rounded-[24px] border border-ink-12 bg-card` },
    title !== undefined ? h('div', { class: 'flex items-center justify-between border-b border-border px-4 py-2.5' },
      h('h2', { class: 'lbl', text: title }),
      iconButton('x', { onClick: close, title: 'Закрыть' })) : null,
    h('div', { class: 'min-h-0 flex-1 overflow-auto p-4' }, body),
    footer ? h('div', { class: 'flex flex-wrap items-center justify-end gap-2 border-t border-border px-4 py-3' }, footer) : null);
  overlay.appendChild(card);
  overlay.addEventListener('mousedown', (e) => { if (e.target === overlay) close(); });
  document.addEventListener('keydown', onKey);
  document.body.appendChild(overlay);
  return { close, overlay, card };
}

// --- Section wrapper used inside panels ---
export function panelBody(...children) {
  return h('div', { class: 'flex h-full min-h-0 flex-col gap-3 p-3' }, ...children);
}

// --- Живой UI: волна от нажатия и волна по дорожке слайдера ---------------
export function initLivingUI() {
  if (typeof document === 'undefined') return;

  // Волна из точки нажатия для кнопок, вкладок и сегментов.
  document.addEventListener('pointerdown', (e) => {
    const el = e.target.closest?.('button, .r2-btn, .sg, .lm_tab, .tile');
    if (!el || el.disabled) return;
    const r = el.getBoundingClientRect();
    const wave = document.createElement('span');
    wave.className = 'ui-wave';
    wave.style.left = `${e.clientX - r.left}px`;
    wave.style.top = `${e.clientY - r.top}px`;
    wave.style.setProperty('--sz', `${Math.max(r.width, r.height) * 2.2}px`);
    el.appendChild(wave);
    wave.addEventListener('animationend', () => wave.remove(), { once: true });
    setTimeout(() => wave.remove(), 900);
  }, true);

  // Значение изменилось — короткая вспышка (тельметрия, углы, счётчики).
  const flash = (el) => {
    if (!el || el._flashUntil > performance.now()) return;
    el._flashUntil = performance.now() + 260;
    el.animate?.(
      [{ opacity: .45, filter: 'brightness(1.8)' }, { opacity: 1, filter: 'brightness(1)' }],
      { duration: 420, easing: 'cubic-bezier(.16,1,.3,1)' },
    );
  };
  const watch = new MutationObserver((muts) => {
    for (const m of muts) {
      let t = m.target.nodeType === 3 ? m.target.parentElement : m.target;
      if (!t) continue;
      if (!(t.classList?.contains('num') || t.matches?.('.num'))) t = t.closest?.('.num') || (t.closest?.('.stat') ? t : null);
      if (t) flash(t);
    }
  });
  const observeValues = () => {
    document.querySelectorAll('.num, .stat').forEach((el) => {
      if (el.dataset.flashWatched) return;
      el.dataset.flashWatched = '1';
      watch.observe(el, { subtree: true, childList: true, characterData: true });
    });
  };
  observeValues();
  new MutationObserver(() => { clearTimeout(observeValues._t); observeValues._t = setTimeout(observeValues, 400); })
    .observe(document.body, { childList: true, subtree: true });

  // Волна по дорожке слайдера: --p — позиция бегунка, --w растёт, --wa гаснет.
  document.addEventListener('input', (e) => {
    const r = e.target;
    if (!(r instanceof HTMLInputElement) || r.type !== 'range') return;
    const min = Number(r.min) || 0;
    const max = Number(r.max) || 100;
    const p = max > min ? (Number(r.value) - min) / (max - min) : 0;
    r.style.setProperty('--p', `${(p * 100).toFixed(1)}%`);
    const t0 = performance.now();
    cancelAnimationFrame(r._waveRaf);
    const step = (t) => {
      const k = Math.min(1, (t - t0) / 620);
      // первое кольцо — быстрое и яркое, второе — догоняющая «волна-эхо»
      r.style.setProperty('--w', `${(k * 85).toFixed(1)}%`);
      r.style.setProperty('--wa', (0.5 * (1 - k) * (1 - k)).toFixed(3));
      const k2 = Math.max(0, Math.min(1, (k - 0.18) / 0.82));
      r.style.setProperty('--w2', `${(k2 * 70).toFixed(1)}%`);
      r.style.setProperty('--wa2', (0.28 * (1 - k2)).toFixed(3));
      if (k < 1) r._waveRaf = requestAnimationFrame(step);
    };
    r._waveRaf = requestAnimationFrame(step);
  }, true);
}

// --- Приближение курсора к рамкам ------------------------------------------
// Рамки светлеют, когда курсор подходит: --near = 1 - d / RADIUS (0..1).
// Прямоугольники кэшируются, пересчёт — на scroll/resize/изменение DOM,
// сама запись --near идёт раз в кадр и только если значение изменилось.
export function initProximity() {
  if (typeof window === 'undefined' || matchMedia('(pointer: coarse)').matches) return;
  const SEL = '.lm_item, .lm_content, .lm_header, .lm_header .lm_tab, .lm_splitter, .seg, .r2-btn, .app-head, .trow, input, select, textarea';
  const RADIUS = 140;
  let items = [];
  let dirty = true;
  const pt = { x: -1e4, y: -1e4 };
  let lastX = -1e4, lastY = -1e4;

  const collect = () => {
    items = [...document.querySelectorAll(SEL)].map((el) => {
      // у сплиттера ::after занят самой линией — там градиент ставится напрямую
      if (el.classList.contains('lm_item')) el.classList.add('ez-frame');
      else if (!el.classList.contains('lm_splitter')) {
        el.classList.add('ez-line');
        // У файлов сияние слабее, чем у папок (is-dir).
        if (el.classList.contains('trow') && !el.classList.contains('is-dir')) {
          el.classList.add('ez-line--dim');
        }
      }
      return { el, r: null, v: -1, mx: -1, my: -1 };
    });
    dirty = true;
  };

  const markDirty = () => { dirty = true; };
  addEventListener('pointermove', (e) => { pt.x = e.clientX; pt.y = e.clientY; }, { passive: true });
  addEventListener('scroll', markDirty, true);
  addEventListener('resize', markDirty);
  new MutationObserver(() => { clearTimeout(collect._t); collect._t = setTimeout(collect, 250); })
    .observe(document.body, { childList: true, subtree: true });

  (function loop() {
    // Курсор не двигался и геометрия не менялась — кадр пропускаем целиком.
    if (!dirty && pt.x === lastX && pt.y === lastY) { requestAnimationFrame(loop); return; }
    lastX = pt.x; lastY = pt.y;
    if (dirty) {
      for (const it of items) it.r = it.el.getBoundingClientRect();
      // новые элементы могли появиться без mutation (перерисовка панели)
      if (items.length < document.querySelectorAll(SEL).length) collect();
      dirty = false;
    }
    for (const it of items) {
      const r = it.r;
      if (!r || !r.width) continue;
      const dx = Math.max(r.left - pt.x, 0, pt.x - r.right);
      const dy = Math.max(r.top - pt.y, 0, pt.y - r.bottom);
      const d = Math.hypot(dx, dy);
      const near = d > RADIUS ? 0 : +(1 - d / RADIUS).toFixed(3);
      if (near !== it.v) {
        it.el.style.setProperty('--near', near);
        it.v = near;
      }
      if (near > 0) {
        // позиция курсора внутри элемента — от неё строится градиент-блик
        const mx = Math.round(pt.x - r.left), my = Math.round(pt.y - r.top);
        if (mx !== it.mx || my !== it.my) {
          it.el.style.setProperty('--mx', `${mx}px`);
          it.el.style.setProperty('--my', `${my}px`);
          it.mx = mx; it.my = my;
        }
      }
    }
    requestAnimationFrame(loop);
  })();

  collect();
}

if (typeof document !== 'undefined') {
  const start = () => { initProximity(); initLivingUI(); };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
}

// --- Переползающая пилюля в .seg: одно поведение на все переключатели ----
// Панели создаются динамически (Golden Layout), поэтому следим за DOM.
export function initSegs(root = document) {
  for (const seg of root.querySelectorAll('.seg:not([data-seg])')) {
    seg.dataset.seg = '1';
    const pill = document.createElement('i');
    pill.className = 'seg-pill is-hidden';
    seg.prepend(pill);

    const active = () => seg.querySelector('.sg.on, .sg:has(input:checked)');
    const place = (instant) => {
      const a = active();
      if (!a) { pill.classList.add('is-hidden'); return; }
      const sr = seg.getBoundingClientRect();
      const ar = a.getBoundingClientRect();
      if (!ar.width) { pill.classList.add('is-hidden'); return; }
      if (instant) pill.style.transition = 'none';
      pill.classList.remove('is-hidden');
      pill.style.width = `${ar.width}px`;
      pill.style.transform = `translateX(${ar.left - sr.left - 2}px)`;
      if (instant) { void pill.offsetWidth; pill.style.transition = ''; }
    };

    seg.addEventListener('click', () => setTimeout(() => place(false), 0));
    seg.addEventListener('change', () => place(false));
    new ResizeObserver(() => place(true)).observe(seg);
    requestAnimationFrame(() => place(true));
  }
}

if (typeof document !== 'undefined') {
  const boot = () => {
    initSegs();
    new MutationObserver((muts) => {
      for (const m of muts) for (const n of m.addedNodes) {
        if (n.nodeType === 1) initSegs(n.matches?.('.seg') ? n.parentElement : n);
      }
    }).observe(document.body, { childList: true, subtree: true });
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
}

// --- Курсор ASBDS: точка (c1) + кольцо (c2), системный скрыт в asbds.css ---
// Механика скопирована из оригинала: точка догоняет курсор быстрее кольца.
(function mountCursor() {
  if (typeof document === 'undefined' || document.getElementById('r2-cur')) return;
  if (matchMedia('(pointer: coarse)').matches) return;
  const cur = h('div', { class: 'cur', id: 'r2-cur' }, h('i', { class: 'c1' }), h('i', { class: 'c2' }));
  document.body.appendChild(cur);
  const dot = cur.firstChild;

  const CLICKABLE = 'a, button, input, textarea, select, label, [role="button"], .lm_tab, .seg .sg';
  const pt = { x: innerWidth / 2, y: innerHeight / 2 };
  // Кольцо и точка — две пружины: скорость с замедлением на концах (S-образно).
  const ring = { x: pt.x, y: pt.y, vx: 0, vy: 0 };
  const core = { x: pt.x, y: pt.y, vx: 0, vy: 0 };
  let downHit = null;
  let last = performance.now();

  addEventListener('pointermove', (e) => { pt.x = e.clientX; pt.y = e.clientY; }, { passive: true });

  addEventListener('pointerdown', (e) => {
    downHit = e.target.closest?.(CLICKABLE) || null;
    cur.classList.add('is-down');
  }, true);

  addEventListener('pointerup', (e) => {
    const upHit = e.target.closest?.(CLICKABLE) || null;
    cur.classList.remove('is-down');
    // Клик успешен (нажали и отпустили на кликабельном) — кольцо схлопывается.
    if (downHit && upHit && (downHit === upHit || downHit.contains(upHit))) {
      cur.classList.add('is-click');
      setTimeout(() => cur.classList.remove('is-click'), 110);
    }
    downHit = null;
  }, true);

  document.addEventListener('pointerover', (e) => {
    const hit = e.target.closest?.(CLICKABLE);
    if (hit) cur.dataset.mode = 'hover';
    else delete cur.dataset.mode;
  });

  const step = (o, tx, ty, k, damp, dt) => {
    o.vx += (tx - o.x) * k * dt;
    o.vy += (ty - o.y) * k * dt;
    const decay = Math.exp(-damp * dt);
    o.vx *= decay; o.vy *= decay;
    o.x += o.vx * dt; o.y += o.vy * dt;
  };

  (function loop(now) {
    const dt = Math.min(0.033, Math.max(0.001, (now - last) / 1000));
    last = now;
    // Точка — цепкая (k 900), кольцо — мягче и с большим демпфированием (k 420).
    step(core, pt.x, pt.y, 900, 46, dt);
    step(ring, pt.x, pt.y, 420, 30, dt);
    cur.style.transform = `translate(${ring.x.toFixed(2)}px,${ring.y.toFixed(2)}px)`;
    dot.style.transform = `translate(${(core.x - ring.x).toFixed(2)}px,${(core.y - ring.y).toFixed(2)}px)`;
    requestAnimationFrame(loop);
  })(performance.now());
})();

