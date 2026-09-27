// R2 UI kit — vanilla DOM helpers and components in the Negentropy language.

export const EASE = 'cubic-bezier(0.23,1,0.32,1)';

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

export function append(parent, children) {
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
  terminal: '<path d="M5 7l5 5-5 5M13 17h7"/>',
  list: '<path d="M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01"/>',
  code: '<path d="M7 8l-4 4 4 4M17 8l4 4-4 4M14 4l-4 16"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.8.4-1 .9-1 1.7M12 17h.01"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h3l2 2h9a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  file: '<path d="M14 3v5h5M6 2h8l5 5v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.6"/><path d="M4 18l5-5 3 3 3-4 5 6"/>',
  database: '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  smile: '<circle cx="12" cy="12" r="9"/><path d="M8.5 14.5a4 4 0 0 0 7 0M9 9h.01M15 9h.01"/>',
  cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="10" y="10" width="4" height="4"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  x: '<path d="M6 6l12 12M18 6l-12 12"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-2.7 1.1V21a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-2.7-1.1l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.6 1.6 0 0 0 3 15a2 2 0 1 1 0-4 1.6 1.6 0 0 0 1.1-2.7l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1A1.6 1.6 0 0 0 9 4.6a2 2 0 1 1 4 0 1.6 1.6 0 0 0 2.7 1.1l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1A1.6 1.6 0 0 0 20 11a2 2 0 1 1 0 4z"/>',
  power: '<path d="M12 3v9M6.3 6.3a8 8 0 1 0 11.4 0"/>',
  refresh: '<path d="M21 12a9 9 0 1 1-2.6-6.4M21 4v5h-5"/>',
  logout: '<path d="M15 4h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-3M10 17l-5-5 5-5M5 12h11"/>',
  command: '<path d="M9 3a3 3 0 1 0 3 3v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6a3 3 0 1 0-3 3h12"/>',
  chevron: '<path d="M6 9l6 6 6-6"/>',
  chevronRight: '<path d="M9 6l6 6-6 6"/>',
  check: '<path d="M5 12l5 5L20 6"/>',
  alert: '<circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/>',
  download: '<path d="M12 3v12M7 10l5 5 5-5M5 21h14"/>',
  external: '<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>',
  trash: '<path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 13a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-13"/>',
  edit: '<path d="M4 20h4L19 9a2 2 0 0 0-3-3L5 17v3zM14 6l3 3"/>',
  upload: '<path d="M12 20V8M7 13l5-5 5 5M5 4h14"/>',
  folderPlus: '<path d="M3 7a2 2 0 0 1 2-2h3l2 2h9a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2zM12 11v5M9.5 13.5h5"/>',
  save: '<path d="M5 3h11l3 3v13a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zM8 3v6h7V3M8 21v-7h8v7"/>',
  database2: '<path d="M12 3c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3zM4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
  activity: '<path d="M3 12h4l3 8 4-16 3 8h4"/>',
  panelLeft: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9.5 4v16"/>',
  monitor: '<rect x="2" y="4" width="20" height="14" rx="2"/><path d="M8 20h8M12 18v2"/>',
  phone: '<rect x="6" y="2" width="12" height="20" rx="3"/><path d="M11 18h2"/>',
};

export function icon(name, cls = 'size-4', stroke = 1.8) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', String(stroke));
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
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
    path.setAttribute('stroke-linecap', 'round');
    path.setAttribute('stroke-linejoin', 'round');
    if (animate) path.setAttribute('pathLength', '1');
    svg.appendChild(path);
  }
  return svg;
}

// --- Buttons ---
const VARIANTS = {
  default: 'bg-primary text-primary-foreground hover:bg-primary/80',
  outline: 'border border-border bg-background hover:bg-muted hover:text-foreground',
  ghost: 'hover:bg-muted hover:text-foreground',
  destructive: 'bg-destructive/10 text-destructive hover:bg-destructive/20',
  link: 'text-primary underline-offset-4 hover:underline',
};
const SIZES = {
  default: 'h-8 gap-1.5 px-2.5 text-sm',
  sm: 'h-7 gap-1 px-2.5 text-[0.8rem]',
  xs: 'h-6 gap-1 px-2 text-xs',
  icon: 'size-8',
  'icon-sm': 'size-7',
  'icon-xs': 'size-6',
};

export function button({ variant = 'default', size = 'default', label, iconName, className = '', onClick, title, disabled } = {}) {
  const btn = h('button', {
    type: 'button',
    class: `press group/button inline-flex shrink-0 items-center justify-center rounded-lg border border-transparent whitespace-nowrap font-medium transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 ${VARIANTS[variant] || VARIANTS.default} ${SIZES[size] || SIZES.default} ${className}`,
    onclick: onClick,
    title,
    disabled,
  });
  if (iconName) btn.appendChild(icon(iconName, size.startsWith('icon') ? 'size-4' : 'size-4'));
  if (label) btn.appendChild(h('span', { text: label }));
  return btn;
}

export function iconButton(iconName, { onClick, title, size = 'icon-sm', variant = 'ghost', className = '' } = {}) {
  return button({ variant, size, iconName, onClick, title, className });
}

// --- Micro-label ---
export function label(text, className = '') {
  return h('span', { class: `font-semibold text-[10px] uppercase tracking-[0.08em] text-muted-foreground/80 ${className}`, text });
}

export function field(labelText, input, className = '') {
  return h('label', { class: `flex flex-col gap-1.5 ${className}` }, label(labelText), input);
}

export const inputClass =
  'h-8 w-full rounded-lg border border-input bg-transparent px-2.5 py-1 text-sm outline-none transition-[color,box-shadow] placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/30 disabled:opacity-50';
export const textareaClass =
  'w-full rounded-lg border border-input bg-transparent px-2.5 py-2 text-sm outline-none transition-[color,box-shadow] placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/30 resize-none';

export function spinner(className = 'size-4') {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('class', `animate-spin ${className}`);
  svg.innerHTML = '<circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="3" opacity="0.2"/><path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>';
  return svg;
}

export function emptyState(text, className = '') {
  return h('div', { class: `flex h-full items-center justify-center p-6 text-center text-[13px] text-muted-foreground/70 ${className}`, text });
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
  const node = h('div', { class: `anim-in pointer-events-auto flex max-w-sm items-start gap-2 rounded-xl border bg-popover/95 px-3.5 py-2.5 text-[13px] shadow-lg backdrop-blur ${tone}` },
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
  const overlay = h('div', { class: 'anim-fade fixed inset-0 z-[90] flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm' });
  const close = () => { overlay.remove(); document.removeEventListener('keydown', onKey); onClose?.(); };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  const card = h('div', { class: `anim-zoom flex max-h-[85vh] w-full ${width} flex-col overflow-hidden rounded-2xl border border-border bg-popover shadow-2xl` },
    title !== undefined ? h('div', { class: 'flex items-center justify-between border-b border-border px-4 py-3' },
      h('h2', { class: 'text-sm font-semibold tracking-tight', text: title }),
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

/** Auto-scroll a container to the bottom only if the user was already there. */
export function stickyScroll(el, threshold = 40) {
  const near = el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
  return () => { if (near) el.scrollTop = el.scrollHeight; };
}
