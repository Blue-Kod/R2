import { api } from '../api.js';
import { h, panelBody, toast } from '../ui.js';

// Каналы колёс (см. servo.py): 10 — правое, 11 — левое (физически
// инвертировано в servo.py). 180 = полный вперёд, 0 = полный назад, 90 = стоп.
const WHEEL_R = 10;
const WHEEL_L = 11;
const STOP = 90;

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/** Клавиши/кнопки -> логические углы колёс (дифференциал). */
function wheelAngles(keys) {
  const throttle = (keys.has('w') ? 1 : 0) - (keys.has('s') ? 1 : 0);
  const steer = (keys.has('d') ? 1 : 0) - (keys.has('a') ? 1 : 0);
  return {
    right: STOP + STOP * clamp(throttle - steer, -1, 1),
    left: STOP + STOP * clamp(throttle + steer, -1, 1),
  };
}

/** Аналоговый вектор (мышь) -> логические углы колёс. */
function analogAngles({ x, y }) {
  const throttle = y;
  const steer = x;
  return {
    right: STOP + STOP * clamp(throttle - steer, -1, 1),
    left: STOP + STOP * clamp(throttle + steer, -1, 1),
  };
}

/** true, если фокус в поле ввода/терминале — тогда WASD не перехватываем. */
function isTyping() {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (el.isContentEditable) return true;
  return !!(el.closest && el.closest('.xterm'));
}

const KEY_CODE = { KeyW: 'w', KeyA: 'a', KeyS: 's', KeyD: 'd' };

export const movePanel = {
  id: 'move',
  title: 'Перемещение',
  icon: 'move',
  group: 'Управление',
  mount(el) {
    const keys = new Set();
    const status = h('span', { class: 'num text-[13px] tracking-normal text-foreground/55' });

    const pads = {};
    const ON = ['bg-foreground', 'text-background', 'border-foreground'];
    const paint = () => {
      for (const [k, b] of Object.entries(pads)) ON.forEach((c) => b.classList.toggle(c, keys.has(k)));
    };

    // Вектор движения: точка в центре, тонкая белая линия — куда едем (X = поворот, Y = тяга).
    const NS = 'http://www.w3.org/2000/svg';
    const svgEl = (tag, attrs, ...kids) => {
      const n = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
      n.append(...kids);
      return n;
    };
    const vecLine = svgEl('line', { x1: 0, y1: 0, x2: 0, y2: 0, stroke: 'var(--fg)', 'stroke-width': 1 });
    const vecTip = svgEl('circle', { cx: 0, cy: 0, r: 2.5, fill: 'var(--fg)' });
    const vecSvg = svgEl('svg', { viewBox: '-110 -110 220 220', class: 'absolute inset-0 size-full touch-none' },
      svgEl('circle', { cx: 0, cy: 0, r: 100, fill: 'none', stroke: 'var(--ink-12)' }),
      svgEl('circle', { cx: 0, cy: 0, r: 2, fill: 'var(--fg)' }),
      vecLine, vecTip,
      svgEl('circle', { cx: 0, cy: 0, r: 100, fill: 'transparent', 'data-grab': '',
        style: 'cursor:crosshair;pointer-events:all' }));
    // Вектор можно тянуть мышью: угол = направление, радиус = сила.
    const analog = { x: 0, y: 0, on: false };
    let vx = 0, vy = 0; // текущее сглаженное направление
    let targetX = 0, targetY = 0;
    const drawVec = () => {
      let x, y;
      if (analog.on) { x = analog.x; y = analog.y; }
      else {
        x = (keys.has('d') ? 1 : 0) - (keys.has('a') ? 1 : 0);
        y = (keys.has('w') ? 1 : 0) - (keys.has('s') ? 1 : 0);
      }
      const len = Math.hypot(x, y);
      if (len > 1) { x /= len; y /= len; }
      targetX = x * 100; targetY = -y * 100;
      vecTip.style.opacity = len > 0.02 ? '1' : '0';
    };

    // Плавное сглаживание вектора: линия догоняет направление клавиш (как курсор в ASBDS).
    (function vecLoop() {
      vx += (targetX - vx) * 0.18;
      vy += (targetY - vy) * 0.18;
      vecLine.setAttribute('x2', vx.toFixed(2));
      vecLine.setAttribute('y2', vy.toFixed(2));
      vecTip.setAttribute('cx', vx.toFixed(2));
      vecTip.setAttribute('cy', vy.toFixed(2));
      requestAnimationFrame(vecLoop);
    })();

    // Перетаскивание внутри круга: положение указателя задаёт направление и силу.
    const hit = vecSvg.querySelector('[data-grab]');
    const toLocal = (e) => {
      const r = vecSvg.getBoundingClientRect();
      const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
      const k = 110 / (r.width / 2); // из пикселей в систему viewBox
      return { x: (e.clientX - cx) * k, y: (e.clientY - cy) * k };
    };
    const setFromPointer = (e) => {
      const p = toLocal(e);
      const d = Math.hypot(p.x, p.y);
      const max = 100;
      const k = d > max ? max / d : 1;
      analog.x = (p.x * k) / max;
      analog.y = -(p.y * k) / max;
      analog.on = true;
      apply();
    };
    hit.addEventListener('pointerdown', (e) => { e.preventDefault(); hit.setPointerCapture(e.pointerId); setFromPointer(e); });
    hit.addEventListener('pointermove', (e) => { if (analog.on && e.buttons) setFromPointer(e); });
    hit.addEventListener('pointerup', () => { analog.on = false; analog.x = analog.y = 0; apply(); });
    hit.addEventListener('pointercancel', () => { analog.on = false; analog.x = analog.y = 0; apply(); });

    let last = { right: null, left: null };
    const apply = () => {
      paint(); drawVec(); // подсветка — из набора keys, до проверки углов (W+S, A+D дают те же углы)
      const a = analog.on ? analogAngles(analog) : wheelAngles(keys);
      const r = Math.round(a.right), l = Math.round(a.left);
      if (r === last.right && l === last.left) return;
      last = { right: r, left: l };
      status.textContent = `Л ${Math.round(a.left)}°  П ${Math.round(a.right)}°`;
      api.setServo(WHEEL_R, Math.round(a.right)).catch((err) => toast(err.message, 'error'));
      api.setServo(WHEEL_L, Math.round(a.left)).catch((err) => toast(err.message, 'error'));
    };

    const padBtn = (key, label, cls) => {
      const btn = h('button', {
        type: 'button',
        class: 'press flex size-12 select-none items-center justify-center rounded-[10px] border border-[var(--ink-12)] text-[11px] text-foreground/65 transition-all duration-300 hover:border-[var(--ink-32)] hover:text-foreground ' + cls,
        text: label,
      });
      pads[key] = btn;
      const down = (e) => { e.preventDefault(); keys.add(key); apply(); };
      const up = () => { keys.delete(key); apply(); };
      btn.addEventListener('pointerdown', down);
      btn.addEventListener('pointerup', up);
      btn.addEventListener('pointerleave', up);
      btn.addEventListener('pointercancel', up);
      return btn;
    };

    const cross = h('div', { class: 'grid grid-cols-3 place-items-center gap-2' },
      h('div'), padBtn('w', 'W'), h('div'),
      padBtn('a', 'A'), h('div'), padBtn('d', 'D'),
      h('div'), padBtn('s', 'S'), h('div'));

    el.appendChild(panelBody(
      h('div', { class: 'flex h-7 items-center gap-2' },
        h('span', { class: 'lbl', text: 'wasd' }),
        h('div', { class: 'grow' }),
        status),
      h('div', { class: 'flex min-h-0 flex-1 items-center justify-center' },
        h('div', { class: 'relative size-[min(78%,320px)] aspect-square' }, vecSvg, h('div', { class: 'pointer-events-none absolute inset-0 m-auto size-[176px] [&_button]:pointer-events-auto' }, cross))),
    ));

    const onKey = (e, pressed) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return; // не глотаем Ctrl+W и пр.
      if (isTyping()) return;
      const key = KEY_CODE[e.code];
      if (!key) return;
      e.preventDefault();
      if (pressed) keys.add(key); else keys.delete(key);
      apply();
    };
    const onDown = (e) => onKey(e, true);
    const onUp = (e) => onKey(e, false);
    document.addEventListener('keydown', onDown);
    document.addEventListener('keyup', onUp);

    apply();

    return () => {
      document.removeEventListener('keydown', onDown);
      document.removeEventListener('keyup', onUp);
      keys.clear();
      // На выходе панели гарантированно останавливаем колёса.
      api.setServo(WHEEL_R, STOP).catch(() => {});
      api.setServo(WHEEL_L, STOP).catch(() => {});
    };
  },
};
