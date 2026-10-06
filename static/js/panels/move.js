import { api } from '../api.js';
import { h, icon, panelBody, toast } from '../ui.js';

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
    const status = h('span', { class: 'font-mono text-[11px] tabular-nums text-muted-foreground' });

    let last = { right: null, left: null };
    const apply = () => {
      const a = wheelAngles(keys);
      if (a.right === last.right && a.left === last.left) return;
      last = a;
      status.textContent = `Л ${a.left}° · П ${a.right}°`;
      api.setServo(WHEEL_R, a.right).catch((err) => toast(err.message, 'error'));
      api.setServo(WHEEL_L, a.left).catch((err) => toast(err.message, 'error'));
    };

    const padBtn = (key, label, cls) => {
      const btn = h('button', {
        type: 'button',
        class: `press flex h-12 w-12 select-none items-center justify-center rounded-xl border border-border bg-card text-base font-semibold text-foreground hover:bg-muted ${cls}`,
        text: label,
      });
      const down = (e) => { e.preventDefault(); keys.add(key); btn.classList.add('bg-primary/20', 'border-primary/40'); apply(); };
      const up = () => { keys.delete(key); btn.classList.remove('bg-primary/20', 'border-primary/40'); apply(); };
      btn.addEventListener('pointerdown', down);
      btn.addEventListener('pointerup', up);
      btn.addEventListener('pointerleave', up);
      btn.addEventListener('pointercancel', up);
      return btn;
    };

    const cross = h('div', { class: 'grid grid-cols-3 gap-1.5 place-items-center' },
      h('div'), padBtn('w', 'W'), h('div'),
      padBtn('a', 'A'), h('div'), padBtn('d', 'D'),
      h('div'), padBtn('s', 'S'), h('div'));

    el.appendChild(panelBody(
      h('div', { class: 'flex items-center gap-2' },
        icon('move', 'size-4 text-muted-foreground'),
        h('span', { class: 'text-[13px] font-medium', text: 'Перемещение' }),
        h('div', { class: 'grow' }),
        status),
      h('p', { class: 'text-[11px] leading-relaxed text-muted-foreground/70',
        text: 'W/S — вперёд/назад, A/D — поворот. Клавиши ловятся, пока окно открыто (кроме ввода в полях).' }),
      cross));

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
