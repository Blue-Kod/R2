import { api } from '../api.js';
import { h, iconButton, panelBody, toast, icon } from '../ui.js';

/** Channel descriptors come from the server config (servo.servo_config()). */
function channelsFromBoot(boot) {
  const cfg = boot?.servo;
  if (cfg?.channels?.length) {
    return cfg.channels.map((c) => ({
      id: c.id,
      name: c.name,
      min: c.command_min ?? c.min ?? 0,
      max: c.command_max ?? c.max ?? 270,
      offset: c.offset ?? 0,
      inverted: !!c.inverted,
    }));
  }
  // Fallback: derive from the legacy servo_limits map.
  const limits = boot?.servo_limits || {};
  const ids = boot?.servo?.order || Object.keys(limits);
  return (ids.length ? ids : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).map((id) => {
    const lim = limits[id] || [0, 270];
    return { id: Number(id), name: `ch${id}`, min: lim[0], max: lim[1], offset: 0, inverted: false };
  });
}

// Fixed width for up to 4 characters ("270°"), so all rows line up and the
// sliders end up the same size.
const VALUE_CLASS = 'w-11 shrink-0 text-right font-mono text-[11px] tabular-nums';
const SETTLED = 'text-foreground/90';
const MOVING = 'text-muted-foreground'; // промежуточный угол — чуть темнее

export const servosPanel = {
  id: 'servos',
  title: 'Сервоприводы',
  icon: 'sliders',
  group: 'Управление',
  mount(el, ctx) {
    const channels = channelsFromBoot(ctx.boot);
    const angles = ctx.boot?.servo_angles || {};
    const list = h('div', { class: 'flex flex-col gap-3 overflow-y-auto pr-1' });
    const sliders = new Map();

    for (const c of channels) {
      const val = angles[c.id] != null ? angles[c.id] : Math.round((c.min + c.max) / 2);
      const out = h('span', { class: `${VALUE_CLASS} ${SETTLED}`, text: `${val}°` });
      const range = h('input', {
        type: 'range', min: c.min, max: c.max, value: val,
        class: 'h-1.5 flex-1 cursor-pointer accent-primary',
        title: c.inverted ? `инверсия, offset ${c.offset}°` : (c.offset ? `offset ${c.offset}°` : ''),
      });
      range.addEventListener('pointerdown', () => { range._dragging = true; });
      range.addEventListener('input', () => {
        range._dragging = true;
        out.textContent = `${range.value}°`;
        out.className = `${VALUE_CLASS} ${SETTLED}`;
      });
      range.addEventListener('change', () => {
        range._dragging = false;
        api.setServo(c.id, parseInt(range.value, 10)).catch((err) => toast(err.message, 'error'));
        burst();
      });
      range.addEventListener('pointerup', () => { range._dragging = false; });
      sliders.set(c.id, { range, out });
      list.appendChild(h('div', { class: 'flex items-center gap-2' },
        h('label', { class: 'w-20 shrink-0 text-[11.5px] text-muted-foreground', text: c.name }),
        range, out));
    }

    const refresh = async () => {
      let d;
      try {
        d = await api.servoAngles();
      } catch {
        return;
      }
      const targets = d.angles || {};
      const positions = d.positions || {};
      for (const [ch, e] of sliders) {
        if (e.range._dragging) continue; // пока тянешь — показываем введённое
        const target = targets[ch];
        const pos = positions[ch];
        if (target != null) e.range.value = target; // слайдер = команда-цель
        const shown = pos != null ? pos : target;
        if (shown == null) continue;
        const moving = pos != null && target != null && pos !== target;
        e.out.textContent = `${shown}°`;
        // Промежуточный угол показываем вместо обычного, чуть приглушённым.
        e.out.className = `${VALUE_CLASS} ${moving ? MOVING : SETTLED}`;
      }
    };

    // Короткий быстрый опрос после команды, чтобы число «бежало» за рукой.
    let fastTimer = null;
    const burst = () => {
      clearInterval(fastTimer);
      let n = 0;
      fastTimer = setInterval(async () => {
        await refresh();
        if (++n >= 16) { clearInterval(fastTimer); fastTimer = null; }
      }, 250);
    };

    const head = h('div', { class: 'flex items-center gap-2' },
      icon('sliders', 'size-4 text-muted-foreground'),
      h('span', { class: 'text-[13px] font-medium', text: 'Сервоприводы' }),
      h('div', { class: 'grow' }),
      iconButton('refresh', { title: 'Обновить углы', onClick: refresh }));

    el.appendChild(panelBody(head, list));

    let alive = true;
    const timer = setInterval(() => { if (alive) refresh(); }, 4000);
    refresh();
    return () => { alive = false; clearInterval(timer); clearInterval(fastTimer); };
  },
};
