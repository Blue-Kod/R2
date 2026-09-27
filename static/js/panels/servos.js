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
      const out = h('span', { class: 'w-10 shrink-0 text-right font-mono text-[11px] tabular-nums text-foreground/90', text: `${val}°` });
      const live = h('span', { class: 'hidden w-16 shrink-0 text-right font-mono text-[10px] tabular-nums text-muted-foreground/70' });
      const range = h('input', {
        type: 'range', min: c.min, max: c.max, value: val,
        class: 'h-1.5 flex-1 cursor-pointer accent-primary',
        title: c.inverted ? `инверсия, offset ${c.offset}°` : (c.offset ? `offset ${c.offset}°` : ''),
      });
      range.addEventListener('input', () => {
        range._touched = Date.now();
        out.textContent = `${range.value}°`;
      });
      range.addEventListener('change', () => {
        range._touched = Date.now();
        api.setServo(c.id, parseInt(range.value, 10)).catch((err) => toast(err.message, 'error'));
        burst();
      });
      sliders.set(c.id, { range, out, live });
      list.appendChild(h('div', { class: 'flex items-center gap-2' },
        h('label', { class: 'w-20 shrink-0 text-[11.5px] text-muted-foreground', text: c.name }),
        range, out, live));
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
        const target = targets[ch];
        const pos = positions[ch];
        // Слайдер = команда-цель. Не перетираем ввод, пока с ним работают.
        const interacting = document.activeElement === e.range
          || (Date.now() - (e.range._touched || 0) < 1200);
        if (target != null && !interacting) {
          e.range.value = target;
          e.out.textContent = `${target}°`;
        }
        // Живой индикатор фактического положения — видно только пока рука едет.
        if (target != null && pos != null && pos !== target) {
          e.live.textContent = `${pos}°`;
          e.live.classList.remove('hidden');
        } else {
          e.live.classList.add('hidden');
        }
      }
    };

    // Короткий быстрый опрос после команды, чтобы индикатор «бежал» за рукой.
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
