import { api } from '../api.js';
import { h, iconButton, panelBody, toast, icon } from '../ui.js';

const NAMES = {
  0: 'Шея', 1: 'Пр. плечо', 2: 'Лев. плечо', 3: 'Наклон',
  4: 'Пов. прав.', 5: 'Пов. лев.', 6: 'Пр. локоть', 7: 'Лев. локоть',
  8: 'Пр. захват', 9: 'Лев. захват',
};
const ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

export const servosPanel = {
  id: 'servos',
  title: 'Сервоприводы',
  icon: 'sliders',
  group: 'Управление',
  mount(el, ctx) {
    const limits = ctx.boot.servo_limits || {};
    const angles = ctx.boot.servo_angles || {};
    const list = h('div', { class: 'flex flex-col gap-3.5 overflow-y-auto pr-1' });
    const sliders = new Map();

    for (const ch of ORDER) {
      const lim = limits[ch] || [0, 270];
      const val = angles[ch] != null ? angles[ch] : Math.round((lim[0] + lim[1]) / 2);
      const out = h('span', { class: 'w-11 shrink-0 text-right font-mono text-[11px] tabular-nums text-foreground/90', text: `${val}°` });
      const range = h('input', {
        type: 'range', min: lim[0], max: lim[1], value: val,
        class: 'h-1.5 flex-1 cursor-pointer accent-primary',
      });
      range.addEventListener('input', () => { out.textContent = `${range.value}°`; });
      range.addEventListener('change', () => {
        api.setServo(ch, parseInt(range.value, 10)).catch((err) => toast(err.message, 'error'));
      });
      sliders.set(ch, { range, out });
      list.appendChild(h('div', { class: 'flex items-center gap-2' },
        h('label', { class: 'w-24 shrink-0 text-[11.5px] text-muted-foreground', text: NAMES[ch] }),
        range, out));
    }

    const refresh = async () => {
      try {
        const d = await api.servoAngles();
        for (const [ch, { range, out }] of sliders) {
          if (d.angles?.[ch] != null) { range.value = d.angles[ch]; out.textContent = `${d.angles[ch]}°`; }
        }
      } catch (err) { toast(err.message, 'error'); }
    };

    const head = h('div', { class: 'flex items-center gap-2' },
      icon('sliders', 'size-4 text-muted-foreground'),
      h('span', { class: 'text-[13px] font-medium', text: 'Сервоприводы' }),
      h('div', { class: 'grow' }),
      iconButton('refresh', { title: 'Обновить углы', onClick: refresh }));

    el.appendChild(panelBody(head, list));

    let alive = true;
    const timer = setInterval(() => { if (alive) refresh(); }, 4000);
    return () => { alive = false; clearInterval(timer); };
  },
};
