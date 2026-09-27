import { api } from '../api.js';
import { h, icon, button, panelBody, toast } from '../ui.js';

/** Channel descriptors from the server config (servo.servo_config()). */
function channels(boot) {
  const list = boot?.servo?.channels;
  if (list?.length) return list.map((c) => ({ id: c.id, name: c.name }));
  return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((id) => ({ id, name: `ch${id}` }));
}

export const calibrationPanel = {
  id: 'calibration',
  title: 'Калибровка',
  icon: 'gear',
  group: 'Отладка',
  mount(el, ctx) {
    const chans = channels(ctx.boot);
    let offsets = {};
    let inverted = new Set();

    const status = h('span', { class: 'text-[11px] text-muted-foreground' });
    const rows = h('div', { class: 'flex flex-col gap-2 overflow-y-auto pr-1' });
    const inputs = new Map();

    const collect = () => {
      for (const [ch, entry] of inputs) {
        const v = Number(entry.offset.value);
        offsets[ch] = Number.isFinite(v) ? v : 0;
        if (entry.inv.checked) inverted.add(ch); else inverted.delete(ch);
      }
    };

    let saveTimer = null;
    const scheduleSave = () => {
      status.textContent = 'изменения…';
      clearTimeout(saveTimer);
      saveTimer = setTimeout(save, 500);
    };
    const save = async () => {
      collect();
      try {
        const res = await api.setCalibration(offsets, [...inverted]);
        offsets = normalizeOffsets(res.offsets);
        inverted = new Set((res.inverted || []).map(Number));
        status.textContent = 'сохранено в config.json';
      } catch (err) {
        status.textContent = 'ошибка сохранения';
        toast(err.message, 'error');
      }
    };

    const normalizeOffsets = (obj) => {
      const out = {};
      for (const [k, v] of Object.entries(obj || {})) out[Number(k)] = Number(v);
      return out;
    };

    const render = () => {
      rows.innerHTML = '';
      inputs.clear();
      for (const c of chans) {
        const value = Number.isFinite(offsets[c.id]) ? offsets[c.id] : 0;
        const offset = h('input', {
          type: 'number', step: '0.5', value: String(value),
          class: 'h-7 w-20 rounded-md border border-input bg-transparent px-2 text-[12px] tabular-nums outline-none focus-visible:border-ring',
        });
        const inv = h('input', { type: 'checkbox', checked: inverted.has(c.id), class: 'accent-primary' });
        offset.addEventListener('change', scheduleSave);
        inv.addEventListener('change', scheduleSave);
        inputs.set(c.id, { offset, inv });
        rows.appendChild(h('div', { class: 'flex items-center gap-2' },
          h('label', { class: 'w-24 shrink-0 text-[11.5px] text-muted-foreground', text: c.name }),
          offset,
          h('span', { class: 'text-[11px] text-muted-foreground', text: '°' }),
          h('label', { class: 'ml-auto flex items-center gap-1.5 text-[11px] text-muted-foreground' }, inv, 'инверсия')));
      }
    };

    const load = async () => {
      try {
        const d = await api.calibration();
        offsets = normalizeOffsets(d.offsets);
        inverted = new Set((d.inverted || []).map(Number));
        status.textContent = 'offsets → config.json';
      } catch {
        // Fall back to defaults from the config snapshot.
        offsets = {};
        inverted = new Set();
        for (const c of ctx.boot?.servo?.channels || []) {
          offsets[c.id] = Number(c.offset) || 0;
          if (c.inverted) inverted.add(c.id);
        }
        status.textContent = 'серво недоступно (значения по умолчанию)';
      }
      render();
    };

    const head = h('div', { class: 'flex items-center gap-2' },
      icon('gear', 'size-4 text-muted-foreground'),
      h('span', { class: 'text-[13px] font-medium', text: 'Калибровка offsets' }),
      h('div', { class: 'grow' }),
      status,
      button({
        size: 'xs', variant: 'ghost', label: 'Сброс',
        onClick: async () => {
          try {
            const d = await api.resetCalibration();
            offsets = normalizeOffsets(d.offsets);
            inverted = new Set((d.inverted || []).map(Number));
            status.textContent = 'сброшено в config.json';
            render();
          } catch (err) { toast(err.message, 'error'); }
        },
      }));

    el.appendChild(panelBody(
      head,
      h('p', { class: 'text-[11px] leading-relaxed text-muted-foreground/70',
        text: 'Offset (°) добавляется к команде канала; «инверсия» — зеркалит направление. Изменения применяются сразу и сохраняются в config.json.' }),
      rows));

    load();
  },
};
