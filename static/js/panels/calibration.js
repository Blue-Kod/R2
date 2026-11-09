import { api } from '../api.js';
import { h, button, panelBody, toast } from '../ui.js';

/** Channel descriptors from the server config (servo.servo_config()). */
function channels(boot) {
  return (boot?.servo?.channels || []).map((c) => ({ id: c.id, name: c.name }));
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

    const status = h('span', { class: 'lbl', text: '' });
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
      status.textContent = '…';
      clearTimeout(saveTimer);
      saveTimer = setTimeout(save, 500);
    };
    const save = async () => {
      collect();
      try {
        const res = await api.setCalibration(offsets, [...inverted]);
        offsets = normalizeOffsets(res.offsets);
        inverted = new Set((res.inverted || []).map(Number));
        status.textContent = 'сохранено';
      } catch (err) {
        status.textContent = 'ошибка';
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
          class: 'h-7 w-16 rounded-full border border-ink-12 bg-[color-mix(in_srgb,var(--fg)_5%,transparent)] px-3 text-center text-[12px] tabular-nums text-foreground outline-none transition-[border-color] duration-300 hover:border-ink-32 focus-visible:border-foreground',
        });
        const inv = h('input', { type: 'checkbox', checked: inverted.has(c.id) });
        offset.addEventListener('change', scheduleSave);
        inv.addEventListener('change', scheduleSave);
        inputs.set(c.id, { offset, inv });
        rows.appendChild(h('div', { class: 'flex items-center gap-2' },
          h('span', { class: 'w-24 shrink-0 truncate text-[11.5px] text-muted-foreground', title: c.name, text: c.name }),
          offset,
          h('span', { class: 'lbl', text: '°' }),
          h('label', { class: 'r2-switch ml-auto text-[11px] text-muted-foreground' }, inv, h('span', { class: 'track' }), h('span', { class: 'lbl', text: 'inv' }))));
      }
    };

    const load = async () => {
      try {
        const d = await api.calibration();
        offsets = normalizeOffsets(d.offsets);
        inverted = new Set((d.inverted || []).map(Number));
        status.textContent = d.connected ? 'шина ок' : 'PCA9685 нет';
      } catch {
        // Fall back to defaults from the config snapshot.
        offsets = {};
        inverted = new Set();
        for (const c of ctx.boot?.servo?.channels || []) {
          offsets[c.id] = Number(c.offset) || 0;
          if (c.inverted) inverted.add(c.id);
        }
        status.textContent = 'нет шины · по умолчанию';
      }
      render();
    };

    const head = h('div', { class: 'flex items-center gap-2' },
      h('span', { class: 'lbl', text: 'offsets' }),
      h('div', { class: 'grow' }),
      status,
      button({
        size: 'xs', variant: 'ghost', iconName: 'refresh', label: 'Переподключить',
        onClick: async () => {
          try {
            const d = await api.reinitServo();
            offsets = normalizeOffsets(d.offsets);
            inverted = new Set((d.inverted || []).map(Number));
            status.textContent = d.connected ? 'шина ок' : 'PCA9685 нет';
            render();
            if (!d.connected) toast('PCA9685 не отвечает по I2C', 'error');
          } catch (err) { toast(err.message, 'error'); }
        },
      }),
      button({
        size: 'xs', variant: 'ghost', label: 'Сброс',
        onClick: async () => {
          try {
            const d = await api.resetCalibration();
            offsets = normalizeOffsets(d.offsets);
            inverted = new Set((d.inverted || []).map(Number));
            status.textContent = 'сброшено';
            render();
          } catch (err) { toast(err.message, 'error'); }
        },
      }));

    el.appendChild(panelBody(
      head,
      h('span', { class: 'lbl', text: 'offset ° · инверсия → config.json' }),
      rows));

    load();
  },
};
