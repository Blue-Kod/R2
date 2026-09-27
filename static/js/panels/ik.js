import { api } from '../api.js';
import { h, icon, button, inputClass, panelBody, label as microLabel } from '../ui.js';

export const ikPanel = {
  id: 'ik',
  title: 'IK',
  icon: 'arm',
  group: 'Управление',
  mount(el, ctx) {
    const baseX = ctx.boot.ik_config?.base_x ?? 115;
    const side = { value: 'right' };
    let xTouched = false;

    const numField = (name, value) => {
      const input = h('input', { type: 'number', step: '1', value, class: inputClass });
      return { input, node: h('label', { class: 'flex flex-col gap-1.5' }, microLabel(name), input) };
    };
    const fx = numField('X, мм', baseX);
    fx.input.addEventListener('input', () => { xTouched = true; });
    const fy = numField('Y, мм', 0);
    const fz = numField('Z, мм', 0);

    // One shared radio name per panel instance (otherwise both look selected).
    const radioGroup = `ik-arm-${Math.random().toString(36).slice(2)}`;
    const armRadio = (text, val) => h('label', { class: 'flex flex-1 cursor-pointer items-center justify-center gap-1.5 rounded-lg border border-border py-1.5 text-[12px] text-muted-foreground has-[:checked]:border-primary/40 has-[:checked]:text-foreground' },
      h('input', {
        type: 'radio', name: radioGroup, checked: side.value === val, class: 'accent-primary',
        onchange: () => {
          side.value = val;
          // If the user hasn't edited X, mirror the default for the new arm.
          if (!xTouched) fx.input.value = String(val === 'left' ? -baseX : baseX);
        },
      }), text);

    const status = h('div', { class: 'min-h-[34px] overflow-y-auto text-[11px] leading-relaxed text-muted-foreground' });
    const btn = button({ label: 'Вычислить IK и переместить', className: 'w-full', onClick: () => run() });

    const run = async () => {
      const x = Number(fx.input.value), y = Number(fy.input.value), z = Number(fz.input.value);
      if (![x, y, z].every(Number.isFinite)) { status.textContent = 'Введите числовые X, Y и Z.'; return; }
      btn.disabled = true;
      try {
        const d = await api.ikMove(x, y, z, side.value === 'left');
        if (d.error) throw new Error(d.error);
        const cmds = d.servo ? Object.entries(d.servo).map(([ch, v]) => `ch${ch}=${v}`).join('  ') : 'углы не определены';
        const dist = d.err_mm ? ` (ближайшая точка ~${Math.round(d.err_mm)} мм)` : '';
        status.textContent = `${d.ok ? '✓ достижима' : '✗ недостижима'}: ${d.message}. ${cmds}${dist}${d.moved ? ' — рука поехала.' : ''}`;
      } catch (err) {
        status.textContent = `Ошибка: ${err.message}`;
      } finally {
        btn.disabled = false;
      }
    };

    el.appendChild(panelBody(
      h('div', { class: 'flex items-center gap-2' }, icon('arm', 'size-4 text-muted-foreground'), h('span', { class: 'text-[13px] font-medium', text: 'Обратная кинематика' })),
      h('div', { class: 'flex gap-2' }, armRadio('ЛЕВАЯ', 'left'), armRadio('ПРАВАЯ', 'right')),
      h('div', { class: 'grid grid-cols-3 gap-2' }, fx.node, fy.node, fz.node),
      btn,
      status,
      h('p', { class: 'text-[11px] leading-relaxed text-muted-foreground/70', text: 'Система координат камеры: X вправо, Y вверх, Z вперёд.' }),
    ));
  },
};
