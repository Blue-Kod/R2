import { api } from '../api.js';
import { h, button, inputClass, panelBody, label as microLabel } from '../ui.js';

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
    const armRadio = (text, val) => h('label', { class: 'sg press cursor-pointer' },
      h('input', {
        type: 'radio', name: radioGroup, checked: side.value === val, class: 'accent-primary',
        onchange: () => {
          side.value = val;
          // If the user hasn't edited X, mirror the default for the new arm.
          if (!xTouched) fx.input.value = String(val === 'left' ? -baseX : baseX);
        },
      }), text);

    // Ветка IK: сверху (локоть высоко) или снизу (натуральная).
    const branchName = `ik-branch-${Math.random().toString(36).slice(2)}`;
    const branchRadio = (text, val) => h('label', { class: 'sg press cursor-pointer' },
      h('input', {
        type: 'radio', name: branchName, value: val, class: 'accent-primary',
        checked: (ctx.boot.ik_config?.branch ?? 'top') === val,
        onchange: () => api.setIkBranch(val).catch((err) => { status.textContent = `Ветка: ${err.message}`; }),
      }), text);

    const status = h('div', { class: 'min-h-[34px] overflow-y-auto text-[12px] leading-relaxed text-muted-foreground' });
    const btn = button({ variant: 'primary', label: 'Переместить', className: 'self-start mt-1', onClick: () => run() });

    const run = async () => {
      const x = Number(fx.input.value), y = Number(fy.input.value), z = Number(fz.input.value);
      if (![x, y, z].every(Number.isFinite)) { status.textContent = 'X, Y, Z — числа'; return; }
      btn.disabled = true;
      try {
        const d = await api.ikMove(x, y, z, side.value === 'left');
        if (d.error) throw new Error(d.error);
        const cmds = d.servo ? Object.entries(d.servo).map(([ch, v]) => `ch${ch}=${v}`).join('  ') : 'углы не определены';
        const dist = d.err_mm ? ` ~${Math.round(d.err_mm)} мм` : '';
        status.textContent = `${d.ok ? '✓ достижима' : '✗ недостижима'}: ${d.message}  ${cmds}${dist}${d.moved ? '  → рука поехала' : ''}`;
      } catch (err) {
        status.textContent = `Ошибка: ${err.message}`;
      } finally {
        btn.disabled = false;
      }
    };

    el.appendChild(panelBody(
      h('div', { class: 'flex flex-wrap items-center gap-2' },
        h('div', { class: 'seg self-start' }, armRadio('левая', 'left'), armRadio('правая', 'right')),
        h('div', { class: 'seg self-start' }, branchRadio('сверху', 'top'), branchRadio('снизу', 'bottom'))),
      h('div', { class: 'grid grid-cols-3 gap-3' }, fx.node, fy.node, fz.node),

      btn,
      status,
      h('span', { class: 'lbl', text: 'X → вправо · Y ↑ вверх · Z ↗ вперёд' }),
    ));
  },
};
