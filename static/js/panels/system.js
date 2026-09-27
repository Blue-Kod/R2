import { api } from '../api.js';
import { h, icon, panelBody, label as microLabel } from '../ui.js';

function tile(name, key, unit = '') {
  const value = h('span', { class: 'font-mono text-lg tabular-nums text-foreground', text: '—' });
  const node = h('div', { class: 'flex flex-col gap-1.5 rounded-xl border border-border bg-card/60 p-3' },
    microLabel(name), h('span', {}, value, h('span', { class: 'ml-1 text-[11px] text-muted-foreground', text: unit })));
  return { node, set: (v) => { value.textContent = v ?? '—'; } };
}

export const systemPanel = {
  id: 'system',
  title: 'Система',
  icon: 'cpu',
  group: 'Отладка',
  mount(el) {
    const cpu = tile('CPU', 'cpu', '%');
    const ram = tile('RAM', 'ram', '%');
    const temp = tile('Температура');
    const fps = tile('Сервер', 'fps', 'fps');
    const stream = tile('Стрим', 'stream', 'fps');
    const cam = tile('Камера');

    let alive = true;
    const tick = async () => {
      if (!alive) return;
      try {
        const d = await api.stats();
        cpu.set(Math.round(d.cpu || 0));
        ram.set(Math.round(d.ram || 0));
        temp.set(d.temp || '—');
        fps.set((d.fps || 0).toFixed(1));
        stream.set(d.stream_fps ?? 0);
        cam.set(`${d.cam_w || 0}×${d.cam_h || 0}`);
      } catch { /* ignore */ }
    };

    const ipValue = h('span', { class: 'font-mono text-[13px] text-foreground', text: '—' });
    el.appendChild(panelBody(
      h('div', { class: 'flex items-center gap-2' }, icon('cpu', 'size-4 text-muted-foreground'), h('span', { class: 'text-[13px] font-medium', text: 'Система' })),
      h('div', { class: 'grid grid-cols-2 gap-2' }, cpu.node, ram.node, temp.node, fps.node, stream.node, cam.node),
      h('div', { class: 'flex items-center justify-between rounded-xl border border-border bg-card/60 px-3 py-2' }, microLabel('IP-адрес'), ipValue)));

    (async () => {
      try { const d = await api.ip(); ipValue.textContent = d.ip || '—'; } catch { /* ignore */ }
    })();

    tick();
    const timer = setInterval(tick, 2000);
    return () => { alive = false; clearInterval(timer); };
  },
};
