import { api } from '../api.js';
import { h, panelBody, label as microLabel } from '../ui.js';

/* Строка .nums из ASBDS: подпись слева, число справа, разделитель — волосок.
   Никаких коробок: значение всегда помещается, обрезать нечего. */
function row(name, unit = '') {
  const value = h('span', { class: 'num', text: '—' });
  const node = h('div', { class: 'stat' },
    microLabel(name),
    h('span', { class: 'flex items-baseline justify-end gap-1' }, value,
      unit ? h('span', { class: 'mono text-foreground/45', text: unit }) : null));
  return { node, set: (v) => { value.textContent = v ?? '—'; } };
}

export const systemPanel = {
  id: 'system',
  title: 'Система',
  icon: 'cpu',
  group: 'Отладка',
  mount(el) {
    const cpu = row('CPU', '%');
    const ram = row('RAM', '%');
    const temp = row('Температура', '°C');
    const fps = row('Сервер', 'fps');
    const stream = row('Стрим', 'fps');
    const cam = row('Камера');

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

    const ipValue = h('span', { class: 'num', text: '—' });

    // Обновление: лаунчер качает его в фоне, здесь только кнопка «применить».
    const updText = h('span', { class: 'mono tabular-nums text-foreground/50 normal-case tracking-normal', text: '' });
    const updBtn = h('button', {
      type: 'button', class: 'r2-btn press hidden', text: 'Обновить',
    });
    const updRow = h('div', { class: 'stat' },
      microLabel('Обновление'), h('div', { class: 'flex items-center gap-2' }, updText, updBtn));
    updBtn.addEventListener('click', async () => {
      updBtn.disabled = true;
      try {
        const d = await api.applyUpdate();
        updText.textContent = d.message || 'обновление принято';
      } catch (err) {
        updText.textContent = `ошибка: ${err.message}`;
        updBtn.disabled = false;
      }
    });
    const checkUpdate = async () => {
      try {
        const d = await api.updateStatus();
        updBtn.classList.toggle('hidden', !d.available);
        if (d.requested) updText.textContent = 'перезапуск...';
        else if (d.available) updText.textContent = `${(d.local || '').slice(0, 8)} → ${(d.remote || '').slice(0, 8)} · ${d.files || 0} файлов`;
        else updText.textContent = 'актуально';
      } catch { /* ignore */ }
    };
    el.appendChild(panelBody(
      h('div', { class: 'stats' }, cpu.node, ram.node, temp.node, fps.node, stream.node, cam.node),
      h('div', { class: 'stat' }, microLabel('IP'), ipValue),
      updRow));

    (async () => {
      try { const d = await api.ip(); ipValue.textContent = d.ip || '—'; } catch { /* ignore */ }
    })();

    tick();
    checkUpdate();
    const timer = setInterval(tick, 2000);
    const updTimer = setInterval(checkUpdate, 10000);
    return () => { alive = false; clearInterval(timer); clearInterval(updTimer); };
  },
};
