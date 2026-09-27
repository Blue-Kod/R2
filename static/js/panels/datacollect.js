import { api } from '../api.js';
import { h, icon, button, inputClass, panelBody, toast, modal, spinner } from '../ui.js';

const fmtDT = (iso) => {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const p = (n) => String(n).padStart(2, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())} ${p(d.getDate())}.${p(d.getMonth() + 1)}.${d.getFullYear()}`;
};

export const datacollectPanel = {
  id: 'datacollect',
  title: 'Сбор данных',
  icon: 'database',
  group: 'Данные',
  mount(el) {
    const task = h('input', { type: 'text', placeholder: 'Название задачи (напр. «сложить кубики»)', class: inputClass });
    const statusLine = h('div', { class: 'rounded-lg border border-border bg-card/60 p-3 text-[12px] leading-relaxed text-muted-foreground' });
    const list = h('div', { class: 'flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto' });

    let collecting = false;
    const toggleBtn = button({ label: 'Начать сбор', className: 'w-full', iconName: 'database', onClick: () => toggle() });

    const renderStatus = (s) => {
      collecting = Boolean(s.collecting);
      toggleBtn.textContent = '';
      toggleBtn.appendChild(icon(collecting ? 'check' : 'database', 'size-4'));
      toggleBtn.appendChild(h('span', { text: collecting ? 'Остановить и сохранить' : 'Начать сбор' }));
      toggleBtn.classList.toggle('bg-destructive/10', collecting);
      toggleBtn.classList.toggle('text-destructive', collecting);
      if (collecting) {
        statusLine.innerHTML = `● Сбор идёт — эпизод <b class="text-foreground">${s.episode_index}</b>, кадров <b class="text-foreground">${s.frames}</b><br>Начат: ${fmtDT(s.episode_start)} · всего эпизодов: ${s.episodes_total}`;
      } else {
        statusLine.textContent = `Сбор остановлен. Эпизодов собрано: ${s.episodes_total}. next_episode=${s.next_episode}`;
      }
    };

    const refresh = async () => {
      try {
        const [s, ds] = await Promise.all([api.dcStatus(), api.dcDatasets()]);
        renderStatus(s);
        renderList(ds.datasets || []);
      } catch (err) { /* ignore */ }
    };

    const renderList = (datasets) => {
      list.innerHTML = '';
      if (!datasets.length) {
        list.appendChild(h('div', { class: 'p-3 text-center text-[12px] text-muted-foreground/70', text: 'Пока нет эпизодов' }));
        return;
      }
      for (const ep of [...datasets].reverse()) {
        list.appendChild(h('div', { class: 'flex items-center gap-2 rounded-lg border border-border bg-card/50 px-2.5 py-2' },
          h('span', { class: 'font-mono text-[11px] text-primary', text: `#${ep.episode_index}` }),
          h('span', { class: 'min-w-0 flex-1 truncate text-[12px]', text: ep.task || ep.file || '' }),
          h('span', { class: 'font-mono text-[11px] tabular-nums text-muted-foreground', text: `${ep.rows} к.` }),
          h('button', {
            class: 'rounded-md px-1.5 py-1 text-[12px] text-muted-foreground hover:text-foreground hover:bg-muted',
            title: 'Просмотр', text: 'просмотр',
            onclick: () => viewEpisode(ep.episode_index),
          }),
          h('a', {
            class: 'rounded-md px-1.5 py-1 text-[12px] text-muted-foreground hover:text-foreground hover:bg-muted',
            title: 'Скачать parquet', text: 'скачать', href: api.dcDownloadUrl(ep.episode_index),
          })));
      }
    };

    const viewEpisode = async (episode) => {
      const box = h('div', { class: 'text-[12px]' }, spinner());
      const m = modal({ title: `Эпизод #${episode}`, body: box, width: 'max-w-2xl' });
      try {
        const d = await api.dcView(episode);
        const p = d.episode?.preview || {};
        box.innerHTML = '';
        box.appendChild(h('div', { class: 'mb-3 grid grid-cols-2 gap-2 text-[12px]' },
          h('div', { class: 'rounded-lg border border-border p-2' }, h('div', { class: 'text-muted-foreground', text: 'Кадров' }), h('div', { class: 'font-mono text-foreground', text: p.rows ?? '—' })),
          h('div', { class: 'rounded-lg border border-border p-2' }, h('div', { class: 'text-muted-foreground', text: 'Схема' }), h('div', { class: 'font-mono text-foreground', text: (p.schema || []).length + ' колонок' }))));
        box.appendChild(h('pre', { class: 'max-h-72 overflow-auto rounded-lg border border-border bg-black/40 p-3 font-mono text-[11px] whitespace-pre-wrap', text: `observation.state[0] = ${JSON.stringify(p.first_state)}\n\nobservation.state[-1] = ${JSON.stringify(p.last_state)}` }));
      } catch (err) {
        box.textContent = `Ошибка: ${err.message}`;
      }
    };

    const toggle = async () => {
      try {
        if (collecting) {
          const d = await api.dcStop();
          toast(`Эпизод #${d.episode?.episode_index} сохранён (${d.episode?.rows} кадров)`, 'success');
        } else {
          await api.dcStart(task.value);
          toast('Сбор данных начат', 'success');
        }
        await refresh();
      } catch (err) { toast(err.message, 'error'); }
    };

    el.appendChild(panelBody(
      h('div', { class: 'flex items-center gap-2' }, icon('database', 'size-4 text-muted-foreground'), h('span', { class: 'text-[13px] font-medium', text: 'Сбор данных (LeRobot / parquet)' })),
      task, toggleBtn, statusLine,
      h('div', { class: 'text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground/80', text: 'Эпизоды' }),
      list));

    refresh();
    const timer = setInterval(() => { if (collecting) refresh(); }, 1000);
    return () => clearInterval(timer);
  },
};
