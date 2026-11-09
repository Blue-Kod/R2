import { api } from '../api.js';
import { h, icon, button, inputClass, panelBody, toast, modal, spinner, label as microLabel } from '../ui.js';

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
    const task = h('input', { type: 'text', placeholder: 'задача', class: inputClass });
    const statusLine = h('div', { class: 'text-[12px] leading-relaxed text-foreground/55' });
    const list = h('div', { class: 'flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto' });

    let collecting = false;
    const toggleBtn = button({ variant: 'primary', label: 'Начать сбор', className: 'w-full', iconName: 'database', onClick: () => toggle() });

    const renderStatus = (s) => {
      collecting = Boolean(s.collecting);
      toggleBtn.textContent = '';
      toggleBtn.appendChild(icon(collecting ? 'check' : 'database', 'size-4'));
      toggleBtn.appendChild(h('span', { text: collecting ? 'Стоп и сохранить' : 'Начать сбор' }));
      toggleBtn.classList.toggle('bg-destructive/10', collecting);
      toggleBtn.classList.toggle('text-destructive', collecting);
      if (collecting) {
        statusLine.innerHTML = `<span class="text-foreground">● ${s.episode_index}</span> · ${s.frames} кадров · всего ${s.episodes_total} · с ${fmtDT(s.episode_start)}`;
      } else {
        statusLine.textContent = `остановлен · эпизодов ${s.episodes_total}`;
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
        list.appendChild(h('div', { class: 'p-3 text-center text-[11px] text-muted-foreground', text: 'эпизодов нет' }));
        return;
      }
      for (const ep of [...datasets].reverse()) {
        list.appendChild(h('div', { class: 'eprow' },
          h('span', { class: 'mono tabular-nums text-foreground/50', text: `#${ep.episode_index}` }),
          h('span', { class: 'min-w-0 truncate text-[12.5px]', text: ep.task || ep.file || '' }),
          h('span', { class: 'flex items-center gap-2' },
            h('span', { class: 'tabular-nums text-foreground/35 tracking-normal', text: `${ep.rows}` }),
            h('button', {
              class: 'r2-btn r2-btn--quiet r2-btn--icon press', title: 'Просмотр', text: '▣',
              onclick: () => viewEpisode(ep.episode_index),
            }),
            h('a', {
              class: 'r2-btn r2-btn--quiet r2-btn--icon press', title: 'Скачать parquet', text: '↓',
              href: api.dcDownloadUrl(ep.episode_index),
            }))));
      }
    };

    const viewEpisode = async (episode) => {
      const box = h('div', { class: 'text-[12px]' }, spinner());
      const m = modal({ title: `Эпизод #${episode}`, body: box, width: 'max-w-2xl' });
      try {
        const d = await api.dcView(episode);
        const p = d.episode?.preview || {};
        box.innerHTML = '';
        box.appendChild(h('div', { class: 'stats mb-3' },
          h('div', { class: 'stat' }, microLabel('Кадров'), h('span', { class: 'num', text: p.rows ?? '—' })),
          h('div', { class: 'stat' }, microLabel('Схема'), h('span', { class: 'num', text: (p.schema || []).length })),
          h('div', { class: 'stat' }, microLabel('Строк'), h('span', { class: 'num', text: (p.rows || 0) }))));
        box.appendChild(h('pre', { class: 'term max-h-72 overflow-auto p-0 text-foreground/70 whitespace-pre-wrap', text: `observation.state[0] = ${JSON.stringify(p.first_state)}\n\nobservation.state[-1] = ${JSON.stringify(p.last_state)}` }));
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
      h('div', { class: 'flex items-center gap-2' }, h('span', { class: 'lbl', text: 'lerobot · parquet' }),
        h('a', { href: '/studio', class: 'press ml-auto rounded-full border border-ink-12 px-3 py-1 text-[11px] text-foreground hover:border-ink-32', title: 'Монтаж датасета', text: 'студия' })),
      h('div', { class: 'flex gap-1.5' }, task),
      toggleBtn, statusLine,
      microLabel('эпизоды'),
      list));

    refresh();
    const timer = setInterval(() => { if (collecting) refresh(); }, 1000);
    return () => clearInterval(timer);
  },
};
