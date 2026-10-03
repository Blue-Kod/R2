import { api } from '../api.js';
import { h, icon, iconButton, panelBody } from '../ui.js';

export const logsPanel = {
  id: 'logs',
  title: 'Логи',
  icon: 'list',
  group: 'Отладка',
  mount(el) {
    const out = h('pre', {
      class: 'min-h-0 flex-1 overflow-auto rounded-lg border border-border bg-black/40 p-3 font-mono text-[11.5px] leading-[1.5] whitespace-pre-wrap text-foreground/85',
      'data-selectable': 'true',
    });
    let alive = true;
    let lastLen = 0;

    const tick = async () => {
      if (!alive) return;
      try {
        const d = await api.logs();
        const logs = d.logs || [];
        if (logs.length !== lastLen) { out.textContent = logs.join('\n'); lastLen = logs.length; }
        const nearBottom = out.scrollHeight - out.scrollTop - out.clientHeight < 60;
        if (nearBottom) out.scrollTop = out.scrollHeight;
      } catch { /* ignore */ }
    };

    const head = h('div', { class: 'flex items-center gap-2' },
      icon('list', 'size-4 text-muted-foreground'),
      h('span', { class: 'text-[13px] font-medium', text: 'Логи сервера' }),
      h('div', { class: 'grow' }),
      iconButton('refresh', { title: 'Обновить', onClick: tick }));

    el.appendChild(panelBody(head, out));
    tick();
    const timer = setInterval(tick, 2000);
    return () => { alive = false; clearInterval(timer); };
  },
};
