import { api } from '../api.js';
import { h } from '../ui.js';

export const logsPanel = {
  id: 'logs',
  title: 'Логи',
  icon: 'list',
  group: 'Отладка',
  mount(el) {
    const out = h('pre', {
      class: 'term m-0 h-full min-h-0 flex-1 overflow-auto whitespace-pre-wrap px-2.5 py-2 text-foreground/85',
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

    el.appendChild(out);
    tick();
    const timer = setInterval(tick, 2000);
    return () => { alive = false; clearInterval(timer); };
  },
};
