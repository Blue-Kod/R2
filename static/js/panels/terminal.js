import { h, icon, iconButton, panelBody } from '../ui.js';

// Full interactive terminal: xterm.js over a WebSocket to a local pty bash.
const XTERM_THEME = {
  background: '#0a0a0a',
  foreground: '#e6e6e6',
  cursor: '#e6e6e6',
  cursorAccent: '#0a0a0a',
  selectionBackground: 'rgba(255,255,255,0.22)',
  black: '#3b3b3b', red: '#e06c75', green: '#98c379', yellow: '#d19a66',
  blue: '#61afef', magenta: '#c678dd', cyan: '#56b6c2', white: '#e6e6e6',
  brightBlack: '#6b6b6b', brightRed: '#ff6b6b', brightGreen: '#5cb85c',
  brightYellow: '#f0c040', brightBlue: '#5b9bd5', brightMagenta: '#b080c0',
  brightCyan: '#50c0c0', brightWhite: '#ffffff',
};

export const terminalPanel = {
  id: 'terminal',
  title: 'Терминал',
  icon: 'terminal',
  group: 'Отладка',
  mount(el) {
    if (!window.Terminal) {
      el.appendChild(h('div', { class: 'p-4 text-sm text-destructive', text: 'xterm.js не загрузился' }));
      return;
    }

    const host = h('div', { class: 'min-h-0 flex-1 overflow-hidden rounded-lg border border-border bg-black/50 p-1.5' });

    const term = new window.Terminal({
      fontFamily: "'JetBrains Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
      fontSize: 12.5,
      lineHeight: 1.25,
      cursorBlink: true,
      convertEol: false,
      scrollback: 5000,
      allowProposedApi: true,
      theme: XTERM_THEME,
    });
    const fit = new window.FitAddon.FitAddon();
    term.loadAddon(fit);
    if (window.WebLinksAddon?.WebLinksAddon) {
      try { term.loadAddon(new window.WebLinksAddon.WebLinksAddon()); } catch { /* ignore */ }
    }
    term.open(host);

    let ws = null;
    let disposed = false;
    let retry = null;

    const send = (obj) => {
      try { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj)); } catch { /* ignore */ }
    };

    const connect = () => {
      if (disposed) return;
      const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
      ws = new WebSocket(`${proto}${location.host}/ws/terminal`);
      ws.onopen = () => {
        fit.fit();
        send({ type: 'resize', cols: term.cols, rows: term.rows });
        term.focus();
      };
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'output') term.write(msg.data);
        } catch { /* ignore */ }
      };
      ws.onclose = () => {
        if (!disposed) { retry = setTimeout(connect, 1200); }
      };
      ws.onerror = () => { try { ws.close(); } catch { /* ignore */ } };
    };

    term.onData((data) => send({ type: 'input', data }));
    term.onResize(({ cols, rows }) => send({ type: 'resize', cols, rows }));

    const doFit = () => { try { fit.fit(); } catch { /* hidden/zero size */ } };
    const ro = new ResizeObserver(doFit);
    ro.observe(host);
    el.addEventListener('r2:resize', doFit);
    host.addEventListener('mousedown', () => setTimeout(() => term.focus(), 0));

    const head = h('div', { class: 'flex items-center gap-2' },
      icon('terminal', 'size-4 text-muted-foreground'),
      h('span', { class: 'text-[13px] font-medium', text: 'Терминал' }),
      h('div', { class: 'grow' }),
      iconButton('refresh', {
        title: 'Переподключить',
        onClick: () => {
          clearTimeout(retry);
          try { ws && ws.close(); } catch { /* ignore */ }
          disposed = false;
          connect();
        },
      }));

    el.appendChild(panelBody(head, host));
    connect();

    return () => {
      disposed = true;
      clearTimeout(retry);
      ro.disconnect();
      el.removeEventListener('r2:resize', doFit);
      try { ws && ws.close(); } catch { /* ignore */ }
      try { term.dispose(); } catch { /* ignore */ }
    };
  },
};
