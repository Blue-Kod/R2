// R2 workspace shell: topbar, dock, a minimal two-pane mobile layout, and the
// command palette (Ctrl+K) used to open / create / arrange windows.

import { api } from './api.js';
import { h, clear, mark, icon, iconButton, button, modal, toast } from './ui.js';
import { Workspace } from './dock.js';
import { PRESETS } from './layout-presets.js';
import { PANELS } from './panels/index.js';

const root = document.getElementById('app');
const MODE_KEY = 'r2.workspace.mode';
const PANES_KEY = 'r2.mobile.panes';

let boot = {};
let workspace = null;
let mobile = null;

function storedMode() {
  try { return localStorage.getItem(MODE_KEY); } catch { return null; }
}
function setStoredMode(mode) {
  try { localStorage.setItem(MODE_KEY, mode); } catch { /* ignore */ }
}
function resolveMode() {
  const stored = storedMode();
  if (stored === 'desktop' || stored === 'mobile') return stored;
  return window.innerWidth < 760 ? 'mobile' : 'desktop';
}

const PANEL_BY_ID = new Map(PANELS.map((p) => [p.id, p]));

function buildHeader() {
  const mode = resolveMode();
  const modeBtn = iconButton(mode === 'mobile' ? 'monitor' : 'phone', {
    title: mode === 'mobile' ? 'Полный интерфейс' : 'Мобильный режим',
    onClick: () => {
      setStoredMode(mode === 'mobile' ? 'desktop' : 'mobile');
      buildShell(boot);
    },
  });

  return h('header', { 'data-chrome': '', class: 'flex h-12 shrink-0 items-center gap-3 border-b border-border px-3' },
    h('div', { class: 'flex items-center gap-2 text-foreground' },
      mark({ intro: true, className: 'h-5 w-auto' })),
    h('div', { class: 'grow' }),
    h('button', {
      type: 'button',
      class: 'press inline-flex h-7 shrink-0 items-center gap-1.5 rounded-lg border border-border bg-background px-2.5 text-[0.8rem] font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground',
      onclick: openPalette,
    },
      h('span', { class: 'font-mono text-foreground/70', text: '/' }),
      h('span', { text: 'Команды' })),
    modeBtn,
    iconButton('refresh', { title: 'Перезагрузить робота', onClick: doReboot }),
    iconButton('power', { title: 'Выключить', onClick: doShutdown }),
    iconButton('logout', { title: 'Выйти', onClick: doLogout }));
}

function buildShell(data) {
  boot = data;
  if (workspace) { workspace.destroy(); workspace = null; }
  mobile = null;
  root.innerHTML = '';
  root.className = 'flex h-dvh w-full flex-col';

  root.appendChild(buildHeader());

  const mode = resolveMode();
  if (mode === 'mobile') {
    const body = h('div', { class: 'min-h-0 flex-1' });
    root.appendChild(body);
    mobile = new MobilePanes(body, data);
  } else {
    const dock = h('main', { class: 'min-h-0 flex-1 p-2' }, h('div', { id: 'r2-dock', class: 'h-full w-full' }));
    root.appendChild(dock);

    workspace = new Workspace(document.getElementById('r2-dock'));
    workspace.boot = data;
    for (const panel of PANELS) workspace.register(panel);
    workspace.init(PRESETS, 'control');
    const requested = new URLSearchParams(location.search).get('panel');
    if (requested) workspace.open(requested);
  }
}

// --- Mobile: two stacked panes with a panel chooser each ---
class MobilePanes {
  constructor(mountEl, data) {
    this.mountEl = mountEl;
    this.boot = data;
    this.cleanups = [null, null];
    this.hosts = [];
    this.selects = [];

    const saved = this._stored();
    const wrap = h('div', { class: 'flex h-full min-h-0 flex-col gap-2 p-2' });

    for (let i = 0; i < 2; i++) {
      const select = h('select', {
        class: 'h-8 min-w-0 flex-1 rounded-lg border border-border bg-card px-2 text-[13px] text-foreground outline-none focus-visible:border-ring',
      }, ...PANELS.map((p) => h('option', { value: p.id }, p.title)));
      select.value = saved[i];
      select.addEventListener('change', () => this.setSlot(i, select.value, false));

      const host = h('div', { class: 'min-h-0 flex-1 overflow-hidden rounded-lg border border-border bg-card' });
      const pane = h('div', { class: 'flex min-h-0 flex-1 flex-col gap-1.5' },
        h('div', { class: 'flex items-center gap-2' }, icon('panelLeft', 'size-4 text-muted-foreground'), select),
        host);
      wrap.appendChild(pane);
      this.hosts.push(host);
      this.selects.push(select);
    }
    mountEl.appendChild(wrap);
    saved.forEach((id, i) => this.setSlot(i, id, true));
  }

  _stored() {
    try {
      const parsed = JSON.parse(localStorage.getItem(PANES_KEY));
      if (Array.isArray(parsed) && parsed.length === 2) return parsed;
    } catch { /* ignore */ }
    return ['video', 'terminal'];
  }

  setSlot(index, id, silent) {
    const host = this.hosts[index];
    if (!host) return;
    try { this.cleanups[index]?.(); } catch { /* ignore */ }
    this.cleanups[index] = null;
    clear(host);
    const panel = PANEL_BY_ID.get(id);
    if (panel) {
      try { this.cleanups[index] = panel.mount(host, { boot: this.boot, workspace: null }) || null; }
      catch (err) { host.appendChild(h('div', { class: 'p-3 text-sm text-destructive', text: `Ошибка панели: ${err.message}` })); }
    }
    if (!silent) {
      const saved = this._stored();
      saved[index] = id;
      try { localStorage.setItem(PANES_KEY, JSON.stringify(saved)); } catch { /* ignore */ }
    }
  }
}

// --- Command palette: open / create windows, layouts, actions ---
function openPalette() {
  if (!workspace) return; // mobile mode has no dock
  const list = h('div', { class: 'flex flex-col gap-0.5' });
  const commands = [];
  for (const p of PANELS) {
    commands.push({ label: `Новое окно: ${p.title}`, icon: p.icon, run: () => workspace.openNew(p.id) });
  }
  for (const [key, p] of Object.entries(PRESETS)) {
    commands.push({ label: `Раскладка: ${p.label}`, icon: 'sliders', run: () => workspace.loadPreset(key) });
  }
  commands.push({ label: 'Сбросить раскладку', icon: 'refresh', run: () => workspace.reset() });
  commands.push({ label: 'Мобильный режим', icon: 'phone', run: () => { setStoredMode('mobile'); buildShell(boot); } });
  commands.push({ label: 'Полный интерфейс', icon: 'monitor', run: () => { setStoredMode('desktop'); buildShell(boot); } });
  commands.push({ label: 'Перезагрузить робота (reboot)', icon: 'refresh', run: doReboot });
  commands.push({ label: 'Обновить код и перезапустить', icon: 'refresh', run: doUpdate });
  commands.push({ label: 'Выключить робота', icon: 'power', run: doShutdown });
  commands.push({ label: 'Выйти', icon: 'logout', run: doLogout });

  const input = h('input', { type: 'text', placeholder: 'Команда…', class: 'h-10 w-full rounded-lg border border-input bg-transparent px-3 text-sm outline-none focus-visible:border-ring' });
  let filtered = commands;
  let index = 0;

  const render = () => {
    list.innerHTML = '';
    if (!filtered.length) {
      list.appendChild(h('div', { class: 'px-2.5 py-3 text-[13px] text-muted-foreground', text: 'Ничего не найдено' }));
      return;
    }
    filtered.forEach((c, i) => {
      list.appendChild(h('button', {
        type: 'button',
        class: `flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-[13px] transition-colors ${i === index ? 'bg-muted text-foreground' : 'text-muted-foreground hover:bg-muted/60'}`,
        onmouseenter: () => { index = i; render(); },
        onclick: () => { m.close(); c.run(); },
      }, h('span', { class: 'w-4 shrink-0 text-center font-mono text-[13px] text-muted-foreground', text: '/' }), h('span', { text: c.label })));
    });
  };
  const filter = () => {
    const q = input.value.trim().toLowerCase();
    filtered = commands.filter((c) => c.label.toLowerCase().includes(q));
    index = 0; render();
  };
  input.addEventListener('input', filter);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { index = Math.min(index + 1, filtered.length - 1); render(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { index = Math.max(index - 1, 0); render(); e.preventDefault(); }
    else if (e.key === 'Enter') { const c = filtered[index]; if (c) { m.close(); c.run(); } }
  });

  const m = modal({ title: 'Команды', body: h('div', { class: 'flex flex-col gap-2' }, input, h('div', { class: 'max-h-96 overflow-y-auto' }, list)), width: 'max-w-md' });
  render();
  setTimeout(() => input.focus(), 0);
}

// --- Actions ---
function confirmAction(message, confirmLabel, fn) {
  const ok = h('button', { class: 'press inline-flex h-8 items-center rounded-lg bg-destructive/10 px-3 text-sm font-medium text-destructive hover:bg-destructive/20', text: confirmLabel });
  const m = modal({
    title: 'Подтверждение', width: 'max-w-sm',
    body: h('p', { class: 'text-[13px] text-muted-foreground', text: message }),
    footer: [ok],
  });
  ok.addEventListener('click', async () => { ok.disabled = true; m.close(); fn(); });
}

async function doUpdate() {
  confirmAction('Обновить репозиторий и перезапустить R2? Соединение прервётся на несколько секунд.', 'Обновить', async () => {
    try { await api.update(); toast('Перезапуск запущен…', 'success', 8000); }
    catch (err) { toast(err.message, 'error'); }
  });
}

async function doReboot() {
  confirmAction('Перезагрузить робота? Выполнится «sudo reboot».', 'Перезагрузить', async () => {
    try { await api.reboot(); toast('Перезагрузка робота…', 'success', 8000); }
    catch (err) { toast(err.message, 'error'); }
  });
}

async function doShutdown() {
  confirmAction('Выключить робота?', 'Выключить', async () => {
    try { await api.shutdown(); toast('Выключение…', 'success', 8000); }
    catch (err) { toast(err.message, 'error'); }
  });
}

async function doLogout() {
  try { await api.logout(); } catch { /* ignore */ }
  window.location.href = '/login';
}

function renderSplash() {
  root.innerHTML = '';
  root.className = 'grid h-dvh place-items-center';
  root.appendChild(mark({ loading: true, className: 'h-10 w-auto text-foreground' }));
}

function showFatal(err) {
  root.innerHTML = '';
  root.className = 'flex h-dvh items-center justify-center p-6';
  root.appendChild(h('div', { class: 'flex max-w-sm flex-col items-center gap-3 text-center' },
    mark({ className: 'h-8 w-auto text-muted-foreground' }),
    h('h1', { class: 'text-lg font-semibold', text: 'Нет связи с роботом' }),
    h('p', { class: 'text-[13px] text-muted-foreground', text: err?.message || 'Не удалось загрузить состояние.' }),
    button({ label: 'Повторить', iconName: 'refresh', onClick: () => window.location.reload() })));
}

async function main() {
  renderSplash();
  try {
    boot = await api.bootstrap();
  } catch (err) {
    if (err.status === 401) { window.location.href = '/login'; return; }
    showFatal(err);
    return;
  }

  buildShell(boot);

  window.addEventListener('resize', () => {
    if (workspace?.layout) { try { workspace.layout.updateSizeFromContainer(); } catch { /* ignore */ } }
  });
}

window.addEventListener('r2:unauthorized', () => { window.location.href = '/login'; });

// Register the palette shortcut in the capture phase before any panel mounts:
// xterm.js listens on the document in capture too and would otherwise swallow
// Ctrl+K and send it to the shell.
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    e.stopPropagation();
    openPalette();
  }
}, true);

main();
