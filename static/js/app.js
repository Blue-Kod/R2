// R2 workspace shell: topbar, dock, a minimal two-pane mobile layout, and the
// command palette (Ctrl+K) used to open / create / arrange windows.

import { api } from './api.js';
import { h, clear, mark, icon, button, modal, toast, inputClass } from './ui.js';
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

  /* Навигация — только значки: в шапке ASBDS текст занимает место,
     а в рабочем приложении его важнее отдать сетке. */
  const action = (name, title, onClick) => {
    const b = h('button', {
      type: 'button',
      class: 'r2-bloom text-foreground/60 transition-colors hover:text-foreground',
      title,
      'aria-label': title,
      onclick: onClick,
    });
    b.appendChild(icon(name, 'size-[18px]', 1.2));
    return b;
  };

  const clock = h('span', { class: 'app-clock' });
  const tick = () => {
    clock.textContent = `MSK ${new Date().toLocaleTimeString('ru-RU', { hour12: false })}`;
  };
  tick();
  setInterval(tick, 1000);

  const logo = h('button', {
    type: 'button',
    class: 'app-logo',
    title: 'R2 — рабочее пространство',
    onclick: () => buildShell(boot),
  }, h('span', { class: 'brand r2-bloom', text: 'R2' }));

  return h('header', { 'data-chrome': '', class: 'head app-head' },
    logo,
    h('nav', { class: 'nav' },
      action('sight', 'Команды (Ctrl+K)', openPalette),
      action(mode === 'mobile' ? 'monitor' : 'phone', mode === 'mobile' ? 'Полный экран' : 'Мобильный вид', () => {
        setStoredMode(mode === 'mobile' ? 'desktop' : 'mobile');
        buildShell(boot);
      }),
      action('refresh', 'Перезагрузить', doReboot),
      action('power', 'Выключить', doShutdown),
      action('logout', 'Выйти', doLogout)),
    clock);
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
    /* Шапка fixed — контейнер дока начинается под ней. */
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
        class: 'r2-select h-8 min-w-0 flex-1',
      }, ...PANELS.map((p) => h('option', { value: p.id }, p.title)));
      select.value = saved[i];
      select.addEventListener('change', () => this.setSlot(i, select.value, false));

      const host = h('div', { class: 'min-h-0 flex-1 overflow-hidden rounded-[22px] border border-ink-12 bg-card' });
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
    commands.push({ label: p.title, hint: 'открыть', icon: p.icon, run: () => workspace.openNew(p.id) });
  }
  for (const [key, p] of Object.entries(PRESETS)) {
    commands.push({ label: p.label, hint: 'раскладка', icon: 'sliders', run: () => workspace.loadPreset(key) });
  }
  commands.push({ label: 'Сбросить раскладку', icon: 'refresh', run: () => workspace.reset() });
  commands.push({ label: 'Мобильный режим', icon: 'phone', run: () => { setStoredMode('mobile'); buildShell(boot); } });
  commands.push({ label: 'Полный интерфейс', icon: 'monitor', run: () => { setStoredMode('desktop'); buildShell(boot); } });
  commands.push({ label: 'Обновить код и перезапустить', icon: 'refresh', run: doUpdate });
  commands.push({ label: 'Перезагрузить', icon: 'refresh', run: doReboot });
  commands.push({ label: 'Выключить', icon: 'power', run: doShutdown });
  commands.push({ label: 'Выйти', icon: 'logout', run: doLogout });

  const input = h('input', { type: 'text', placeholder: 'Команда…', class: `${inputClass} h-9` });
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
        class: `flex w-full items-center gap-2 rounded-full px-3 py-1.5 text-left text-[12.5px] transition-colors ${i === index ? 'bg-muted text-foreground' : 'text-muted-foreground hover:bg-muted/60'}`,
        onmouseenter: () => { index = i; render(); },
        onclick: () => { m.close(); c.run(); },
      }, h('span', { class: 'w-4 shrink-0', text: '' }), icon(c.icon, 'size-3.5 shrink-0'), h('span', { class: 'truncate', text: c.label }),
        c.hint ? h('span', { class: 'lbl ml-auto shrink-0', text: c.hint }) : null));
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

  render();
  const m = modal({ title: 'Команды', body: h('div', { class: 'flex flex-col gap-2' }, input, h('div', { class: 'max-h-96 overflow-y-auto' }, list)), width: 'max-w-md' });
  requestAnimationFrame(() => input.focus());
}

// --- Actions ---
function confirmAction(message, confirmLabel, fn) {
  /* Красный — только аварийные действия (reboot, shutdown). */
  const ok = button({ label: confirmLabel, variant: 'destructive', size: 'default' });
  const m = modal({
    title: 'Подтверждение', width: 'max-w-sm',
    body: h('p', { class: 'text-[12.5px] text-muted-foreground', text: message }),
    footer: [ok],
  });
  ok.addEventListener('click', async () => { ok.disabled = true; m.close(); fn(); });
}

async function doUpdate() {
  confirmAction('Обновить репозиторий и перезапустить? Связь прервётся.', 'Обновить', async () => {
    try { await api.update(); toast('Перезапуск…', 'success', 8000); }
    catch (err) { toast(err.message, 'error'); }
  });
}

async function doReboot() {
  confirmAction('Перезагрузить робота (sudo reboot)?', 'Перезагрузить', async () => {
    try { await api.reboot(); toast('Перезагрузка…', 'success', 8000); }
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
    h('h1', { class: 'display text-[15px]', text: 'Нет связи' }),
    h('p', { class: 'text-[12px] text-muted-foreground', text: err?.message || 'Робот не отвечает.' }),
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
