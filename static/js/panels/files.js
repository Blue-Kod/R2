import { api } from '../api.js';
import { h, icon, iconButton, inputClass, panelBody, toast, modal, emptyState } from '../ui.js';

const IMAGE = new Set(['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'svg', 'ico']);
const AUDIO = new Set(['mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac']);
const VIDEO = new Set(['mp4', 'webm', 'mkv', 'mov', 'avi']);
const CODE_MODE = { py: 'python', pyw: 'python' };

const ext = (name) => (name.split('.').pop() || '').toLowerCase();
const fmtSize = (n) => {
  if (!n) return '0 B';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
};

function promptModal({ title, value = '', placeholder, onOk }) {
  const input = h('input', { type: 'text', value, placeholder, class: inputClass });
  const ok = h('button', { class: 'press inline-flex h-8 items-center rounded-lg bg-primary px-3 text-sm font-medium text-primary-foreground', type: 'button', text: 'OK' });
  const m = modal({
    title,
    width: 'max-w-sm',
    body: h('div', { class: 'flex flex-col gap-2' }, input),
    footer: [ok],
  });
  const submit = () => { const v = input.value.trim(); if (!v) return; m.close(); onOk(v); };
  ok.addEventListener('click', submit);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') submit(); });
  setTimeout(() => input.focus(), 0);
}

export const filesPanel = {
  id: 'files',
  title: 'Файлы',
  icon: 'folder',
  group: 'Файлы',
  mount(el) {
    let cwd = '';
    let entries = [];
    let showHidden = false;
    let editor = null;
    const tabs = [];
    let activePath = null;

    const tree = h('div', { class: 'flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto' });
    const crumb = h('div', { class: 'truncate rounded-md bg-black/30 px-2 py-1 font-mono text-[11px] text-muted-foreground', text: '/' });

    const tabBar = h('div', { class: 'flex items-center gap-1 overflow-x-auto border-b border-border pb-1' });
    const editorHost = h('div', { class: 'min-h-0 flex-1 overflow-hidden' });
    const previewHost = h('div', { class: 'hidden min-h-0 flex-1 items-center justify-center overflow-auto rounded-lg border border-border bg-black/30 p-3' });

    const fileInput = h('input', { type: 'file', multiple: true, class: 'hidden' });

    // --- editor (created lazily) ---
    const ensureEditor = () => {
      if (editor || !window.CodeMirror) return;
      const ta = h('textarea');
      editorHost.appendChild(ta);
      editor = window.CodeMirror.fromTextArea(ta, {
        lineNumbers: true, theme: 'monokai', indentUnit: 4, tabSize: 4,
        matchBrackets: true, autoCloseBrackets: true, viewportMargin: Infinity,
      });
      editor.setSize('100%', '100%');
      editor.on('change', () => {
        const t = tabs.find((x) => x.path === activePath);
        if (t) { t.content = editor.getValue(); t.dirty = t.content !== t.saved; renderTabs(); }
      });
    };

    const activeTab = () => tabs.find((t) => t.path === activePath) || null;

    const showContent = () => {
      const t = activeTab();
      editorHost.classList.add('hidden');
      previewHost.classList.add('hidden');
      previewHost.innerHTML = '';
      if (!t) { previewHost.classList.remove('hidden'); previewHost.appendChild(emptyState('Файл не выбран')); return; }
      if (t.kind === 'text') {
        editorHost.classList.remove('hidden');
        ensureEditor();
        if (editor) { editor.setValue(t.content ?? ''); editor.setOption('mode', CODE_MODE[ext(t.name)] || null); setTimeout(() => editor.refresh(), 0); }
        else { editorHost.appendChild(h('pre', { class: 'h-full overflow-auto p-3 text-[12px]', text: t.content })); }
      } else if (t.kind === 'image') {
        previewHost.classList.remove('hidden');
        previewHost.classList.remove('items-center');
        previewHost.appendChild(h('img', { src: api.fileUrl(t.path), class: 'max-h-full max-w-full object-contain', alt: t.name }));
      } else if (t.kind === 'audio') {
        previewHost.classList.remove('hidden');
        previewHost.appendChild(h('audio', { controls: true, src: api.fileUrl(t.path), class: 'w-full' }));
      } else if (t.kind === 'video') {
        previewHost.classList.remove('hidden');
        previewHost.appendChild(h('video', { controls: true, src: api.fileUrl(t.path), class: 'max-h-full max-w-full' }));
      } else {
        previewHost.classList.remove('hidden');
        previewHost.appendChild(h('div', { class: 'flex flex-col items-center gap-3 text-center' },
          icon('file', 'size-8 text-muted-foreground'),
          h('span', { class: 'text-[13px] text-muted-foreground', text: `${t.name} — ${fmtSize(t.size)}` }),
          h('a', { class: 'press inline-flex h-8 items-center gap-1.5 rounded-lg bg-primary px-3 text-sm font-medium text-primary-foreground', href: api.fileUrl(t.path), text: 'Скачать' })));
      }
    };

    const renderTabs = () => {
      tabBar.innerHTML = '';
      if (!tabs.length) { tabBar.appendChild(h('span', { class: 'px-1 text-[11px] text-muted-foreground/70', text: 'Файлы' })); return; }
      for (const t of tabs) {
        const active = t.path === activePath;
        const tab = h('div', {
          class: `group/tab flex shrink-0 cursor-pointer items-center gap-1.5 rounded-md border px-2 py-1 text-[12px] transition-colors ${active ? 'border-primary/30 bg-primary/10 text-foreground' : 'border-transparent text-muted-foreground hover:bg-muted'}`,
          onclick: () => { activePath = t.path; renderTabs(); showContent(); },
        }, icon(t.kind === 'image' ? 'image' : 'file', 'size-3.5'),
          h('span', { class: 'max-w-[140px] truncate', text: t.name }),
          t.dirty ? h('span', { class: 'text-primary', text: '•' }) : null,
          h('button', {
            class: 'rounded p-0.5 text-muted-foreground opacity-0 group-hover/tab:opacity-100 hover:text-foreground',
            title: 'Закрыть', text: '×',
            onclick: (e) => { e.stopPropagation(); closeTab(t.path); },
          }));
        tabBar.appendChild(tab);
      }
    };

    const closeTab = (path) => {
      const idx = tabs.findIndex((t) => t.path === path);
      if (idx < 0) return;
      tabs.splice(idx, 1);
      if (activePath === path) { activePath = tabs[Math.max(0, idx - 1)]?.path ?? null; }
      renderTabs(); showContent();
    };

    const openEntry = async (entry) => {
      const kind = IMAGE.has(ext(entry.name)) ? 'image'
        : AUDIO.has(ext(entry.name)) ? 'audio'
          : VIDEO.has(ext(entry.name)) ? 'video' : 'text';
      let t = tabs.find((x) => x.path === entry.path);
      if (!t) {
        if (kind === 'text') {
          try {
            const d = await api.fileRead(entry.path);
            t = { path: entry.path, name: entry.name, kind, content: d.content, saved: d.content, size: d.size, dirty: false };
          } catch (err) { toast(err.message, 'error'); return; }
        } else {
          t = { path: entry.path, name: entry.name, kind, size: entry.size };
        }
        tabs.push(t);
      }
      activePath = entry.path;
      renderTabs(); showContent();
    };

    const saveActive = async () => {
      const t = activeTab();
      if (!t || t.kind !== 'text') return;
      try {
        await api.fileWrite(t.path, t.content ?? '');
        t.saved = t.content; t.dirty = false; renderTabs();
        toast('Файл сохранён', 'success');
      } catch (err) { toast(err.message, 'error'); }
    };

    const navigate = async (path) => {
      try {
        const d = await api.filesList(path);
        cwd = d.path;
        entries = d.items || [];
        crumb.textContent = cwd;
        renderTree();
      } catch (err) {
        toast(err.message, 'error');
        if (path) navigate('');
      }
    };

    const renderTree = () => {
      tree.innerHTML = '';
      const visible = entries.filter((e) => showHidden || !e.name.startsWith('.'));
      if (cwd !== '/' && cwd) {
        tree.appendChild(row({ name: '..', path: parentOf(cwd), type: 'directory' }, true));
      }
      if (!visible.length) {
        tree.appendChild(h('div', { class: 'p-3 text-center text-[12px] text-muted-foreground/70', text: 'Пусто' }));
      }
      for (const e of visible) tree.appendChild(row(e));
    };

    const parentOf = (p) => p.replace(/\/[^/]+\/?$/, '') || '/';

    const row = (entry, isUp = false) => {
      const isDir = entry.type === 'directory';
      const node = h('div', {
        class: 'group/row flex cursor-pointer items-center gap-2 rounded-md px-2 py-1 text-[12.5px] hover:bg-muted',
        onclick: () => { if (isDir) navigate(entry.path); else openEntry(entry); },
        oncontextmenu: (e) => { e.preventDefault(); contextMenu(e, entry, isDir); },
      },
        icon(isDir ? 'folder' : IMAGE.has(ext(entry.name)) ? 'image' : 'file', 'size-4 shrink-0 text-muted-foreground'),
        h('span', { class: 'min-w-0 flex-1 truncate', text: isUp ? '..' : entry.name }),
        !isDir ? h('span', { class: 'shrink-0 font-mono text-[10.5px] text-muted-foreground/70', text: fmtSize(entry.size) }) : null);
      return node;
    };

    const contextMenu = (event, entry, isDir) => {
      const items = [];
      if (isDir) items.push(['Открыть', () => navigate(entry.path)]);
      else items.push(['Открыть', () => openEntry(entry)]);
      if (!isDir) items.push(['Скачать', () => window.open(api.fileUrl(entry.path), '_blank')]);
      items.push(['Переименовать', () => promptModal({
        title: 'Переименовать', value: entry.name,
        onOk: async (name) => { try { await api.fileRename(entry.path, name); toast('Переименовано', 'success'); navigate(cwd); } catch (err) { toast(err.message, 'error'); } },
      })]);
      items.push(['Удалить', async () => {
        const m = modal({
          title: 'Удалить?', width: 'max-w-sm',
          body: h('p', { class: 'text-[13px] text-muted-foreground', text: `Удалить «${entry.name}»${isDir ? ' со всем содержимым' : ''}?` }),
          footer: [h('button', { class: 'press h-8 rounded-lg px-3 text-sm text-destructive hover:bg-destructive/10', text: 'Удалить', onclick: async (e) => { e.currentTarget.disabled = true; try { await api.fileDelete(entry.path); m.close(); toast('Удалено', 'success'); navigate(cwd); } catch (err) { toast(err.message, 'error'); } } })],
        });
      }]);

      const menu = h('div', { class: 'anim-zoom fixed z-[95] min-w-40 rounded-lg border border-border bg-popover p-1 shadow-xl' });
      for (const [label, fn] of items) {
        menu.appendChild(h('button', { class: 'block w-full rounded-md px-2.5 py-1.5 text-left text-[12.5px] text-muted-foreground hover:bg-muted hover:text-foreground', text: label, onclick: () => { menu.remove(); fn(); } }));
      }
      document.body.appendChild(menu);
      menu.style.left = `${Math.min(event.clientX, window.innerWidth - 170)}px`;
      menu.style.top = `${Math.min(event.clientY, window.innerHeight - 160)}px`;
      const off = () => { menu.remove(); document.removeEventListener('mousedown', off); };
      setTimeout(() => document.addEventListener('mousedown', off), 0);
    };

    // --- toolbar ---
    const toolbar = h('div', { class: 'flex items-center gap-1' },
      iconButton('chevronRight', { title: 'Вверх', className: 'rotate-[-90deg]', onClick: () => navigate(parentOf(cwd)) }),
      iconButton('refresh', { title: 'Обновить', onClick: () => navigate(cwd) }),
      iconButton('file', { title: 'Новый файл', onClick: () => promptModal({ title: 'Новый файл', placeholder: 'имя.txt', onOk: async (name) => { try { await api.fileCreate(cwd, name, 'file'); navigate(cwd); } catch (err) { toast(err.message, 'error'); } } }) }),
      iconButton('folderPlus', { title: 'Новая папка', onClick: () => promptModal({ title: 'Новая папка', placeholder: 'папка', onOk: async (name) => { try { await api.fileCreate(cwd, name, 'directory'); navigate(cwd); } catch (err) { toast(err.message, 'error'); } } }) }),
      iconButton('upload', { title: 'Загрузить', onClick: () => fileInput.click() }),
      iconButton('folder', { title: 'Скрытые файлы', onClick: (e) => { showHidden = !showHidden; e.currentTarget.classList.toggle('text-foreground', showHidden); renderTree(); } }),
      h('div', { class: 'grow' }),
      iconButton('save', { title: 'Сохранить (Ctrl+S)', onClick: saveActive }));

    fileInput.addEventListener('change', async () => {
      if (!fileInput.files?.length) return;
      try { const d = await api.fileUpload(cwd, fileInput.files); toast(`Загружено: ${d.uploaded_count}`, 'success'); navigate(cwd); }
      catch (err) { toast(err.message, 'error'); }
      fileInput.value = '';
    });

    // --- splitter ---
    const treeCol = h('div', { class: 'flex w-64 shrink-0 flex-col gap-1.5' }, toolbar, crumb, tree);
    const divider = h('div', { class: 'w-1.5 shrink-0 cursor-col-resize rounded bg-transparent hover:bg-primary/30' });
    divider.addEventListener('mousedown', (down) => {
      down.preventDefault();
      const startX = down.clientX;
      const startW = treeCol.getBoundingClientRect().width;
      const move = (e) => { const w = Math.max(160, Math.min(560, startW + e.clientX - startX)); treeCol.style.width = `${w}px`; };
      const up = () => { document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); editor?.refresh(); };
      document.addEventListener('mousemove', move); document.addEventListener('mouseup', up);
    });

    const editorCol = h('div', { class: 'flex min-w-0 flex-1 flex-col gap-1.5' }, tabBar, h('div', { class: 'relative flex min-h-0 flex-1' }, editorHost, previewHost));

    el.appendChild(panelBody(h('div', { class: 'flex min-h-0 flex-1 gap-2' }, treeCol, divider, editorCol), fileInput));

    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') { e.preventDefault(); saveActive(); }
    };
    document.addEventListener('keydown', onKey);
    const onResize = () => editor?.refresh();
    el.addEventListener('r2:resize', onResize);

    renderTabs(); showContent(); navigate('');

    return () => {
      document.removeEventListener('keydown', onKey);
      el.removeEventListener('r2:resize', onResize);
      try { editor?.toTextArea(); } catch { /* ignore */ }
    };
  },
};
