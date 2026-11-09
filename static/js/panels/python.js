import { api } from '../api.js';
import { h, button, spinner, panelBody, toast } from '../ui.js';

// Код панели переживает перезагрузку страницы и пересоздание панели.
const STORAGE_KEY = 'r2.python.code';

const DEFAULT_CODE = `# Пример управления роботом:
print(get_servo_angles())
angle(0, 90)`;

// Имена для автодополнения: что реально доступно в exec — публичные имена
// robov_core.high_level + встроенные функции и ключевые слова (отдаёт сервер).
let NAMES = null;

async function loadNames() {
  if (NAMES) return;
  try {
    NAMES = (await api.pythonNames()).names || [];
  } catch { /* панель работает и без подсказок */ }
}

// Подсказка по текущему слову. Атрибуты (obj.…) не подсказываем.
function pythonHint(cm) {
  if (!NAMES || !NAMES.length) {
    loadNames();
    return null;
  }
  const cur = cm.getCursor();
  const line = cm.getLine(cur.line);
  let start = cur.ch;
  while (start > 0 && /[A-Za-z0-9_]/.test(line.charAt(start - 1))) start--;
  const word = line.slice(start, cur.ch);
  if (!word || line.charAt(start - 1) === '.') return null;
  const low = word.toLowerCase();
  const list = NAMES
    .filter((n) => n !== word && n.toLowerCase().startsWith(low))
    .sort((a, b) => a.length - b.length || a.localeCompare(b));
  if (!list.length) return null;
  return { list, from: CodeMirror.Pos(cur.line, start), to: CodeMirror.Pos(cur.line, cur.ch) };
}

export const pythonPanel = {
  id: 'python',
  title: 'Python',
  icon: 'code',
  group: 'Отладка',
  mount(el) {
    if (!window.CodeMirror) {
      el.appendChild(h('div', { class: 'p-4 text-sm text-destructive', text: 'CodeMirror не загрузился' }));
      return;
    }
    let restored = null;
    try { restored = localStorage.getItem(STORAGE_KEY); } catch { /* приватный режим */ }

    const ta = h('textarea', { text: restored ?? DEFAULT_CODE });
    // Редактор и вывод — два разных слоя: редактор на фоне панели,
    // вывод — на своей тёмной поверхности с подписью и линией сверху.
    const editorWrap = h('div', { class: 'min-h-0 flex-1 overflow-hidden bg-[var(--bg)]' }, ta);
    const out = h('div', {
      class: 'term h-40 shrink-0 overflow-auto whitespace-pre-wrap bg-[var(--bg-2)] px-3 py-2 text-foreground/80',
      'data-selectable': 'true',
    });
    const outWrap = h('div', { class: 'flex shrink-0 flex-col border-t border-[var(--ink-20)]' },
      h('div', { class: 'flex items-center justify-between px-3 pt-2' },
        h('span', { class: 'lbl', text: 'вывод' })),
      out);

    el.appendChild(panelBody(
      h('div', { class: 'flex items-center gap-2' },
        h('span', { class: 'lbl', text: 'код' }),
        h('div', { class: 'grow' }),
        button({
          size: 'sm', variant: 'primary', label: 'Запустить',
          onClick: async (e) => {
            const btn = e.currentTarget;
            btn.disabled = true;
            const old = btn.querySelector('svg');
            btn.textContent = '';
            btn.appendChild(spinner('size-4'));
            out.textContent = 'Выполнение…';
            try {
              const d = await api.pythonExec(editor.getValue());
              const text = (d.stdout || '') + (d.stderr ? `\n--- ОШИБКИ ---\n${d.stderr}` : '');
              out.textContent = text || '✓ ok';
            } catch (err) {
              out.textContent = `Ошибка: ${err.message}`;
              toast(err.message, 'error');
            } finally {
              btn.disabled = false;
              btn.textContent = '';
              if (old) btn.appendChild(old);
              btn.appendChild(document.createTextNode('Запустить'));
              out.scrollTop = out.scrollHeight;
            }
          },
        })),
      editorWrap, outWrap));

    const editor = window.CodeMirror.fromTextArea(ta, {
      mode: 'python', theme: 'monokai', lineNumbers: true,
      indentUnit: 4, tabSize: 4, matchBrackets: true, autoCloseBrackets: true,
      viewportMargin: Infinity,
      hintOptions: { hint: pythonHint, completeSingle: false, closeOnUnfocus: true },
      extraKeys: { 'Ctrl-Space': () => { loadNames(); editor.showHint(); } },
    });
    editor.setSize(null, '100%');
    setTimeout(() => editor.refresh(), 0);
    loadNames();

    // --- сохранение в localStorage (с задержкой, чтобы не писать на каждый символ)
    let saveTimer = 0;
    const persist = () => {
      try { localStorage.setItem(STORAGE_KEY, editor.getValue()); } catch { /* ignore */ }
    };
    editor.on('change', () => {
      clearTimeout(saveTimer);
      saveTimer = setTimeout(persist, 400);
    });

    // --- автодополнение по начала слова; дальше CodeMirror фильтрует сам
    editor.on('inputRead', (cm, change) => {
      if (cm.state.completionActive) return;
      const typed = change.text.join('');
      if (!/^[A-Za-z_]$/.test(typed)) return;
      const cur = cm.getCursor();
      if (/[A-Za-z0-9_]/.test(cm.getLine(cur.line).charAt(cur.ch - 2))) return;
      loadNames();
      cm.showHint({ hint: pythonHint, completeSingle: false });
    });

    const onResize = () => editor.refresh();
    el.addEventListener('r2:resize', onResize);

    return () => {
      el.removeEventListener('r2:resize', onResize);
      clearTimeout(saveTimer);
      persist();                       // не теряем последние символы при закрытии
      try { editor.toTextArea(); } catch { /* ignore */ }
    };
  },
};
