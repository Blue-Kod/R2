import { api } from '../api.js';
import { h, icon, button, spinner, panelBody, toast } from '../ui.js';

const DEFAULT_CODE = `# Пример управления роботом:
print(get_servo_angles())
angle(0, 90)`;

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
    const ta = h('textarea', { text: DEFAULT_CODE });
    const editorWrap = h('div', { class: 'min-h-0 flex-1 overflow-hidden rounded-lg border border-border bg-black/40' }, ta);
    const out = h('div', {
      class: 'h-40 shrink-0 overflow-auto rounded-lg border border-border bg-black/40 p-3 font-mono text-[11.5px] leading-[1.5] whitespace-pre-wrap text-foreground/85',
      'data-selectable': 'true',
    });

    el.appendChild(panelBody(
      h('div', { class: 'flex items-center gap-2' },
        icon('code', 'size-4 text-muted-foreground'),
        h('span', { class: 'text-[13px] font-medium', text: 'Python' }),
        h('div', { class: 'grow' }),
        button({ size: 'sm', label: 'Очистить', variant: 'ghost', onClick: () => { editor.setValue(''); out.textContent = ''; } }),
        button({
          size: 'sm', iconName: 'check', label: 'Запустить',
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
              out.textContent = text || '✓ Успешно';
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
      editorWrap, out));

    const editor = window.CodeMirror.fromTextArea(ta, {
      mode: 'python', theme: 'monokai', lineNumbers: true,
      indentUnit: 4, tabSize: 4, matchBrackets: true, autoCloseBrackets: true,
      viewportMargin: Infinity,
    });
    editor.setSize(null, '100%');
    setTimeout(() => editor.refresh(), 0);

    const onResize = () => editor.refresh();
    el.addEventListener('r2:resize', onResize);

    return () => {
      el.removeEventListener('r2:resize', onResize);
      try { editor.toTextArea(); } catch { /* ignore */ }
    };
  },
};
