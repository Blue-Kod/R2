// Переключатель темы для панели R2: ползунок с солнцем и луной, без текста.
// Живёт в шапке (перед часами). Состояние — в localStorage.
//
// Светлые токены — в static/css/theme.css; тема задаётся data-theme на <html>.

const KEY = 'r2-theme';

(function apply() {
  let saved = null;
  try { saved = localStorage.getItem(KEY); } catch { /* ignore */ }
  const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
  document.documentElement.dataset.theme = saved || (prefersLight ? 'light' : 'dark');
})();

const SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.4v2.2M12 19.4v2.2M2.4 12h2.2M19.4 12h2.2M5.2 5.2l1.6 1.6M17.2 17.2l1.6 1.6M18.8 5.2l-1.6 1.6M6.8 17.2l-1.6 1.6"/></svg>';
const MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20.4 14.6A8.6 8.6 0 1 1 9.4 3.6a7 7 0 0 0 11 11z"/></svg>';

function sync(btn) {
  const light = document.documentElement.dataset.theme === 'light';
  btn.setAttribute('aria-checked', String(light));
  btn.title = light ? 'Светлая тема' : 'Тёмная тема';
}

function applyTheme(next) {
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem(KEY, next); } catch { /* ignore */ }
  document.querySelectorAll('.tsw').forEach(sync);
}

function createSwitch() {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'tsw';
  btn.setAttribute('aria-label', 'Тема');
  btn.setAttribute('role', 'switch');
  btn.innerHTML = `<i class="tsw-sun">${SUN}</i><i class="tsw-moon">${MOON}</i><span class="tsw-k"></span>`;
  sync(btn);
  btn.addEventListener('click', () => {
    applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
  });
  return btn;
}

// Вставляем в шапку перед часами (или в конец, если часов нет).
function mount(host) {
  const header = host || document.querySelector('.app-head') || document.querySelector('header');
  if (!header || header.querySelector('.tsw')) return null;
  const btn = createSwitch();
  const clock = header.querySelector('.app-clock');
  if (clock) header.insertBefore(btn, clock);
  else header.appendChild(btn);
  return btn;
}

window.R2Theme = { mount, apply: applyTheme };

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => mount(), { once: true });
} else {
  mount();
}
