// Линза-курсор на встроенном `backdrop-filter` (без LiquidGlass и без захвата DOM).
//
// Плюсы: ноль JS-работы на кадр (всё делает композитор GPU), настоящие пиксели
// страницы, нет задержки и нет снимков. Минусы: нет увеличения и преломления —
// только матовое стекло с блюром и подсветкой краёв.
//
// Если нужен эффект LiquidGlass — в base.html подключи glass-lens.js вместо этого.

const SIZE = 40;
const HALF = SIZE / 2;

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const coarse = window.matchMedia('(pointer: coarse)');

if (!reduceMotion.matches && !coarse.matches) {
  start();

  function start() {
    const style = document.createElement('style');
    style.id = 'r2-lens-blur-style';
    style.textContent = `
      #r2-lens {
        position: fixed; left: 0; top: 0; width: ${SIZE}px; height: ${SIZE}px;
        border-radius: 50%; pointer-events: none; z-index: 9997;
        opacity: 0; will-change: transform;
        -webkit-backdrop-filter: blur(7px) saturate(0.75) brightness(1.12) contrast(1.05);
        backdrop-filter: blur(7px) saturate(0.75) brightness(1.12) contrast(1.05);
        box-shadow:
          inset 0 1px 0 rgba(241, 239, 234, 0.14),
          inset 0 0 0 1px rgba(241, 239, 234, 0.10),
          0 6px 18px rgba(0, 0, 0, 0.45);
        transition: opacity 140ms ease, width 140ms ease, height 140ms ease;
      }
      #r2-lens.is-shown { opacity: 1; }
      html.has-lens .cur .c2 { opacity: 0 !important; }
    `;
    document.head.appendChild(style);

    const lens = document.createElement('div');
    lens.id = 'r2-lens';
    document.body.appendChild(lens);

    const state = { x: innerWidth / 2, y: innerHeight / 2 };
    const cur = { x: state.x, y: state.y, vx: 0, vy: 0 };

    const place = (x, y) => {
      lens.style.transform = `translate3d(${x - HALF}px,${y - HALF}px,0)`;
    };
    place(cur.x, cur.y);

    let shown = false;
    let hoverEl = null;

    addEventListener('pointermove', (e) => {
      state.x = e.clientX;
      state.y = e.clientY;
      const el = e.target;
      hoverEl = el && el.closest ? el.closest('a, button, input, textarea, select, .r2-btn, .lm_tab, .seg label') : null;
      if (!shown) {
        shown = true;
        lens.classList.add('is-shown');
        document.documentElement.classList.add('has-lens');
      }
    }, { passive: true });

    let lastT = performance.now();
    (function tick(now) {
      const dt = Math.min(0.033, Math.max(0.001, (now - lastT) / 1000));
      lastT = now;
      if (!document.hidden) {
        cur.vx += (state.x - cur.x) * 420 * dt;
        cur.vy += (state.y - cur.y) * 420 * dt;
        const decay = Math.exp(-30 * dt);
        cur.vx *= decay;
        cur.vy *= decay;
        cur.x += cur.vx * dt;
        cur.y += cur.vy * dt;
        place(cur.x, cur.y);
      }
      requestAnimationFrame(tick);
    })(performance.now());

    // Над интерактивом линза чуть увеличивается — как «оживление» стекла.
    setInterval(() => {
      const want = hoverEl ? 1 : 0;
      if (lens.dataset.hot === String(want)) return;
      lens.dataset.hot = String(want);
      const size = want ? SIZE + 8 : SIZE;
      lens.style.width = `${size}px`;
      lens.style.height = `${size}px`;
      lens.style.marginLeft = `${(SIZE - size) / 2}px`;
      lens.style.marginTop = `${(SIZE - size) / 2}px`;
    }, 120);

    document.addEventListener('visibilitychange', () => {
      lens.style.opacity = document.hidden ? '0' : '';
    });

    addEventListener('pagehide', () => {
      lens.remove();
      document.documentElement.classList.remove('has-lens');
    }, { once: true });
  }
}
