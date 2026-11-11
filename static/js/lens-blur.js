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
        -webkit-backdrop-filter: blur(5px);
        backdrop-filter: blur(5px);
        box-shadow:
          inset 0 1px 0 rgba(241, 239, 234, 0.14),
          inset 0 0 0 1px rgba(241, 239, 234, 0.10),
          0 6px 18px rgba(0, 0, 0, 0.45);
        transition: opacity 140ms ease;
      }
      #r2-lens.is-shown { opacity: 1; }
      html.has-lens .cur .c2 { opacity: 0 !important; }
    `;
    document.head.appendChild(style);

    let shown = false;
    let hoverEl = null;
    let down = false;
    let scale = 1;          // текущий масштаб (плавно едет к цели)
    let scaleTo = 1;

    const lens = document.createElement('div');
    lens.id = 'r2-lens';
    document.body.appendChild(lens);

    const state = { x: innerWidth / 2, y: innerHeight / 2 };
    const cur = { x: state.x, y: state.y, vx: 0, vy: 0 };

    const place = (x, y) => {
      lens.style.transform = `translate3d(${x - HALF}px,${y - HALF}px,0) scale(${scale.toFixed(3)})`;
    };
    place(cur.x, cur.y);



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
        // масштаб едет к цели плавно (~120мс), без layout — только transform
        scale += (scaleTo - scale) * Math.min(1, dt * 14);
        place(cur.x, cur.y);
      }
      requestAnimationFrame(tick);
    })(performance.now());

    // Зажатие сжимает стекло, наведение на интерактив — чуть увеличивает.
    const setScaleTo = () => {
      scaleTo = down ? 0.62 : (hoverEl ? 1.14 : 1);
    };
    addEventListener('pointerdown', () => { down = true; setScaleTo(); }, true);
    addEventListener('pointerup', () => { down = false; setScaleTo(); }, true);
    addEventListener('pointercancel', () => { down = false; setScaleTo(); }, true);
    setInterval(setScaleTo, 120);

    document.addEventListener('visibilitychange', () => {
      lens.style.opacity = document.hidden ? '0' : '';
    });

    addEventListener('pagehide', () => {
      lens.remove();
      document.documentElement.classList.remove('has-lens');
    }, { once: true });
  }
}
