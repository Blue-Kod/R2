import { api } from '../api.js';
import { h, panelBody, toast } from '../ui.js';

/* Камера: кадр занимает всю панель, текст лежит поверх на скриме.
 * Всё, что раньше было отдельной строкой под кадром, стало HUD'ом. */
export const videoPanel = {
  id: 'video',
  title: 'Видео',
  icon: 'video',
  group: 'Управление',
  mount(el, ctx) {
    const img = h('img', { class: 'cam-frame', src: '/video_feed', alt: 'Видеопоток камеры' });
    const stage = h('div', { class: 'cam h-full w-full' }, img);
    let showLeft = ctx.boot.camera_params?.show_left !== false;

    /* Переключатель глаза — тот же .seg/.sg, что у IK («сверху / снизу»). */
    const eye = (label, left) => {
      const b = h('button', {
        type: 'button',
        class: 'sg press',
        text: label,
        onclick: async () => {
          showLeft = left;
          paint();
          try { await api.setCamera(left); } catch (err) { toast(err.message, 'error'); }
        },
      });
      return b;
    };
    const left = eye('левый', true);
    const right = eye('правый', false);
    const seg = h('div', { class: 'seg' }, left, right);
    const paint = () => {
      left.classList.toggle('on', showLeft);
      right.classList.toggle('on', !showLeft);
    };
    paint();

    const fps = h('span', { class: 'mono text-foreground/70', text: '— fps' });
    const layout = h('span', { class: 'mono text-foreground/45' });
    const xy = h('span', {
      class: 'num text-[13px] tracking-normal text-foreground/85',
      text: 'X1 — Y1 —',
      title: 'X1 Y1 — доли кадра 0..1: их принимают depth(), coords(), hand_go_to_point(), pick_point(), place_point().',
    });

    const overlay = h('div', { class: 'cam-overlay' },
      h('div', { class: 'cam-hud' }, seg, h('span', { class: 'grow' }), layout),
      h('div', { class: 'cam-hud', style: { marginTop: 'auto' } }, xy, h('span', { class: 'grow' }), fps));
    overlay.classList.add('cam-scrim');
    stage.appendChild(overlay);

    /* Coordinates under the pointer: X1/Y1 are 0..1 fractions of the frame.
       Слушаем сцену, а не картинку: HUD лежит поверх и иначе «съедает» движения
       там, где нарисован текст. */
    const onMove = (e) => {
      const r = img.getBoundingClientRect();
      const nw = img.naturalWidth || 0, nh = img.naturalHeight || 0;
      if (!nw || !nh) return;
      const sc = Math.min(r.width / nw, r.height / nh);
      const dw = nw * sc, dh = nh * sc;
      const ox = r.left + (r.width - dw) / 2, oy = r.top + (r.height - dh) / 2;
      const x1 = (e.clientX - ox) / dw, y1 = (e.clientY - oy) / dh;
      if (x1 < 0 || x1 > 1 || y1 < 0 || y1 > 1) return;
      xy.textContent = `X1 ${x1.toFixed(3)}  Y1 ${y1.toFixed(3)}`;
    };
    const onLeave = () => { xy.textContent = 'X1 — Y1 —'; };
    stage.addEventListener('mousemove', onMove);
    stage.addEventListener('mouseleave', onLeave);

    el.appendChild(stage);

    let alive = true;
    const tick = async () => {
      if (!alive) return;
      try {
        const d = await api.stats();
        fps.textContent = `${(d.fps ?? 0).toFixed(1)} fps`;
      } catch { /* ignore */ }
    };
    (async () => {
      try {
        const p = await api.cameraParams();
        if (p?.layout) {
          const L = p.layout;
          layout.textContent = `${L.frame_w}×${L.frame_h} · ${L.w}px · ${L.fov_h}°`;
        }
      } catch { /* ignore */ }
    })();
    const timer = setInterval(tick, 2000);
    tick();

    return () => {
      alive = false;
      clearInterval(timer);
      stage.removeEventListener('mousemove', onMove);
      stage.removeEventListener('mouseleave', onLeave);
    };
  },
};
