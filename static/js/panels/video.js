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
    let mode = ctx.boot.camera_params?.mode === 'depth' ? 'depth' : 'color';

    /* Цвет / глубина: у Astra одна камера, поэтому вместо глаз — режим кадра. */
    const pick = (label, value) => h('button', {
      type: 'button',
      class: 'sg press',
      text: label,
      onclick: async () => {
        mode = value;
        paint();
        updateLayout();
        placeCursor();
        pollDepth();
        try { await api.setCamera(value); } catch (err) { toast(err.message, 'error'); }
      },
    });
    const color = pick('Цвет', 'color');
    const depth = pick('Глубина', 'depth');
    const seg = h('div', { class: 'seg' }, color, depth);
    const paint = () => {
      color.classList.toggle('on', mode === 'color');
      depth.classList.toggle('on', mode === 'depth');
    };
    paint();

    const fps = h('span', {
      class: 'num text-foreground',
      style: { display: 'inline-block', minWidth: '6ch', textAlign: 'right' },
      text: '— fps',
    });
    const layout = h('span', { class: 'num text-foreground' });
    let camLayout = null;
    const updateLayout = () => {
      if (!camLayout) return;
      layout.textContent = mode === 'depth'
        ? ''
        : `${camLayout.frame_w}×${camLayout.frame_h} · ${camLayout.fov_h}°`;
    };
    const xy = h('span', {
      class: 'num text-[13px] tracking-normal text-foreground',
      text: 'X1 — Y1 —',
      title: 'X1 Y1 — доли кадра 0..1: их принимают depth(), coords(), hand_go_to_point(), pick_point(), place_point().',
    });
    const dist = h('span', {
      class: 'num text-[13px] tracking-normal text-foreground',
      style: { display: 'inline-block', minWidth: '8ch', textAlign: 'right' },
      text: '— мм',
      title: 'Расстояние в точке под курсором.',
    });
    const cursorMm = h('div', {
      class: 'num',
      style: {
        position: 'absolute', left: '0', top: '0', display: 'none',
        zIndex: '4', pointerEvents: 'none', whiteSpace: 'nowrap',
        padding: '2px 6px', borderRadius: '6px',
        background: 'rgba(13,13,13,.72)', color: '#fff', fontSize: '12px',
        transform: 'translate(12px, 12px)',
        minWidth: '8ch', textAlign: 'center',
      },
      text: '— мм',
    });
    stage.appendChild(cursorMm);

    /* Расстояние под курсором обновляется постоянно, пока он над кадром:
       раз в 120 мс читаем последнюю точку, даже если мышь стоит. */
    const setText = (node, text) => {
      if (node.textContent !== text) node.textContent = text;
    };
    const depthText = (d) => {
      if (d.mm == null || d.status === 1) return '— мм';
      if (d.status === 3) return `≤${d.mm} мм`;   // неизвестно, ближняя зона
      if (d.status === 2) return `~${d.mm} мм`;   // неизвестно, оценка
      return `${d.mm} мм`;                        // измерено
    };
    let depthBusy = false, hover = null, lastClient = null;
    const pollDepth = async () => {
      if (!hover || depthBusy) return;
      const p = hover;
      depthBusy = true;
      try {
        const d = await api.depth(p.x, p.y);
        const text = depthText(d);
        setText(dist, text);
        setText(cursorMm, text);
      } catch {
        /* одна неудача — не сбрасываем последнее значение, иначе мигает */
      } finally {
        depthBusy = false;
      }
    };
    const depthPoll = setInterval(pollDepth, 120);
    const placeCursor = () => {
      if (!hover || !lastClient) return;
      const sr = stage.getBoundingClientRect();
      if (cursorMm.style.display !== 'block') cursorMm.style.display = 'block';
      cursorMm.style.left =
        `${Math.max(0, Math.min(lastClient.x - sr.left, sr.width - 90))}px`;
      cursorMm.style.top =
        `${Math.max(0, Math.min(lastClient.y - sr.top, sr.height - 30))}px`;
    };

    const overlay = h('div', { class: 'cam-overlay' },
      h('div', { class: 'cam-hud' }, seg, h('span', { class: 'grow' }), layout),
      h('div', { class: 'cam-hud', style: { marginTop: 'auto' } }, xy, dist, h('span', { class: 'grow' }), fps));
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
      hover = { x: x1, y: y1 };
      lastClient = { x: e.clientX, y: e.clientY };
      placeCursor();
    };
    const onLeave = () => {
      xy.textContent = 'X1 — Y1 —';
      dist.textContent = '— мм';
      hover = null;
      lastClient = null;
      cursorMm.style.display = 'none';
    };
    stage.addEventListener('mousemove', onMove);
    stage.addEventListener('mouseleave', onLeave);

    el.appendChild(stage);

    let alive = true;
    const tick = async () => {
      if (!alive) return;
      try {
        const d = await api.stats();
        setText(fps, `${(d.fps ?? 0).toFixed(1)} fps`);
      } catch { /* ignore */ }
    };
    (async () => {
      try {
        const p = await api.cameraParams();
        if (p?.layout) {
          camLayout = p.layout;
          updateLayout();
        }
      } catch { /* ignore */ }
    })();
    const timer = setInterval(tick, 2000);
    tick();

    return () => {
      alive = false;
      clearInterval(timer);
      clearInterval(depthPoll);
      stage.removeEventListener('mousemove', onMove);
      stage.removeEventListener('mouseleave', onLeave);
    };
  },
};
