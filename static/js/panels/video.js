import { api } from '../api.js';
import { h, iconButton, panelBody, toast, modal, icon } from '../ui.js';

export const videoPanel = {
  id: 'video',
  title: 'Видео',
  icon: 'video',
  group: 'Управление',
  mount(el, ctx) {
    const uid = Math.random().toString(36).slice(2);
    const img = h('img', { class: 'h-full w-full object-contain', src: '/video_feed', alt: 'Видеопоток камеры' });
    const stage = h('div', { class: 'relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-lg bg-black' }, img);

    let showLeft = ctx.boot.camera_params?.show_left !== false;
    const eyeRadio = (label, left) => h('label', { class: 'flex cursor-pointer items-center gap-1.5 text-[12px] text-muted-foreground' },
      h('input', {
        type: 'radio', name: `eye-${uid}`, checked: showLeft === left,
        class: 'accent-primary',
        onchange: async () => {
          showLeft = left;
          onEye();
          try { await api.setCamera(left); } catch (err) { toast(err.message, 'error'); }
        },
      }), label);

    const fpsTag = h('span', { class: 'font-mono text-[11px] tabular-nums text-muted-foreground' });
    const layoutTag = h('span', { class: 'text-[11px] text-muted-foreground/70' });

    const debugBtn = iconButton('image', {
      title: 'Диагностика кропа глаз',
      onClick: () => {
        const body = h('div', { class: 'flex flex-col gap-2' },
          h('img', { class: 'w-full rounded-lg border border-border', src: `/api/camera/debug?t=${Date.now()}` }),
          h('p', { class: 'text-[12px] text-muted-foreground', text: 'Зелёные рамки — что камера считает левым/правым глазом.' }));
        modal({ title: 'Кроп глаз', body, width: 'max-w-3xl' });
      },
    });

    // Координаты под кадром: X1 Y1 — доли 0..1 (как у depth/coords в high_level),
    // X Y — пиксели самого кадра.
    const eyeName = () => (showLeft ? 'левый' : 'правый');
    const xyTag = h('span', {
      class: 'font-mono text-[11px] tabular-nums text-muted-foreground',
      text: 'X1 — Y1 —   ·   X — Y — px',
      title: 'X1 Y1 — доли кадра 0..1: ровно их принимают depth(), coords(), '
        + 'hand_go_to_point(), pick_point(), place_point(). Отсчёт — в том глазу, '
        + 'который сейчас показывается.',
    });
    const readout = h('div', { class: 'flex items-center px-1 pt-1' }, xyTag);
    const onMove = (e) => {
      const r = img.getBoundingClientRect();
      const nw = img.naturalWidth || 0, nh = img.naturalHeight || 0;
      if (!nw || !nh) return;
      // object-contain: кадр вписан и отцентрован, края — поля
      const sc = Math.min(r.width / nw, r.height / nh);
      const dw = nw * sc, dh = nh * sc;
      const ox = r.left + (r.width - dw) / 2, oy = r.top + (r.height - dh) / 2;
      const x1 = (e.clientX - ox) / dw, y1 = (e.clientY - oy) / dh;
      if (x1 < 0 || x1 > 1 || y1 < 0 || y1 > 1) return;
      xyTag.textContent = `X1 ${x1.toFixed(3)}  Y1 ${y1.toFixed(3)}   ·   `
        + `X ${Math.round(x1 * nw)}  Y ${Math.round(y1 * nh)} px   ·   ${eyeName()} глаз`;
    };
    const onLeave = () => { xyTag.textContent = 'X1 — Y1 —   ·   X — Y — px'; };
    const onEye = () => {
      if (!xyTag.textContent.includes('·')) return;
      xyTag.textContent = xyTag.textContent.replace(/(левый|правый) глаз/, `${eyeName()} глаз`);
    };
    img.addEventListener('mousemove', onMove);
    img.addEventListener('mouseleave', onLeave);

    const bar = h('div', { class: 'flex flex-wrap items-center gap-3' },
      h('div', { class: 'flex items-center gap-3' }, eyeRadio('Левый', true), eyeRadio('Правый', false)),
      h('div', { class: 'grow' }),
      fpsTag, debugBtn);

    el.appendChild(panelBody(
      h('div', { class: 'flex items-center gap-2' }, icon('video', 'size-4 text-muted-foreground'), h('span', { class: 'text-[13px] font-medium', text: 'Камера' }), h('div', { class: 'grow' }), layoutTag),
      stage,
      bar,
      readout,
    ));

    let alive = true;
    const tick = async () => {
      if (!alive) return;
      try {
        const d = await api.stats();
        fpsTag.textContent = `${(d.fps ?? 0).toFixed(1)} fps`;
      } catch { /* ignore */ }
    };
    (async () => {
      try {
        const p = await api.cameraParams();
        if (p?.layout) {
          const L = p.layout;
          layoutTag.textContent = `кадр ${L.frame_w}×${L.frame_h} · глаз ${L.w}px · FOV ${L.fov_h}°`;
        }
      } catch { /* ignore */ }
    })();
    const timer = setInterval(tick, 2000);
    tick();

    return () => {
      alive = false;
      clearInterval(timer);
      img.removeEventListener('mousemove', onMove);
      img.removeEventListener('mouseleave', onLeave);
    };
  },
};
