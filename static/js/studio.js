// R2 dataset studio — CapCut-like timeline editor.
// Player on top; below: servo-motion graph, frame filmstrip, editable task track.
// Segments are exported non-destructively to `library` or `dataset`.

import { h, button, inputClass, label, toast, emptyState } from './ui.js';
import { api } from './api.js';

const root = document.getElementById('studio');

const SERVO_COLORS = ['#60a5fa', '#f472b6', '#34d399', '#fbbf24', '#a78bfa', '#f87171',
  '#22d3ee', '#fb923c', '#4ade80', '#e879f9', '#38bdf8', '#facc15'];

let store = 'library';
let episodes = [];
let epMeta = null;
let stateData = null;
let frame = 0;
let playing = false;
let playTimer = null;
let segments = [];                  // {start, end, task} — start/end inclusive
let pending = { start: null, end: null };
const selected = new Set();
let eye = 'right';
let bboxVal = null;                 // сохранённый bbox (нормированный)
let bboxDraft = null;               // рисуемый сейчас
let bboxMode = false;
const hidden = new Set();
let playhead = null;
let drag = null;

// ---------- header ----------
const exportLib = button({ label: '→ library', iconName: 'save', onClick: () => exportAll('library') });
const exportDs = button({ label: '→ dataset', iconName: 'database', onClick: () => exportAll('dataset') });
const header = h('header', { class: 'flex shrink-0 items-center gap-3 border-b border-border px-4 py-2.5' },
  h('a', { href: '/', class: 'flex items-center gap-1 text-[12px] text-muted-foreground hover:text-foreground', title: 'В workspace' },
    h('span', { text: '←' }), h('span', { text: 'workspace' })),
  h('div', { class: 'flex items-baseline gap-2' },
    h('span', { class: 'text-sm font-semibold tracking-tight', text: 'Студия датасета' }),
    h('span', { class: 'text-[11px] text-muted-foreground', text: 'вырезание · деление · разметка задач' })),
  h('div', { class: 'ml-auto flex items-center gap-2' }, exportLib, exportDs));

// ---------- sidebar ----------
const libTab = h('button', { class: '', text: 'Library', onclick: () => setStore('library') });
const dsTab = h('button', { class: '', text: 'Dataset', onclick: () => setStore('dataset') });
const tabs = h('div', { class: 'flex gap-1 rounded-lg border border-border p-0.5' }, libTab, dsTab);
const epList = h('div', { class: 'flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto' });
const sidebar = h('aside', { class: 'flex w-56 shrink-0 flex-col gap-2 border-r border-border p-3' },
  tabs, label('Эпизоды'), epList);

// ---------- player ----------
const previewImg = h('img', { class: 'max-h-full max-w-full select-none object-contain', alt: '', ondragstart: () => false });
const bboxRect = h('div', { class: 'pointer-events-none absolute z-10 border-2', style: { display: 'none', borderColor: '#ff3b30' } });
const previewWrap = h('div', { class: 'relative inline-block max-h-full max-w-full' }, previewImg, bboxRect);
const eyeL = h('button', { class: 'rounded-md px-2 py-0.5 text-[11px]', text: 'L', onclick: () => setEye('left') });
const eyeR = h('button', { class: 'rounded-md px-2 py-0.5 text-[11px]', text: 'R', onclick: () => setEye('right') });
const tcChip = h('div', { class: 'font-mono text-[12px] tabular-nums text-foreground', text: '00:00.00' });
const player = h('div', { class: 'relative flex min-h-0 flex-1 items-center justify-center bg-black' },
  previewWrap,
  h('div', { class: 'absolute bottom-3 left-3 flex items-center gap-2 rounded-lg border border-border bg-popover/80 px-2 py-1 backdrop-blur' },
    eyeL, eyeR, h('span', { class: 'mx-1 h-3 w-px bg-border' }), tcChip));

// ---------- bbox цели (рисуем на первом кадре) ----------
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const bboxBtn = button({ label: 'bbox', title: 'Нарисовать рамку цели', onClick: () => setBboxMode(!bboxMode) });
const bboxSave = button({ label: '✓ bbox', title: 'Сохранить bbox (кадр 0)', onClick: () => saveBbox() });
const bboxClear = button({ label: '✕', title: 'Стереть bbox', onClick: () => clearBbox() });

function imgRect() { return previewImg.getBoundingClientRect(); }

function renderBbox() {
  const bb = bboxDraft || bboxVal;
  const r = imgRect();
  const wr = previewWrap.getBoundingClientRect();
  if (!bb || !epMeta || !r.width) { bboxRect.style.display = 'none'; return; }
  bboxRect.style.display = 'block';
  bboxRect.style.left = `${r.left - wr.left + bb[0] * r.width}px`;
  bboxRect.style.top = `${r.top - wr.top + bb[1] * r.height}px`;
  bboxRect.style.width = `${Math.max(2, (bb[2] - bb[0]) * r.width)}px`;
  bboxRect.style.height = `${Math.max(2, (bb[3] - bb[1]) * r.height)}px`;
}

function setBboxMode(on) {
  bboxMode = on;
  if (on && epMeta) seek(0);  // bbox хранится на кадре 0 — рисуем его там
  previewWrap.style.cursor = on ? 'crosshair' : '';
  const on_ = 'rounded-md px-2 py-0.5 text-[11px] bg-primary/30 text-foreground';
  const off_ = 'rounded-md px-2 py-0.5 text-[11px] text-muted-foreground hover:text-foreground';
  bboxBtn.className = on ? on_ : off_;
}

function relPt(e) {
  const r = imgRect();
  return [clamp01((e.clientX - r.left) / r.width),
          clamp01((e.clientY - r.top) / r.height)];
}

previewWrap.addEventListener('pointerdown', (e) => {
  if (!bboxMode || !epMeta) return;
  const [x, y] = relPt(e);
  bboxDraft = [x, y, x, y];
  previewWrap.setPointerCapture(e.pointerId);
});
previewWrap.addEventListener('pointermove', (e) => {
  if (!bboxDraft) return;
  const [x, y] = relPt(e);
  bboxDraft = [Math.min(bboxDraft[0], x), Math.min(bboxDraft[1], y),
               Math.max(bboxDraft[0], x), Math.max(bboxDraft[1], y)];
  renderBbox();
});
previewWrap.addEventListener('pointerup', () => {
  if (bboxDraft) { bboxVal = bboxDraft; bboxDraft = null; renderBbox(); }
});

async function saveBbox() {
  if (!epMeta || !bboxVal) { toast('Сначала нарисуй рамку (кнопка bbox)', 'error'); return; }
  try {
    const d = await api.studioSetBbox(store, epMeta.episode_index, bboxVal);
    epMeta.bbox = d.bbox;
    toast('bbox сохранён', 'success');
  } catch (err) { toast(err.message, 'error'); }
}

async function clearBbox() {
  if (!epMeta) return;
  try {
    await api.studioSetBbox(store, epMeta.episode_index, null);
    epMeta.bbox = null;
    bboxVal = null;
    bboxDraft = null;
    renderBbox();
    toast('bbox стёрт', 'success');
  } catch (err) { toast(err.message, 'error'); }
}

// ---------- toolbar ----------
const scrub = h('input', {
  type: 'range', min: '0', max: '0', value: '0', step: '1',
  class: 'w-full accent-[var(--primary)]',
  oninput: (e) => seek(Number(e.target.value)),
});
const playBtn = button({ label: '▶', title: 'Play/Pause (Space)', onClick: () => togglePlay() });
const taskInput = h('input', {
  type: 'text', placeholder: 'task (напр. grab apple)', class: `${inputClass} min-w-[140px] flex-1`,
  oninput: () => applyTaskFromField(),
});
const pendingInfo = h('span', { class: 'font-mono text-[11px] text-muted-foreground' });

const toolbar = h('div', { class: 'flex shrink-0 flex-wrap items-center gap-2 border-t border-border px-3 py-2' },
  button({ label: '−10', onClick: () => seek(frame - 10) }),
  button({ label: '−1', onClick: () => seek(frame - 1) }),
  playBtn,
  button({ label: '+1', onClick: () => seek(frame + 1) }),
  button({ label: '+10', onClick: () => seek(frame + 10) }),
  h('div', { class: 'min-w-[140px] flex-1' }, scrub),
  h('span', { class: 'mx-1 h-4 w-px bg-border' }),
  button({ label: 'IN', title: 'Начало сегмента (I)', onClick: () => mark('start') }),
  button({ label: 'OUT', title: 'Конец сегмента (O)', onClick: () => mark('end') }),
  button({ label: '+ Сегмент', iconName: 'folderPlus', onClick: () => addSegment() }),
  button({ label: '✂', title: 'Разделить в playhead', onClick: () => splitAtPlayhead() }),
  button({ label: '🗑', title: 'Удалить выделенные (Del)', onClick: () => deleteSelected() }),
  h('span', { class: 'mx-1 h-4 w-px bg-border' }),
  bboxBtn, bboxSave, bboxClear,
  h('div', { class: 'min-w-[140px] flex-[2]' }, taskInput),
  pendingInfo);

// ---------- timeline ----------
const ruler = h('div', { class: 'relative h-5 select-none' });
const servoCanvas = h('canvas', { class: 'absolute inset-0 h-full w-full' });
const servoLane = h('div', { class: 'relative h-[72px] overflow-hidden rounded-md border border-border bg-card/30' }, servoCanvas);
const frameLane = h('div', { class: 'relative h-12 overflow-hidden rounded-md border border-border bg-black/40' });
const taskLane = h('div', { class: 'relative h-9 overflow-hidden rounded-md border border-border bg-card/30' });
const overlay = h('div', { class: 'pointer-events-none absolute inset-0' });
const lanes = h('div', { class: 'relative flex touch-none select-none flex-col gap-1' }, servoLane, frameLane, taskLane, overlay);
const legend = h('div', { class: 'flex flex-wrap gap-1 pt-1.5' });
const timeline = h('div', { class: 'shrink-0 border-t border-border px-3 pb-3 pt-1' }, ruler, lanes, legend);

const stage = h('main', { class: 'flex min-h-0 flex-1 flex-col' }, player, toolbar);

root.appendChild(h('div', { class: 'anim-in flex h-dvh flex-col' },
  header, h('div', { class: 'flex min-h-0 flex-1' }, sidebar, stage), timeline));

// ---------- helpers ----------
const clampInt = (v, lo, hi) => {
  const n = parseInt(v, 10);
  return Number.isNaN(n) ? lo : Math.max(lo, Math.min(hi, n));
};

function formatTC(sec) {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(2).padStart(5, '0')}`;
}

function pct(v) { return `${(v / (epMeta?.rows || 1)) * 100}%`; }

function frameAt(clientX) {
  const rect = lanes.getBoundingClientRect();
  const ratio = (clientX - rect.left) / rect.width;
  return clampInt(Math.floor(ratio * (epMeta?.rows || 0)), 0, (epMeta?.rows || 1) - 1);
}

// ---------- stores / episodes ----------
function setStore(next) {
  store = next;
  epMeta = null;
  stateData = null;
  segments = [];
  selected.clear();
  pending = { start: null, end: null };
  taskInput.value = '';
  syncTabs();
  resetTimeline();
  renderEpisodes();
  loadEpisodes();
}

function syncTabs() {
  const active = 'rounded-md px-2.5 py-1 text-[12px] bg-primary/20 text-foreground';
  const idle = 'rounded-md px-2.5 py-1 text-[12px] text-muted-foreground hover:text-foreground';
  libTab.className = store === 'library' ? active : idle;
  dsTab.className = store === 'dataset' ? active : idle;
}

function renderEpisodes() {
  epList.innerHTML = '';
  if (!episodes.length) {
    epList.appendChild(emptyState(store === 'dataset' ? 'Dataset пуст' : 'Нет эпизодов'));
    return;
  }
  for (const ep of [...episodes].reverse()) {
    const active = epMeta && epMeta.episode_index === ep.episode_index;
    epList.appendChild(h('button', {
      class: `press flex flex-col gap-0.5 rounded-lg border px-2.5 py-2 text-left transition-colors ${active ? 'border-primary/40 bg-primary/10' : 'border-border bg-card/50 hover:bg-muted'}`,
      onclick: () => selectEpisode(ep.episode_index),
    },
      h('span', { class: 'font-mono text-[11px] text-primary', text: `#${ep.episode_index}` }),
      h('span', { class: 'min-w-0 truncate text-[12px]', text: ep.task || '—' }),
      h('span', { class: 'font-mono text-[10px] text-muted-foreground', text: `${ep.rows} кадров` })));
  }
}

async function loadEpisodes() {
  try {
    episodes = (await api.studioEpisodes(store)).datasets || [];
    renderEpisodes();
  } catch (err) { /* ignore */ }
}

async function selectEpisode(ep) {
  pause();
  try {
    epMeta = await api.studioEpisode(store, ep);
  } catch (err) { toast(err.message, 'error'); return; }
  frame = 0;
  segments = [];
  selected.clear();
  pending = { start: null, end: null };
  taskInput.value = '';
  bboxVal = epMeta.bbox || null;
  bboxDraft = null;
  setBboxMode(false);
  scrub.max = String(Math.max(0, epMeta.rows - 1));
  stateData = null;
  hidden.clear();
  renderEpisodes();
  renderRuler();
  renderFrames();
  renderTask();
  renderLegend();
  seek(0);
  try {
    stateData = await api.studioState(store, ep);
    renderServo();
    renderLegend();
  } catch (err) { /* no servo data */ }
}

function resetTimeline() {
  previewImg.removeAttribute('src');
  tcChip.textContent = '00:00.00';
  scrub.max = '0';
  scrub.value = '0';
  ruler.innerHTML = '';
  frameLane.innerHTML = '';
  taskLane.innerHTML = '';
  overlay.innerHTML = '';
  legend.innerHTML = '';
  const ctx = servoCanvas.getContext('2d');
  ctx.clearRect(0, 0, servoCanvas.width, servoCanvas.height);
}

// ---------- playback ----------
function seek(f) {
  if (!epMeta) return;
  const rows = epMeta.rows;
  frame = Math.max(0, Math.min(rows - 1, f));
  scrub.value = String(frame);
  previewImg.src = api.studioFrameUrl(store, epMeta.episode_index, frame, eye);
  tcChip.textContent = formatTC(epMeta.fps ? frame / epMeta.fps : 0);
  updatePlayhead();
  renderBbox();
}

previewImg.addEventListener('load', () => renderBbox());
window.addEventListener('resize', () => renderBbox());

function setEye(e) {
  eye = e;
  eyeL.className = `rounded-md px-2 py-0.5 text-[11px] ${eye === 'left' ? 'bg-primary/30 text-foreground' : 'text-muted-foreground'}`;
  eyeR.className = `rounded-md px-2 py-0.5 text-[11px] ${eye === 'right' ? 'bg-primary/30 text-foreground' : 'text-muted-foreground'}`;
  if (epMeta) previewImg.src = api.studioFrameUrl(store, epMeta.episode_index, frame, eye);
  renderBbox();
}

function togglePlay() { playing ? pause() : play(); }

function play() {
  if (!epMeta || epMeta.rows < 2) return;
  playing = true;
  playBtn.textContent = '❚❚';
  const interval = Math.max(20, 1000 / (epMeta.fps || 10));
  playTimer = setInterval(() => {
    if (frame >= epMeta.rows - 1) { pause(); return; }
    seek(frame + 1);
  }, interval);
}

function pause() {
  playing = false;
  if (playTimer) clearInterval(playTimer);
  playTimer = null;
  playBtn.textContent = '▶';
}

// ---------- ruler ----------
function renderRuler() {
  ruler.innerHTML = '';
  if (!epMeta || !epMeta.rows) return;
  const rows = epMeta.rows;
  const fps = epMeta.fps || 10;
  const totalSec = rows / fps;
  const nice = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  const stepSec = nice.find((t) => t >= totalSec / 8) || nice[nice.length - 1];
  const step = Math.max(1, Math.round(stepSec * fps));
  for (let f = 0; f < rows; f += step) {
    ruler.appendChild(h('span', {
      class: 'absolute top-0 font-mono text-[9px] text-muted-foreground',
      style: { left: pct(f) }, text: formatTC(f / fps),
    }));
  }
}

// ---------- frames filmstrip ----------
function renderFrames() {
  frameLane.innerHTML = '';
  if (!epMeta || !epMeta.rows) return;
  const rows = epMeta.rows;
  const step = Math.max(1, Math.ceil(rows / 70));
  for (let f = 0; f < rows; f += step) {
    frameLane.appendChild(h('img', {
      src: api.studioFrameUrl(store, epMeta.episode_index, f, 'right', 96),
      class: 'absolute top-0 h-full min-w-0 object-cover',
      style: { left: pct(f), width: `${Math.min(100, (step / rows) * 100)}%` },
      draggable: 'false', loading: 'lazy', alt: '',
    }));
  }
}

// ---------- servo graph ----------
function renderServo() {
  if (!epMeta) return;
  const w = servoLane.clientWidth || 800;
  const hpx = servoLane.clientHeight || 72;
  const dpr = window.devicePixelRatio || 1;
  servoCanvas.width = Math.round(w * dpr);
  servoCanvas.height = Math.round(hpx * dpr);
  const ctx = servoCanvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, hpx);
  if (!stateData || !stateData.values || !stateData.values.length) return;

  const samples = stateData.values;
  const n = samples.length;
  const dim = samples[0].length;
  for (let ch = 0; ch < dim; ch++) {
    if (hidden.has(ch)) continue;
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of samples) {
      const x = v[ch];
      if (x < lo) lo = x;
      if (x > hi) hi = x;
    }
    if (!Number.isFinite(lo)) continue;
    const span = (hi - lo) || 1;
    ctx.beginPath();
    ctx.strokeStyle = SERVO_COLORS[ch % SERVO_COLORS.length];
    ctx.lineWidth = 1.25;
    ctx.lineJoin = 'round';
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1 || 1)) * w;
      const y = hpx - 3 - ((samples[i][ch] - lo) / span) * (hpx - 6);
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    }
    ctx.stroke();
  }
}

function renderLegend() {
  legend.innerHTML = '';
  if (!stateData || !stateData.channels) return;
  stateData.channels.forEach(({ ch, name }) => {
    const off = hidden.has(ch);
    legend.appendChild(h('button', {
      class: `press flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] transition-colors ${off ? 'border-border text-muted-foreground/50' : 'border-border text-muted-foreground hover:text-foreground'}`,
      title: 'Показать/скрыть',
      onclick: () => { off ? hidden.delete(ch) : hidden.add(ch); renderLegend(); renderServo(); },
    },
      h('span', { class: 'size-2 rounded-full', style: { background: off ? 'transparent' : SERVO_COLORS[ch % SERVO_COLORS.length] } }),
      h('span', { text: name })));
  });
}

// ---------- task track + overlay ----------
function segColor(i) {
  const hue = (i * 137.5) % 360;
  return { bg: `hsl(${hue} 70% 45% / 0.32)`, border: `hsl(${hue} 75% 62%)` };
}

function renderTask() {
  taskLane.innerHTML = '';
  if (!epMeta) return;
  segments.forEach((s, i) => {
    const c = segColor(i);
    const block = h('div', {
      dataset: { seg: '1' },
      class: `absolute top-0.5 bottom-0.5 flex items-center overflow-hidden rounded-md border px-1.5 ${selected.has(i) ? 'ring-2 ring-foreground' : ''}`,
      style: { left: pct(s.start), width: pct(Math.max(1, s.end - s.start + 1)), background: c.bg, borderColor: c.border },
      title: `#${i + 1}: ${s.task || '—'}`,
      ondblclick: (e) => { e.stopPropagation(); editTaskInline(i, block); },
    },
      h('div', { dataset: { handle: 'l' }, class: 'absolute left-0 top-0 h-full w-2 cursor-ew-resize bg-foreground/30' }),
      h('span', { class: 'pointer-events-none truncate px-2 text-[11px] text-foreground', text: s.task || `#${i + 1}` }),
      h('div', { dataset: { handle: 'r' }, class: 'absolute right-0 top-0 h-full w-2 cursor-ew-resize bg-foreground/30' }));
    block.addEventListener('pointerdown', (e) => startBlockDrag(e, i));
    block.addEventListener('click', (e) => e.stopPropagation());
    taskLane.appendChild(block);
  });
  renderOverlay();
}

function editTaskInline(i, block) {
  const inp = h('input', { type: 'text', value: segments[i].task,
    class: 'absolute inset-0 h-full w-full rounded-md border border-ring bg-popover px-1.5 text-[11px] text-foreground outline-none' });
  block.textContent = '';
  block.appendChild(inp);
  inp.focus();
  inp.select();
  const commit = () => { segments[i].task = inp.value.trim(); renderTask(); syncTaskField(); };
  inp.addEventListener('blur', commit);
  inp.addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Enter') inp.blur();
    else if (e.key === 'Escape') { e.preventDefault(); renderTask(); }
  });
}

function renderOverlay() {
  overlay.innerHTML = '';
  if (!epMeta) return;
  segments.forEach((s) => {
    overlay.appendChild(h('div', { class: 'absolute top-0 bottom-0 w-px bg-foreground/15', style: { left: pct(s.end + 1) } }));
  });
  for (const key of ['start', 'end']) {
    if (pending[key] !== null) {
      overlay.appendChild(h('div', { class: 'absolute top-0 bottom-0 w-px border-l border-dashed border-foreground/70', style: { left: pct(pending[key] + 1) } }));
    }
  }
  playhead = h('div', { class: 'absolute top-0 bottom-0 z-10 w-0.5 bg-foreground shadow', style: { left: '0%' } });
  overlay.appendChild(playhead);
  updatePlayhead();
}

function updatePlayhead() {
  if (!playhead || !epMeta || !epMeta.rows) return;
  playhead.style.left = pct(frame);
  if (epMeta.fps) tcChip.textContent = formatTC(frame / epMeta.fps);
}

// ---------- drag: scrub + block trim/move ----------
lanes.addEventListener('pointerdown', (e) => {
  if (!epMeta || e.target.closest('[data-seg]')) return;
  drag = { mode: 'scrub' };
  lanes.setPointerCapture(e.pointerId);
  seek(frameAt(e.clientX));
});
lanes.addEventListener('pointermove', (e) => {
  if (drag && drag.mode === 'scrub') seek(frameAt(e.clientX));
});
lanes.addEventListener('pointerup', () => { if (drag && drag.mode === 'scrub') drag = null; });
lanes.addEventListener('pointercancel', () => { if (drag && drag.mode === 'scrub') drag = null; });

function startBlockDrag(e, i) {
  if (!epMeta) return;
  e.stopPropagation();
  e.preventDefault();
  const rect = lanes.getBoundingClientRect();
  const mode = e.target.dataset ? e.target.dataset.handle : null;
  if (!selected.has(i)) {
    if (!e.shiftKey) selected.clear();
    selected.add(i);
  } else if (e.shiftKey) {
    selected.delete(i);
  }
  const orig = new Map();
  for (const idx of selected) orig.set(idx, { start: segments[idx].start, end: segments[idx].end });
  if (!orig.has(i)) orig.set(i, { start: segments[i].start, end: segments[i].end });
  drag = {
    mode: mode === 'l' ? 'trim-l' : mode === 'r' ? 'trim-r' : 'move',
    i, orig, startX: e.clientX, pxPerFrame: rect.width / epMeta.rows,
  };
  renderTask();
  syncTaskField();
  window.addEventListener('pointermove', onBlockDragMove);
  window.addEventListener('pointerup', onBlockDragUp, { once: true });
}

function onBlockDragMove(e) {
  if (!drag || !epMeta) return;
  const rows = epMeta.rows;
  const dF = Math.round((e.clientX - drag.startX) / drag.pxPerFrame);
  if (drag.mode === 'trim-l' || drag.mode === 'trim-r') {
    const s = segments[drag.i];
    const o = drag.orig.get(drag.i);
    if (drag.mode === 'trim-l') s.start = clampInt(o.start + dF, 0, s.end);
    else s.end = clampInt(o.end + dF, s.start, rows - 1);
  } else {
    let minStart = Infinity;
    let maxEnd = -Infinity;
    for (const [, o] of drag.orig) {
      minStart = Math.min(minStart, o.start);
      maxEnd = Math.max(maxEnd, o.end);
    }
    const delta = Math.max(-minStart, Math.min(rows - 1 - maxEnd, dF));
    for (const [idx, o] of drag.orig) {
      segments[idx].start = o.start + delta;
      segments[idx].end = o.end + delta;
    }
  }
  renderTask();
}

function onBlockDragUp() {
  window.removeEventListener('pointermove', onBlockDragMove);
  drag = null;
  renderTask();
}

// ---------- segment ops ----------
function syncTaskField() {
  if (selected.size === 1) {
    const idx = [...selected][0];
    taskInput.value = segments[idx] ? segments[idx].task : '';
  } else if (selected.size === 0) {
    taskInput.value = '';
  }
}

function renderPending() {
  pendingInfo.textContent = `IN ${pending.start ?? '—'} · OUT ${pending.end ?? '—'}`;
  renderOverlay();
}

function mark(which) {
  if (selected.size === 1) {
    const s = segments[[...selected][0]];
    if (which === 'start') s.start = Math.min(frame, s.end);
    else s.end = Math.max(frame, s.start);
    renderTask();
    return;
  }
  pending[which] = frame;
  renderPending();
}

function addSegment() {
  if (!epMeta) return;
  const a = pending.start;
  const b = pending.end;
  if (a === null || b === null) { toast('Задай IN и OUT (I/O)', 'error'); return; }
  segments.push({ start: Math.min(a, b), end: Math.max(a, b), task: (taskInput.value || '').trim() });
  pending = { start: null, end: null };
  selected.clear();
  selected.add(segments.length - 1);
  syncTaskField();
  renderTask();
  renderPending();
}

function applyTaskFromField() {
  for (const idx of selected) {
    if (segments[idx]) segments[idx].task = taskInput.value;
  }
  if (selected.size) renderTask();
}

function splitAtPlayhead() {
  let i = selected.size === 1 ? [...selected][0] : segments.findIndex((s) => frame > s.start && frame <= s.end);
  if (i < 0) { toast('Нет сегмента под playhead', 'error'); return; }
  const s = segments[i];
  if (frame <= s.start || frame > s.end) { toast('Поставь playhead внутрь сегмента', 'error'); return; }
  segments.splice(i + 1, 0, { start: frame, end: s.end, task: s.task });
  s.end = frame - 1;
  selected.clear();
  selected.add(i);
  selected.add(i + 1);
  renderTask();
}

function deleteSelected() {
  if (!selected.size) { toast('Выдели сегмент', 'error'); return; }
  for (const idx of [...selected].sort((a, b) => b - a)) segments.splice(idx, 1);
  selected.clear();
  taskInput.value = '';
  renderTask();
}

async function exportAll(target) {
  if (!epMeta) { toast('Выбери эпизод', 'error'); return; }
  const idxs = selected.size
    ? [...selected].sort((a, b) => segments[a].start - segments[b].start)
    : segments.map((_, i) => i);
  if (!idxs.length) { toast('Нет сегментов', 'error'); return; }
  exportLib.disabled = true;
  exportDs.disabled = true;
  try {
    const payload = idxs.map((i) => ({ start: segments[i].start, end: segments[i].end + 1, task: segments[i].task }));
    const d = await api.studioApply(store, epMeta.episode_index, payload, target);
    toast(`→ ${target}: эпизодов ${d.episodes.length}`, 'success');
    for (const i of [...idxs].sort((a, b) => b - a)) segments.splice(i, 1);
    selected.clear();
    taskInput.value = '';
    renderTask();
    if (target === store) await loadEpisodes();
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    exportLib.disabled = false;
    exportDs.disabled = false;
  }
}

// ---------- keyboard ----------
window.addEventListener('keydown', (e) => {
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return;
  if (e.key === 'ArrowLeft') { e.preventDefault(); seek(frame - 1); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); seek(frame + 1); }
  else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  else if ('iIшШ'.includes(e.key)) { e.preventDefault(); mark('start'); }
  else if ('oOщЩ'.includes(e.key)) { e.preventDefault(); mark('end'); }
  else if (e.key === 'Delete' || e.key === 'Backspace') { deleteSelected(); }
});

if (window.ResizeObserver) new ResizeObserver(() => renderServo()).observe(servoLane);

syncTabs();
setEye('right');
renderTask();
renderLegend();
loadEpisodes();
