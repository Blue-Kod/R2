// Workspace dock: a thin wrapper over Golden Layout 2.6.
//
// Panels register a mount()/unmount() pair; the dock handles create / close /
// drag / group-into-tabs / split, multiple instances of the same panel,
// layout persistence and presets. Golden Layout reparents DOM nodes when items
// move, so every panel is mounted into a fresh host element and torn down via
// the container's 'destroy' event.
//
// Golden Layout 2.6 usage: construct from the container element only (this
// auto-initialises an empty layout), then loadLayout() the root config.

import { h, toast } from './ui.js';

const STORAGE_KEY = 'r2.workspace.layout.v2';
const PRESET_KEY = 'r2.workspace.preset.v1';
const HEADER_HEIGHT = 30;

let _instSeq = 0;

export class Workspace {
  constructor(rootEl) {
    this.root = rootEl;
    this.registry = new Map();
    this.instances = new Map(); // instanceId -> { panel, container, cleanup }
    this.layout = null;
    this.preset = 'control';
    this.boot = {};
    this._presets = {};
    this._saveTimer = null;
    this._suspendPersist = false;
    this._options = {
      dimensions: {
        headerHeight: HEADER_HEIGHT,
        borderWidth: 4,
        defaultMinItemWidth: '64px',
        defaultMinItemHeight: '48px',
      },
      header: {
        show: 'top',
        popout: false,
        maximise: 'Развернуть',
        close: 'Закрыть',
        tabDropdown: 'Ещё вкладки',
      },
      settings: { reorderEnabled: true, constrainDragToContainer: false },
    };
  }

  register(panel) {
    this.registry.set(panel.id, panel);
    return this;
  }

  _newInstId() {
    _instSeq += 1;
    return `${_instSeq}-${Math.random().toString(36).slice(2, 7)}`;
  }

  /** Build a Golden Layout item config from a preset tree of panel ids. */
  buildConfig(node) {
    if (node.id) {
      const panel = this.registry.get(node.id);
      const inst = this._newInstId();
      const cfg = {
        type: 'component',
        componentType: node.id,
        componentState: { inst },
        id: inst,
        title: panel?.title || node.id,
        isClosable: true,
        reorderEnabled: true,
      };
      if (node.size) cfg.size = node.size;
      return cfg;
    }
    const cfg = {
      type: node.dir,
      content: (node.children || []).map((c) => this.buildConfig(c)),
    };
    if (node.size) cfg.size = node.size;
    return cfg;
  }

  _layoutConfig(rootConfig) {
    return {
      root: rootConfig,
      dimensions: this._options.dimensions,
      header: this._options.header,
      settings: this._options.settings,
    };
  }

  init(presets, initialPreset = 'control') {
    this._presets = presets;
    // The IIFE bundle exposes a module namespace; unwrap the class if needed.
    const GoldenLayout = window.GoldenLayout?.GoldenLayout ?? window.GoldenLayout;
    if (typeof GoldenLayout !== 'function') {
      throw new Error('GoldenLayout не загрузился');
    }

    const startPreset = this._presets[initialPreset] ? initialPreset : Object.keys(this._presets)[0];
    this.preset = startPreset;

    // Construct from the element only -> the layout auto-initialises empty.
    this.layout = new GoldenLayout(this.root);

    this.layout.registerComponentFactoryFunction('__r2_missing__', (container) => {
      container.element.appendChild(h('div', { class: 'p-4 text-sm text-muted-foreground', text: 'Панель недоступна' }));
    });
    for (const panel of this.registry.values()) {
      // Factory signature: (container, state, virtual). state = { inst }.
      this.layout.registerComponentFactoryFunction(panel.id, (container, state) => {
        this._mount(panel, container, state);
      });
    }

    this.layout.on('stateChanged', () => this._schedulePersist());

    // --- Привязка размеров к сетке 8px -------------------------------------
    // Golden Layout считает размер окна из координат указателя, поэтому
    // достаточно квантовать clientX/clientY во время перетаскивания сплиттера:
    // границы окон встают на общую сетку, соседние выравниваются между собой.
    const GRID = 8;
    const quantize = (e) => {
      const qx = Math.round(e.clientX / GRID) * GRID;
      const qy = Math.round(e.clientY / GRID) * GRID;
      const dx = qx - e.clientX;
      const dy = qy - e.clientY;
      for (const [k, v] of [['clientX', qx], ['clientY', qy],
                            ['pageX', e.pageX + dx], ['pageY', e.pageY + dy],
                            ['screenX', e.screenX + dx], ['screenY', e.screenY + dy],
                            ['x', qx], ['y', qy]]) {
        try { Object.defineProperty(e, k, { value: v, configurable: true }); } catch { /* пропускаем */ }
      }
    };
    let dragging = false;
    const onDown = (e) => { dragging = !!e.target.closest?.('.lm_splitter'); };
    const onMove = (e) => { if (dragging) quantize(e); };
    const onUp = () => { dragging = false; };
    for (const target of [this.root, document]) {
      target.addEventListener('pointerdown', onDown, true);
      target.addEventListener('pointermove', onMove, true);
      target.addEventListener('mousemove', onMove, true);   // GL слушает mousemove
      target.addEventListener('pointerup', onUp, true);
      target.addEventListener('mouseup', onUp, true);
    }
    window.__r2 = this;   // отладочный доступ к раскладке (не участвует в UI)

    const baseConfig = this._layoutConfig(this.buildConfig(this._presets[startPreset].root));
    let config = baseConfig;
    const saved = this._loadSaved();
    if (saved) {
      config = saved;
      this.preset = localStorage.getItem(PRESET_KEY) || startPreset;
    }

    this._suspendPersist = true;
    try {
      this.layout.loadLayout(config);
    } catch (err) {
      console.warn('[dock] раскладка не загрузилась, беру пресет по умолчанию:', err);
      this.preset = startPreset;
      this.layout.loadLayout(baseConfig);
    } finally {
      this._suspendPersist = false;
    }
    this.persist();
  }

  _mount(panel, container, state) {
    const inst = (state && state.inst) || this._newInstId();
    const host = h('div', { class: 'h-full w-full overflow-hidden' });
    container.element.classList.add('bg-card');
    container.element.appendChild(host);

    let cleanup = null;
    try {
      cleanup = panel.mount(host, { container, workspace: this, boot: this.boot, instanceId: inst });
    } catch (err) {
      console.error(`[dock] панель ${panel.id} упала при монтировании:`, err);
      host.appendChild(h('div', { class: 'p-4 text-sm text-destructive', text: `Ошибка панели: ${err.message}` }));
    }
    this.instances.set(inst, { panel, container, cleanup });

    container.on('resize', () => host.dispatchEvent(new CustomEvent('r2:resize')));
    container.on('destroy', () => {
      this.instances.delete(inst);
      try { cleanup?.(); } catch (err) { console.warn('[dock] cleanup error', panel.id, err); }
    });
  }

  instancesOf(id) {
    return [...this.instances.values()].filter((entry) => entry.panel.id === id);
  }

  focus(id) {
    const [first] = this.instancesOf(id);
    if (first) {
      try { first.container.focus(); } catch { /* ignore */ }
      return true;
    }
    return false;
  }

  _add(id, title) {
    const panel = this.registry.get(id);
    if (!panel) return;
    const state = { inst: this._newInstId() };
    const selectors = window.GoldenLayout?.LayoutManager?.afterFocusedItemIfPossibleLocationSelectors;
    let location;
    try {
      if (selectors && typeof this.layout.addComponentAtLocation === 'function') {
        location = this.layout.addComponentAtLocation(id, state, title, selectors);
      }
    } catch (err) {
      console.warn('[dock] addComponentAtLocation failed', err);
    }
    if (!location) {
      try {
        this.layout.addComponent(id, state, title);
      } catch (err) {
        console.error('[dock] addComponent failed', err);
        toast(`Не удалось открыть «${panel.title}»`, 'error');
      }
    }
  }

  /** Focus the panel if it is open, otherwise open it. */
  open(id) {
    if (this.focus(id)) return;
    const panel = this.registry.get(id);
    if (!panel) return;
    this._add(id, panel.title);
  }

  /** Always open another instance (e.g. a second terminal). */
  openNew(id) {
    const panel = this.registry.get(id);
    if (!panel) return;
    const count = this.instancesOf(id).length;
    this._add(id, count > 0 ? `${panel.title} ${count + 1}` : panel.title);
  }

  close(id) {
    for (const entry of this.instancesOf(id)) {
      try { entry.container.close(); } catch { /* ignore */ }
    }
  }

  loadPreset(name) {
    const preset = this._presets?.[name];
    if (!preset) return;
    this.preset = name;
    this._suspendPersist = true;
    try {
      this.layout.loadLayout(this._layoutConfig(this.buildConfig(preset.root)));
    } catch (err) {
      console.error('[dock] preset load failed', err);
      toast('Не удалось применить раскладку', 'error');
    } finally {
      this._suspendPersist = false;
    }
    try { localStorage.setItem(PRESET_KEY, name); } catch { /* ignore */ }
    this.persist();
  }

  reset() {
    this.loadPreset('control');
  }

  destroy() {
    try { this.layout?.destroy(); } catch { /* ignore */ }
    this.instances.clear();
    this.layout = null;
  }

  // Ни одно окно не должно быть уже 40px — иначе раскладка «слиплась».
  static _looksBroken(node) {
    if (!node || typeof node !== 'object') return true;
    if (node.type === 'component') return false;
    const kids = node.content || node.contentItems || [];
    if (!kids.length) return true;
    for (const kid of kids) {
      const size = Number(kid.size ?? 0);
      if (!Number.isFinite(size) || size < 6) return true;   // окно сжато в ничто
      if (Dock._looksBroken(kid)) return true;
    }
    return false;
  }

  _loadSaved() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed?.root) return null;
      // Раскладка могла испортиться (баг, эксперимент, ручной drag в ноль).
      // Проверяем, что ни одно окно не сжато меньше осмысленного минимума;
      // иначе отбрасываем сохранение и грузим пресет по умолчанию.
      if (Dock._looksBroken(parsed.root)) {
        console.warn('[dock] сохранённая раскладка выглядит сломанной — беру пресет');
        try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
        return null;
      }
      return parsed;
    } catch {
      return null;
    }
  }

  _schedulePersist() {
    if (this._suspendPersist) return;
    clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => this.persist(), 400);
  }

  persist() {
    try {
      // saveLayout() returns a ResolvedLayoutConfig; loadLayout() expects a
      // LayoutConfig, so convert before storing (sizes etc. differ in shape).
      const resolved = this.layout.saveLayout();
      const ns = window.GoldenLayout;
      const config = ns?.LayoutConfig?.fromResolved ? ns.LayoutConfig.fromResolved(resolved) : resolved;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
    } catch (err) {
      console.warn('[dock] persist failed', err);
    }
  }
}

export { STORAGE_KEY, PRESET_KEY };
