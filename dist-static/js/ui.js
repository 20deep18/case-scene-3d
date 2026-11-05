/**
 * 界面层：把 DOM 与事件集中封装，主程序只调用方法。
 */
import { defaultColor, kindName } from './annotations.js';

const $ = (sel) => document.querySelector(sel);

/**
 * 打开对话框（带老内核兜底）。
 *
 * 为什么不让各面板直接调 `dlg.showModal()`：
 *   1) 对话框已经打开时再调 `showModal()` 会抛 InvalidStateError；
 *   2) 部分嵌入式内核 / 旧 WebView **没有实现 `<dialog>`**，
 *      直接调 `showModal()` 会抛 TypeError，表现为「点了按钮没反应」。
 *      这时退化为「加 open 属性 + CSS 定位」，功能仍然可用。
 */
export function openDialogModal(dlg) {
  if (!dlg) return false;
  if (dlg.open) return true;
  if (typeof dlg.showModal === 'function') {
    try {
      dlg.showModal();
      return true;
    } catch {
      /* 落到下面的兜底 */
    }
  }
  if (typeof dlg.show === 'function') {
    try {
      dlg.show();
      return true;
    } catch {
      /* 落到下面的兜底 */
    }
  }
  dlg.setAttribute('open', '');
  dlg.classList.add('dialog-fallback');
  return true;
}

/** 关闭对话框（带老内核兜底）。 */
export function closeDialogModal(dlg, returnValue = 'cancel') {
  if (!dlg) return;
  const wasOpen = dlg.hasAttribute('open');
  try {
    dlg.returnValue = returnValue;
  } catch {
    /* ignore */
  }
  let closed = false;
  try {
    if (typeof dlg.close === 'function') {
      dlg.close(returnValue);
      closed = !dlg.hasAttribute('open');
    }
  } catch {
    /* 落到下面的兜底 */
  }
  if (!closed && wasOpen) {
    // 内核没有 close()：手动摘掉 open 并补发 close 事件，
    // 否则依赖 close 回调的逻辑（如 ui.confirm 的 Promise）会永远挂着。
    dlg.removeAttribute('open');
    dlg.classList.remove('dialog-fallback');
    try {
      dlg.dispatchEvent(new Event('close'));
    } catch {
      /* ignore */
    }
  }
}

/**
 * 让对话框里的「取消 / 关闭」按钮**一定**能关掉窗口。
 *
 * 背景（用户反馈的 bug：点「取消」关不掉建模对话框）：
 *   这些对话框用的是 `<form method="dialog"><button value="cancel">取消</button>`，
 *   规范上「提交即关闭对话框」。但只要内核**不支持 `method="dialog"`** 或
 *   把该按钮当成普通提交处理，这个默认行为就不发生 —— 窗口就关不掉了。
 *   这里再补一层**显式**的 JS 关闭，不依赖浏览器对 dialog 表单的支持。
 *
 * 判定规则（按优先级）：
 *   1) 按钮带 `data-dialog-close`；
 *   2) `value` 是 cancel / close；
 *   3) 按钮文字正好是 取消 / 关闭 / 关闭窗口 / cancel / close。
 */
export function ensureDialogClosable(dlg) {
  if (!dlg || dlg.__closableWired) return dlg;
  dlg.__closableWired = true;
  // 用捕获阶段：即使别的处理器 preventDefault，也照样能关
  dlg.addEventListener(
    'click',
    (ev) => {
      const btn = ev.target?.closest?.('button, [data-dialog-close]');
      if (!btn || !dlg.contains(btn)) return;
      const value = btn.getAttribute?.('value') ?? '';
      const text = (btn.textContent ?? '').trim();
      const isClose =
        btn.hasAttribute('data-dialog-close') ||
        value === 'cancel' ||
        value === 'close' ||
        /^(取消|关闭|关闭窗口|cancel|close)$/i.test(text);
      if (isClose) closeDialogModal(dlg, value || 'cancel');
    },
    true
  );
  return dlg;
}

export class UI {
  constructor() {
    this.el = {
      caseSubtitle: $('#caseSubtitle'),
      projectSelect: $('#projectSelect'),
      modeSwitch: $('#modeSwitch'),
      statFps: $('#statFps'),
      statGauss: $('#statGauss'),
      presence: $('#presence'),
      connState: $('#connState'),
      btnNewCase: $('#btnNewCase'),
      btnRefreshCase: $('#btnRefreshCase'),
      modelSelect: $('#modelSelect'),
      btnLoadModel: $('#btnLoadModel'),
      modelHint: $('#modelHint'),
      layerList: $('#layerList'),
      userList: $('#userList'),
      rGaussScale: $('#rGaussScale'),
      vScale: $('#vScale'),
      rOpacity: $('#rOpacity'),
      vOpacity: $('#vOpacity'),
      rExposure: $('#rExposure'),
      vExposure: $('#vExposure'),
      cSort: $('#cSort'),
      cPoints: $('#cPoints'),
      toolBar: $('#toolBar'),
      toolHint: $('#toolHint'),
      annoList: $('#annoList'),
      annoFilter: $('#annoFilter'),
      annoCount: $('#annoCount'),
      inspector: $('#inspector'),
      chatLog: $('#chatLog'),
      chatText: $('#chatText'),
      chatSend: $('#chatSend'),
      cameraStrip: $('#cameraStrip'),
      btnAddCamera: $('#btnAddCamera'),
      hoverTip: $('#hoverTip'),
      toast: $('#toast'),
      loader: $('#loader'),
      loaderText: $('#loaderText'),
      loaderBar: $('#loaderBar'),
      loaderSub: $('#loaderSub'),
      dlgNewCase: $('#dlgNewCase'),
      dlgImport: $('#dlgImport'),
      ncName: $('#ncName'),
      ncDesc: $('#ncDesc'),
      ncOperator: $('#ncOperator'),
      impFile: $('#impFile'),
      impName: $('#impName'),
      impKind: $('#impKind'),
      impBar: $('#impBar'),
      staleBanner: $('#staleBanner'),
      staleText: $('#staleText'),
      staleReload: $('#staleReload'),
    };
    this.handlers = {};
    this.currentProject = null;
    this.annotations = [];
    this.filter = '';
    this.pendingImportTarget = null;
    this.toastTimer = null;
    this.labelHost = document.createElement('div');
    this.labelHost.id = 'labelLayer';
    document.getElementById('app').appendChild(this.labelHost);

    this.el.annoFilter.addEventListener('input', () => {
      this.filter = this.el.annoFilter.value.trim().toLowerCase();
      this.renderAnnotations(this.annotations);
    });

    // 所有对话框都保证「取消 / 关闭」按钮点得动（详见 ensureDialogClosable）
    for (const dlg of [this.el.dlgNewCase, this.el.dlgImport, document.getElementById('dlgConfirm')]) {
      ensureDialogClosable(dlg);
    }

    // 版本过期横幅的「立即刷新」
    this.el.staleReload?.addEventListener('click', () => {
      // 硬刷新：普通 reload 在个别嵌入式内核里可能仍用内存里的旧模块
      try {
        location.reload();
      } catch {
        location.href = location.href;
      }
    });
  }

  /**
   * 对比「页面里的前端版本」与「服务端版本」，不一致就挂横幅。
   *
   * 这不是锦上添花：本项目是单机本地部署，页面一开就是几小时。
   * 改了前端 + 重启服务之后，用户那个标签页里跑的仍是旧 JS，
   * 现象是「你说修好了，我这边还是老样子」——而且完全看不出原因。
   * 实测踩过一次：旧页面的「取消」用的是被内核屏蔽的 window.confirm，
   * 点下去毫无反应，服务端连请求都没收到。
   */
  watchVersion(serverVersion, pageVersion) {
    // 静态演示版的"服务器版本"取自打包时注入的常量，永远等于页面版本；
    // 万一打包后页面被 CDN 缓存成旧的，这条会误报"系统已更新"（实测截图抓到）。
    // 静态模式下不挂横幅：用户刷新也拿不到"新版本"，只会白折腾。
    if (typeof window !== 'undefined' && window.__DSH_STATIC__ === true) return;
    this.pageVersion = pageVersion;
    const apply = (v) => {
      if (!v) return;
      if (String(v) === String(pageVersion)) this.hideStaleBanner();
      else this.showStaleBanner(v, pageVersion);
    };
    apply(serverVersion);
    clearInterval(this._versionTimer);
    this._versionTimer = setInterval(async () => {
      try {
        const h = await fetch('/api/health', { cache: 'no-store' }).then((r) => r.json());
        apply(h?.version);
      } catch {
        /* 网络抖动忽略 */
      }
    }, 60_000);
  }

  showStaleBanner(serverVersion, pageVersion) {
    const el = this.el.staleBanner;
    if (!el) return;
    if (this.el.staleText) {
      this.el.staleText.innerHTML =
        `系统已更新：<b>本页面是旧版本</b>（页面 ${escapeHtml(String(pageVersion))} / 服务器 ${escapeHtml(String(serverVersion))}），` +
        `行为可能与当前代码不一致。请点右侧按钮刷新。`;
    }
    el.classList.remove('hidden');
  }

  hideStaleBanner() {
    this.el.staleBanner?.classList.add('hidden');
  }

  /**
   * 拍摄引导：把「哪些方向没拍到」这类**只能靠补拍解决**的问题明确说出来。
   *
   * 为什么值得单独一个弹窗：实测某个案件的留出视角 PSNR 从 12.99 到 25.02 dB，
   * 归因下来最差那帧的视锥里只有 106 个稀疏点（最好那帧 10,563）——
   * 那个方向根本没人拍到，模型只能插值。用户很容易误以为是「参数没调好」
   * 而反复重跑，每次白等二十分钟。所以要在载入模型后立刻讲清楚。
   */
  showShootingAdvice({ title = '拍摄建议', body = '', tip = '' } = {}) {
    const dlg = document.getElementById('dlgShooting');
    const t = document.getElementById('saTitle');
    const b = document.getElementById('saBody');
    const p = document.getElementById('saTip');
    if (!dlg || !t || !b || !p) {
      // 元素缺失时至少把结论写进 toast，别静默丢掉
      this.toast(`${title}：${body}`, { sticky: true });
      return;
    }
    t.textContent = title;
    b.textContent = body;
    p.textContent = tip;
    p.classList.toggle('hidden', !tip);
    ensureDialogClosable(dlg);
    openDialogModal(dlg);
  }

  // ---------------------------------------------------------------- 绑定

  bind(handlers) {
    this.handlers = handlers;
    const h = handlers;

    this.el.projectSelect.addEventListener('change', () => h.onSelectProject?.(this.el.projectSelect.value));
    this.el.btnRefreshCase.addEventListener('click', () => h.onRefresh?.());
    this.el.btnNewCase.addEventListener('click', () => openDialogModal(this.el.dlgNewCase));
    this.el.dlgNewCase.addEventListener('close', () => {
      if (this.el.dlgNewCase.returnValue !== 'ok') return;
      const name = this.el.ncName.value.trim();
      if (!name) return;
      h.onNewProject?.({ name, description: this.el.ncDesc.value.trim(), operator: this.el.ncOperator.value.trim() });
      this.el.ncName.value = '';
      this.el.ncDesc.value = '';
    });
    this.el.btnLoadModel.addEventListener('click', () => {
      const url = this.el.modelSelect.value;
      if (url) h.onLoadModel?.(url);
    });
    // 双击 = 点「载入三维现场」（原有行为）
    this.el.modelSelect.addEventListener('dblclick', () => this.el.btnLoadModel.click());
    // 单选即载入：实测评审/用户都预期"选了就出现"，不会再去找按钮。
    this.el.modelSelect.addEventListener('change', () => {
      const url = this.el.modelSelect.value;
      if (url) h.onLoadModel?.(url);
    });

    this.el.modeSwitch.addEventListener('click', (ev) => {
      const btn = ev.target.closest('button[data-mode]');
      if (btn) h.onMode?.(btn.dataset.mode);
    });
    this.el.toolBar.addEventListener('click', (ev) => {
      const btn = ev.target.closest('button[data-tool]');
      if (btn) h.onTool?.(btn.dataset.tool);
    });

    const sliders = [
      [this.el.rGaussScale, this.el.vScale, 'gaussScale', (v) => v.toFixed(2)],
      [this.el.rOpacity, this.el.vOpacity, 'opacity', (v) => v.toFixed(2)],
      [this.el.rExposure, this.el.vExposure, 'exposure', (v) => v.toFixed(2)],
    ];
    for (const [input, label, key, fmt] of sliders) {
      input.addEventListener('input', () => {
        label.textContent = fmt(Number(input.value));
        h.onFilterChange?.({ [key]: Number(input.value) });
      });
    }
    this.el.cSort.addEventListener('change', () => h.onFilterChange?.({ sort: this.el.cSort.checked }));
    this.el.cPoints.addEventListener('change', () => h.onFilterChange?.({ showPoints: this.el.cPoints.checked }));

    this.el.chatSend.addEventListener('click', () => this._sendChat());
    this.el.chatText.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') this._sendChat();
    });

    this.el.btnAddCamera.addEventListener('click', () => this.openImportDialog(null));
    this.el.impFile.addEventListener('change', () => {
      const f = this.el.impFile.files?.[0];
      if (f && !this.el.impName.value) this.el.impName.value = f.name;
    });
    this.el.dlgImport.addEventListener('close', async () => {
      if (this.el.dlgImport.returnValue !== 'ok') return;
      const file = this.el.impFile.files?.[0];
      if (!file) return;
      // 防重入：上传期间再次触发 close 会导致同一个视频被导入两次
      // （表现为「只导入一个视频却出现两个相机对象」）
      if (this._importing) {
        this.toast('上一个视频还在上传中，请稍候…', { error: true });
        return;
      }
      // 用「提交时冻结」的目标，避免对话框复用导致的目标漂移
      const target = this.pendingImportTarget;
      this.pendingImportTarget = null;
      this._importing = true;
      try {
        await h.onImportMedia?.(file, target);
      } finally {
        this._importing = false;
        this.el.impFile.value = '';
        this.el.impName.value = '';
      }
    });
  }

  _sendChat() {
    const text = this.el.chatText.value.trim();
    if (!text) return;
    this.handlers.onSendChat?.(text);
    this.el.chatText.value = '';
  }

  openImportDialog(annotation) {
    this.pendingImportTarget = annotation;
    openDialogModal(this.el.dlgImport);
  }

  // ---------------------------------------------------------------- 案件

  renderProjects(projects, selectedId) {
    const sel = this.el.projectSelect;
    sel.innerHTML = '';
    if (!projects.length) {
      const opt = document.createElement('option');
      opt.textContent = '（暂无案件）';
      opt.value = '';
      sel.appendChild(opt);
      return;
    }
    for (const p of projects) {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = `${p.name} · ${p.annotationCount} 标注`;
      if (p.id === selectedId) opt.selected = true;
      sel.appendChild(opt);
    }
  }

  setProject(project) {
    this.currentProject = project;
    this.annotations = project.annotations ?? [];
    this.el.caseSubtitle.textContent = project
      ? `${project.name}${project.description ? ' · ' + project.description : ''} · 编号 ${project.id}`
      : '未加载案件';
    localStorage.setItem('dsh:lastProject', project?.id ?? '');
    this.renderLayerList();
  }

  setModelOptions(models) {
    const sel = this.el.modelSelect;
    // 记住原来的选中值：重建选项后要**恢复**它，否则每次刷新模型列表都会跳回第一项。
    const previous = sel.value;
    sel.innerHTML = '';
    if (!models.length) {
      const opt = document.createElement('option');
      opt.textContent = '（data/models 下暂无模型）';
      opt.value = '';
      sel.appendChild(opt);
      this.el.btnLoadModel.disabled = true;
      return;
    }
    this.el.btnLoadModel.disabled = false;
    for (const m of models) {
      const opt = document.createElement('option');
      opt.value = m.url;
      const size = m.bytes ? ` · ${(m.bytes / 1048576).toFixed(1)}MB` : '';
      const count = m.gaussianCount ? ` · ${Number(m.gaussianCount).toLocaleString('zh-CN')} 高斯` : '';
      opt.textContent = `${m.name}${size}${count}`;
      sel.appendChild(opt);
    }
    if (previous && models.some((m) => m.url === previous)) sel.value = previous;
  }

  /**
   * 让模型下拉框显示**当前真正载入的那个模型**。
   *
   * 为什么需要：原来**没有任何地方写过这个下拉框的值** ——
   * 选项列表重建后就停在第一项，而实际载入的可能是上次用的模型，
   * 于是界面上"下拉框显示的模型"与"场里的模型"是两回事
   * （截图实测：下拉框写着 A、左栏提示写着 B）。
   * 这种不一致会让人误判"我现在看的是哪个模型"，对研判是硬伤。
   */
  syncModelSelect(url) {
    const sel = this.el.modelSelect;
    if (url && [...sel.options].some((o) => o.value === url)) sel.value = url;
  }

  setModelHint(text) {
    this.el.modelHint.textContent = text;
  }

  /**
   * 图层面板：按类型分组，可展开查看该类型下的**具体条目**。
   *
   * 之前这里只显示「物证点（3）」这样的计数，看不到究竟是哪几个，
   * 也没法单独隐藏某一类 —— 用户反馈「图层板块缺乏显示具体项目的能力」。
   * 现在：点击分类标题展开条目列表，点条目定位，右侧开关控制该类显隐。
   */
  renderLayerList() {
    const p = this.currentProject;
    if (!p) return;
    const anns = p.annotations ?? [];
    const groups = [
      { key: 'model', name: '三维现场（3DGS）', color: '#37c2d8', items: [] },
      { key: 'point', name: '物证点', color: defaultColor('point'), kind: 'point' },
      { key: 'polyline', name: '行动轨迹', color: defaultColor('polyline'), kind: 'polyline' },
      { key: 'polygon', name: '区域', color: defaultColor('polygon'), kind: 'polygon' },
      { key: 'measure', name: '测距', color: defaultColor('measure'), kind: 'measure' },
      { key: 'camera', name: '监控/勘查相机', color: defaultColor('camera'), kind: 'camera' },
    ];
    for (const g of groups) {
      if (!g.kind) continue;
      g.items = anns.filter((a) => a.kind === g.kind);
    }
    const hidden = this.hiddenLayers ?? (this.hiddenLayers = new Set());
    const expanded = this.expandedLayers ?? (this.expandedLayers = new Set(['point']));

    this.el.layerList.innerHTML = groups
      .map((g) => {
        const n = g.key === 'model' ? (this.currentProject?.model ? 1 : 0) : g.items.length;
        const isHidden = hidden.has(g.key);
        const isOpen = expanded.has(g.key);
        const canExpand = n > 0 && g.kind;
        const sub = canExpand
          ? `<div class="layer-sub" ${isOpen ? '' : 'hidden'}>` +
            g.items
              .map(
                (a) =>
                  `<div class="layer-subitem" data-anno="${a.id}" title="${escapeAttr(a.label ?? '')}">
                     <span class="subdot" style="background:${a.color ?? g.color}"></span>
                     <span class="subname">${escapeHtml(a.label || kindName(a.kind, a))}</span>
                   </div>`
              )
              .join('') +
            `</div>`
          : '';
        return `<div class="layer-group${isHidden ? ' is-hidden' : ''}">
          <div class="layer-item" data-layer="${g.key}">
            <button class="layer-caret" data-caret="${g.key}" ${canExpand ? '' : 'disabled'} aria-label="展开">${canExpand ? (isOpen ? '▾' : '▸') : '·'}</button>
            <span class="dot" style="background:${g.color};color:${g.color}"></span>
            <span class="label">${escapeHtml(g.name)}</span>
            <span class="count">${n}</span>
            <button class="layer-eye" data-eye="${g.key}" title="${isHidden ? '显示' : '隐藏'}该图层">${isHidden ? '🚫' : '👁'}</button>
          </div>
          ${sub}
        </div>`;
      })
      .join('');

    for (const caret of this.el.layerList.querySelectorAll('[data-caret]')) {
      caret.addEventListener('click', (ev) => {
        ev.stopPropagation();
        const key = caret.dataset.caret;
        if (expanded.has(key)) expanded.delete(key);
        else expanded.add(key);
        this.renderLayerList();
      });
    }
    for (const eye of this.el.layerList.querySelectorAll('[data-eye]')) {
      eye.addEventListener('click', (ev) => {
        ev.stopPropagation();
        const key = eye.dataset.eye;
        if (hidden.has(key)) hidden.delete(key);
        else hidden.add(key);
        this.renderLayerList();
        this.handlers.onLayerToggle?.(key, !hidden.has(key));
      });
    }
    for (const item of this.el.layerList.querySelectorAll('[data-anno]')) {
      item.addEventListener('click', () => {
        this.handlers.onSelectAnnotation?.(item.dataset.anno);
      });
    }
  }

  // ---------------------------------------------------------------- 标注

  renderAnnotations(annotations) {
    this.annotations = annotations ?? [];
    this.el.annoCount.textContent = String(this.annotations.length);
    // 图层树也要跟着刷新，否则新增/删除标注后，图层里的条目会过期
    if (this.currentProject) this.currentProject.annotations = this.annotations;
    this.renderLayerList();
    const list = this.el.annoList;
    const filter = this.filter;
    const items = this.annotations.filter((a) => {
      if (!filter) return true;
      return `${a.label} ${a.note} ${kindName(a.kind, a)} ${a.measured ? "已实测" : ""}`.toLowerCase().includes(filter);
    });
    if (!items.length) {
      list.innerHTML = '<div class="hint">还没有标注。选择左侧工具后，在三维现场里点击即可添加。</div>';
      return;
    }
    list.innerHTML = items
      .map((a) => {
        const color = a.color ?? defaultColor(a.kind);
        const sel = a.id === this.selectedId ? ' selected' : '';
        return `<div class="anno-item${sel}" data-id="${a.id}" title="单击选中并定位，双击也只会定位（删除请用右侧的删除按钮）">
          <span class="dot" style="background:${color};color:${color}"></span>
          <span class="label">${escapeHtml(a.label || kindName(a.kind, a))}</span>
          <span class="kind${a.kind === "dimension" ? (a.measured ? " ok" : " warn") : ""}">${kindName(a.kind, a)}</span>
          <button class="anno-del" data-del="${a.id}" title="删除该标注">✕</button>
        </div>`;
      })
      .join('');
    for (const node of list.querySelectorAll('.anno-item')) {
      node.addEventListener('click', (ev) => {
        if (ev.target.closest('[data-del]')) return;
        this.handlers.onSelectAnnotation?.(node.dataset.id);
      });
      // 双击原本是删除，极易误触（尤其是想「双击定位」的时候），改成定位
      node.addEventListener('dblclick', (ev) => {
        if (ev.target.closest('[data-del]')) return;
        this.handlers.onFocusAnnotation?.(node.dataset.id);
      });
    }
    for (const btn of list.querySelectorAll('[data-del]')) {
      btn.addEventListener('click', async (ev) => {
        ev.stopPropagation();
        ev.preventDefault();
        const id = btn.dataset.del;
        const ann = this.annotations.find((a) => a.id === id);
        const name = ann?.label || (ann ? kindName(ann.kind, ann) : '该对象');
        const ok = await this.confirm({
          title: '删除该对象',
          body: `确定删除「${name}」吗？`,
          warn: '该操作会同步给同案其他人员，且不可撤销。',
        });
        if (ok) this.handlers.onDeleteAnnotation?.(id);
      });
    }
  }

  highlightAnnotation(id) {
    this.selectedId = id;
    for (const node of this.el.annoList.querySelectorAll('.anno-item')) {
      node.classList.toggle('selected', node.dataset.id === id);
    }
  }

  renderInspector(ann, opts = {}) {
    const box = this.el.inspector;
    if (!ann) {
      box.innerHTML = '<div class="hint">未选中任何对象</div>';
      return;
    }
    const color = ann.color ?? defaultColor(ann.kind);
    const pts = ann.points ?? [];
    const coordsText = pts
      .slice(0, 6)
      .map((p, i) => `#${i + 1} (${p.map((v) => v.toFixed(3)).join(', ')})`)
      .join('<br>');
    let extra = '';
    if (ann.kind === 'measure' && pts.length >= 2) {
      let total = 0;
      for (let i = 1; i < pts.length; i++) total += dist(pts[i - 1], pts[i]);
      extra = `<div class="coords">总距离 ${total.toFixed(4)} m</div>`;
    }
    // 尺寸标注：显示模型值与实测量，并提供「填写实测值」表单
    if (ann.kind === 'dimension' && pts.length >= 2) {
      const scale = Number(this.currentProject?.world?.scaleToMeters ?? 1) || 1;
      const modelM = Number(ann.modelDistance ?? dist(pts[0], pts[1]));
      const scaled = modelM * scale;
      const measured = Number(ann.measured);
      const has = Number.isFinite(measured) && measured > 0;
      const err = has ? scaled - measured : null;
      extra =
        `<div class="dim-box${has ? ' done' : ' pending'}">` +
        `<div class="dim-row"><span>模型侧距离</span><b>${scaled.toFixed(4)} m</b></div>` +
        (has
          ? `<div class="dim-row"><span>民警实测</span><b class="ok">${measured.toFixed(4)} m</b></div>` +
            `<div class="dim-row"><span>偏差</span><b class="${Math.abs(err) < 0.005 ? 'ok' : 'warn'}">${err >= 0 ? '+' : ''}${err.toFixed(4)} m</b></div>` +
            `<div class="dim-row"><span>测量人</span><b>${escapeHtml(ann.measuredBy ?? '-')}</b></div>`
          : `<div class="dim-row warn"><span>状态</span><b>待民警实测</b></div>`) +
        `</div>` +
        `<label class="dim-input"><span>实测距离 (m)</span>
           <input id="insMeasured" type="number" step="0.001" min="0" placeholder="施工现场量出的米数"
                  value="${has ? measured : ''}" />
         </label>` +
        `<div class="row dim-actions">
           <button id="insMeasuredSave" class="primary small">${has ? '更新实测值' : '提交实测值'}</button>
           <button id="insCalibrate" class="ghost small" title="用所有已实测的尺寸反算比例尺">按实测校正比例尺</button>
         </div>`;
    }
    if (ann.kind === 'polyline' && pts.length >= 2) {
      let total = 0;
      for (let i = 1; i < pts.length; i++) total += dist(pts[i - 1], pts[i]);
      extra = `<div class="coords">轨迹长度 ${total.toFixed(3)} m · ${pts.length} 个点</div>`;
    }
    if (ann.kind === 'camera') {
      const pose = ann.pose ?? {};
      const err = pose.reprojectionError;
      extra = `<div class="coords">标定方式：${pose.source ?? 'manual'}${err != null ? ` · 重投影误差 ${Number(err).toFixed(3)} px` : ''}<br>
        机位：${(pose.position ?? []).map((v) => Number(v).toFixed(3)).join(', ')}<br>
        视场角：${pose.fovY ?? 45}° · 宽高比 ${(pose.aspect ?? 1.78).toFixed(2)}</div>`;
    }

    box.innerHTML = `
      <label><span>名称</span><input id="insLabel" value="${escapeAttr(ann.label)}"></label>
      <label><span>备注</span><input id="insNote" value="${escapeAttr(ann.note ?? '')}"></label>
      <label><span>颜色</span><input id="insColor" type="color" value="${color}"></label>
      <div class="coords">${coordsText}${extra ? '<br>' + extra : ''}</div>
      ${
        ann.kind === 'camera' && ann.media?.[0]
          ? `<div class="ins-media">
               <button id="insPlay" class="ghost small">贴回现场并播放</button>
               <button id="insCalib" class="ghost small">PnP 标定…</button>
               <span class="hint">画面文件：${escapeHtml(ann.media[0].url?.split('/').pop() ?? '')}</span>
             </div>`
          : ''
      }
      <div class="row">
        <button id="insFocus" class="ghost small">定位</button>
        <button id="insVideo" class="ghost small">${ann.kind === 'camera' ? (ann.media?.[0] ? '换视频' : '导入视频') : '附照片'}</button>
        <button id="insDelete" class="danger small">删除</button>
      </div>
      <div class="hint">创建者：${escapeHtml(ann.createdBy ?? '-')}</div>
    `;
    $('#insLabel')?.addEventListener('change', (e) => opts.onUpdate?.({ label: e.target.value }));
    $('#insNote')?.addEventListener('change', (e) => opts.onUpdate?.({ note: e.target.value }));
    $('#insColor')?.addEventListener('change', (e) => opts.onUpdate?.({ color: e.target.value }));
    $('#insFocus')?.addEventListener('click', () => opts.onFocus?.());
    $('#insVideo')?.addEventListener('click', () => this.openImportDialog(ann));
    $('#insPlay')?.addEventListener('click', () => opts.onAttachVideo?.(ann));
    $('#insCalib')?.addEventListener('click', () => opts.onCalibrate?.(ann));
    // 尺寸标注：提交实测值 / 触发比例尺校正
    const submit = () => {
      const raw = $('#insMeasured')?.value;
      const v = Number(raw);
      if (!Number.isFinite(v) || v <= 0) {
        this.toast('请填写一个大于 0 的实测距离（米）', { error: true });
        return;
      }
      opts.onUpdate?.({ measured: v });
    };
    $('#insMeasuredSave')?.addEventListener('click', submit);
    $('#insMeasured')?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        submit();
      }
    });
    $('#insCalibrate')?.addEventListener('click', () => opts.onRecalibrate?.());
    $('#insDelete')?.addEventListener('click', async () => {
      const name = ann.label || kindName(ann.kind, ann);
      // 同样不用 window.confirm：被屏蔽时按钮会「毫无反应」
      const ok = await this.confirm({
        title: '删除标注',
        body: `确定删除「${name}」吗？该操作会同步给同案其他人员。`,
        okText: '确认删除',
      });
      if (ok) opts.onDelete?.();
    });
  }

  // ---------------------------------------------------------------- 人员/协同

  renderUsers(roster, selfId) {
    const list = this.el.userList;
    list.innerHTML = (roster ?? [])
      .map((u) => {
        const color = u.color ?? '#4dd0e1';
        const self = u.id === selfId ? '（我）' : '';
        return `<div class="user-item">
          <span class="avatar" style="background:${color}">${escapeHtml((u.name ?? '?').slice(0, 1))}</span>
          <span class="nm">${escapeHtml(u.name ?? '匿名')}${self}</span>
          <span class="role">${u.role === 'commander' ? '指挥' : u.role === 'investigator' ? '研判' : '观察'}</span>
        </div>`;
      })
      .join('');
    this.el.presence.innerHTML = (roster ?? [])
      .slice(0, 8)
      .map((u) => `<span class="avatar" style="background:${u.color ?? '#4dd0e1'}" title="${escapeAttr(u.name)}">${escapeHtml((u.name ?? '?').slice(0, 1))}</span>`)
      .join('');
  }

  setIdentity(name, color) {
    this._identity = { name, color };
  }

  setConnState(state, detail = '') {
    const el = this.el.connState;
    el.className = `conn ${state === 'online' ? 'online' : state === 'offline' ? 'offline' : ''}`;
    el.textContent =
      state === 'online' ? '协同在线' : state === 'connecting' ? '连接中…' : state === 'reconnecting' ? `重连 ${detail}` : '离线';
  }

  setServerInfo(info) {
    this._server = info;
  }

  pushChat(msg) {
    const div = document.createElement('div');
    div.className = 'msg';
    div.innerHTML = `<span class="who" style="color:${msg.color ?? '#4dd0e1'}">${escapeHtml(msg.name)}</span>${escapeHtml(msg.text)}`;
    this.el.chatLog.appendChild(div);
    this.el.chatLog.scrollTop = this.el.chatLog.scrollHeight;
  }

  pushSystem(text) {
    const div = document.createElement('div');
    div.className = 'sys';
    div.textContent = `· ${text}`;
    this.el.chatLog.appendChild(div);
    this.el.chatLog.scrollTop = this.el.chatLog.scrollHeight;
  }

  // ---------------------------------------------------------------- 相机条

  renderCameras(cameras, overlay) {
    const strip = this.el.cameraStrip;
    if (!cameras?.length) {
      strip.innerHTML = '<div class="hint" style="padding:8px">尚无监控/勘查画面。点击右上「+ 导入监控/勘查视频」上传后，可在此预览并做 PnP 标定。</div>';
      return;
    }
    strip.innerHTML = cameras
      .map((c) => {
        const media = c.media?.[0];
        const pnp = c.pose?.source === 'pnp' ? 'ok' : 'no';
        const pnpText = c.pose?.source === 'pnp' ? `PnP ${Number(c.pose.reprojectionError ?? 0).toFixed(2)}px` : '未标定';
        const inner = media
          ? media.kind === 'image'
            ? `<img src="${media.url}" alt="">`
            : `<video src="${media.url}" muted loop playsinline preload="metadata"></video>`
          : '<div class="placeholder"></div>';
        return `<div class="cam-card" data-id="${c.id}">
          ${inner}
          <div class="cap"><span class="nm">${escapeHtml(c.label || '画面')}</span><span class="state ${pnp}">${pnpText}</span></div>
          <div class="cam-actions">
            <button class="ghost small" data-act="goto">贴回现场</button>
            <button class="ghost small" data-act="calib" title="在画面上点像素、在三维现场点对应点，解算相机位姿">PnP 标定</button>
            <button class="ghost small" data-act="import">${media ? '换视频' : '导入视频'}</button>
            <button class="ghost small danger" data-act="delete" title="从本案删除这个画面（含已上传的文件）">✕ 删除</button>
          </div>
        </div>`;
      })
      .join('');
    for (const card of strip.querySelectorAll('.cam-card')) {
      const id = card.dataset.id;
      card.addEventListener('click', (ev) => {
        if (ev.target.closest('button')) return;
        this.handlers.onCameraSelect?.(id);
        for (const c of strip.querySelectorAll('.cam-card')) c.classList.toggle('active', c.dataset.id === id);
      });
      for (const btn of card.querySelectorAll('button[data-act]')) {
        btn.addEventListener('click', (ev) => {
          ev.stopPropagation();
          ev.preventDefault();
          const act = btn.dataset.act;
          const ann = this.annotations.find((a) => a.id === id);
          if (act === 'goto') this.handlers.onCameraGoto?.(id);
          else if (act === 'calib') this.handlers.onCameraCalibrate?.(ann);
          else if (act === 'delete') this.handlers.onCameraDelete?.(id);
          else this.openImportDialog(ann);
        });
      }
      const video = card.querySelector('video');
      if (video) {
        video.addEventListener('mouseenter', () => video.play().catch(() => {}));
        video.addEventListener('mouseleave', () => video.pause());
        video.addEventListener('click', (ev) => {
          ev.stopPropagation();
          this.handlers.onCameraPlayToggle?.(id);
        });
      }
    }
  }

  // ---------------------------------------------------------------- 浮层

  /**
   * 弹出确认对话框（Promise 风格）。
   *
   * 为什么不用 window.confirm：
   *   · 部分环境/嵌入式浏览器会屏蔽原生弹窗，表现为「点了删除毫无反应」——
   *     这正是用户反馈「点叉号没反应」的可能原因之一；
   *   · 原生弹窗无法展示「会影响什么」，而删除媒体文件需要说清楚后果。
   *
   * @returns {Promise<boolean>} 用户是否确认
   */
  confirm({ title = '确认操作', body = '', warn = '', okText = '确认删除' } = {}) {
    const dlg = document.getElementById('dlgConfirm');
    const t = document.getElementById('cfTitle');
    const b = document.getElementById('cfBody');
    const w = document.getElementById('cfWarn');
    const ok = document.getElementById('cfOk');
    const cancel = document.getElementById('cfCancel');
    if (!dlg || !t || !b || !w || !ok || !cancel) {
      // 兜底：对话框元素缺失时退回原生确认，至少不静默失败
      console.warn('[ui] 缺少确认对话框元素，回退到 window.confirm');
      return Promise.resolve(window.confirm(`${title}\n\n${body}${warn ? '\n\n' + warn : ''}`));
    }
    t.textContent = title;
    b.textContent = body;
    w.textContent = warn;
    w.classList.toggle('hidden', !warn);
    ok.textContent = okText;

    return new Promise((resolve) => {
      let settled = false;
      const finish = (value) => {
        if (settled) return;
        settled = true;
        dlg.removeEventListener('close', onClose);
        resolve(value);
      };
      const onClose = () => finish(dlg.returnValue === 'ok');
      dlg.addEventListener('close', onClose);
      // 点遮罩（dialog 自身）视为取消
      dlg.addEventListener('click', (ev) => {
        if (ev.target === dlg) closeDialogModal(dlg, 'cancel');
      });
      dlg.returnValue = 'cancel';
      openDialogModal(dlg);
      // 默认聚焦「取消」，避免回车误删
      cancel.focus();
    });
  }

  toast(message, { error = false, sticky = false, duration = 3200 } = {}) {
    const el = this.el.toast;
    el.textContent = message;
    el.className = `toast show${error ? ' error' : ''}`;
    clearTimeout(this.toastTimer);
    if (!sticky) this.toastTimer = setTimeout(() => (el.className = 'toast'), duration);
  }

  moveHover(ev, text) {
    const el = this.el.hoverTip;
    el.style.display = 'block';
    el.style.left = `${ev.clientX + 14}px`;
    el.style.top = `${ev.clientY + 14}px`;
    el.textContent = text;
  }

  hideHover() {
    this.el.hoverTip.style.display = 'none';
  }

  setToolHint(text) {
    this.el.toolHint.textContent = text;
  }

  setMode(mode) {
    for (const b of this.el.modeSwitch.querySelectorAll('button')) {
      b.classList.toggle('active', b.dataset.mode === mode);
    }
  }

  setTool(tool) {
    for (const b of this.el.toolBar.querySelectorAll('button')) {
      b.classList.toggle('active', b.dataset.tool === tool);
    }
  }

  setPresentation(f) {
    this.el.rGaussScale.value = String(f.gaussScale ?? 1);
    this.el.vScale.textContent = Number(f.gaussScale ?? 1).toFixed(2);
    this.el.rOpacity.value = String(f.opacity ?? 1);
    this.el.vOpacity.textContent = Number(f.opacity ?? 1).toFixed(2);
    this.el.rExposure.value = String(f.exposure ?? 1);
    this.el.vExposure.textContent = Number(f.exposure ?? 1).toFixed(2);
    this.el.cSort.checked = f.sort !== false;
  }

  // ---------------------------------------------------------------- 载入遮罩

  showLoader(text) {
    this.el.loader.classList.remove('hidden');
    this.el.loaderText.textContent = text ?? '处理中…';
    this.setLoaderProgress(0, '');
  }

  setLoaderStage(stage, progress) {
    const map = {
      downloading: '正在下载模型数据…',
      decoding: '正在解析高斯数据…',
      building: '正在构建渲染缓冲…',
      sorting: '正在排序高斯球…',
    };
    if (map[stage]) this.el.loaderText.textContent = map[stage];
    if (typeof progress === 'number') this.setLoaderProgress(progress, '');
  }

  setLoaderProgress(p, sub) {
    this.el.loaderBar.style.width = `${Math.max(0, Math.min(1, p)) * 100}%`;
    if (sub !== undefined) this.el.loaderSub.textContent = sub;
  }

  hideLoader() {
    this.el.loader.classList.add('hidden');
    this.el.loaderBar.style.width = '0%';
  }

  // ---------------------------------------------------------------- HUD

  renderHud({ fps, gaussians, drawn, sortMs }) {
    this._hudTick = (this._hudTick ?? 0) + 1;
    if (this._hudTick % 12 !== 0) return;
    this.el.statFps.textContent = `${fps.toFixed(0)} FPS`;
    this.el.statGauss.textContent = gaussians
      ? `${fmtCount(drawn || gaussians)} 高斯 · 排序 ${sortMs.toFixed(1)}ms`
      : '未载入模型';
  }

  startClock() {
    setInterval(() => {
      if (!this._hudTick) this.el.statFps.textContent = '-- FPS';
    }, 2000);
  }

  resize() {
    /* 预留：面板折叠等响应式处理 */
  }
}

function dist(a, b) {
  return Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
}

function fmtCount(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function escapeAttr(s) {
  return escapeHtml(s).replace(/`/g, '&#96;');
}
