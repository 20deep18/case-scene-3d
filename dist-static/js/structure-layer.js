/**
 * 房间结构图层：把 `*.structure.json` 画成**屏幕上的空间线条**。
 *
 * 为什么需要它
 * ------------
 * 3D 高斯泼溅呈现的是「照片云」——能看见材质，看不见结构。
 * 用户的原话是「没有呈现出房间的整体布局和空间线条」。
 * `tools/scene_structure.py` 已经从 COLMAP 稀疏点云里提炼出地面 / 房间盒 /
 * 主墙面 / 相机轨迹，这一层负责把它们画出来。
 *
 * ⚠️ 为什么是 **DOM/SVG** 而不是 Three.js 网格（踩过的坑）
 * ------------------------------------------------------
 * 本项目**没有 Three.js 渲染器**：`splat-renderer.js` 自己持有 WebGL2 上下文，
 * `render()` 里第一件事就是 `gl.clear()` 然后只画高斯；全项目搜不到
 * `new THREE.WebGLRenderer`。所以往 `THREE.Scene` 里加网格**根本不会被绘制** ——
 * 我第一版就是这么写的，测试 12/12 全过、截图里却一条线都看不到
 * （那些断言只证明了「场景图里有对象」，不证明可见）。
 * 标注、标签这些叠加层实际都走 DOM，因此这一层也走 DOM：
 * 每帧把 COLMAP 世界坐标投影到屏幕，更新 SVG 的 line/polyline/circle。
 *
 * 坐标系：structure.json 是 **COLMAP 世界系**（x 右 / y 下 / z 前），
 * 与标注、相机位姿同一套；投影前统一乘 WORLD_MATRIX 转到 Three.js 场景系。
 */
import * as THREE from 'three';
import { WORLD_MATRIX, WORLD_QUAT } from './camera-overlay.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const COLOR_BOX = '#4dd0e1';   // 房间盒：青
const COLOR_GRID = '#2f6f7a';  // 地面网格：暗青
const COLOR_WALL = '#ffcc33';  // 墙面：黄
const COLOR_TRAJ = '#ff8a80';  // 相机轨迹：橙红
const COLOR_PLAN = '#b388ff';  // 地面平面轮廓：紫（与房间盒的青、墙的黄区分开）
const COLOR_START = '#69f0ae'; // 起点：绿
const COLOR_END = '#ff5252';   // 终点：红
const COLOR_FRUSTUM = '#ffd54f';   // 相机视锥（PnP 标定）：亮黄，虚线
const COLOR_CANDIDATE = '#00e5ff'; // 候选取点：亮青
const COLOR_HALO = '#0b1620';      // 描边光晕：深色，垫在亮线/亮点下面提高对比度
// 物体（柜子）专用色：与房间盒的青、墙的黄、轨迹的橙红都拉开距离，
// 否则像素级回归测试会把它们混在一起数（测试里数的是前三个颜色的增量）。
const COLOR_OBJ = '#ff9f43';       // 物体盒：橙
const COLOR_FURN = '#ff7043';      // 家具盒默认色（未知类别时用）

/**
 * 家具盒的**类别配色**（专家 Q5 建议：按类别编码颜色，评审一眼能分出床/门/柜子）。
 * 未列出的类别用确定性哈希落到同一组色板里 —— 保证同一类别在不同模型上颜色一致。
 */
const CATEGORY_COLORS = {
  bed: '#4fc3f7', 'bunk bed': '#29b6f6',
  door: '#ffd54f', wardrobe: '#a5d6a7', cabinet: '#81c784',
  desk: '#ba68c8', table: '#ba68c8', chair: '#ff8a65',
  sofa: '#4db6ac', toilet: '#e57373', sink: '#90a4ae',
};
const CATEGORY_PALETTE = ['#4fc3f7', '#a5d6a7', '#ba68c8', '#ff8a65', '#4db6ac', '#ffd54f', '#e57373', '#90a4ae'];

function categoryColor(label) {
  const key = String(label ?? '').trim().toLowerCase();
  if (!key) return COLOR_FURN;
  if (CATEGORY_COLORS[key]) return CATEGORY_COLORS[key];
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
  return CATEGORY_PALETTE[h % CATEGORY_PALETTE.length];
}
const COLOR_OBJ_LAYER = '#7cf2c0'; // 物体水平层（搁板/顶面）：薄荷绿

export class StructureLayer {
  /**
   * @param {HTMLElement} host 覆盖层容器（ui.labelHost）
   * @param {THREE.Camera} camera
   */
  constructor(host, camera) {
    this.host = host;
    this.camera = camera;
    this.visible = true;
    this.data = null;
    /** 场景判定：'room' | 'object' | 'uncertain' | 'none'（未载入） */
    this.mode = 'none';
    /** 不画房间盒时，给界面用的解释（用户要能看到"为什么"） */
    this.notice = '';
    /** 判定被否的理由（来自 structure.json 的 sceneModeEvidence.rejectReasons） */
    this.reasons = [];
    /** @type {Array<{a:number[],b:number[],color:string,width:number,opacity:number}>} */
    this.segments = [];
    /** @type {Array<{pts:number[][],color:string,width:number,opacity:number}>} */
    this.polylines = [];
    /** @type {Array<{p:number[],color:string,r:number}>} */
    this.dots = [];
    /** @type {Array<{p:number[],text:string,color:string}>} */
    this.labels = [];
    /**
     * 临时图层：PnP 标定时的**相机视锥**与**候选点**。
     * 与结构线框分开存 —— 重建结构不会清掉它，反之亦然。
     */
    this.transientSegments = [];
    this.transientDots = [];
    /**
     * 家具盒（来自 `*.furniture.json`，由 tools/furniture_lines.py 产出）。
     *
     * 为什么和 structure.json 分开：家具是从**稠密表面点**聚类出来的，坐标系可能与
     * structure.json 不同（VGGT/稠密重建没有曼哈顿系）。两者独立加载、独立开关 ——
     * 这样"结构判定不通过"时家具盒仍能显示，它不依赖房间盒那个假设。
     */
    this.furniture = null;
    this.furnitureVisible = true;
    this.furnitureDrawn = 0;
    // 竖直刻度尺（让评委一眼看出"这是多高的空间"）。见 setRuler()。
    this.ruler = null;
    this.rulerDrawn = 0;
    /**
     * 重建单位 → 米的换算系数（来自「按实测测量校正」）。
     *
     * 为什么要它：结构文件里的尺寸都是**重建单位**，而白瓷砖那份重建
     * 1 单位 ≈ 0.09 m —— 直接标成"24.91 m"会严重误导（实测截图抓到）。
     * 校正过就乘系数；没校正过就是 1（此时标签写"单位"不写"m"）。
     */
    this.scaleToMeters = 1;
    this._tels = [];
    this._els = [];
    this._v = new THREE.Vector3();
    this._svg = null;
    this._buildSvg();
  }

  _buildSvg() {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('id', 'structureLayer');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.style.position = 'absolute';
    svg.style.inset = '0';
    svg.style.pointerEvents = 'none';
    svg.style.overflow = 'visible';
    // 显式抬到 canvas 之上：原来只靠 DOM 顺序，容易被后续插入的节点压下去。
    // 用户反馈「视锥和高斯球重合（看不出哪个在上）」——层级别再靠运气。
    svg.style.zIndex = '60';
    (this.host ?? document.getElementById('app')).appendChild(svg);
    this._svg = svg;
  }

  setVisible(v) {
    this.visible = !!v;
    if (this._svg) this._svg.style.display = this.visible ? 'block' : 'none';
    return this.visible;
  }

  /**
   * 载入/清空家具盒（`*.furniture.json`）。传 null 表示该模型没有家具数据。
   *
   * 与 `set()` 分开是因为两者**来自不同文件、不同坐标系**：结构文件是 COLMAP 世界系 +
   * 曼哈顿系，家具文件是它自己的帧（稠密重建/VGGT 没有曼哈顿系）。
   * 结构已载入时会立刻重画，保证切换模型/开关后画面一致。
   */
  setFurniture(data) {
    const ok = data && Array.isArray(data.objects) && data.objects.length > 0;
    this.furniture = ok ? data : null;
    // 有结构文件就走完整重建；**没有结构文件也要能单独画家具盒** ——
    // 稠密重建/VGGT 的模型可能根本没有 `.structure.json`（实测就踩到了：
    //  没有结构时 `set()` 根本不会被调用，家具永远画不出来）。
    if (this.data) this.set(this.data);
    else this._renderFurnitureOnly();
    return this.furnitureDrawn;
  }

  /** 没有结构文件时，只画家具盒（房间盒/墙面/轨迹都没有，这是诚实的）。 */
  _renderFurnitureOnly() {
    this.segments = [];
    this.polylines = [];
    this.dots = [];
    this.labels = [];
    for (const el of this._els) el.remove();
    this._els = [];
    this.mode = this.data ? this.mode : 'none';
    this.furnitureDrawn = 0;
    if (this.ruler) this._addRuler();
    if (this.furniture && this.furnitureVisible) {
      this.furnitureDrawn = this._addFurniture(this.furniture);
    }
    this._createElements();
    return this.furnitureDrawn;
  }

  /** 设置「重建单位 → 米」的换算系数；有结构数据时立刻重画。 */
  setCalibration(scaleToMeters) {
    const s = Number(scaleToMeters);
    this.scaleToMeters = Number.isFinite(s) && s > 0 ? s : 1;
    if (this.data) this.set(this.data);
    else this._renderFurnitureOnly();
    return this.scaleToMeters;
  }

  /** 尺寸文本：校正过就写成米，否则标成"单位"（不谎称是米）。 */
  _sizeText(v) {
    const s = this.scaleToMeters ?? 1;
    const calibrated = Math.abs(s - 1) > 1e-6;
    return calibrated ? `${(v * s).toFixed(2)} m` : `${v.toFixed(2)} 单位`;
  }

  /**
   * 竖直刻度尺：在场景一角画一条从地面到顶部的竖线，每 `spacing` 画一条横刻度并标注。
   *
   * 为什么需要：评委看到的是一团点云，很难建立"多高"的空间感；
   * 一条带米数刻度的竖线能立刻把"这是 2.4 m 高的房间、床面在 0.9 m"讲清楚。
   *
   * @param {{origin:number[], up:number[], height:number, spacing:number, axisU?:number[]}} opts
   *   origin 为**地面**上的锚点（世界坐标），up 为竖直方向，height 为总高（重建单位）。
   */
  setRuler(opts) {
    if (!opts || !opts.origin || !opts.up || !(opts.height > 0)) {
      this.ruler = null;
    } else {
      this.ruler = {
        origin: opts.origin,
        up: opts.up,
        height: opts.height,
        spacing: opts.spacing > 0 ? opts.spacing : opts.height / 4,
        axisU: opts.axisU ?? [1, 0, 0],
      };
    }
    if (this.data) this.set(this.data);
    else this._renderFurnitureOnly();
    return this.rulerDrawn;
  }

  /** 家具盒的显示开关（独立于结构线框的开关）。 */
  setFurnitureVisible(v) {
    this.furnitureVisible = !!v;
    if (this.data) this.set(this.data);
    else this._renderFurnitureOnly();
    return this.furnitureVisible;
  }

  clear() {
    this.data = null;
    this.segments = [];
    this.polylines = [];
    this.dots = [];
    this.labels = [];
    for (const el of this._els) el.remove();
    this._els = [];
  }

  // ---------------------------------------------------------------- 临时图层

  /**
   * 画一组**临时**几何（相机视锥 + 候选点），供 PnP 标定时参考。
   *
   * 为什么需要：用户反馈「很难选一个合适的三维标定点」—— 因为屏幕上
   * 看不出「从这台相机看出去，哪块空间是拍到的、可以当对应点」。
   * 视锥线框给出几何上正确的可选范围；候选点给出点云实际存在的位置。
   *
   * @param {Array<{a:number[],b:number[]}>} segments 世界坐标（COLMAP 系）线段
   * @param {Array<{p:number[]}>} dots 世界坐标点
   * @param {{line?:string,dot?:string}} [style]
   */
  setTransient(segments = [], dots = [], style = {}) {
    this.clearTransient();
    if (!this._svg) return 0;
    const lineColor = style.line ?? COLOR_FRUSTUM;
    const dotColor = style.dot ?? COLOR_CANDIDATE;
    // 每条线画**两遍**：底下一遍深色粗描边（halo），上面一遍亮色细线。
    // 只画一遍亮色细线的话，压在花花绿绿的高斯上根本看不清
    // （用户反馈「视锥和候选点不明显，和高斯球重合了」）。
    for (const s of segments) {
      const halo = document.createElementNS(SVG_NS, 'line');
      halo.setAttribute('stroke', COLOR_HALO);
      halo.setAttribute('stroke-width', '5.5');
      halo.setAttribute('stroke-opacity', '0.75');
      halo.setAttribute('stroke-linecap', 'round');
      this._svg.appendChild(halo);
      this._tels.push(halo);
      const haloRef = { a: s.a, b: s.b, el: halo, kind: 'line' };

      const el = document.createElementNS(SVG_NS, 'line');
      el.setAttribute('stroke', lineColor);
      el.setAttribute('stroke-width', '2.6');
      el.setAttribute('stroke-opacity', '1');
      // 虚线：与结构线框（实线）一眼区分，表明这是「临时提示」
      el.setAttribute('stroke-dasharray', '9 6');
      el.setAttribute('stroke-linecap', 'round');
      this._svg.appendChild(el);
      this._tels.push(el);
      this.transientSegments.push({ a: s.a, b: s.b, el });
      this.transientSegments.push(haloRef);
    }
    for (const d of dots) {
      const halo = document.createElementNS(SVG_NS, 'circle');
      halo.setAttribute('fill', COLOR_HALO);
      halo.setAttribute('r', '5.2');
      halo.setAttribute('fill-opacity', '0.6');
      this._svg.appendChild(halo);
      this._tels.push(halo);
      this.transientDots.push({ p: d.p, el: halo });

      const el = document.createElementNS(SVG_NS, 'circle');
      el.setAttribute('fill', dotColor);
      el.setAttribute('r', '3.2');
      el.setAttribute('fill-opacity', '1');
      this._svg.appendChild(el);
      this._tels.push(el);
      this.transientDots.push({ p: d.p, el });
    }
    return this.transientSegments.length + this.transientDots.length;
  }

  clearTransient() {
    for (const el of this._tels) el.remove();
    this._tels = [];
    this.transientSegments = [];
    this.transientDots = [];
  }

  get hasTransient() {
    return this._tels.length > 0;
  }

  /**
   * 用一份 structure.json 重建图层；传 null 等于清空。返回构造出的图形数量。
   *
   * ⚠️ 2026-10-06：**按场景判定的可信度决定画什么**（以前是无条件画房间盒）。
   * 起因是实测发现：柜子那份素材被画上了一个假的房间盒 ——
   * `sceneMode` 字段在 `app/` 里当时零引用，`set()` 无条件 `_addRoomBox`。
   * 用户对这个问题的原话是「明明是个柜子，但是模型是不规则的」，
   * 而界面上再糊一个 6.0 × 3.7 × 3.7 m 的"房间盒"只会更误导。
   *
   * 三态（来自 tools/scene_structure.py 的证据式判定）：
   *   room      → 房间盒 / 地面网格 / 墙面 / 轨迹 / 平面轮廓（全都画）
   *   object    → **只画相机轨迹**（房间盒、地面网格、墙面都来自"房间"假设，不能用）
   *   uncertain → **只画相机轨迹**，并把"为什么不可信"写进 notice 让界面说出来
   * 另外 `sane === false`（上方向/地面可疑）时一律降级为"只画轨迹"：
   * 那种情况下房间盒可能是倒扣的，画出来比不画更糟。
   *
   * 兼容：历史结构文件没有 `sceneMode` 字段，此时退回按 `sane` 判断（保持旧行为），
   * 这样老模型不会因为这次改动突然变成一片空白。
   */
  set(data) {
    this.clear();
    this.mode = 'none';
    this.notice = '';
    this.reasons = [];
    if (!data || !data.room) return 0;
    this.data = data;
    const sane = data.sane !== false;
    const mode = data.sceneMode ?? (sane ? 'room' : 'uncertain');
    this.mode = mode;
    // auto = 证据自动判的；manual = 用户在前端改判（服务端 sidecar）或 --scene-mode 指定
    this.modeSource = data.sceneModeSource === 'manual' ? 'manual' : 'auto';
    this.overriddenBy = data.sceneModeOverrideBy ?? null;
    // **改判前**的自动判定（界面要能同时显示"现在按什么处理"和"原本自动判成什么"）。
    // 优先取结构文件里记录的 autoDecision；老文件没有这个字段时，
    // 只有"非人工改判"才等于当前 mode（人工改判时当前 mode 已经不是自动判定了）。
    this.autoDecision =
      data.sceneModeEvidence?.autoDecision ?? (this.modeSource === 'manual' ? null : mode);
    const ev = data.sceneModeEvidence ?? {};
    this.reasons = Array.isArray(ev.rejectReasons) ? ev.rejectReasons.slice(0, 3) : [];
    // 物体几何的可信度（见 tools/structure_object.py 的 evaluate_object_trust）
    const trust = data.object?.trust ?? null;
    this.objectTrust = trust;
    this.objectDrawn = false;
    // 房间盒是否为「推测」（只有 1 面可信墙 + 曼哈顿补全）：
    // 实测精度只有 10~19%（5 面墙全拟合是 ~1%），所以必须**画成虚线并写明**，
    // 不能让看的人把它当成已核实的户型。
    this.estimated = data.estimated === true;
    this.estimation = data.estimation ?? null;
    // ⚠️ `roomDrawn` 是补上的：以前"到底画没画房间盒"只能靠"mode 是不是 room"来猜，
    //    而 mode=room 但自检不通过（sane=false）时**一个房间盒都没画** ——
    //    界面却据此宣称"已叠加房间结构线框 6.0×16.5×2.6 m"（实测截图抓到）。
    //    判定是判定，画没画是画没画，两者必须分开记账。
    this.roomDrawn = false;
    const cov = ev.surroundCoverage?.median;
    let n = 0;
    // 推测盒若**远大于观测范围**就不画：一个比所有观测点大 2 倍以上的"推测房间"，
    // 画出来只会占满屏幕、还把模型衬得更小（实测厕所：盒 24.9 单位 vs 点云半径 9.8）。
    const suppressBox = data.roomSuppressed === true;
    if (suppressBox) {
      this.notice = '推测房间盒**超出观测范围**（比点云大 2 倍以上），已隐藏；'
        + '需要房间尺寸请补拍或人工圈定';
    }
    if (mode === 'room' && sane && !suppressBox) {
      n += this._addRoomBox(data.room);
      n += this._addFloorGrid(data.room);
      n += this._addWalls(data.walls, data.room);
      n += this._addTrajectory(data.trajectory);
      n += this._addPlanOutline(data.planOutline);
      this.roomDrawn = true;
      if (this.estimated) {
        const err = this.estimation?.measuredError;
        const pct = Array.isArray(err) && err.length
          ? `，实测尺寸误差约 ${Math.round(Math.min(...err) * 100)}~${Math.round(Math.max(...err) * 100)}%`
          : '';
        this.notice =
          '房间盒为「推测」：只有 1 面可信竖墙，按曼哈顿正交 + 地面/相机高度补全'
          + `${pct}。虚线表示未经核实，可人工改判或拖拽校正`;
      }
      if (this.ruler) n += this._addRuler();
    } else if (mode !== 'room' && trust?.ok) {
      // ---- 几何**自检通过** → 按「物体」呈现 ----------------------------------
      // 这里**不**要求 sceneMode 恰好是 'object'。为什么：
      //   场景模式（房间/物体）是按"方向包围度 + 竖墙 + 相机高度自洽"综合判的，
      //   实测一份**已知尺寸的柜子**（合成渲染 → 真 COLMAP，36/36 注册）会被判成
      //   `uncertain`（包围度 0.117 像物体，但"相机高度自洽得像房间"把它压回不确定）。
      //   而物体盒自带的 trust 判据是**直接**测"这个盒子可不可信"：
      //   盒长/相机半径 0.68、盒心偏移 0.07、相机附近点占比 1.00 —— 全部通过。
      //   用更模糊的场景模式去否决更直接的几何证据，就会让"几何明明可信却什么都不画"。
      //
      // ⚠️ 但**必须排除 `mode === 'room'`**：房间模型里画一个"物体盒"是误导
      //    （房间的点云也可能凑巧通过那几条比值），实测把 sane=false + room 的
      //    组合放进来时，物体分支会接手画出一个盒子 —— 这正是要避免的。
      n += this._addTrajectory(data.trajectory);
      n += this._addObjectBox(data.object);
      n += this._addObjectLayers(data.object);
      this.objectDrawn = true;
      this.notice =
        mode === 'object'
          ? '已按「物体」呈现：包围盒 + 水平层（搁板/顶面）。房间盒与地面网格不适用'
          : '场景判定不确定，但物体几何自检通过 → 按「物体」呈现：包围盒 + 水平层';
    } else {
      // 只画由**相机位姿**直接得到的东西（轨迹）：它不依赖任何"房间/物体"假设。
      n += this._addTrajectory(data.trajectory);
      if (mode === 'room') {
        // 判定是房间，但自检没过 → 不许画那个可能倒扣/错位的房间盒。
        // 也**不**退回画物体盒（房间模型画物体盒同样误导）。
        this.notice = '结构自检未通过（上方向或地面可疑）：不叠加房间盒，避免画出倒扣/错位的户型线';
        if (this.modeSource === 'manual') {
          this.notice += '（已人工改判为房间，但自检没过，仍不画）';
        }
      } else if (mode === 'object') {
        this.notice =
          '这个现场更像「贴着物体拍」：房间盒与地面网格来自"房间"假设，已不叠加（只显示相机轨迹）';
        if (trust && !trust.ok) {
          // 物体盒算出来了但不可信 —— **不画**，并说清为什么（否则用户以为功能坏了）
          this.notice += '。物体盒也没画：几何不可信';
          if (trust.reasons?.length) this.notice += `（${trust.reasons[0]}）`;
        } else if (!trust) {
          this.notice += '。没有物体几何可用，故未画物体盒';
        }
      } else {
        this.notice = '结构判定为「不确定」：不叠加房间盒与地面网格（只显示相机轨迹）';
        if (this.reasons.length) this.notice += ` —— ${this.reasons[0]}`;
      }
      // 人工改判必须显式说出来：用户改了判定却看到"没画"，要能一眼看出是为什么。
      if (this.modeSource === 'manual' && mode === 'object') {
        this.notice = '已人工改判为「物体」：不叠加房间盒与地面网格（只显示相机轨迹）';
        if (trust && !trust.ok) this.notice += '。物体盒也没画：几何不可信';
      }
    }
    // ---- 家具盒（独立数据源，见 this.furniture 的注释）------------------------
    // 放在所有"房间/物体"分支之后：它不属于那两个假设，有数据就画。
    this.furnitureDrawn = 0;
    if (this.furniture && this.furnitureVisible) {
      this.furnitureDrawn = this._addFurniture(this.furniture);
      n += this.furnitureDrawn;
    }
    this._createElements();
    return n;
  }

  /**
   * 地面平面轮廓（俯视态势下的「现场平面示意图」）。
   *
   * 为什么不是「房间轮廓」：实测墙面检测只找到 2~4 个几乎平行的平面，
   * 曼哈顿房间分解没有原料（详见 tools/scene_structure.py 里 plan_outline 的说明）。
   * 这里画的是**人活动高度层在地面上的占用边界** —— 对重建质量不敏感，稳得多。
   */
  _addPlanOutline(plan) {
    const segs = plan?.segmentsWorld;
    if (!Array.isArray(segs) || !segs.length) return 0;
    for (const s of segs) {
      if (!Array.isArray(s) || s.length < 2) continue;
      this.segments.push({ a: s[0], b: s[1], color: COLOR_PLAN, width: 2.6, opacity: 0.9 });
    }
    return segs.length;
  }

  // ---------------------------------------------------------------- 几何构造
  // 全部以「世界坐标线段」的形式存起来，投影放到每帧做（相机在动）。

  _addRoomBox(room) {
    const c = room.corners ?? [];
    if (c.length < 8) return 0;
    const idx = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    for (const [a, b] of idx) {
      this.segments.push({
        a: c[a], b: c[b], color: COLOR_BOX, width: 2.2, opacity: 0.95,
        dash: this.estimated ? '8 5' : undefined,   // 推测 → 虚线
      });
    }
    // 尺寸标注：把长/宽/高直接写在对应的棱上，让「空间线条」自带数据。
    // 这比让民警去数格子直观得多，也和已有的「精确测量」闭环对得上。
    const size = room.size ?? [];
    const mid = (i, j) => [(c[i][0] + c[j][0]) / 2, (c[i][1] + c[j][1]) / 2, (c[i][2] + c[j][2]) / 2];
    if (size.length >= 3) {
      this.labels.push({ p: mid(0, 1), text: this._sizeText(size[0]), color: COLOR_BOX });
      this.labels.push({ p: mid(1, 2), text: this._sizeText(size[1]), color: COLOR_BOX });
      this.labels.push({ p: mid(1, 5), text: this._sizeText(size[2]), color: COLOR_BOX });
    }
    return idx.length;
  }

  _addFloorGrid(room) {
    const { origin, axisU, axisV, size } = room;
    if (!origin || !axisU || !axisV || !size) return 0;
    const [w, d] = size;
    if (!(w > 0) || !(d > 0)) return 0;
    const o = new THREE.Vector3(...origin);
    const u = new THREE.Vector3(...axisU).normalize();
    const v = new THREE.Vector3(...axisV).normalize();
    const step = Math.max(0.5, Math.round(Math.max(w, d) / 24) * 0.5);
    const V = (vec) => [vec.x, vec.y, vec.z];
    let n = 0;
    for (let a = 0; a <= w + 1e-6; a += step) {
      const p = o.clone().addScaledVector(u, a);
      const q = p.clone().addScaledVector(v, d);
      this.segments.push({ a: V(p), b: V(q), color: COLOR_GRID, width: 1, opacity: 0.5 });
      n++;
    }
    for (let b = 0; b <= d + 1e-6; b += step) {
      const p = o.clone().addScaledVector(v, b);
      const q = p.clone().addScaledVector(u, w);
      this.segments.push({ a: V(p), b: V(q), color: COLOR_GRID, width: 1, opacity: 0.5 });
      n++;
    }
    return n;
  }

  /**
   * 墙面。
   *
   * ⚠️ 2026-10-06 改法：优先用 `w.rect.corners`（结构提取器算出的**真实矩形**），
   * 而不是原来的"以 center 为中心、沿切向画 ±span/2 的固定长度线段"。
   * 为什么必须改：`span` 是点云包围盒对角线的长度，而墙是有边界的矩形 ——
   * 用 span 画出来的墙**长度与真实墙不符**（短墙画长、长墙画短），
   * 而这些线的用途恰恰是让人读出"这是户型"。
   *
   * 同时**跳过 kind === "patch"**：提取器把"太窄/太小"的平面标成 patch
   * （家具、门板、镜面残影），原文注释就写了"标出来让前端可以只画真墙"。
   * 以前这里没过滤，于是把柜门、桌子面也当成墙画了出来。
   */
  _addWalls(walls, room) {
    if (!Array.isArray(walls) || !walls.length || !room.axisU) return 0;
    const u = new THREE.Vector3(...room.axisU).normalize();
    const v = new THREE.Vector3(...room.axisV).normalize();
    const worldUp = new THREE.Vector3().crossVectors(u, v).normalize();
    const V = (vec) => [vec.x, vec.y, vec.z];
    let n = 0;
    for (const w of walls) {
      if (w.kind === 'patch') continue;      // 家具/门板/残影，不是墙
      const corners = w.rect?.corners;
      if (Array.isArray(corners) && corners.length === 4) {
        // 真实矩形：画 4 条边（顺序为 [h0v0, h1v0, h1v1, h0v1]）
        for (let i = 0; i < 4; i++) {
          const a = corners[i];
          const b = corners[(i + 1) % 4];
          this.segments.push({ a, b, color: COLOR_WALL, width: 1.6, opacity: 0.85 });
        }
        n += 4;
        continue;
      }
      // 兜底：老结构文件没有 rect，退回原来的 span 画法
      const nrm = new THREE.Vector3(...w.normal).normalize();
      const c = new THREE.Vector3(...w.center);
      const along = new THREE.Vector3().crossVectors(worldUp, nrm).normalize();
      if (along.lengthSq() < 0.5) continue;
      const half = Math.min(w.span ?? 2, 8) / 2;
      const p0 = c.clone().addScaledVector(along, -half);
      const p1 = c.clone().addScaledVector(along, half);
      this.segments.push({ a: V(p0), b: V(p1), color: COLOR_WALL, width: 1.8, opacity: 0.85 });
      const hh = Math.min(room.size?.[2] ?? 2.6, 2.6) / 2;
      this.segments.push({
        a: V(c.clone().addScaledVector(worldUp, -hh)),
        b: V(c.clone().addScaledVector(worldUp, hh)),
        color: COLOR_WALL, width: 1.4, opacity: 0.6,
      });
      n += 2;
    }
    return n;
  }

  /**
   * 物体（柜子之类）的包围盒：8 个角点 → 12 条棱。
   *
   * 只应在 `object.trust.ok === true` 时调用（见 set()）：实测那份柜子的盒子
   * 是 4.42 × 16.77 × 8.60 m，而柜子本身约 1×0.5×2 m —— 画出来就是假的。
   */
  /**
   * 家具盒（`*.furniture.json`，来自 tools/furniture_lines.py）。
   *
   * 数据形态：`{ frame: {origin, axisU, axisV, up}, objects: [{center, size, points}] }`，
   * 其中 `center` 已是**世界坐标**，`size` 沿 frame 的三个轴 —— 所以这里只要把
   * 8 个角点按 center ± 轴×size/2 算出来，再连 12 条棱即可。
   *
   * 为什么不做"物体盒"那套可信度判据：家具盒本身就是**几何聚类 + 尺寸筛选**的产物，
   * 它的可信度信息在 `points`（每个盒子多少个点）里，画出来并把点数写在标签上，
   * 让看的人自己判断 —— 不在这里假装"可信/不可信"。
   */
  /**
   * 画竖直刻度尺：一条竖线 + 若干横刻度 + 数字标签。
   * 尺度用 `_sizeText` 的同一套规则：没标定就写"单位"，标定了才写"m"——不谎称米。
   */
  _addRuler() {
    const r = this.ruler;
    if (!r) return 0;
    const o = r.origin;
    const up = new THREE.Vector3(...r.up).normalize();
    const u = new THREE.Vector3(...r.axisU);
    u.addScaledVector(up, -u.dot(up));            // 让 u 落在水平面内
    if (u.lengthSq() < 1e-12) u.set(1, 0, 0);
    u.normalize();
    const col = '#ffffff';
    const tickLen = Math.max(r.height * 0.08, r.spacing * 0.35);
    const base = new THREE.Vector3(o[0], o[1], o[2]);
    const tip = base.clone().addScaledVector(up, r.height);
    let n = 0;
    // 竖线
    this.segments.push({ a: [base.x, base.y, base.z], b: [tip.x, tip.y, tip.z],
                         color: col, width: 2, opacity: 0.85 });
    n++;
    // 刻度：只画 3~4 条，且**不标 0**（地面在哪儿看得出来），避免挤成一团（实测 7 条会糊）
    const steps = Math.min(4, Math.max(2, Math.round(r.height / r.spacing)));
    const labelEvery = steps >= 4 ? 2 : 1;
    for (let i = 1; i <= steps; i++) {
      const h = (i / steps) * r.height;
      const p = base.clone().addScaledVector(up, h);
      const q = p.clone().addScaledVector(u, tickLen);
      this.segments.push({ a: [p.x, p.y, p.z], b: [q.x, q.y, q.z],
                           color: col, width: 1.6, opacity: 0.75 });
      n++;
      if (i % labelEvery === 0 || i === steps) {
        this.labels.push({ p: [q.x, q.y, q.z], text: this._sizeText(h), color: '#ffffff', dx: 12 });
        n++;
      }
    }
    this.rulerDrawn = n;
    return n;
  }

  _addFurniture(f) {
    const objs = f?.objects ?? [];
    const fr = f?.frame ?? {};
    const au = fr.axisU, av = fr.axisV, aw = fr.up;
    if (!objs.length || !au || !av || !aw) return 0;
    const U = new THREE.Vector3(...au).normalize();
    const V = new THREE.Vector3(...av).normalize();
    const W = new THREE.Vector3(...aw).normalize();
    const signs = [
      [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ];
    const idx = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    let n = 0;
    // 标签只给"点数最多的前几个"：实测 10 个盒子一起标会把中间糊成一团
    // （截图里字叠字完全读不出来）。按点数排序取前 5 个，且要求点数够多。
    const labeled = new Set(
      objs
        .map((o, i) => ({ i, p: Number(o?.points) || 0 }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 5)
        .filter((x) => x.p >= 2000)
        .map((x) => x.i)
    );
    for (let oi = 0; oi < objs.length; oi++) {
      const obj = objs[oi];
      const c = obj.center, s0 = obj.size;
      if (!c || !s0 || s0.length < 3) continue;
      // 内缩 1.5%：相邻两个盒子贴在一起时，边线不会粘成一条（实测很糊）
      const s = [s0[0] * 0.97, s0[1] * 0.97, s0[2] * 0.97];
      const col = categoryColor(obj.label);
      const corners = signs.map(([sx, sy, sz]) => {
        const p = new THREE.Vector3(c[0], c[1], c[2]);
        p.addScaledVector(U, (sx * s[0]) / 2)
          .addScaledVector(V, (sy * s[1]) / 2)
          .addScaledVector(W, (sz * s[2]) / 2);
        return [p.x, p.y, p.z];
      });
      for (const [a, b] of idx) {
        this.segments.push({ a: corners[a], b: corners[b], color: col, width: 1.8, opacity: 0.92 });
      }
      // 标签：尺寸 + 点数。点数就是"这个盒子由多少个高斯撑起来"，代表了它的实心程度。
      if (labeled.has(oi)) {
        const top = corners[4].map((v, k) => (v + corners[5][k] + corners[6][k] + corners[7][k]) / 4);
        const pts = Number.isFinite(obj.points) ? ` · ${obj.points} 点` : '';
        // 检测路线（Grounding DINO）会给 label / score；几何路线没有。
        // 有标签时优先显示"类别 + 置信度"，这比只有尺寸有用得多。
        const lab = obj.label
          ? `${obj.label}${Number.isFinite(obj.score) ? ` ${(obj.score * 100).toFixed(0)}%` : ''} · `
          : '';
        this.labels.push({
          p: top,
          text: `${lab}${this._sizeText(s[0]).replace(' 单位', '')}×${this._sizeText(s[1]).replace(' 单位', '')}×${this._sizeText(s[2]).replace(' 单位', '')}${pts}`,
          color: COLOR_FURN,
        });
      }
      n += idx.length;
    }
    return n;
  }

  _addObjectBox(obj) {
    const c = obj?.corners ?? [];
    if (c.length < 8) return 0;
    const idx = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    for (const [a, b] of idx) {
      this.segments.push({ a: c[a], b: c[b], color: COLOR_OBJ, width: 2.2, opacity: 0.95 });
    }
    // 三边尺寸直接写在棱上：一眼能读出物体多大（与房间盒同一套做法）
    const size = obj.size ?? [];
    const mid = (i, j) => [(c[i][0] + c[j][0]) / 2, (c[i][1] + c[j][1]) / 2, (c[i][2] + c[j][2]) / 2];
    if (size.length >= 3) {
      this.labels.push({ p: mid(0, 1), text: `${size[0].toFixed(2)} m`, color: COLOR_OBJ });
      this.labels.push({ p: mid(1, 2), text: `${size[1].toFixed(2)} m`, color: COLOR_OBJ });
      this.labels.push({ p: mid(1, 5), text: `${size[2].toFixed(2)} m`, color: COLOR_OBJ });
    }
    return 12 + 3;
  }

  /**
   * 物体的**水平层**（顶面 / 搁板 / 底面）—— 柜子这类物体的结构信息就在这里，
   * 竖墙那套对它没用（这正是"场景/物体两套算法"的理由之一）。
   *
   * 每层在结构文件里的 `height` 是**相对锚点**的竖直坐标（点云在物体轴上的投影），
   * 而盒子的局部原点在 `center`，所以这里要先换算：
   *     层在盒内的竖直偏移 = height − (center − anchor)·up
   * 然后把该层画成盒子水平截面上的一块矩形。
   */
  _addObjectLayers(obj) {
    const layers = obj?.layers ?? [];
    const corners = obj?.corners ?? [];
    const axes = obj?.axes ?? {};
    if (!layers.length || corners.length < 8 || !axes.u || !axes.v || !axes.up) return 0;
    const anchor = obj?.anchor?.center ?? obj?.center ?? [0, 0, 0];
    const size = obj?.size ?? [];
    if (size.length < 3) return 0;
    const u = new THREE.Vector3(...axes.u).normalize();
    const v = new THREE.Vector3(...axes.v).normalize();
    const up = new THREE.Vector3(...axes.up).normalize();
    const center = new THREE.Vector3(...(obj.center ?? [0, 0, 0]));
    const anc = new THREE.Vector3(...anchor);
    const centerOffsetUp = center.clone().sub(anc).dot(up);   // 盒心相对锚点的竖直偏移
    // 层要落在盒子的竖直范围内，否则是脏数据（不外扩画出去）
    const halfUp = size[2] / 2;
    const V = (vec) => [vec.x, vec.y, vec.z];
    // 水平方向按盒子尺寸略缩一点，免得层线与盒子棱完全重合、看不出是两层信息
    const hu = (size[0] / 2) * 0.98;
    const hv = (size[1] / 2) * 0.98;
    let n = 0;
    for (const ly of layers) {
      const local = Number(ly?.height) - centerOffsetUp;
      if (!Number.isFinite(local) || Math.abs(local) > halfUp * 1.02) continue;
      const origin = center.clone().addScaledVector(up, local);
      const pts = [
        origin.clone().addScaledVector(u, -hu).addScaledVector(v, -hv),
        origin.clone().addScaledVector(u, hu).addScaledVector(v, -hv),
        origin.clone().addScaledVector(u, hu).addScaledVector(v, hv),
        origin.clone().addScaledVector(u, -hu).addScaledVector(v, hv),
      ];
      for (let i = 0; i < 4; i++) {
        this.segments.push({
          a: V(pts[i]), b: V(pts[(i + 1) % 4]),
          color: COLOR_OBJ_LAYER, width: 1.6, opacity: 0.8,
        });
      }
      this.labels.push({ p: V(pts[0]), text: `层 ${Number(ly.height).toFixed(2)} m`, color: COLOR_OBJ_LAYER });
      n += 5;
    }
    return n;
  }

  _addTrajectory(traj) {
    const pts = traj?.points ?? [];
    if (pts.length < 2) return 0;
    this.polylines.push({ pts, color: COLOR_TRAJ, width: 2, opacity: 0.95 });
    const r = Math.max(3, Math.min(9, (this.data?.room?.size?.[0] ?? 8) * 0.5));
    this.dots.push({ p: pts[0], color: COLOR_START, r });
    this.dots.push({ p: pts[pts.length - 1], color: COLOR_END, r });
    return this.segments.length + 3;
  }

  // ---------------------------------------------------------------- DOM

  _createElements() {
    if (!this._svg) return;
    for (const s of this.segments) {
      const el = document.createElementNS(SVG_NS, 'line');
      el.setAttribute('stroke', s.color);
      el.setAttribute('stroke-width', String(s.width));
      el.setAttribute('stroke-opacity', String(s.opacity));
      if (s.dash) el.setAttribute('stroke-dasharray', s.dash);
      el.setAttribute('stroke-linecap', 'round');
      this._svg.appendChild(el);
      this._els.push(el);
      s.el = el;
    }
    for (const p of this.polylines) {
      const el = document.createElementNS(SVG_NS, 'polyline');
      el.setAttribute('fill', 'none');
      el.setAttribute('stroke', p.color);
      el.setAttribute('stroke-width', String(p.width));
      el.setAttribute('stroke-opacity', String(p.opacity));
      this._svg.appendChild(el);
      this._els.push(el);
      p.el = el;
    }
    for (const d of this.dots) {
      const el = document.createElementNS(SVG_NS, 'circle');
      el.setAttribute('fill', d.color);
      el.setAttribute('r', String(d.r));
      this._svg.appendChild(el);
      this._els.push(el);
      d.el = el;
    }
    for (const lb of this.labels) {
      const el = document.createElementNS(SVG_NS, 'text');
      el.textContent = lb.text;
      el.setAttribute('fill', lb.color);
      el.setAttribute('font-size', '13');
      el.setAttribute('font-family', 'Consolas, monospace');
      el.setAttribute('text-anchor', 'middle');
      el.setAttribute('dominant-baseline', 'middle');
      // 描边保证在任何背景下都读得清（泼溅画面明暗变化很大）
      el.setAttribute('stroke', 'rgba(3,6,9,0.85)');
      el.setAttribute('stroke-width', '3');
      el.setAttribute('paint-order', 'stroke');
      this._svg.appendChild(el);
      this._els.push(el);
      lb.el = el;
    }
  }

  /** 世界坐标（COLMAP 系）→ 屏幕像素；返回 null 表示在相机后方 */
  _project(p) {
    const v = this._v.set(p[0], p[1], p[2]).applyMatrix4(WORLD_MATRIX);
    v.project(this.camera);
    if (v.z < -1 || v.z > 1) return null;
    const w = window.innerWidth;
    const h = window.innerHeight;
    return [((v.x + 1) / 2) * w, ((-v.y + 1) / 2) * h];
  }

  /** 每帧调用（放在 labelLayer.update() 旁边） */
  update() {
    if (!this.visible || !this._svg) return;
    // 临时图层（PnP 视锥/候选点）与结构线框独立：即使没有 structure.json
    // （比如模型没有结构骨架）也要能画视锥。
    this._updateTransient();
    // ⚠️ 这里**不能**用 `if (!this.data) return;` —— 家具盒是独立数据源，
    // 稠密重建/VGGT 的模型可能没有 structure.json（this.data 为 null），
    // 那种情况下家具线段会被整体跳过，画面上一条橙线都看不到（实测踩到：
    // 测试断言"画了 120 条线段"通过，但截图里什么都没有）。
    if (!this.data && !this.segments.length && !this.polylines.length
        && !this.dots.length && !this.labels.length) return;
    for (const s of this.segments) {
      const el = s.el;
      if (!el) continue;
      const a = this._project(s.a);
      const b = this._project(s.b);
      if (!a || !b) {
        if (el.style.display !== 'none') el.style.display = 'none';
        continue;
      }
      if (el.style.display === 'none') el.style.display = '';
      el.setAttribute('x1', a[0].toFixed(1));
      el.setAttribute('y1', a[1].toFixed(1));
      el.setAttribute('x2', b[0].toFixed(1));
      el.setAttribute('y2', b[1].toFixed(1));
    }
    for (const p of this.polylines) {
      const el = p.el;
      if (!el) continue;
      const parts = [];
      for (const pt of p.pts) {
        const s = this._project(pt);
        if (s) parts.push(`${s[0].toFixed(1)},${s[1].toFixed(1)}`);
      }
      if (parts.length < 2) {
        if (el.style.display !== 'none') el.style.display = 'none';
        continue;
      }
      if (el.style.display === 'none') el.style.display = '';
      el.setAttribute('points', parts.join(' '));
    }
    for (const d of this.dots) {
      const el = d.el;
      if (!el) continue;
      const s = this._project(d.p);
      if (!s) {
        if (el.style.display !== 'none') el.style.display = 'none';
        continue;
      }
      if (el.style.display === 'none') el.style.display = '';
      el.setAttribute('cx', s[0].toFixed(1));
      el.setAttribute('cy', s[1].toFixed(1));
    }
    for (const lb of this.labels) {
      const el = lb.el;
      if (!el) continue;
      const s = this._project(lb.p);
      if (!s) {
        if (el.style.display !== 'none') el.style.display = 'none';
        continue;
      }
      if (el.style.display === 'none') el.style.display = '';
      // 标签支持屏幕像素偏移（刻度尺用 dx=12，避免刻度数字压在竖线上）
      el.setAttribute('x', (s[0] + (lb.dx || 0)).toFixed(1));
      el.setAttribute('y', (s[1] + (lb.dy || 0)).toFixed(1));
    }
  }

  /** 临时图层每帧重投影 */
  _updateTransient() {
    for (const s of this.transientSegments) {
      const el = s.el;
      if (!el) continue;
      const a = this._project(s.a);
      const b = this._project(s.b);
      if (!a || !b) {
        if (el.style.display !== 'none') el.style.display = 'none';
        continue;
      }
      if (el.style.display === 'none') el.style.display = '';
      el.setAttribute('x1', a[0].toFixed(1));
      el.setAttribute('y1', a[1].toFixed(1));
      el.setAttribute('x2', b[0].toFixed(1));
      el.setAttribute('y2', b[1].toFixed(1));
    }
    for (const d of this.transientDots) {
      const el = d.el;
      if (!el) continue;
      const s = this._project(d.p);
      if (!s) {
        if (el.style.display !== 'none') el.style.display = 'none';
        continue;
      }
      if (el.style.display === 'none') el.style.display = '';
      el.setAttribute('cx', s[0].toFixed(1));
      el.setAttribute('cy', s[1].toFixed(1));
    }
  }

  /** 便于测试/排查：当前有多少可见图形 */
  stats() {
    const shown = this._els.filter((e) => e.style.display !== 'none').length;
    const tShown = this._tels.filter((e) => e.style.display !== 'none').length;
    return {
      visible: this.visible,
      transientSegments: this.transientSegments.length,
      transientDots: this.transientDots.length,
      transientElements: this._tels.length,
      transientShown: tShown,
      segments: this.segments.length,
      polylines: this.polylines.length,
      dots: this.dots.length,
      labels: this.labels.length,
      labelTexts: this.labels.map((l) => l.text),
      elements: this._els.length,
      shown,
      roomSize: this.data?.room?.size ?? null,
      trajPoints: this.data?.trajectory?.points?.length ?? 0,
      upVector: this.data ? [WORLD_QUAT.x, WORLD_QUAT.y, WORLD_QUAT.z, WORLD_QUAT.w] : null,
    };
  }
}
