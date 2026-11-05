/**
 * 案发现场三维重建与协同研判系统 · 前端主程序
 *
 * 职责：
 *  1. 三维现场加载与渲染（自研 WebGL2 3DGS 渲染器 + Three.js 场景）
 *  2. 多源画面（监控/勘查视频）贴回三维现场（PnP 标定位姿）
 *  3. 研判工具：打点、轨迹、区域、测距、相机标定
 *  4. WebSocket 多人实时协同：视角同步、标注同步、在线名单、文字交流
 *
 * 坐标系说明：
 *  - 3DGS/COLMAP 世界系：x 右、y 下、z 前；Three.js 场景系：x 右、y 上、z 后
 *  - 两者之间用四元数 q=(√2/2,0,0,-√2/2) 绕 x 轴 -90° 转换，见 WORLD_QUAT
 *  - 标注与相机位姿在数据里一律使用 COLMAP 系，渲染时统一乘 worldMatrix
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SplatRenderer } from './splat-renderer.js';
import { loadGaussians, computeBounds } from './splat-loader.js';
import { SplatPicker } from './picker.js';
import { AnnotationLayer, defaultColor, kindName, applyPose } from './annotations.js';
import { CollabClient } from './collab.js';
import { api } from './api.js';
import { UI } from './ui.js';
import { Drawing } from './drawing.js';
import { CameraOverlay, WORLD_QUAT, WORLD_MATRIX, WORLD_MATRIX_INV, peerPoseToScene } from './camera-overlay.js';
import { PnpPanel } from './pnp-panel.js';
import { ReconPanel } from './recon-panel.js';
import { MeasurePanel } from './measure-panel.js';
import { StructureLayer } from './structure-layer.js';
import { APP_VERSION } from './version.js';

// COLMAP(3DGS) 世界系 -> Three.js 场景系：绕 X 轴 -90°
// （WORLD_QUAT / WORLD_MATRIX 由 camera-overlay.js 统一导出，避免两处定义漂移）

const state = {
  projects: [],
  project: null,
  model: null,
  bounds: null,
  renderer: null,
  picker: null,
  scene: null,
  camera: null,
  controls: null,
  mode: 'orbit',
  tool: 'select',
  selectedId: null,
  peers: new Map(),
  // 渲染参数默认值：实测对比（1/1/1、0.75/0.72/1.15、0.62/0.62/1.05）后选的甜点。
    // 高斯尺寸调小 + 不透明度降一点 + 曝光略提 → 表面细节出来、白雾感下降；
    // 再小就开始颗粒化（0.62 那档能看到单个高斯）。
    filters: { sort: true, gaussScale: 0.8, opacity: 0.75, exposure: 1.1, showCameras: true, showPoints: true },
  loading: false,
};

/**
 * 取一个「有限数」，否则用兜底值。
 *
 * 存在的理由：`state.bounds?.radius * 0.4 ?? 1` 这种写法看着很安全，其实有个大坑 ——
 * `undefined * 0.4` 得到 **NaN**，而 `NaN ?? 1` 仍然是 NaN（`??` 只处理 null/undefined）。
 * 一旦这个 NaN 进了 `camera.position`，整条矩阵链都会是 NaN，
 * 渲染器照样发出 12 万个 draw，但全部落在无效坐标上 —— 表现就是**整个三维区黑屏**，
 * 而且必须重新载入模型才能恢复。所以凡是「可选字段参与算术」的地方都用这个函数。
 */
function finiteOr(value, fallback) {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

const conn = {
  name: localStorage.getItem('dsh:name') || '',
  color: localStorage.getItem('dsh:color') || pickColor(),
  role: 'investigator',
  client: null,
  throttle: { pose: 0 },
};

function pickColor() {
  const palette = ['#4dd0e1', '#ffcc33', '#ff8a65', '#7ef0a0', '#c792ea', '#ff5f6d', '#64b5f6', '#ffd54f'];
  return palette[Math.floor(Math.random() * palette.length)];
}

// ------------------------------------------------------------------ 初始化

/** 把三维标注锚点投影成屏幕标签，用于显示名称与测距结果 */
class LabelProjector {
  constructor(host, camera) {
    this.host = host;
    this.camera = camera;
    this.els = new Map();
  }

  update() {
    if (!this.host) return;
    const w = window.innerWidth;
    const h = window.innerHeight;
    const v = new THREE.Vector3();
    for (const item of annotationLayer.list()) {
      if (item.data.visible === false) {
        const el = this.els.get(item.data.id);
        if (el) el.style.display = 'none';
        continue;
      }
      const anchor = item.labelAnchor;
      if (!anchor) continue;
      v.copy(anchor);
      v.applyMatrix4(WORLD_MATRIX);
      v.project(this.camera);
      if (v.z < -1 || v.z > 1) {
        const el = this.els.get(item.data.id);
        if (el) el.style.display = 'none';
        continue;
      }
      let el = this.els.get(item.data.id);
      if (!el) {
        el = document.createElement('div');
        el.className = 'label3d';
        this.host.appendChild(el);
        this.els.set(item.data.id, el);
      }
      const text = this.textFor(item.data);
      if (el.dataset.text !== text) {
        el.dataset.text = text;
        el.innerHTML = text;
      }
      el.style.display = 'block';
      el.style.borderColor = item.data.color ?? defaultColor(item.data.kind);
      el.style.transform = `translate(-50%, -100%) translate(${((v.x + 1) / 2) * w}px, ${((-v.y + 1) / 2) * h}px)`;
    }
    for (const [id, el] of this.els) {
      if (!annotationLayer.get(id)) {
        el.remove();
        this.els.delete(id);
      }
    }
  }

  textFor(ann) {
    const name = escapeHtml(ann.label || kindName(ann.kind, ann));
    // 尺寸标注：三行信息 —— 名称 / 模型值 / 实测值（待测时提示）
    if (ann.kind === 'dimension') {
      const scale = Number(state.project?.world?.scaleToMeters ?? 1) || 1;
      const modelM = Number(ann.modelDistance ?? 0);
      const scaledM = modelM * scale;
      if (ann.measured) {
        const err = scaledM - Number(ann.measured);
        const flat = Math.abs(err) < 0.005;
        return (
          `<b>${name}</b>` +
          `<span class="dist ok">实测 ${Number(ann.measured).toFixed(3)} m</span>` +
          `<span class="dist">模型 ${scaledM.toFixed(3)} m（${flat ? '一致' : `差 ${err >= 0 ? '+' : ''}${err.toFixed(3)} m`}）</span>`
        );
      }
      return `<b>${name}</b><span class="dist warn">待民警实测</span><span class="dist">模型 ${scaledM.toFixed(3)} m</span>`;
    }
    if (ann.kind === 'measure' || ann.kind === 'polyline') {
      // 测距要按现场实测校正过的比例尺换算，否则标定完数字不会变
      const scale = Number(state.project?.world?.scaleToMeters ?? 1) || 1;
      let total = 0;
      for (let i = 1; i < ann.points.length; i++) {
        const a = ann.points[i - 1];
        const b = ann.points[i];
        total += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
      }
      if (ann.closed && ann.points.length > 2) {
        const a = ann.points.at(-1);
        const b = ann.points[0];
        total += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
      }
      const meters = total * scale;
      return `<b>${name}</b><span class="dist">${meters.toFixed(ann.kind === 'measure' ? 3 : 2)} m</span>`;
    }
    if (ann.kind === 'camera') {
      const err = ann.pose?.reprojectionError;
      const tag = ann.pose?.source === 'pnp' ? `PnP ${err != null ? err.toFixed(2) + 'px' : 'OK'}` : '未标定';
      return `<b>${name}</b><span class="dist">${tag}</span>`;
    }
    return `<b>${name}</b>`;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function formatInt(n) {
  return Number(n).toLocaleString('zh-CN');
}

const canvas = document.getElementById('viewport');
const ui = new UI();
const scene = new THREE.Scene();
scene.background = new THREE.Color('#080c11');
const camera = new THREE.PerspectiveCamera(55, 1, 0.02, 20000);
camera.position.set(3, 2.5, 3);
camera.up.set(0, 1, 0);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;
controls.maxDistance = 5000;
controls.minDistance = 0.05;
controls.zoomToCursor = true;

const gridHelper = new THREE.GridHelper(10, 20, 0x2a3a4a, 0x18242e);
gridHelper.visible = true;
scene.add(gridHelper);

const axesHelper = new THREE.AxesHelper(0.6);
axesHelper.visible = false;
scene.add(axesHelper);

const annotationLayer = new AnnotationLayer(scene);
annotationLayer.setLabelHost(ui.labelHost);
const labelLayer = new LabelProjector(ui.labelHost, camera, scene);

// 房间结构图层：把「地面 / 房间盒 / 墙面 / 相机轨迹」画成屏幕上的空间线条。
// 泼溅只能看材质，这一层才让「整体布局与空间线条」看得见。
// 注意：它是 DOM/SVG 覆盖层，不是 Three.js 网格 —— 本项目没有 Three.js 渲染器，
// 往场景里加网格不会被绘制（详见 structure-layer.js 文件头）。
const structureLayer = new StructureLayer(ui.labelHost, camera);
// 图层开关（默认开）
document.getElementById('cStructure')?.addEventListener('change', (ev) => {
  structureLayer.setVisible(ev.target.checked);
  updateStructureBadge();
  ui.toast(ev.target.checked ? '已显示房间结构线框' : '已隐藏房间结构线框');
});

// ---------------------------------------------------------------- 移动端抽屉
//
// 窄屏把左右两栏做成抽屉（CSS 在 app.css 的 `@media (max-width: 900px)` 里）。
// 这里只负责开关状态：body 上挂 `drawer-open-left/right`，CSS 据此平移面板。
// 为什么用 body 上的 class 而不是直接改面板样式：CSS 与 JS 的职责分开，
// 面板的位移/过渡全部留在样式表里，回归测试也能只查 class 与 rect。
const DRAWER_MQ = window.matchMedia('(max-width: 900px)');
const isNarrow = () => DRAWER_MQ.matches;

function closeDrawers() {
  document.body.classList.remove('drawer-open-left', 'drawer-open-right');
}

function toggleDrawer(side) {
  const cls = side === 'left' ? 'drawer-open-left' : 'drawer-open-right';
  const other = side === 'left' ? 'drawer-open-right' : 'drawer-open-left';
  const willOpen = !document.body.classList.contains(cls);
  document.body.classList.remove(other);
  document.body.classList.toggle(cls, willOpen);
  return willOpen;
}

document.getElementById('btnDrawerLeft')?.addEventListener('click', (ev) => {
  ev.stopPropagation();
  toggleDrawer('left');
});
document.getElementById('btnDrawerRight')?.addEventListener('click', (ev) => {
  ev.stopPropagation();
  toggleDrawer('right');
});
document.getElementById('drawerBackdrop')?.addEventListener('click', closeDrawers);
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape') closeDrawers();
});
// 点画布即收起（手机上最常见的操作就是"把面板点掉看模型"）
canvas.addEventListener(
  'pointerdown',
  () => {
    if (isNarrow()) closeDrawers();
  },
  { passive: true }
);
// 从窄屏转回桌面时，别把 body 的类留成"抽屉打开"的状态
DRAWER_MQ.addEventListener('change', (e) => {
  if (!e.matches) closeDrawers();
});

// 家具盒开关（独立于结构线框：家具来自另一个文件、另一套坐标系）
document.getElementById('cFurniture')?.addEventListener('change', (ev) => {
  structureLayer.setFurnitureVisible(ev.target.checked);
  const n = structureLayer.furniture?.objects?.length ?? 0;
  ui.toast(
    ev.target.checked
      ? n
        ? `已显示家具盒（${n} 件）`
        : '该模型没有家具盒数据（*.furniture.json）'
      : '已隐藏家具盒'
  );
});

// 场景判定的人工改判（房间 / 物体 / 自动）。
// 判不准时由拍摄的人指定 —— 几何分不开这两类素材（实测 0.2344 vs 0.2270）。
document.getElementById('sceneModeSelect')?.addEventListener('change', async (ev) => {
  const sel = ev.target;
  const want = sel.value;
  const ok = await setSceneMode(want);
  if (!ok) {
    // 失败要把下拉框**弹回真实状态**，否则界面会显示一个没有生效的判定
    syncSceneModeControl(state.model);
  }
});

const peerGroup = new THREE.Group();
scene.add(peerGroup);

const drawing = new Drawing(scene, {
  getWorldMatrix: () => WORLD_MATRIX,
  onCommit: (draft) => commitAnnotation(draft),
  onHint: (text) => ui.setToolHint(text),
  camera,
  canvas,
});

const cameraOverlay = new CameraOverlay(scene, { worldMatrix: WORLD_MATRIX });

// ------------------------------------------------------------------ PnP 标定面板

/**
 * 位姿合理性检查（Bug C）。返回 null 表示可用，否则返回**拒绝原因**。
 *
 * 为什么必须有：`onSolved` 以前把 PnP 解出的位姿**不做任何检查**就写回标注；
 * 对应点选得差时（点到远处野点、或几个 3D 点近乎共线）会解出飞到天外的位姿。
 * 实测本案件里就有 `pose.position = [21.9, 283.4, 37.0]`（Y=283 米，模型半径 15 米）：
 * 一点「贴回现场」/「转到该机位」，视角就飞到 283 米外 —— **整个模型黑掉**，
 * 用户只能刷新页面。这个函数把这类位姿在两处（写入前、跳转前）都拦下来。
 */
function poseRejectReason(pose) {
  const pos = pose?.position;
  if (!Array.isArray(pos) || pos.length < 3) return '位姿缺少位置';
  if (!pos.every((v) => Number.isFinite(v))) return '位姿含非法数值（NaN/Inf）';
  const c = state.bounds?.center ?? [0, 0, 0];
  const r = Number.isFinite(state.bounds?.radius) ? state.bounds.radius : 10;
  const dist = Math.hypot(pos[0] - c[0], pos[1] - c[1], pos[2] - c[2]);
  const limit = Math.max(30, r * 4);
  if (dist > limit) {
    return `机位离现场 ${dist.toFixed(0)} 米（场景半径 ${r.toFixed(1)} 米）`;
  }
  return null;
}

const pnpPanel = new PnpPanel({
  onToast: (msg, opts) => ui.toast(msg, opts),
  // 解算成功后把位姿写回相机标注，并让画面立刻按新位姿贴回现场
  onSolved: async (ann, pose) => {
    if (!ann?.id) return;
    // ⚠️ 位姿合理性校验（Bug C）：PnP 在对应点选得很差时（3D 点近乎共线、
    // 或误点了远处的野点）会解出一个**飞到天外**的位姿，而这里以前**不做任何检查**
    // 就写回标注。后果是「贴回现场」把视角送到几百米外 —— 用户看到的就是
    // 「点了没反应」（其实是飞走了、画面全空）。
    // 实测本案件里有一个标注的 pose.position = [21.9, 283.4, 37.0]，
    // Y 高达 283 米，而模型半径只有 15 米。
    const pos = pose?.position;
    const why = poseRejectReason(pose);
    if (why) {
      ui.toast(
        `解算结果不可用（${why}），已放弃写入。通常是三维对应点选到了远处的野点，` +
          `或几个点几乎共线 —— 请挑分布开、且有明确特征的位置重新标定`,
        { error: true, sticky: true }
      );
      return;
    }
    await updateAnnotation(ann.id, { pose, points: [[pose.position[0], pose.position[1], pose.position[2]]] });
    const fresh = state.project.annotations.find((a) => a.id === ann.id);
    if (fresh?.media?.[0]?.url) await cameraOverlay.attach(fresh, fresh.media[0].url);
    cameraOverlay.select(ann.id);
    ui.renderCameras(state.project.annotations.filter((a) => a.kind === 'camera'), cameraOverlay);
  },
  // 「把视角转到该机位附近」：先按当前（或已解算的）位姿飞到机位，便于挑特征点
  onFlyTo: (ann) => {
    if (!ann) return;
    // ⚠️ 飞之前先做合理性校验：位姿要是飞到场景外（比如 Y=283 米），
    // 视角会落到一片空处 —— 用户看到的就是「整个模型黑掉了」。
    // 以前这里直接照飞不误，黑屏后只能刷新页面才能恢复。
    const why = poseRejectReason(ann.pose);
    if (why) {
      ui.toast(`这个机位的位姿不可用（${why}），已取消跳转。请重新做 PnP 标定`, {
        error: true,
        sticky: true,
      });
      return;
    }
    const view = cameraOverlay.flyTo(ann);
    if (!view) return;
    camera.position.copy(view.position);
    camera.quaternion.copy(view.quaternion);
    camera.fov = view.fovY ?? camera.fov;
    camera.updateProjectionMatrix();
    if (view.target) controls.target.copy(view.target);
    controls.update();
    ui.toast('已把视角转到该机位附近，方便对照画面挑特征点');
  },
  // 面板进入「等待三维点击」状态时：
  //   ① 给三维视口加提示描边；
  //   ② **用透视线框画出该相机的视锥**，并高亮视锥内可选的点 ——
  //      用户反馈「很难选一个合适的三维标定点」，就是因为屏幕上看不出
  //      「从这台相机看出去，哪块空间是拍到的」。选完点立刻撤掉。
  onArmedChange: (armed) => {
    canvas.classList.toggle('pnp-armed', !!armed);
    if (armed) {
      ui.setToolHint('已选好画面像素点，现在到三维视口里点它的对应位置（黄色虚线内是可选范围）');
      showPnpPickHelper();
    } else {
      ui.setToolHint('PnP 标定中：先在面板画面上点像素，再到三维视口点对应位置');
      structureLayer.clearTransient();
    }
  },
});
pnpPanel.onClose = () => {
  canvas.classList.remove('pnp-armed');
  structureLayer.clearTransient();
  ui.setToolHint('左键选择，右键/ESC 取消当前绘制');
};

/**
 * 相机视锥的 8 个角点在**世界坐标（COLMAP 系）**下的位置。
 *
 * 约定（与 camera-overlay.js 完全一致，别自己另立一套）：
 *   pose 的局部系里 **-Z 是观察方向**、+Y 向上、+X 向右；
 *   视频幕布就挂在局部 z = -far 处（见 camera-overlay.attach）。
 * 所以视锥也按这个约定构造，再由 structureLayer 统一乘 WORLD_MATRIX。
 */
function cameraFrustumCorners(pose, far) {
  const q = new THREE.Quaternion(
    pose.quaternion?.[0] ?? 0,
    pose.quaternion?.[1] ?? 0,
    pose.quaternion?.[2] ?? 0,
    pose.quaternion?.[3] ?? 1
  ).normalize();
  const origin = new THREE.Vector3(...(pose.position ?? [0, 0, 0]));
  const fovY = ((pose.fovY ?? 45) * Math.PI) / 180;
  const aspect = pose.aspect || 16 / 9;
  const near = Math.max(0.15, far * 0.06);
  const corner = (z, sx, sy) => {
    const hh = Math.tan(fovY / 2) * z;
    const hw = hh * aspect;
    const local = new THREE.Vector3(sx * hw, sy * hh, -z); // -Z 向前
    return local.applyQuaternion(q).add(origin).toArray();
  };
  const pts = [];
  for (const z of [near, far]) {
    pts.push(corner(z, -1, -1), corner(z, 1, -1), corner(z, 1, 1), corner(z, -1, 1));
  }
  return { origin: origin.toArray(), near: pts.slice(0, 4), far: pts.slice(4, 8) };
}

/** 视锥线框：4 条从机位出发的棱 + 近/远两个矩形 */
function frustumSegments(pose, far) {
  const f = cameraFrustumCorners(pose, far);
  const segs = [];
  for (let i = 0; i < 4; i++) {
    segs.push({ a: f.origin, b: f.near[i] });
    segs.push({ a: f.near[i], b: f.near[(i + 1) % 4] });
    segs.push({ a: f.far[i], b: f.far[(i + 1) % 4] });
    segs.push({ a: f.near[i], b: f.far[i] });
  }
  return segs;
}

/**
 * 从已加载的高斯里抽稀挑出「落在视锥内」的点，作为可选点提示（T2B）。
 *
 * 为什么抽稀：一个模型有 4 万个高斯，全画成 SVG 圆点会直接把浏览器拖死。
 * 用户只需要看到「这块空间有点云、可以点」，几百个点足够表达。
 */
function samplePointsInFrustum(pose, far, maxDots = 260) {
  const pos = state.gaussianPositions;
  if (!pos || pos.length < 3) return [];
  const q = new THREE.Quaternion(
    pose.quaternion?.[0] ?? 0,
    pose.quaternion?.[1] ?? 0,
    pose.quaternion?.[2] ?? 0,
    pose.quaternion?.[3] ?? 1
  ).normalize();
  const inv = q.clone().invert();
  const origin = new THREE.Vector3(...(pose.position ?? [0, 0, 0]));
  const fovY = ((pose.fovY ?? 45) * Math.PI) / 180;
  const aspect = pose.aspect || 16 / 9;
  const tanY = Math.tan(fovY / 2);
  const total = Math.floor(pos.length / 3);
  const stride = Math.max(1, Math.floor(total / 12000)); // 先粗筛，控制循环量
  const v = new THREE.Vector3();
  const hits = [];
  for (let i = 0; i < total; i += stride) {
    v.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]).sub(origin).applyQuaternion(inv);
    const z = -v.z; // 局部 -Z 为前方
    if (z <= 0.2 || z > far) continue;
    if (Math.abs(v.x) > tanY * aspect * z || Math.abs(v.y) > tanY * z) continue;
    hits.push([pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]]);
  }
  if (!hits.length) return [];
  const step = Math.max(1, Math.ceil(hits.length / maxDots));
  const out = [];
  for (let i = 0; i < hits.length; i += step) out.push({ p: hits[i] });
  return out;
}

function showPnpPickHelper() {
  const ann = pnpPanel.annotation;
  if (!ann?.pose) return 0;
  const r = finiteOr(state.bounds?.radius, 10);
  // 视锥长度取场景尺度：太短看不出方向，太长会糊满屏幕
  const far = Math.min(30, Math.max(3, r * 1.6));
  const segs = frustumSegments(ann.pose, far);
  const dots = samplePointsInFrustum(ann.pose, far);
  const n = structureLayer.setTransient(segs, dots);
  if (!dots.length) {
    ui.toast('已显示相机视锥（黄色虚线）。该范围内没抽到明显点云，可换个特征点试试');
  }
  return n;
}


// ------------------------------------------------------------------ 现场建模面板
//
// 「依据视频和图片建立现场模型」是本系统的核心能力：
// 上传素材 → COLMAP 位姿 → gsplat 训练 → 导出 .splat → 出现在「模型文件」里可直接载入。

const reconPanel = new ReconPanel({
  currentProjectId: () => state.project?.id ?? null,
  getProject: () => state.project,
  operator: () => conn.name || '未署名',
  // 建模完成后刷新模型下拉，让新模型立刻可选
  refreshModels: async () => {
    try {
      await updateModelList();
      ui.toast('模型列表已刷新，可在「模型文件」里选择新模型');
    } catch (err) {
      console.error('[recon] 刷新模型列表失败', err);
    }
  },
  toast: (msg, opts) => ui.toast(msg, opts),
  // 统一走自定义确认框：原生 window.confirm 在部分环境会被静默屏蔽，
  // 表现为「点了取消毫无反应」——建模任务的取消与素材体积预警都必须可靠。
  confirm: (opts) => ui.confirm(opts),
  onBusy: (open, text) => {
    if (open) ui.showLoader(text ?? '正在启动建模任务…');
    else ui.hideLoader();
  },
});
reconPanel.init();

// ------------------------------------------------------------------ 精确测量面板
//
// 「粗模上标出需精确测量的位置 → 民警实测填写 → 反算比例尺提升精度」这条闭环的前端部分。

const measurePanel = new MeasurePanel({
  currentProjectId: () => state.project?.id ?? null,
  getProject: () => state.project,
  operator: () => conn.name || '未署名',
  onUpdateAnnotation: (id, patch) => updateAnnotation(id, patch),
  onCalibrated: async (result) => {
    // 比例尺变了：更新本地 world，重绘所有标注（测距/尺寸会立刻按真实尺寸显示）
    if (result.world) state.project.world = { ...(state.project.world ?? {}), ...result.world };
    annotationLayer.setAll(state.project.annotations);
    ui.renderAnnotations(state.project.annotations);
    if (state.selectedId) selectAnnotation(state.selectedId, { silent: true });
  },
  toast: (msg, opts) => ui.toast(msg, opts),
});
measurePanel.init();

state.scene = scene;
state.camera = camera;
state.controls = controls;

// ------------------------------------------------------------------ 渲染循环

const TARGET_FPS = 60; // 上限，避免在无 vsync 限制的机器上空转（会把 GPU 与电量吃满）
const FRAME_MIN_MS = 1000 / TARGET_FPS;

let lastFrame = performance.now();
let running = true;
let rafId = 0;
let timerId = 0;

// FPS 采用「固定时间窗计数」，比逐帧 1/dt 累加稳定得多
let fpsWindowStart = performance.now();
let fpsWindowFrames = 0;
let fpsDisplay = 0;

function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  if (state.renderer) state.renderer.resize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  state.canvasWidth = w;
  state.canvasHeight = h;
  state.pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
}

/** 用「下一帧何时该画」来控制节奏：距上一帧不足 FRAME_MIN_MS 就先交给定时器等待 */
function scheduleNextFrame() {
  if (!running) return;
  const elapsed = performance.now() - lastFrame;
  if (elapsed >= FRAME_MIN_MS) {
    rafId = requestAnimationFrame(frame);
  } else {
    // 用 setTimeout 补足剩余时间；误差 <= 1ms 且不会空转
    timerId = setTimeout(() => {
      rafId = requestAnimationFrame(frame);
    }, Math.max(0, FRAME_MIN_MS - elapsed - 1));
  }
}

function frame(now) {
  if (!running) return;
  scheduleNextFrame();

  const rawDt = (now - lastFrame) / 1000;
  lastFrame = now;
  // dt 下限保护：某些环境（高刷屏、首帧、切标签页回来）会出现极小或极大的间隔，
  // 直接拿去做 1/dt 会算出几万 FPS 的荒谬值
  const dt = Math.min(0.1, Math.max(rawDt, 1 / 1000));

  if (state.mode === 'fly') flyUpdate(dt);
  else controls.update();

  if (state.renderer?.chunks) {
    state.renderer.modelMatrix.copy(WORLD_MATRIX);
    state.renderer.maybeSort(camera);
    if (!state.filters.sort) state.renderer.chunkOrder = state.renderer.chunks.map((_, i) => i);
    state.renderer.render(camera);
  }

  drawing.update(dt);
  cameraOverlay.update(dt);
  labelLayer.update();
  structureLayer.update();
  updatePeers();

  // 固定时间窗统计 FPS（每 500ms 刷新一次显示）
  fpsWindowFrames++;
  const windowMs = now - fpsWindowStart;
  if (windowMs >= 500) {
    fpsDisplay = (fpsWindowFrames * 1000) / windowMs;
    fpsWindowFrames = 0;
    fpsWindowStart = now;
  }

  ui.renderHud({
    fps: fpsDisplay,
    gaussians: state.renderer?.stats.gaussians ?? 0,
    drawn: state.renderer?.stats.drawn ?? 0,
    sortMs: state.renderer?.stats.sortMs ?? 0,
  });

  // 视角同步（节流 12Hz）
  if (conn.client?.connected && now - conn.throttle.pose > 80) {
    conn.throttle.pose = now;
    publishPose();
  }
}

function startRenderLoop() {
  fpsWindowStart = performance.now();
  fpsWindowFrames = 0;
  lastFrame = performance.now();
  rafId = requestAnimationFrame(frame);
}

window.addEventListener('resize', () => {
  ui.resize();
  resize();
});
// 页面切到后台时暂停渲染，回来再恢复（省电，也避免回来时 dt 巨大）
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    running = false;
    cancelAnimationFrame(rafId);
    clearTimeout(timerId);
  } else if (!running) {
    running = true;
    lastFrame = performance.now();
    fpsWindowStart = performance.now();
    fpsWindowFrames = 0;
    rafId = requestAnimationFrame(frame);
  }
});

// ------------------------------------------------------------------ 视角同步

const _tmpQuat = new THREE.Quaternion();
const _tmpVec = new THREE.Vector3();
const _invWorld = new THREE.Matrix4();
const WORLD_QUAT_INV = WORLD_QUAT.clone().invert();

function publishPose() {
  // scene 系 -> local 系：q_local = WORLD_QUAT⁻¹ · q_scene,  p_local = WORLD_MATRIX⁻¹ · p_scene
  // （与 camera-overlay.js 的 peerPoseToScene / annotationPoseToScene 严格互逆）
  _tmpQuat.copy(camera.quaternion).premultiply(WORLD_QUAT_INV);
  _tmpVec.copy(camera.position).applyMatrix4(WORLD_MATRIX_INV);
  conn.client?.send({
    type: 'pose',
    p: [round4(_tmpVec.x), round4(_tmpVec.y), round4(_tmpVec.z)],
    q: [round4(_tmpQuat.x), round4(_tmpQuat.y), round4(_tmpQuat.z), round4(_tmpQuat.w)],
    fov: round4(camera.fov),
    mode: state.mode,
    name: conn.name,
  });
}

function round4(v) {
  return Math.round(v * 10000) / 10000;
}

function updatePeers() {
  const now = performance.now();
  for (const peer of state.peers.values()) {
    if (!peer.object) continue;
    const age = (now - (peer.lastSeen ?? 0)) / 1000;
    peer.object.visible = age < 8;
    if (peer.target) {
      peer.object.position.lerp(peer.targetPos, 0.15);
      peer.object.quaternion.slerp(peer.targetQuat, 0.15);
    }
  }
}

function ensurePeer(peer) {
  let p = state.peers.get(peer.id);
  if (!p) {
    const obj = cameraOverlay.makePeerFrustum(peer.color ?? '#4dd0e1', peer.name ?? '?');
    peerGroup.add(obj);
    p = { ...peer, object: obj, targetPos: new THREE.Vector3(), targetQuat: new THREE.Quaternion(), lastSeen: performance.now() };
    state.peers.set(peer.id, p);
  }
  Object.assign(p, peer);
  return p;
}

// ------------------------------------------------------------------ 案件与模型

async function refreshProjects(selectId = null) {
  const res = await api.projects();
  state.projects = res.projects ?? [];
  ui.renderProjects(state.projects, state.project?.id ?? selectId);
}

async function openProject(id) {
  const res = await api.project(id);
  state.project = res.project;
  ui.setProject(state.project);
  annotationLayer.setAll(state.project.annotations ?? []);
  ui.renderAnnotations(state.project.annotations ?? []);
  updateModelList();
  // 切案件时把建模面板重置，并尝试接回该案件正在跑的建模任务
  reconPanel.onProjectChanged();
  measurePanel.onProjectChanged();
  applyPresentation(state.project.presentation);
  selectAnnotation(null, { silent: true });
  connectCollab();
}

async function updateModelList() {
  const models = await api.models();
  // ⚠️ 必须存进 state：下拉框加载时要按 URL 找回**完整对象**（含 structureUrl），
  //    否则结构线永远不出现（实测：`onLoadModel` 只拼了 {name,url}，把 structureUrl 丢了）。
  state.models = models;
  ui.setModelOptions(models);
}

async function loadModel(model, { announce = false } = {}) {
  if (!model?.url) return;
  ui.showLoader('正在载入三维现场…');
  try {
    const gauss = await loadGaussians(model.url, {
      onProgress: (p, got, total) => ui.setLoaderProgress(p, `${(got / 1048576).toFixed(1)}MB / ${(total / 1048576).toFixed(1)}MB`),
      onStage: (stage, p) => ui.setLoaderStage(stage, p),
    });
    const bounds = computeBounds(gauss.position);
    // 记住高斯位置：PnP 标定时要抽稀出「视锥内的可选点」（T2B）
    state.gaussianPositions = gauss.position;
    ui.setLoaderStage('building', 0.2);

    if (!state.renderer) {
      // 把「渲染参数」滑杆的当前值带进构造函数：渲染器是**加载模型时**才创建的，
      // 只靠滑杆的 input 事件，首次加载前调过的值会丢（实测）。
      state.renderer = new SplatRenderer(canvas, {
        maxPixelRatio: 2,
        fov: 55,
        gaussScale: state.filters.gaussScale ?? 1,
        opacityGain: state.filters.opacity ?? 1,
        exposure: state.filters.exposure ?? 1,
      });
      // 关键：渲染器是拿到 WebGL2 上下文之后才存在的，而 resize() 依赖 state.renderer。
      // 启动时的那次 resize() 发生在渲染器创建之前，因此**必须在创建后立刻重算一次**，
      // 否则 canvas 的绘制缓冲会一直是默认的 300×150 —— 表现为「模型已载入但画面全黑」。
      resize();
    }
    state.renderer.setData(gauss, { useSH: false });
    state.renderer._boundRadius = bounds.radius;

    state.picker = new SplatPicker(gauss.position, gauss.count);
    state.picker.bounds = bounds;
    // 场景尺度必须存进 state：focusAnnotation / 飞行速度 / 网格尺寸等多处都要用。
    // 之前只赋给了 state.picker.bounds，导致 state.bounds 一直是 null，
    // 而 `state.bounds?.radius * 0.4 ?? 1` 里的 **NaN 不会被 ?? 兜住**（?? 只处理 null/undefined），
    // 于是 dist 变成 NaN → 相机位置 NaN → 所有矩阵 NaN → 整屏黑。
    state.bounds = bounds;

    scene.background = new THREE.Color(state.project?.presentation?.background ?? '#080c11');
    gridHelper.scale.setScalar(Math.max(bounds.radius * 2, 0.5));
    gridHelper.position.set(bounds.center[0], bounds.max[1], bounds.center[2]);
    gridHelper.updateMatrix();
    axesHelper.position.set(bounds.center[0], bounds.max[1], bounds.center[2]);
    axesHelper.scale.setScalar(Math.max(bounds.radius * 0.35, 0.2));

    const thumbnail = captureThumbnail();
    state.model = model;
    // ⚠️ 静态演示版没有案件写入能力：state.project 可能为 null，
    //    这里裸访问 .id 会抛错，把后面的 applyStructure（画结构线）整段跳掉
    //    —— 表现为"模型能看到、结构线一条没有"，实测踩过。
    if (state.project?.id) localStorage.setItem(`dsh:lastModel:${state.project.id}`, model.url);

    // 记录"本案件上次用的模型"：这是**持久化**，静态演示版没有服务端。
    // ⚠️ 这里必须容错：静态模式下 api.setModel 会抛"只读演示版"，
    //    如果让它冒泡，下面的 applyStructure（画结构线）会被整段跳过 ——
    //    表现就是"模型能看、结构线一条没有"。
    if (state.project?.id) {
      try {
        await api.setModel(state.project.id, {
          ...model,
          center: bounds.center,
          gaussianCount: gauss.count,
          bounds,
          shDegree: gauss.shDegree ?? 0,
          thumbnail,
        });
      } catch (err) {
        // 静态演示版预期会走到这里（只读）；服务端正常时不抛。
        state.modelPersistSkipped = String(err?.message ?? err);
      }
    }

    frameCameraOnBounds(bounds);
    // 房间结构骨架（可选）：有就画出来，没有就静默跳过。
    // 抽成函数是因为**人工改判**之后要能只重画结构层、不重载高斯（见 setSceneMode）。
    const structureShort = await applyStructure(model, { announce });
    ui.hideLoader();
    const covNote = state.coverageWarn;
    state.coverageWarn = null;
    ui.toast(
      `已载入 ${formatInt(gauss.count)} 个高斯球`
      + (structureShort ? ` · ${structureShort}` : '')
      + (covNote ? ` · ⚠ ${covNote}` : ''),
      { duration: covNote ? 9000 : 3200 },
    );
    // 下拉框同步到**真正载入的这个模型**：否则界面会"下拉框写着 A、场里是 B"
    // （实测截图就是这个问题，对研判是硬伤）。
    ui.syncModelSelect(model.url);
    ui.setModelHint(`${model.name} · ${formatInt(gauss.count)} 高斯 · 半径 ${bounds.radius.toFixed(2)}m${state.structureNote ?? ''}`);
  } catch (err) {
    ui.hideLoader();
    updateStructureBadge();
    ui.toast(`载入失败：${err.message}`, { error: true, sticky: true });
    console.error(err);
  }
}

/**
 * 刷新顶栏的「结构状态牌」。
 *
 * 为什么需要：用户反馈「房间处理和物体处理没区别，可能是没显示出来」。
 * 实测确实如此 —— 柜子那两个模型在 自动/强制房间/强制物体 三种判定下**都画 0 条线**
 * （几何不可信 → 不画物体盒；强制房间时自检不通过 → 也不画房间盒），
 * 屏幕上完全一样。原因原来只写在左栏一行小字里，很容易被当成"功能没生效"。
 * 现在：**按哪种处理、画了没有、为什么没画**，常驻顶栏一眼可见（鼠标悬停看完整原因）。
 */
function updateStructureBadge() {
  const el = document.getElementById('structBadge');
  if (!el) return;
  const L = structureLayer;
  const seg = L.segments?.length ?? 0;
  const mode = L.mode ?? 'none';
  const modeCn = { room: '房间', object: '物体', uncertain: '不确定', none: '未载入' }[mode] ?? mode;
  let text;
  let cls;
  if (mode === 'none') {
    text = '结构：未载入';
    cls = 'none';
  } else if (L.objectDrawn) {
    text = `结构：${modeCn} · 已画物体盒${L.layers?.length ? `+${L.layers.length}层` : ''}`;
    cls = 'obj';
  } else if (L.roomDrawn && L.estimated) {
    // 推测出来的房间盒：必须和"已核实"区分开（实测误差 10~19%，5 面墙全拟合才 ~1%）
    text = '结构：房间（推测）· 虚线房间盒';
    cls = 'warn';
  } else if (L.roomDrawn) {
    text = '结构：房间 · 已画房间盒';
    cls = 'ok';
  } else if (mode === 'room') {
    text = '结构：房间 · 自检未通过，未画';
    cls = 'none';
  } else if (mode === 'object') {
    text = '结构：物体 · 几何不可信，未画';
    cls = 'none';
  } else {
    text = '结构：不确定 · 未画';
    cls = 'none';
  }
  if (L.modeSource === 'manual') text += ' · 人工';
  if (L.visible === false) text += ' · 已隐藏';
  el.textContent = text;
  el.className = `stat struct-badge ${cls}`;
  const why = [L.notice, ...(L.objectTrust?.reasons ?? [])].filter(Boolean);
  el.title =
    (why.length ? why.join('\n') : text) +
    '\n\n线色：青=房间盒/地面（虚线=推测） · 黄=墙 · 橙=物体盒 · 深橙=家具盒 · 按类别配色（床蓝/门黄/柜绿/桌紫/椅橙）';
}

/**
 * 叠加「房间结构骨架」图层，并返回一句**很短的**状态说明给提示条用。
 *
 * 为什么单独成函数：人工改判场景模式之后，只需要重画这一层，
 * 没必要把几万个高斯重新下载解码一遍（那是几秒到几十秒的差别）。
 *
 * ⚠️ 两处历史教训写在这里，别再犯：
 *   1. 说明文字必须**短**。第一版把几十个字的完整理由拼进 `setModelHint`，
 *      提示条多折一行把「建模进度区」推出了面板可视区，
 *      结果 `#btnReconCancel` 的坐标落到页脚上、点不动（browser_recon_gate 当场失败）。
 *      长理由走面板日志与 docs。
 *   2. 非 room 的说明**不能在这条 toast 里发** —— 载入流程紧接着还有一句
 *      「已载入 N 个高斯球」，会把它立刻覆盖，等于没说。所以这里只**返回**说明，
 *      由调用方拼到最后那条 toast 上。
 */
/**
 * `*.structure.json` → 同名的 `*.furniture.json`（家具级线框盒子）。
 *
 * 为什么要单独一个文件：家具盒由 tools/furniture_lines.py 从**稠密表面点**聚类而来，
 * 坐标系与结构文件不同（稠密重建/VGGT 没有曼哈顿系），两者必须能独立加载、独立开关。
 */
function structureToFurnitureUrl(model) {
  // 优先用结构文件同名（两者通常并排放）；没有结构文件时退回**模型文件同名** ——
  // 实测：VGGT/稠密重建的模型 structureUrl 是 null（只有模型 URL），
  // 从结构推的话永远推不出家具文件（这条踩过一次，测试才抓到）。
  const u = model?.structureUrl;
  if (u) return u.replace(/\.structure\.json(\?.*)?$/, '.furniture.json$1');
  const base = model?.url;
  if (!base) return null;
  return base.replace(/\.(splat|ply|ksplat)(\?.*)?$/i, '.furniture.json$2');
}

/**
 * 把「重建单位 → 米」的系数同步给结构/家具图层。
 *
 * 结构文件里的尺寸是重建单位；没校正过就标"单位"，校正过才标"m"。
 * 之前直接写死 "m"，在白瓷砖那份 1 单位≈0.09 m 的重建上标出了"24.91 m"（实测截图抓到）。
 */
function syncLayerCalibration() {
  const s = Number(state.project?.world?.scaleToMeters);
  structureLayer.setCalibration(Number.isFinite(s) && s > 0 ? s : 1);
}

async function loadFurnitureSidecar(model) {
  const url = structureToFurnitureUrl(model);
  if (!url) {
    structureLayer.setFurniture(null);
    return 0;
  }
  try {
    const data = await fetch(url, { cache: 'no-store' }).then((r) => (r.ok ? r.json() : null));
    return structureLayer.setFurniture(data);
  } catch {
    // 没有就当作"该模型没有家具数据"，不报警（回归测试对多余 warn 敏感）
    structureLayer.setFurniture(null);
    return 0;
  }
}

async function applyStructure(model, { announce = false } = {}) {
  structureLayer.clear();
  // 家具盒是**独立数据源**（`*.furniture.json`），换模型时必须一起清掉，
  // 否则上一个模型的家具会留在屏幕上 —— 那比不画更误导。
  structureLayer.setFurniture(null);
  syncLayerCalibration();
  state.structureNote = '';
  let structureShort = '';
  // 家具盒必须**先加载**、且不依赖结构文件：稠密重建/VGGT 的模型常常没有
  // `.structure.json`（实测：`applyStructure` 在下面那段 early-return 里就退出了，
  // 家具永远画不出来）。所以这里先取家具，再决定结构走哪条路。
  await loadFurnitureSidecar(model);
  if (!model?.structureUrl) {
    // 没有结构骨架：整个改判控件藏起来（它只对"能判房间/物体"的模型有意义）
    // 但家具盒可能仍然有（稠密/VGGT 模型）→ 要在提示里说出来，别让用户以为"没画"
    if (structureLayer.furnitureDrawn) {
      const cnt = structureLayer.furniture?.objects?.length ?? 0;
      structureShort = `家具 ${cnt} 件（按类别配色）`;
      state.structureNote = ` · ${structureShort}`;
      ui.setModelHint(
        `${model.name} · ${formatInt(state.gaussianPositions?.length / 3 || 0)} 高斯 · 半径 ${state.bounds?.radius?.toFixed(2) ?? '?'}m${state.structureNote}`
      );
    }
    syncSceneModeControl(model);
    updateStructureBadge();
    return structureShort;
  }
  try {
    const st = await fetch(model.structureUrl, { cache: 'no-store' }).then((r) => (r.ok ? r.json() : null));
    if (!st) return structureShort;
    // ---- 人工改判优先于结构文件里的自动判定 ----
    // 结构文件是"那次重建的证据"，改判是"使用者的判断"，两者分开存：
    // 这里只覆盖用于渲染的字段，**不回写结构文件**。
    const ov = model.sceneModeOverride;
    if (ov && (ov.mode === 'room' || ov.mode === 'object')) {
      st.sceneMode = ov.mode;
      st.sceneModeSource = 'manual';
      st.sceneModeOverrideBy = ov.operator ?? null;
    }
    // ---- 竖直刻度尺：让评委建立"多高"的空间感 ----------------------------
    // 高度来源优先级：房间盒（可信时）> 点云包围盒沿重力方向的跨度。
    // 重力：结构文件的 frame.up；没有就退回世界 Z 轴（大多数重建的默认竖直方向）。
    try {
      const up = st?.frame?.up ?? [0, 0, 1];
      let height = 0;
      if (st?.room?.size && st.room.size.length === 3 && st.sane !== false) {
        height = Number(st.room.size[2]) || 0;
      } else if (state.bounds) {
        const b = state.bounds;
        const proj = [0, 1, 2].map((k) => Math.abs(up[k] || 0));
        const kMax = proj.indexOf(Math.max(...proj));
        height = Math.max(0, (b.max?.[kMax] ?? 0) - (b.min?.[kMax] ?? 0));
      }
      if (height > 0) {
        // 刻度间隔取"好数"：让刻度数落在 3~7 之间
        const nice = [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2, 2.5, 5, 10, 20, 50];
        let spacing = nice.find((v) => height / v <= 7) ?? height / 5;
        // 锚点优先级：房间盒"离地最近的那个角"（贴着地面、就在房间里，最直观）
        //            > 包围盒最小角。刻度方向朝场景中心，避免指向画面外。
        let origin = null;
        let axisU = [1, 0, 0];
        const corners = st?.room?.corners;
        if (Array.isArray(corners) && corners.length === 8) {
          const upV = [up[0], up[1], up[2]];
          const proj = (p) => p[0] * upV[0] + p[1] * upV[1] + p[2] * upV[2];
          let best = corners[0];
          for (const c of corners) if (proj(c) < proj(best)) best = c;
          origin = [best[0], best[1], best[2]];
          const center = state.bounds?.center;
          if (center) {
            axisU = [
              (center[0] ?? 0) - best[0],
              (center[1] ?? 0) - best[1],
              (center[2] ?? 0) - best[2],
            ];
          }
        } else if (state.bounds) {
          const b = state.bounds;
          origin = [b.min[0], b.min[1], b.min[2]];
        }
        origin = origin ?? [0, 0, 0];
        structureLayer.setRuler({ origin, up, height, spacing, axisU });
      } else {
        structureLayer.setRuler(null);
      }
    } catch (err) {
      console.warn('[ruler] 刻度尺参数计算失败，跳过：', err?.message ?? err);
    }

    const parts = structureLayer.set(st);
    // ---- 家具盒：同名的 `*.furniture.json`（可选）------------------------------
    // 单独一个文件、单独一套坐标系：由 tools/furniture_lines.py 从**稠密表面点**
    // 聚类而来。没有这个文件 = 该模型没有家具数据，静默跳过（不是错误）。
    await loadFurnitureSidecar(model);
    // ⚠️ 只有**真的画了房间盒**才可以说"已叠加房间结构线框"。
    // 判定为 object / uncertain 时只画相机轨迹，此时谎报"已叠加房间盒"
    // 会让用户以为看到的是户型，而实际上没有（甚至曾经画的是错的）。
    const sMode = structureLayer.mode;
    const manual = structureLayer.modeSource === 'manual';
    if (sMode === 'object') {
      // ⚠️ 物体模式有两种结果，必须分开说：
      //    几何可信 → 真的画了物体盒与水平层；
      //    几何不可信 → **什么都不画**（实测那份柜子的盒子 16.8 m 长，画出来是假的）。
      //    含混地说"已叠加"会让用户以为看到了柜子结构。
      structureShort = structureLayer.objectDrawn
        ? (manual ? '已按「物体」呈现（人工改判）' : '已按「物体」呈现：包围盒 + 水平层')
        : '未叠加房间盒，物体几何不可信（只显示轨迹）';
    } else if (sMode === 'room') {
      // ⚠️ 必须看 `roomDrawn`（真的画没画），不能只看"判定是不是 room"。
      //    我第一版这里只写了 `sMode === 'room'`，于是**房间盒画出来了也照说
      //    "未叠加房间盒（结构自检未通过）"** —— 左栏提示与顶栏状态牌互相打脸
      //    （截图实测：顶栏"已画房间盒"、左栏"未叠加…自检未通过"）。
      //    这和之前那条"没画却说已叠加"是同一类错误的镜像：**判定 ≠ 画没画**。
      structureShort = structureLayer.roomDrawn ? '' : '未叠加房间盒（结构自检未通过）';
    } else {
      structureShort = '未叠加房间盒（判定：结构不确定）';
    }
    // ⚠️ 只有**真的画了房间盒**才可以说"已叠加房间结构线框"。
    //    以前的条件是 `parts && sMode === 'room'`，而 `parts` 在"只画了相机轨迹"时
    //    也是非零（轨迹本身算一条折线），于是 mode=room 但自检不通过时
    //    界面会谎报"已叠加房间结构线框 6.0×16.5×2.6 m"（截图实测抓到）。
    if (structureLayer.roomDrawn) {
      const size = st?.room?.size ?? [];
      const eye = st?.trajectory?.heightAboveFloor?.median;
      ui.toast(
        `已叠加房间结构线框：${size.map((v) => v.toFixed(1)).join(' × ')} m` +
          (eye != null ? `（相机离地 ${eye} m）` : '') +
          (manual ? '（人工改判）' : '')
      );
    }
    // 拍摄引导：覆盖度不好就直接说出来 —— 这是「模型为什么糊」最可能的原因，
    // 而且**调参救不了**，只能补拍。
    //
    // ⚠️ 只在**用户主动载入**（announce=true）时弹，而且同一个模型只弹一次：
    // 否则开机自动恢复上次模型时也会弹一个**模态**窗口，
    // 把用户所有操作都挡住（这个 bug 当场被 browser_recon_gate 的 15 项失败抓到）。
    const cov = st?.coverage;
    if (cov && cov.verdict && cov.verdict !== '覆盖良好') {
      state.structureNote = ` · ⚠ ${cov.verdict}`;
      const seenKey = `dsh:adviceSeen:${model.url}`;
      const seen = (() => {
        try {
          return localStorage.getItem(seenKey) === '1';
        } catch {
          return false;
        }
      })();
      if (announce && !seen) {
        const pct = ((cov.weakRatio ?? 0) * 100).toFixed(1);
        // ⚠️ 这里原来弹的是**模态**框（ui.showShootingAdvice）—— 演示时它正好盖在模型和
        //    结构线上面，评审以为"结构线没了"（实测踩到）。改成非阻塞 toast：
        //    结论照旧写进 state.structureNote（提示条里常驻），只是不再挡画面。
        // ⚠️ 不能单独 toast：紧接着 loadModel 会弹「已载入 N 个高斯球」把它覆盖掉
        //    （实测：用户只看到"已载入"，从没见过覆盖提示）。改为暂存，合并进载入提示。
        state.coverageWarn =
          `覆盖提示：${cov.verdict}（${cov.weakFrames}/${cov.frames} 帧几乎看不到点云，${pct}%）`
          + ' —— 没拍到的方向会缺面，这不是参数问题，建议补拍这些方向';
        try {
          // 控制台别刷警告：回归测试会对多余 warn 报错
          document.getElementById('btnDrawerLeft')?.classList.remove('pulse');
        } catch { /* 忽略 */ }
        try {
          localStorage.setItem(seenKey, '1');
        } catch {
          /* 隐私模式下写不了，忽略 */
        }
      }
      // 若已看过建议：不再打断用户。结论常驻在提示条里（见 setModelHint），
      // 这里**不要** console.warn —— 每次开机都刷一条警告只会污染控制台（回归测试会报错）。
    }
    // 家具盒画出来了就在提示里报数：用户看到橙框时能立刻知道有几件、是不是空的
    if (structureLayer.furnitureDrawn) {
      const cnt = structureLayer.furniture?.objects?.length ?? 0;
      structureShort = `${structureShort}${structureShort ? ' · ' : ''}家具 ${cnt} 件（按类别配色）`;
    }
    if (structureShort) state.structureNote = ` · ${structureShort}`;
    // ⚠️ 必须**在 set() 之后**同步下拉框：`syncSceneModeControl` 要读
    //    `structureLayer.mode`（刚算出来的判定）。我第一版把它放在 set() 之前，
    //    结果提示条显示的是**上一个模型**的判定（载入 room-real 却写着"物体（贴拍）"）。
    syncSceneModeControl(model);
    updateStructureBadge();
    ui.syncModelSelect(model.url);
    ui.setModelHint(
      `${model.name} · ${formatInt(state.gaussianPositions?.length / 3 || 0)} 高斯 · 半径 ${state.bounds?.radius?.toFixed(2) ?? '?'}m${state.structureNote}`
    );
  } catch (err) {
    console.warn('[structure] 结构骨架加载失败', err);
  }
  return structureShort;
}

/** 把「场景判定」下拉框同步成当前模型的状态（没有结构骨架时整块隐藏）。 */
function syncSceneModeControl(model) {
  const field = document.getElementById('sceneModeField');
  const sel = document.getElementById('sceneModeSelect');
  const hint = document.getElementById('sceneModeHint');
  if (!field || !sel || !hint) return;
  const usable = Boolean(model?.structureUrl);
  field.hidden = !usable;
  hint.hidden = !usable;
  if (!usable) return;
  const ov = model?.sceneModeOverride ?? null;
  sel.value = ov?.mode === 'room' || ov?.mode === 'object' ? ov.mode : 'auto';
  // ⚠️ 用 `autoDecision`（结构文件里记的**改判前**判定），不能拿 `structureLayer.mode` ——
  //    人工改判之后那个值已经是"改判后的"，会让界面谎报"自动判定为 X"。
  const autoLabel = { room: '房间', object: '物体（贴拍）', uncertain: '不确定' };
  const auto = autoLabel[structureLayer.autoDecision] ?? '—';
  hint.textContent = ov
    ? `已人工改判为「${ov.mode === 'room' ? '房间' : '物体'}」${ov.operator ? `（${ov.operator}）` : ''} · 自动判定为「${auto}」`
    : `自动判定：${auto} · 判不准时可以在这里人工指定`;
  // 物体几何不可信时把**原因**摆在这里：用户看不到物体盒，得知道是为什么
  // （这条提示在面板里可换行、面板本身可滚动，所以比主题提示条更适合放长理由）。
  const trust = structureLayer.objectTrust;
  if (structureLayer.mode === 'object' && trust && !trust.ok) {
    const why = trust.reasons?.[0] ?? '几何不可信';
    hint.textContent += ` · 物体盒未画：${why}`;
  }
}

/**
 * 人工改判场景模式（房间 / 物体 / 自动）。
 *
 * 为什么需要这个入口：几何本身**分不开**这两类素材 —— 实测柜子素材的方向包围度 0.2344，
 * 而某段"以旋转为主"的房间素材是 0.2270。判不准时不该硬猜，由拍摄的人指定最可靠。
 * 改判结果由服务端存 sidecar，**多人共享**（不是只存在本机浏览器里）。
 */
async function setSceneMode(mode, { silent = false } = {}) {
  const model = state.model;
  if (!model?.url) {
    if (!silent) ui.toast('请先载入一个三维现场', { error: true });
    return false;
  }
  if (!['auto', 'room', 'object'].includes(mode)) {
    if (!silent) ui.toast('场景判定只能是 自动 / 房间 / 物体', { error: true });
    return false;
  }
  const file = model.url.replace(/^\/models\//, '');
  try {
    const res = await fetch('/api/models/scene-mode', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: decodeURIComponent(file), mode, operator: conn.name || '未署名' }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.ok === false) throw new Error(data.error ?? `改判失败 ${res.status}`);
    state.model = { ...model, sceneModeOverride: data.sceneModeOverride ?? null };
    await applyStructure(state.model, { announce: false });
    if (!silent) {
      // ⚠️ toast 必须与**实际画出来的东西**一致，不能只说"已重画"：
      //    改判为「房间」时若自检不过，其实一条线都没画 ——
      //    说"已按新判定重画"会让人以为成功了（同类问题刚在房间盒 toast 上踩过一次）。
      const drew = structureLayer.roomDrawn
        ? '，已叠加房间盒'
        : structureLayer.objectDrawn
          ? `，已叠加物体盒${structureLayer.layers?.length ? `+${structureLayer.layers.length}层` : ''}`
          : '，但结构线未叠加（原因见顶栏状态牌）';
      ui.toast(
        mode === 'auto'
          ? '已恢复「自动判定」（按证据重新判定）'
          : `已改判为「${mode === 'room' ? '房间' : '物体'}」${drew}`
      );
    }
    return true;
  } catch (err) {
    if (!silent) ui.toast(`改判失败：${err.message}`, { error: true });
    else console.warn('[structure] 改判失败', err);
    return false;
  }
}

function captureThumbnail() {
  try {
    const c = document.createElement('canvas');
    const w = 320;
    const h = Math.round((w * window.innerHeight) / window.innerWidth);
    c.width = w;
    c.height = h;
    const ctx = c.getContext('2d');
    ctx.drawImage(canvas, 0, 0, w, h);
    return c.toDataURL('image/jpeg', 0.72);
  } catch {
    return null;
  }
}

function frameCameraOnBounds(bounds) {
  const center = new THREE.Vector3(bounds.center[0], bounds.center[1], bounds.center[2]);
  const worldCenter = center.clone().applyMatrix4(WORLD_MATRIX);
  const d = Math.max(bounds.radius * 2.2, 0.6);
  camera.position.set(worldCenter.x + d * 0.7, worldCenter.y + d * 0.55, worldCenter.z + d * 0.7);
  camera.near = Math.max(bounds.radius / 2000, 0.01);
  camera.far = Math.max(bounds.radius * 200, 100);
  camera.updateProjectionMatrix();
  controls.target.copy(worldCenter);
  controls.update();
}

// ------------------------------------------------------------------ 标注操作

async function commitAnnotation(draft) {
  if (!state.project) return ui.toast('请先选择案件', { error: true });
  const payload = {
    kind: draft.kind,
    label: draft.label || `${kindName(draft.kind)} ${(state.project.annotations?.length ?? 0) + 1}`,
    color: draft.color || defaultColor(draft.kind),
    points: draft.points,
    pose: draft.pose,
    media: draft.media ?? [],
    note: draft.note ?? '',
    createdBy: conn.name,
    meta: draft.meta ?? {},
  };
  try {
    const res = await api.addAnnotation(state.project.id, payload);
    const ann = res.annotation;
    // 按 id 去重后再插入：服务端已做幂等，但网络重试/协同回显仍可能把同一条推两次
    const idx = state.project.annotations.findIndex((a) => a.id === ann.id);
    if (idx >= 0) state.project.annotations[idx] = ann;
    else state.project.annotations.push(ann);
    annotationLayer.upsert(ann);
    ui.renderAnnotations(state.project.annotations);
    ui.setToolHint('左键选择，右键/ESC 取消当前绘制');
    return ann;
  } catch (err) {
    ui.toast(`保存失败：${err.message}`, { error: true });
  }
}

function selectAnnotation(id, { silent = false } = {}) {
  state.selectedId = id;
  ui.highlightAnnotation(id);
  ui.renderInspector(id ? state.project?.annotations.find((a) => a.id === id) : null, {
    onUpdate: (patch) => updateAnnotation(id, patch),
    onDelete: () => deleteAnnotation(id),
    onFocus: () => focusAnnotation(id),
    onAttachVideo: (ann) => openCameraVideo(ann),
    onCalibrate: (ann) => openPnpCalibration(ann),
    onRecalibrate: () => recalibrateScale(),
  });
  if (!silent) conn.client?.send({ type: 'select', annotationId: id });
}

/**
 * 用所有已实测的「尺寸标注」反算模型比例尺。
 *
 * 这是「民警实测数据 → 提升建模精度」闭环的最后一步：
 * 后端做最小二乘 scale = Σ(d_model·d_real) / Σ(d_model²)，
 * 结果写进 project.world.scaleToMeters，前端所有测距/尺寸立刻按真实尺寸显示。
 */
async function recalibrateScale() {
  if (!state.project) return;
  const measuredCount = (state.project.annotations ?? []).filter(
    (a) => a.kind === 'dimension' && Number.isFinite(Number(a.measured)) && Number(a.measured) > 0
  ).length;
  if (!measuredCount) {
    ui.toast('还没有任何已实测的尺寸。请先用「尺寸标注」标出两点并填写民警实测值。', { error: true });
    return;
  }
  ui.showLoader('正在依据实测数据校正比例尺…');
  try {
    const res = await api.calibrate(state.project.id, { operator: conn.name || '未署名' });
    // 比例尺变了，模型与所有标注的显示都要跟着更新
    state.project.world = { ...(state.project.world ?? {}), ...(res.world ?? {}) };
    annotationLayer.setAll(state.project.annotations);
    ui.renderAnnotations(state.project.annotations);
    if (state.selectedId) selectAnnotation(state.selectedId, { silent: true });
    syncLayerCalibration();
    const s = Number(res.scaleToMeters);
    const pct = ((s - 1) * 100).toFixed(2);
    const confText = { high: '高', medium: '中', low: '偏低（建议复核标注点）' }[res.confidence] ?? res.confidence;
    ui.toast(
      `比例尺已校正：${s.toFixed(4)}（相对原来 ${pct >= 0 ? '+' : ''}${pct}%）· 用了 ${res.count} 条实测数据 · 可信度 ${confText}`
    );
    ui.setToolHint(`比例尺已校正为 ${s.toFixed(4)}，所有测距已按真实尺寸显示`);
  } catch (err) {
    ui.toast(`校正失败：${err.message}`, { error: true });
  } finally {
    ui.hideLoader();
  }
}

async function updateAnnotation(id, patch) {
  try {
    const res = await api.updateAnnotation(state.project.id, id, patch);
    const ann = res.annotation;
    const idx = state.project.annotations.findIndex((a) => a.id === id);
    if (idx >= 0) state.project.annotations[idx] = ann;
    annotationLayer.upsert(ann);
    ui.renderAnnotations(state.project.annotations);
    ui.renderInspector(ann, {
      onUpdate: (p) => updateAnnotation(id, p),
      onDelete: () => deleteAnnotation(id),
      onFocus: () => focusAnnotation(id),
      onAttachVideo: (a) => openCameraVideo(a),
      onCalibrate: (a) => openPnpCalibration(a),
      onRecalibrate: () => recalibrateScale(),
    });
  } catch (err) {
    ui.toast(`更新失败：${err.message}`, { error: true });
  }
}

async function deleteAnnotation(id) {
  if (!id) return;
  try {
    await api.deleteAnnotation(state.project.id, id);
    state.project.annotations = state.project.annotations.filter((a) => a.id !== id);
    annotationLayer.remove(id);
    ui.renderAnnotations(state.project.annotations);
    selectAnnotation(null, { silent: true });
  } catch (err) {
    ui.toast(`删除失败：${err.message}`, { error: true });
  }
}

/** 主动删除一条标注（带确认框），供快捷键等入口使用 */
async function deleteAnnotationConfirmed(id) {
  const ann = state.project?.annotations.find((a) => a.id === id);
  if (!ann) return;
  const name = ann.label || kindName(ann.kind, ann);
  const ok = await ui.confirm({
    title: '删除该对象',
    body: `确定删除「${name}」吗？`,
    warn: '该操作会同步给同案其他人员，且不可撤销。',
  });
  if (ok) await deleteAnnotation(id);
}

function focusAnnotation(id) {
  const ann = state.project?.annotations.find((a) => a.id === id);
  if (!ann) return;
  let target;
  if (ann.kind === 'camera' && ann.pose?.position) target = new THREE.Vector3(...ann.pose.position);
  else target = new THREE.Vector3(...(ann.points[0] ?? [0, 0, 0]));
  // 标注坐标可能损坏（NaT/NaN/缺字段）——一旦把 NaN 塞进相机，
  // 整个矩阵链都是 NaN，画面直接全黑，而且很难恢复（必须重新载入模型）。
  // 这里做入口校验，宁可不动相机也不要污染它。
  if (!Number.isFinite(target.x) || !Number.isFinite(target.y) || !Number.isFinite(target.z)) {
    console.warn('[focusAnnotation] 标注坐标无效，已跳过定位：', ann.kind, ann.points?.[0]);
    ui.toast('该标注的坐标无效，无法定位', { error: true });
    return;
  }
  const world = target.applyMatrix4(WORLD_MATRIX);
  // 注意：不能写成 `state.bounds?.radius * 0.4 ?? 1` —— NaN 不会被 ?? 兜住
  // （?? 只处理 null/undefined），dist 会变成 NaN 从而污染整条相机矩阵链。
  const radius = finiteOr(state.bounds?.radius, 2.5);
  const center = state.bounds?.center
    ? new THREE.Vector3(...state.bounds.center).applyMatrix4(WORLD_MATRIX)
    : new THREE.Vector3();

  // ⚠️ 必须同时挡住「飞到天外」—— 不只是 NaN。
  //
  // 为什么：`onSelectAnnotation` 里 **单击标注列表就会调 focusAnnotation** 移动相机。
  // 相机标注的 pose 一旦是坏的（实测本案件有一个 Y=283 米的位姿，模型半径才 15 米），
  // 单击就把相机送到 283 米外 → **整个模型黑掉**；而且 controls.target 也跟着飞走，
  // 用户几乎没法自己找回视角。原来只挡了 NaN，所以「数值合法但荒谬」的坐标长驱直入。
  // 这条路径与 PnP 面板的「贴回现场」是两个不同入口，都要挡。
  //
  // 比较必须在**同一个坐标系**里做：`world` 和 `center` 都已乘过 WORLD_MATRIX（场景系）。
  // （`target` 被 applyMatrix4 原地改写过，拿它跟世界系的 bounds.center 比是错的。）
  const targetDist = world.distanceTo(center);
  const farLimit = Math.max(30, radius * 4);
  if (targetDist > farLimit) {
    ui.toast(
      `该标注离现场 ${targetDist.toFixed(0)} 米（场景半径 ${radius.toFixed(1)} 米），` +
        `定位会让视角飞出模型，已取消。这类坐标多半来自一次失败的 PnP 标定 —— ` +
        `请重新标定该机位；若画面已经飞走，点顶栏「回到全景」`,
      { error: true, sticky: true }
    );
    return;
  }

  // 关键：定位距离既要适配场景尺度，也要保证「标注确实在视野里」。
  // 只取 0.4*radius 时，若标注坐标明显偏离模型（例如落在模型内部或远处），
  // 相机会正好停在模型里 → 依然是一片黑。这里把距离限制在场景直径的一个合理区间内。
  const dist = Math.min(Math.max(radius * 0.4, 0.35), radius * 2.2);
  let dir = camera.position.clone().sub(controls.target).normalize();
  if (dir.lengthSq() < 1e-12) {
    // 相机正好在 target 上，方向退化 —— 给一个默认斜视角
    dir.set(0.7, 0.55, 0.7).normalize();
  }

  // 让「标注点」和「场景中心」都在视野里：把视线方向稍微偏向场景中心，
  // 避免标注在场景边缘时相机整个转出模型之外。
  if (state.bounds && world.distanceTo(center) > radius) {
    const toCenter = center.clone().sub(world).normalize();
    dir.lerp(toCenter, 0.65).normalize();
  }

  const newPos = world.clone().addScaledVector(dir, dist);
  if (!Number.isFinite(newPos.x) || !Number.isFinite(newPos.y) || !Number.isFinite(newPos.z)) {
    // 兜底：宁可不动相机，也绝不把 NaN 写进相机矩阵（写进去就是整屏黑）
    console.warn('[focusAnnotation] 相机位置计算异常，已跳过定位', {
      world: world.toArray(),
      dir: [dir.x, dir.y, dir.z],
      dist,
      boundsRadius: state.bounds?.radius ?? null,
    });
    ui.toast('无法定位到该标注', { error: true });
    return;
  }
  controls.target.copy(world);
  camera.position.copy(newPos);
  controls.update();
  // 兜底：万一 OrbitControls 内部把位置算坏了，直接回滚，避免整屏变黑
  if (!Number.isFinite(camera.position.x) || !Number.isFinite(camera.position.y) || !Number.isFinite(camera.position.z)) {
    console.error('[focusAnnotation] controls.update() 产生了无效相机位置，已回滚');
    camera.position.copy(center).addScaledVector(new THREE.Vector3(0.7, 0.55, 0.7).normalize(), Math.max(radius * 1.6, 1));
    controls.target.copy(center);
    controls.update();
  }

  // ---- 「绝不让用户对着空气」自检 ----------------------------------------
  //
  // 为什么需要：点击标注列表会调 focusAnnotation 移动相机，而标注可能在**模型之外**
  // （实测某个相机标注的位姿距场景中心 17.8 m，而模型半径只有 9.7 m）。
  // 相机被放到那个标注附近、视线也朝着它 —— 模型就落在视野之外，屏幕一片空。
  // 既有的距离阈值挡不住这种情况（17.8 m 没超过 max(30, 4×半径)），
  // 因为「距离多远算离谱」本来就没有一个放之四海皆准的数。
  //
  // 换个判据：不看距离，直接问「场景包围球还在不在视锥里」。这才是用户真正在意的
  // ——「点了之后我还看得见模型吗」。看不见就退回全景，宁可少一个定位动作，
  // 也不要给一个黑屏。
  if (state.bounds && !sceneSphereVisible(center, radius)) {
    frameCameraOnBounds(state.bounds);
    ui.toast('该标注在模型范围之外，已改为回到全景（避免视角落到空处）');
  }
}

/**
 * 场景包围球是否与当前视锥相交（保守判断：相交即算可见）。
 *
 * 用「包围球中心与视线的夹角 − 球的角半径 < 半个视场角」而不是精确的六平面测试：
 * 只需判断"看不看得见"，保守一点（宁可认为可见）才不会误伤正常定位。
 */
function sceneSphereVisible(centerWorld, radiusWorld) {
  const toC = centerWorld.clone().sub(camera.position);
  const dist = toC.length();
  if (!Number.isFinite(dist) || dist <= 1e-6) return true;
  if (dist > camera.far) return false;
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  const cosAng = Math.max(-1, Math.min(1, forward.dot(toC.clone().normalize())));
  const angleToCenter = Math.acos(cosAng);
  // 球的角半径：asin(r/d)，比值 >1 时说明相机在球内 → 必然可见
  const ratio = Math.min(1, Math.max(0, radiusWorld / dist));
  const halfFov = (camera.fov * Math.PI) / 180 / 2;
  // 竖直半视场角已足够（水平方向只会更宽）
  return angleToCenter - Math.asin(ratio) < halfFov;
}

// ------------------------------------------------------------------ 拾取与交互

const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.03;
raycaster.params.Line.threshold = 0.04;

let dragTarget = null;
const _ndc = new THREE.Vector2();

function eventNdc(ev) {
  const rect = canvas.getBoundingClientRect();
  return _ndc.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -(((ev.clientY - rect.top) / rect.height) * 2 - 1));
}

/**
 * 点击坐标是否落在三维视口的可见区域内。
 * 用它替代 `ev.target === canvas`：后者在有任何覆盖元素时会静默失效。
 */
function pointInsideCanvas(ev) {
  const rect = canvas.getBoundingClientRect();
  return (
    ev.clientX >= rect.left &&
    ev.clientX <= rect.right &&
    ev.clientY >= rect.top &&
    ev.clientY <= rect.bottom
  );
}

function splatPick(ev) {
  if (!state.picker) return null;
  eventNdc(ev);
  return state.picker.pick({ x: _ndc.x, y: _ndc.y }, camera, WORLD_MATRIX, { screenRadius: 0.03 });
}

/** COLMAP 世界系下的拾取点（标注数据统一用该坐标系） */
function pickWorldPoint(ev) {
  const hit = splatPick(ev);
  if (!hit) return null;
  const local = hit.position.clone().applyMatrix4(_invWorld.copy(WORLD_MATRIX).invert());
  return { point: local, normal: hit.normal, world: hit.position };
}

let pointerDownInfo = null;

canvas.addEventListener('pointerdown', (ev) => {
  if (ev.button !== 0) return;
  pointerDownInfo = { x: ev.clientX, y: ev.clientY, at: performance.now(), moved: false };
  if (state.mode === 'fly') return;

  // 顶点拖拽优先级最高
  eventNdc(ev);
  raycaster.setFromCamera(_ndc, camera);
  const hit = annotationLayer.hitTest(raycaster);
  if (hit && hit.handleIndex >= 0) {
    const item = annotationLayer.get(hit.id);
    dragTarget = { id: hit.id, handleIndex: hit.handleIndex, item };
    controls.enabled = false;
    selectAnnotation(hit.id);
  }
});

canvas.addEventListener('pointermove', (ev) => {
  if (pointerDownInfo) {
    if (Math.hypot(ev.clientX - pointerDownInfo.x, ev.clientY - pointerDownInfo.y) > 4) pointerDownInfo.moved = true;
  }
  if (dragTarget) {
    const pick = pickWorldPoint(ev);
    if (pick) {
      const handle = dragTarget.item.handles[dragTarget.handleIndex];
      const ann = dragTarget.item.data;
      if (handle) {
        if (ann.kind === 'camera' && handle.type === 'pose') {
          dragTarget.item.camGroup.position.copy(pick.world);
          ann.pose = { ...(ann.pose ?? {}), position: [pick.point.x, pick.point.y, pick.point.z] };
        } else if (handle.rotateOnly) {
          // 照片/画面仅跟随平移
          handle.position.copy(pick.world);
          ann.points[0] = [pick.point.x, pick.point.y, pick.point.z];
        } else {
          handle.position.copy(pick.world);
          ann.points[handle.index] = [pick.point.x, pick.point.y, pick.point.z];
        }
      }
      ui.setToolHint(`拖动中：${pick.point.toArray().map((v) => v.toFixed(3)).join(', ')}`);
    }
    return;
  }

  if (state.tool !== 'select' && state.tool !== 'camera' && state.picker) {
    const pick = pickWorldPoint(ev);
    if (pick) {
      drawing.previewAt(pick.point, pick.normal);
      ui.setToolHint(`${state.tool === 'measure' ? '测距' : '绘制'}：${pick.point.toArray().map((v) => v.toFixed(3)).join(', ')}`);
      ui.moveHover(ev, `${pick.point.x.toFixed(3)}, ${pick.point.y.toFixed(3)}, ${pick.point.z.toFixed(3)}`);
    } else {
      ui.hideHover();
    }
  } else {
    ui.hideHover();
  }
});

window.addEventListener('pointerup', async (ev) => {
  const info = pointerDownInfo;
  pointerDownInfo = null;
  if (dragTarget) {
    const { id, item } = dragTarget;
    dragTarget = null;
    controls.enabled = true;
    const ann = item.data;
    if (ann.kind === 'camera') {
      // 相机只移动机位，朝向保持不变
      const pos = item.camGroup.position.clone().applyMatrix4(_invWorld.copy(WORLD_MATRIX).invert());
      await updateAnnotation(id, { pose: { ...ann.pose, position: [pos.x, pos.y, pos.z], source: 'manual-drag' } });
    } else {
      await updateAnnotation(id, { points: ann.points.map((p) => [p[0], p[1], p[2]]) });
    }
    return;
  }
  // 注意：这里刻意**不**要求 ev.target 必须是 canvas。
  // 标定面板的遮罩层虽然已是 pointer-events:none，但将来若再引入覆盖元素，
  // 用 target 判断会让「三维点击」静默失效（正是之前标定做不下去的原因之一）。
  // 改为判断点击坐标是否落在 canvas 的可见区域内。
  if (!info || info.moved || ev.button !== 0) return;
  if (!pointInsideCanvas(ev)) return;
  if (state.mode === 'fly') return;

  eventNdc(ev);

  // PnP 标定模式优先：把这次点击当作「三维侧的对应点」
  if (pnpPanel.active) {
    if (pnpPanel.handleSceneClick(_ndc)) return;
    // 还没有在画面上点过像素点 → 提示而不是走正常标注流程
    ui.toast('请先在左侧「PnP 标定」面板的画面里点一个特征点', { error: true });
    return;
  }

  // 单击：选择已有标注或新增
  raycaster.setFromCamera(_ndc, camera);
  const hit = annotationLayer.hitTest(raycaster);
  if (hit) {
    if (state.tool !== 'camera') {
      selectAnnotation(hit.id);
      return;
    }
  }

  const world = pickWorldPoint(ev);
  if (!world) {
    if (state.tool === 'select') selectAnnotation(null);
    return;
  }

  if (state.tool === 'camera') {
    drawing.cameraClick(world.point, world.normal);
    return;
  }
  if (state.tool === 'select') {
    selectAnnotation(null);
    ui.setToolHint('未选中对象。可切换到打点/轨迹/测距工具开始标注。');
    return;
  }
  drawing.click(state.tool, world.point, world.normal);
});

canvas.addEventListener('dblclick', () => {
  drawing.finish();
});

canvas.addEventListener('contextmenu', (ev) => {
  if (drawing.isActive()) {
    ev.preventDefault();
    drawing.cancel();
  }
});

window.addEventListener('keydown', (ev) => {
  if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement) return;
  if (ev.key === 'Escape') {
    drawing.cancel();
    selectAnnotation(null);
  } else if (ev.key === 'Enter') {
    drawing.finish();
  } else if (ev.key === 'Delete' && state.selectedId) {
    deleteAnnotation(state.selectedId);
  } else if (ev.code === 'KeyW' || ev.code === 'KeyA' || ev.code === 'KeyS' || ev.code === 'KeyD') {
    flyKeys.add(ev.code);
  } else if (ev.code === 'Space') {
    flyKeys.add('Space');
  } else if (ev.key === '1') setMode('orbit');
  else if (ev.key === '2') setMode('fly');
  else if (ev.key === '3') setMode('top');
  else if (ev.key === 'p' || ev.key === 'P') setTool('point');
  else if (ev.key === 'l' || ev.key === 'L') setTool('polyline');
  else if (ev.key === 'm' || ev.key === 'M') setTool('measure');
  else if (ev.key === 'd' || ev.key === 'D') setTool('dimension');
  else if (ev.key === 'g' || ev.key === 'G') setTool('polygon');
  else if (ev.key === 'c' || ev.key === 'C') setTool('camera');
  else if (ev.key === 'f' || ev.key === 'F') focusSelected();
});

window.addEventListener('keyup', (ev) => flyKeys.delete(ev.code));

function focusSelected() {
  if (state.selectedId) focusAnnotation(state.selectedId);
}

// ------------------------------------------------------------------ 第一人称飞行

const flyKeys = new Set();
const flyEuler = new THREE.Euler(0, 0, 0, 'YXZ');
let flySpeed = 1.2;

canvas.addEventListener('click', () => {
  if (state.mode === 'fly' && document.pointerLockElement !== canvas) canvas.requestPointerLock();
});

document.addEventListener('mousemove', (ev) => {
  if (state.mode !== 'fly' || document.pointerLockElement !== canvas) return;
  flyEuler.setFromQuaternion(camera.quaternion);
  flyEuler.y -= ev.movementX * 0.0022;
  flyEuler.x -= ev.movementY * 0.0022;
  flyEuler.x = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, flyEuler.x));
  camera.quaternion.setFromEuler(flyEuler);
});

function flyUpdate(dt) {
  const fast = flyKeys.has('ShiftLeft') || flyKeys.has('ShiftRight');
  const speed = finiteOr(state.bounds?.radius, 2) * 0.35 * (fast ? 2.5 : 1);
  const dir = new THREE.Vector3();
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
  if (flyKeys.has('KeyW')) dir.add(fwd);
  if (flyKeys.has('KeyS')) dir.sub(fwd);
  if (flyKeys.has('KeyD')) dir.add(right);
  if (flyKeys.has('KeyA')) dir.sub(right);
  if (flyKeys.has('Space')) dir.y += 1;
  if (flyKeys.has('ControlLeft')) dir.y -= 1;
  if (dir.lengthSq() > 0) camera.position.addScaledVector(dir.normalize(), speed * dt);
  // 注意：这里**不再**改写 controls.target。
  //
  // 原来这里每帧执行 controls.target = camera.position + fwd*k，
  // 后果是飞行把 OrbitControls 的旋转中心一路拖到相机前方约 4.5 米处。
  // 一旦切回环绕/俯视，相机就绕着这个贴近自己的点转 —— 表现为「视角乱飞」
  //（实测：进入飞行前 dist(target)=37.7，切回后只剩 4.5，
  //  再切俯视时相机被抬到 22 米外，Δpos 达 22.4）。
  //
  // 飞行模式自己用四元数控制朝向，本来就不需要 controls.target；
  // 环绕/俯视所需的 target 由 setMode 在切换时恢复。
}

/**
 * 切换视角模式：环绕漫游 / 第一人称 / 俯视态势。
 *
 * 三种模式共用同一个相机对象，但「谁在控制相机」不同：
 *   · orbit / top → OrbitControls 控制（用 camera.position 与 controls.target 决定球坐标）
 *   · fly         → 自己用四元数控制朝向 + WASD 平移
 *
 * 因此切换时必须把状态交接干净，否则会出现「视角乱飞」：
 *   1) 进 fly 前记下 orbit 的 target，离开时恢复（否则会绕着飞行时残留的近点旋转）；
 *   2) 进 top 时按 target 重新摆位，并让 camera.up 回到 (0,1,0)；
 *   3) 任何非法数值都夹掉，绝不把 NaN 写进相机。
 */
function setMode(mode) {
  const prev = state.mode;
  if (prev === mode) {
    ui.setMode(mode);
    return;
  }
  state.mode = mode;
  ui.setMode(mode);

  // ---------- 离开 fly：把 OrbitControls 的旋转中心还回去 ----------
  if (prev === 'fly' && mode !== 'fly') {
    const saved = state.savedOrbitTarget;
    const radius = finiteOr(state.bounds?.radius, 2);
    if (saved && saved.every((v) => Number.isFinite(v))) {
      controls.target.set(saved[0], saved[1], saved[2]);
    } else {
      // 没有存档（例如刷新后直接切）→ 用相机正前方一段距离当作合理中心
      const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
      controls.target.copy(camera.position).addScaledVector(fwd, Math.max(radius * 0.8, 0.8));
    }
    // 飞行中相机可能被转成「头上脚下」，环绕模式必须回到竖直向上
    camera.up.set(0, 1, 0);
    // 距离过近会让 OrbitControls 抖得很厉害，兜一个最小半径
    if (camera.position.distanceTo(controls.target) < Math.max(radius * 0.15, 0.2)) {
      const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
      controls.target.copy(camera.position).addScaledVector(dir, Math.max(radius * 0.8, 0.8));
    }
  }

  // ---------- 进入 fly：记下 orbit 的 target，准备接管 ----------
  if (mode === 'fly' && prev !== 'fly') {
    state.savedOrbitTarget = controls.target.toArray();
    controls.minPolarAngle = 0;
    controls.maxPolarAngle = Math.PI;
  }

  controls.enabled = mode !== 'fly';

  // ---------- 进入 top：从正上方俯视 target ----------
  if (mode === 'top') {
    const c = controls.target.clone();
    if (![c.x, c.y, c.z].every((v) => Number.isFinite(v))) c.set(0, 0, 0);
    const r = Math.max(finiteOr(state.bounds?.radius, 2) * 1.6, 1);
    camera.up.set(0, 1, 0);
    camera.position.set(c.x + 0.001, c.y + r, c.z + 0.001);
    controls.target.copy(c);
    controls.update();
  } else if (mode === 'orbit' && prev === 'top') {
    // 从俯视回到环绕：稍微倾斜一点，避免正好在极点上（极点处方位角退化，会抖）
    const dir = camera.position.clone().sub(controls.target);
    if (dir.lengthSq() < 1e-8) dir.set(0.7, 0.55, 0.7);
    if (Math.abs(dir.clone().normalize().y) > 0.999) {
      dir.y = dir.length() * 0.55;
      dir.x += dir.length() * 0.4;
      camera.position.copy(controls.target).add(dir);
    }
    camera.up.set(0, 1, 0);
    controls.update();
  }

  if (document.pointerLockElement === canvas && mode !== 'fly') document.exitPointerLock();

  ui.toast(
    mode === 'fly'
      ? '第一人称模式：点击画面锁定鼠标，WASD 移动，Esc 退出'
      : `已切换：${mode === 'orbit' ? '环绕漫游' : '俯视态势'}`
  );
}

function setTool(tool) {
  state.tool = tool;
  drawing.setTool(tool);
  ui.setTool(tool);
  const hints = {
    select: '左键选择，右键/ESC 取消当前绘制',
    point: '左键在三维现场打点标注物证',
    polyline: '沿嫌疑人路线依次点击，双击或回车结束',
    measure: '点击两个点测量真实距离（米）',
    dimension: '尺寸标注：点两个端点标出「需要精确测量」的位置，民警实测后可在属性面板回填，系统据此校正比例尺',
    polygon: '依次点击圈定区域，双击或回车闭合',
    camera: '先点相机位置，再点画面朝向目标（可导入视频后用 PnP 精确标定）',
  };
  ui.setToolHint(hints[tool] ?? '');
}

// ------------------------------------------------------------------ 协同

let pendingAnnotations = new Map();
let flushTimer = null;

function connectCollab() {
  if (conn.client) conn.client.disconnect();
  if (!state.project) return;
  if (!conn.name) {
    conn.name = `民警${Math.floor(Math.random() * 900 + 100)}`;
    localStorage.setItem('dsh:name', conn.name);
  }
  const client = new CollabClient({
    projectId: state.project.id,
    name: conn.name,
    color: conn.color,
    role: conn.role,
    onState: (s, detail) => ui.setConnState(s, detail),
    onRoster: (roster) => {
      ui.renderUsers(roster, client.you?.id);
      const ids = new Set(roster.map((r) => r.id));
      for (const [id, p] of state.peers) {
        if (!ids.has(id)) {
          peerGroup.remove(p.object);
          state.peers.delete(id);
        }
      }
      for (const r of roster) if (r.id !== client.you?.id) ensurePeer(r);
    },
    onSystem: (msg) => ui.pushSystem(msg.message),
    onEvent: handleCollabEvent,
  });
  client.connect();
  conn.client = client;
  ui.setIdentity(conn.name, conn.color);
}

function handleCollabEvent(msg) {
  switch (msg.type) {
    case 'welcome': {
      if (msg.project && state.project && msg.project.revision > (state.project.revision ?? 0)) {
        state.project = msg.project;
        annotationLayer.setAll(msg.project.annotations ?? []);
        ui.renderAnnotations(msg.project.annotations ?? []);
      }
      // 同步一次当前视角，便于他人定位
      publishPose();
      break;
    }
    case 'pose': {
      const peer = ensurePeer({ id: msg.id });
      // local 系（COLMAP）下的相机位置与朝向 -> scene 系。
      // 使用 camera-overlay.js 的纯函数：与 makePeerFrustum 的「局部 -Z 为观察方向」
      // 语义保持一致，保证远端视锥朝向与对方真实观察方向一致。
      const scene = peerPoseToScene(msg.p, msg.q, WORLD_MATRIX);
      peer.targetPos = scene.position;
      peer.targetQuat = scene.quaternion;
      peer.lastSeen = performance.now();
      peer.fov = msg.fov;
      break;
    }
    case 'roster':
      ui.renderUsers(msg.roster ?? [], conn.client?.you?.id);
      break;
    case 'chat':
      ui.pushChat(msg);
      break;
    case 'system':
      ui.pushSystem(msg.message);
      break;
    case 'annotation.add':
      mergeAnnotation(msg.annotation);
      ui.toast(`${msg.by?.name ?? '同事'} 新增了标注：${msg.annotation.label || kindName(msg.annotation.kind, msg.annotation)}`);
      break;
    case 'annotation.update':
      mergeAnnotation(msg.annotation);
      break;
    case 'annotation.delete':
      state.project.annotations = state.project.annotations.filter((a) => a.id !== msg.id);
      annotationLayer.remove(msg.id);
      ui.renderAnnotations(state.project.annotations);
      break;
    case 'annotation.batch':
      for (const op of msg.ops ?? []) {
        if (op.op === 'delete') {
          state.project.annotations = state.project.annotations.filter((a) => a.id !== op.id);
          annotationLayer.remove(op.id);
        } else if (op.annotation) {
          mergeAnnotation(op.annotation);
        }
      }
      ui.renderAnnotations(state.project.annotations);
      break;
    case 'model.set':
      if (msg.model && msg.model.url !== state.model?.url) {
        ui.toast('同案人员切换了三维现场模型');
        updateModelList();
      }
      break;
    // 现场建模任务的进度（同案人员都能看到）
    case 'recon.progress':
      if (msg.job?.id === reconPanel.jobId) reconPanel.render(msg.job);
      break;
    case 'recon.start':
      if (msg.job?.id !== reconPanel.jobId) {
        ui.toast(`${msg.job?.by?.name ?? '同案人员'} 启动了现场建模任务：${msg.job?.name ?? ''}`);
        reconPanel.jobId = msg.job?.id ?? reconPanel.jobId;
        reconPanel.render(msg.job);
        reconPanel.poll();
      }
      break;
    case 'recon.done':
      reconPanel.render(msg.job);
      ui.toast(`现场建模完成：${msg.job?.result?.model ?? msg.job?.name ?? ''}`);
      updateModelList();
      break;
    case 'recon.failed':
      reconPanel.render(msg.job);
      ui.toast(`现场建模失败：${msg.job?.error ?? '详见日志'}`, { error: true });
      break;
    case 'error':
      ui.toast(msg.message, { error: true });
      break;
    default:
      break;
  }
}

function mergeAnnotation(ann) {
  if (!ann) return;
  const idx = state.project.annotations.findIndex((a) => a.id === ann.id);
  if (idx >= 0) state.project.annotations[idx] = ann;
  else state.project.annotations.push(ann);
  annotationLayer.upsert(ann);
  ui.renderAnnotations(state.project.annotations);
  if (state.selectedId === ann.id) selectAnnotation(ann.id, { silent: true });
}

// ------------------------------------------------------------------ 多源画面

async function openCameraVideo(ann) {
  const media = ann.media?.[0];
  if (!media?.url) {
    ui.toast('该相机尚未关联视频，请先导入监控/勘查视频', { error: true });
    return;
  }
  await cameraOverlay.attach(ann, media.url);
  cameraOverlay.select(ann.id);
  ui.renderCameras(state.project.annotations.filter((a) => a.kind === 'camera'), cameraOverlay);
}

/**
 * 打开 PnP 标定面板。
 *
 * 流程：面板里点像素 → 主视口里点三维对应点 → 采够 4 组 → 解算 → 写回相机位姿。
 * 解算完全在浏览器完成（app/web/js/pnp.js）。
 */
async function openPnpCalibration(ann) {
  if (!ann) return;
  if (!state.renderer?.chunks || !state.picker) {
    ui.toast('请先载入三维现场，再进行 PnP 标定', { error: true });
    return;
  }
  const media = ann.media?.[0];
  if (!media?.url) {
    ui.toast('该相机还没关联画面，请先「导入视频」', { error: true });
    return;
  }
  // 标定期间用选择模式，避免误触发绘制
  setMode('orbit');
  setTool('select');
  try {
    await pnpPanel.open({
      annotation: ann,
      mediaUrl: media.url,
      ctx: {
        scene,
        camera,
        canvas,
        picker: state.picker,
        worldMatrix: WORLD_MATRIX,
        overlay: cameraOverlay,
      },
    });
    ui.setToolHint('PnP 标定中：在面板画面上点像素 → 在三维视口里点对应点（需 4 组以上）');
    ui.toast('① 在面板画面上点一个特征点 ② 在三维视口里点它的对应位置');
  } catch (err) {
    ui.toast(`打开标定面板失败：${err.message}`, { error: true });
    console.error(err);
  }
}

async function importMedia(ann, file) {
  ui.showLoader(`正在上传 ${file.name}…`);
  try {
    const res = await api.uploadMedia(state.project.id, file, (p) =>
      ui.setLoaderProgress(p, `${((p * file.size) / 1048576).toFixed(1)}MB`)
    );
    const patch = { media: [{ url: res.url, kind: res.kind, name: res.name }] };
    let target = ann;
    if (ann) {
      await updateAnnotation(ann.id, patch);
    } else {
      const meta = await probeMedia(file);
      // 新相机默认放在「场景包围盒边缘、朝向中心」的位置，
      // 这样视锥与画面一定在场景可见范围内（放在原点常常被墙挡在里面，看起来像黑屏/没反应）
      const pose = defaultCameraPose(meta);
      target = await commitAnnotation({
        kind: 'camera',
        points: [pose.position],
        pose,
        media: patch.media,
        label: file.name,
      });
    }
    ui.hideLoader();
    // 创建/更新后立刻挂载视频并让相机可见
    const cameras = state.project.annotations.filter((a) => a.kind === 'camera');
    for (const c of cameras) {
      if (c.media?.[0]?.url && !cameraOverlay.entries.has(c.id)) {
        await cameraOverlay.attach(c, c.media[0].url);
      }
    }
    cameraOverlay.setVisible(true);
    ui.renderCameras(cameras, cameraOverlay);
    if (target?.id) {
      cameraOverlay.select(target.id);
      selectAnnotation(target.id, { silent: true });
    }
    ui.toast('上传完成。点画面卡片上的「贴回现场」即可站到该机位核对是否重合。');
  } catch (err) {
    ui.hideLoader();
    ui.toast(`上传失败：${err.message}`, { error: true });
    console.error(err);
  }
}

/**
 * 为一个新导入的监控/勘查画面算一个默认机位：
 * 放在场景包围盒外侧、略高于中心的位置，朝向场景中心。
 */
function defaultCameraPose(mediaMeta) {
  const b = state.bounds;
  const aspect = mediaMeta?.aspect ?? 16 / 9;
  const center = b ? new THREE.Vector3(...b.center) : new THREE.Vector3(0, 0, 0);
  const radius = b?.radius ?? 3;
  // 在「相机标注所在的 local 系」里计算
  const localCenter = center.clone().applyMatrix4(WORLD_MATRIX_INV);
  const offset = new THREE.Vector3(radius * 0.75, -radius * 0.35, radius * 0.75)
    .applyQuaternion(WORLD_QUAT.clone().invert())
    .normalize()
    .multiplyScalar(radius * 1.4);
  const position = localCenter.clone().add(offset);
  return {
    source: 'manual',
    frame: 'gaussian',
    position: [position.x, position.y, position.z],
    target: [localCenter.x, localCenter.y, localCenter.z],
    up: [0, 0, 1],
    fovY: 45,
    aspect,
  };
}

function probeMedia(file) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    let settled = false;
    const finish = (info) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(info);
      // 注意：不要在 onloadedmetadata 里立刻 revoke，部分浏览器会因此拿不到后续帧
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    };
    // 兜底超时：视频元数据一直不回来时不要让上传流程卡死
    const timer = setTimeout(() => finish({ aspect: 16 / 9 }), 8000);
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.muted = true;
    video.onloadedmetadata = () => {
      const w = video.videoWidth || 1920;
      const h = video.videoHeight || 1080;
      finish({ aspect: w / h, width: w, height: h });
    };
    video.onerror = () => {
      const img = new Image();
      img.onload = () => finish({ aspect: img.width / img.height, width: img.width, height: img.height });
      img.onerror = () => finish({ aspect: 16 / 9 });
      img.src = url;
    };
    video.src = url;
  });
}

// ------------------------------------------------------------------ 表现参数

function applyPresentation(p) {
  if (!p) return;
  state.filters.exposure = p.exposure ?? 1;
  scene.background = new THREE.Color(p.background ?? '#080c11');
  gridHelper.visible = p.grid !== false;
  ui.setPresentation(state.filters);
}
// ------------------------------------------------------------------ UI 事件绑定

// 「回到全景」：一键把镜头拉回现场。位姿异常/误操作把视角送到场景外导致
// 画面全黑时，这是唯一的自救入口（以前只能刷新页面，标定进度全丢）。
document.getElementById('btnFitView')?.addEventListener('click', () => {
  if (!state.bounds) {
    ui.toast('还没有载入模型', { error: true });
    return;
  }
  frameCameraOnBounds(state.bounds);
  ui.toast('已把镜头拉回现场全景');
});

ui.bind({
  onSelectProject: async (id) => {
    await openProject(id);
  },
  onNewProject: async ({ name, description, operator }) => {
    const res = await api.createProject({ name, description, operator: operator || conn.name });
    await refreshProjects(res.project.id);
    await openProject(res.project.id);
    ui.toast(`案件「${name}」已创建`);
  },
  onRefresh: () => refreshProjects(),
  onLoadModel: async (url) => {
    // 优先用**清单里的完整对象**（带 structureUrl / furnitureUrl / sceneMode）；
    // 找不到就**再取一次清单**（用户可能在清单还没加载完时就点了下拉框 —— 实测静态版
    // 会慢一拍，这时候退回裸对象就丢 structureUrl，结构线永远不出现）。
    let full = state.models?.find((m) => m.url === url);
    if (!full) {
      try {
        await updateModelList();
      } catch (err) {
        console.warn('[models] 重新取模型清单失败：', err?.message ?? err);
      }
      full = state.models?.find((m) => m.url === url);
    }
    const model = full ?? {
      name: url.split('/').pop(),
      url,
      format: url.endsWith('.ply') ? 'ply' : 'splat',
    };
    await loadModel(model, { announce: true });
  },
  onMode: setMode,
  onTool: setTool,
  onFilterChange: (patch) => {
    Object.assign(state.filters, patch);
    if (state.renderer) {
      state.renderer.chunkOrder = state.filters.sort ? state.renderer.chunkOrder : state.renderer.chunks.map((_, i) => i);
    }
    cameraOverlay.setVisible(state.filters.showCameras);
    ui.setPresentation(state.filters);
  },
  onSelectAnnotation: (id) => {
    selectAnnotation(id);
    if (id) focusAnnotation(id);
  },
  // 图层显隐：把「图层 key」映射到标注类型，直接控制三维里的可见性
  onLayerToggle: (key, visible) => {
    const kindByLayer = {
      point: 'point',
      polyline: 'polyline',
      polygon: 'polygon',
      measure: 'measure',
      camera: 'camera',
    };
    if (key === 'model') {
      const mesh = state.renderer;
      if (mesh) mesh.groupVisible = visible;
      return;
    }
    const kind = kindByLayer[key];
    if (kind) annotationLayer.setKindVisible(kind, visible);
    if (key === 'camera') {
      state.filters.showCameras = visible;
      cameraOverlay.setVisible(visible);
    }
    ui.setToolHint(visible ? `已显示「${key}」图层` : `已隐藏「${key}」图层`);
  },
  // 列表中双击 = 定位（不再是删除，避免误删）
  onFocusAnnotation: (id) => {
    selectAnnotation(id);
    focusAnnotation(id);
  },
  onDeleteAnnotation: (id) => deleteAnnotation(id),
  /**
   * 删除「多源画面」里的一个监控/勘查画面。
   *
   * 用户反馈：导入的视频和图片无法删除（点 ✕ 没反应）。
   * 根因是**界面上根本没有删除入口** —— 画面卡片只有「贴回现场 / PnP 标定 / 换视频」。
   * 这里补上入口，并在确认后：
   *   1) 从三维里摘掉相机与视锥（cameraOverlay.detach）
   *   2) 删除该相机标注
   *   3) 询问是否连服务器上的媒体文件一起删掉（默认删，避免占空间）
   */
  onCameraDelete: async (id) => {
    const ann = state.project?.annotations.find((a) => a.id === id);
    if (!ann) return;
    const media = ann.media?.[0];
    const name = ann.label || media?.name || '该画面';
    // 中文文件名在 URL 里是 %XX 编码的，提示里要还原成可读文本
    const rawFile = media?.url?.split('/').pop() ?? media?.name ?? '';
    let fileName = rawFile;
    if (/%[0-9A-Fa-f]{2}/.test(rawFile)) {
      try {
        fileName = decodeURIComponent(rawFile);
      } catch {
        /* 保持原样 */
      }
    }
    const ok = await ui.confirm({
      title: '删除画面',
      body: `确定从本案删除「${name}」吗？`,
      warn: media
        ? `对应的媒体文件也会一并从服务器删除：${fileName}`
        : '该画面没有关联的媒体文件，只会移除相机对象。',
    });
    if (!ok) return;
    ui.showLoader('正在删除画面…');
    try {
      // 先摘掉三维里的相机/视锥，避免残留
      cameraOverlay.detach?.(id);
      if (media?.url) {
        // 媒体文件删除失败不影响标注删除，只提示
        try {
          await api.deleteMedia(state.project.id, media.url);
        } catch (err) {
          ui.toast(`媒体文件删除失败（标注仍会删除）：${err.message}`, { error: true });
        }
      }
      await deleteAnnotation(id);
      ui.renderCameras(state.project.annotations.filter((a) => a.kind === 'camera'), cameraOverlay);
      ui.toast(`已删除画面「${name}」`);
    } catch (err) {
      ui.toast(`删除失败：${err.message}`, { error: true });
    } finally {
      ui.hideLoader();
    }
  },
  onSendChat: (text) => {
    conn.client?.send({ type: 'chat', text });
  },
  onRename: (name) => {
    conn.name = name || conn.name;
    localStorage.setItem('dsh:name', conn.name);
    ui.setIdentity(conn.name, conn.color);
    conn.client?.send({ type: 'pose', p: [camera.position.x, camera.position.y, camera.position.z], q: [0, 0, 0, 1], name: conn.name });
  },
  onImportMedia: async (file, ann) => {
    await importMedia(ann, file);
  },
  onCameraSelect: (id) => {
    cameraOverlay.select(id);
    const ann = state.project.annotations.find((a) => a.id === id);
    if (ann) selectAnnotation(id, { silent: true });
  },
  onCameraGoto: (id) => {
    const ann = state.project.annotations.find((a) => a.id === id);
    if (!ann) return;
    const view = cameraOverlay.flyTo(ann);
    setMode('orbit');
    camera.position.copy(view.position);
    camera.quaternion.copy(view.quaternion);
    camera.fov = view.fovY;
    camera.updateProjectionMatrix();
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(view.quaternion);
    controls.target.copy(view.position.clone().addScaledVector(forward, Math.max(finiteOr(state.bounds?.radius, 2) * 0.35, 0.4)));
    controls.update();
    ui.toast(`已切换到「${ann.label || '监控画面'}」的拍摄视角，视频画面应与现场重合`);
  },
  onCameraPlayToggle: (id) => cameraOverlay.togglePlay(id),
  onCameraCalibrate: (ann) => openPnpCalibration(ann),
});

// ------------------------------------------------------------------ 启动

(async function boot() {
  try {
    const health = await api.health();
    ui.setServerInfo(health);
    // 页面里的 JS 与服务端版本不一致 → 顶部挂横幅要求刷新（详见 version.js）
    ui.watchVersion(health.version, APP_VERSION);
    await refreshProjects();
    const lastId = localStorage.getItem('dsh:lastProject');
    const target = state.projects.find((p) => p.id === lastId) ?? state.projects[0];
    if (target) {
      await openProject(target.id);
      const lastModel = localStorage.getItem(`dsh:lastModel:${target.id}`);
      const models = await api.models();
      const pick = models.find((m) => m.url === lastModel) ?? models[0];
      if (pick) await loadModel(pick);
    } else {
      ui.toast('还没有案件，请先点击「新建案件」', { sticky: true });
    }
  } catch (err) {
    ui.toast(`后端连接失败：${err.message}`, { error: true, sticky: true });
    console.error(err);
  }
  ui.startClock();
  startRenderLoop();
})();

// 截图导出（供 UI 调用）
// 另外暴露少量内部入口给自动化测试使用（浏览器端回归测试会调用它们）
window.__caseSystem = {
  state,
  // 供浏览器回归测试直接验证"只读演示版会拒绝写操作"（静态托管时）
  api,
  captureThumbnail,
  screenshot() {
    return canvas.toDataURL('image/png');
  },
  // --- 以下供 tests/browser_pnp_panel.mjs 等浏览器回归测试使用 ---
  pnpPanel,
  reconPanel,
  measurePanel,
  recalibrateScale,
  // 场景判定改判（tests/browser_structure_layer.mjs 会驱动它验证门控）
  setSceneMode,
  applyStructure,
  syncSceneModeControl,
  // 暴露标签投影器（变量名是 labelLayer）：测试用它验证
  // 「校正比例尺后测距文字按真实尺寸变化」
  labelProjector: labelLayer,
  openPnpCalibration,
  loadModel,
  updateModelList,
  WORLD_MATRIX,
  WORLD_QUAT,
  selectAnnotation,
  updateAnnotation,
  ui,
  APP_VERSION,
  structureLayer,
};
// ---- 渲染参数滑杆（高斯尺寸 / 不透明度 / 曝光）------------------------------
// ⚠️ 这三个滑杆在 index.html 里一直存在，但**从来没有事件绑定** ——
//    拖动它们画面纹丝不动（实测确认），只往 state.filters 写了值没人读。
//    现在直接接到渲染器的 options 上：渲染循环每帧读取，改完立刻生效。
function bindRenderSliders() {
  const bind = (id, labelId, key, optKey, fmt = (v) => v.toFixed(2)) => {
    const el = document.getElementById(id);
    const lab = document.getElementById(labelId);
    if (!el) return;
    const apply = () => {
      const v = Number(el.value);
      state.filters[key] = v;
      if (lab) lab.textContent = fmt(v);
      if (state.renderer) state.renderer.options[optKey] = v;
    };
    el.addEventListener('input', apply);
    el.value = String(state.filters[key] ?? 1);
    apply();
  };
  bind('rGaussScale', 'vScale', 'gaussScale', 'gaussScale');
  bind('rOpacity', 'vOpacity', 'opacity', 'opacityGain');
  bind('rExposure', 'vExposure', 'exposure', 'exposure');
}


bindRenderSliders();
