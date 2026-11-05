/**
 * 浏览器端 PnP 标定面板回归测试
 *
 * 验证：
 *   1. 面板能打开、画面能抽帧、UI 元素齐全
 *   2. 在画布上点击能记录像素点
 *   3. 在三维视口点击能记录对应点（走真实的高斯拾取）
 *   4. 采够 4 组后能解算出位姿并把重投影误差显示出来
 *   5. 解算结果能在画面上画出重投影标记
 *
 * 用法: node tests/browser_pnp_panel.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9229;
const PROFILE = path.join(process.cwd(), '.browsertmp', 'pnp-panel');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

let pass = 0, fail = 0;
const check = (name, ok, detail = '') => {
  if (ok) { pass++; console.log(`  [OK]   ${name}${detail ? '  ' + detail : ''}`); }
  else { fail++; console.log(`  [FAIL] ${name}${detail ? '  ' + detail : ''}`); }
};

// ---------------------------------------------------------------- 准备数据
// 造一个案件 + 带视频的相机标注，然后用该相机位姿反算出一组像素点，
// 用来验证「像素点 → 三维点 → 解算回来」这条链路。
const fs = await import('node:fs');

const proj = await (
  await fetch(`${BASE}/api/projects`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'pnp-panel-browser', operator: 'test' }),
  })
).json();
const pid = proj.project.id;

// 生成/复用一个真实小视频作为画面
const SCENE_DIR = path.join(process.cwd(), 'data', 'scene-test');
const VIDEO = path.join(SCENE_DIR, 'pnp-panel-demo.mp4');
const mediaDir = path.join(process.cwd(), 'data', 'projects', pid, 'media');
mkdirSync(mediaDir, { recursive: true });
mkdirSync(SCENE_DIR, { recursive: true });

const findVideo = () => {
  if (fs.existsSync(VIDEO)) return VIDEO;
  const pools = [
    [SCENE_DIR, fs.existsSync(SCENE_DIR) ? fs.readdirSync(SCENE_DIR) : []],
    [path.join(process.cwd(), 'data', 'projects'), fs.existsSync(path.join(process.cwd(), 'data', 'projects')) ? fs.readdirSync(path.join(process.cwd(), 'data', 'projects')) : []],
  ];
  for (const [dir, names] of pools) {
    const hit = names.find((f) => /\.(mp4|webm|mov|m4v)$/i.test(f));
    if (hit) {
      const src = path.join(dir, hit);
      if (fs.statSync(src).isFile()) return src;
    }
  }
  // 再到各案件目录的 media 下找
  const projectsRoot = path.join(process.cwd(), 'data', 'projects');
  if (fs.existsSync(projectsRoot)) {
    for (const p of fs.readdirSync(projectsRoot)) {
      const md = path.join(projectsRoot, p, 'media');
      if (!fs.existsSync(md)) continue;
      const hit = fs.readdirSync(md).find((f) => /\.(mp4|webm|mov|m4v)$/i.test(f));
      if (hit) return path.join(md, hit);
    }
  }
  return null;
};

const srcVideo = findVideo();
if (!srcVideo) {
  console.error('缺少可用的测试视频，请先运行 tools/make_test_video 生成 data/scene-test/pnp-panel-demo.mp4');
  await fetch(`${BASE}/api/projects/${pid}`, { method: 'DELETE' });
  process.exit(2);
}
if (srcVideo !== VIDEO) fs.copyFileSync(srcVideo, VIDEO);
fs.copyFileSync(VIDEO, path.join(mediaDir, 'pnp-panel-demo.mp4'));
const mediaUrl = `/media/${pid}/media/pnp-panel-demo.mp4`;
console.log(`测试视频来源：${path.relative(process.cwd(), srcVideo)}`);

const ann = await (
  await fetch(`${BASE}/api/projects/${pid}/annotations`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      annotation: {
        kind: 'camera',
        points: [[2, 1, 2]],
        pose: {
          source: 'manual',
          frame: 'gaussian',
          position: [2, 1, 2],
          target: [0, 0, 0],
          up: [0, 0, 1],
          fovY: 60,
          aspect: 16 / 9,
        },
        media: [{ url: mediaUrl, kind: 'video', name: 'pnp-panel-demo.mp4' }],
        label: 'PnP 面板测试相机',
      },
      operator: 'test',
    }),
  })
).json();
const annId = ann.annotation.id;
console.log(`测试案件 ${pid}，相机 ${annId.slice(0, 8)}，媒体 ${mediaUrl}\n`);

// ---------------------------------------------------------------- 启动浏览器
const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1400,900', BASE,
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let page = null;
for (let i = 0; i < 40 && !page; i++) {
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
  } catch { /* 重试 */ }
  if (!page) await sleep(400);
}
if (!page) {
  console.error('无法连接到浏览器调试端口');
  edge.kill();
  process.exit(1);
}

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const consoleErrors = [];
const consoleAll = [];
const pageExceptions = [];
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Runtime.consoleAPICalled') {
    const text = m.params.args.map((a) => a.value ?? a.description ?? '').join(' ');
    consoleAll.push(`[${m.params.type}] ${text}`);
    if (m.params.type === 'error') consoleErrors.push(text);
  }
  if (m.method === 'Runtime.exceptionThrown') {
    pageExceptions.push(m.params.exceptionDetails?.exception?.description ?? JSON.stringify(m.params.exceptionDetails));
  }
};
const send = (method, params = {}) => new Promise((res) => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
await send('Runtime.enable');
// 等应用启动完成：轮询 window.__caseSystem
let booted = false;
for (let i = 0; i < 40 && !booted; i++) {
  const r = await send('Runtime.evaluate', {
    expression: 'String(!!(window.__caseSystem && window.__caseSystem.state && window.__caseSystem.pnpPanel))',
    returnByValue: true,
  });
  booted = r.result?.result?.value === 'true';
  if (!booted) await sleep(500);
}
console.log('应用已启动:', booted);
if (!booted) {
  console.error('应用未在 20 秒内启动');
  console.log('\n--- 页面控制台输出 ---');
  for (const l of consoleAll.slice(-40)) console.log('  ' + l);
  console.log('\n--- 页面未捕获异常 ---');
  for (const e of pageExceptions.slice(0, 10)) console.log('  ' + String(e).split('\n').slice(0, 5).join('\n  '));
  await fetch(`${BASE}/api/projects/${pid}`, { method: 'DELETE' });
  edge.kill();
  process.exit(1);
}
// 再给模型与 WebSocket 一点时间
await sleep(3000);

// ⚠️ 显式把页面切到**本测试自己建的案件**。
// 以前这里依赖"应用启动时会自动选中最近的案件"—— 那是个隐藏的顺序依赖：
// 只要有别的案件比本测试的案件更晚更新（例如某个截图脚本忘了清理临时案件），
// 下面就会报「找不到测试相机标注」。
// 2026-10-06 实测踩过一次（我自己的 shot_scene_mode.mjs 遗留了案件）。
// 走真实 UI（#projectSelect 的 change 事件），不做任何测试专用后门。
const selected = await (async () => {
  const r = await send('Runtime.evaluate', {
    expression: `(async () => {
      const sel = document.getElementById('projectSelect');
      if (!sel) return 'no-select';
      for (let i = 0; i < 40; i++) {
        const opt = [...sel.options].find(o => o.value === ${JSON.stringify(pid)});
        if (opt) {
          sel.value = ${JSON.stringify(pid)};
          sel.dispatchEvent(new Event('change', { bubbles: true }));
          return 'selected';
        }
        await new Promise(r => setTimeout(r, 250));
      }
      return 'not-found';
    })()`,
    returnByValue: true,
    awaitPromise: true,
  });
  return r.result?.result?.value;
})();
console.log('切换到本测试案件:', selected);
await sleep(1500);

// ---------------------------------------------------------------- 断言
console.log('浏览器端 PnP 标定面板回归测试\n');

const script = `
(async () => {
  const S = window.__caseSystem;
  const out = { steps: [] };
  if (!S?.state) { out.error = '应用未启动'; return JSON.stringify(out); }
  const st = S.state;

  // 载入三维现场（面板拾取需要 picker）
  const models = await (await fetch('/models/index.json')).json();
  const target = models.models.find(m => /sample-room/.test(m.name)) ?? models.models[0];
  out.model = target.name;
  if (typeof S.loadModel === 'function') await S.loadModel(target);
  if (!st.renderer?.chunks) { out.error = '三维现场未载入'; return JSON.stringify(out); }

  // 打开面板（走真实入口）
  const ann = st.project.annotations.find(a => a.kind === 'camera' && a.label === 'PnP 面板测试相机');
  if (!ann) { out.error = '找不到测试相机标注'; return JSON.stringify(out); }
  await S.openPnpCalibration(ann);

  const panel = document.querySelector('.pnp-panel');
  out.panelVisible = !!panel && !panel.hidden;
  // 两个不同的画布，务必区分清楚：
  //   canvas      = 三维视口（点击对应点的地方）
  //   frameCanvas = 面板里显示监控画面的画布（点击像素点的地方）
  const canvas = document.getElementById('viewport');
  const frameCanvas = panel.querySelector('[data-framecanvas]');
  out.frameCanvas = frameCanvas ? { w: frameCanvas.width, h: frameCanvas.height } : null;
  out.viewportCanvas = canvas ? { w: canvas.width, h: canvas.height } : null;
  out.canvasIsDistinct = canvas !== frameCanvas;
  out.hasSolveBtn = !!panel.querySelector('[data-solve]');
  out.solveDisabledAtStart = panel.querySelector('[data-solve]').disabled;
  out.solveTextAtStart = panel.querySelector('[data-solve]').textContent.trim();
  // 复现用户报的 bug：不够 4 组时点「解算并贴回现场」必须给出**可见**提示。
  // 旧实现把按钮设成 disabled，浏览器根本不派发 click → 什么都不发生。
  out.beforeSolveToast = document.getElementById('toast')?.textContent ?? '';
  panel.querySelector('[data-solve]').click();
  out.afterSolveToast = document.getElementById('toast')?.textContent ?? '';
  out.pairCountAtStart = panel.querySelector('[data-pairs]').children.length;
  // 面板结构完整性：所有 data-* 元素都必须被缓存（否则运行到一半会 undefined）
  out.missingDomKeys = [];
  for (const k of ['frameCanvas','marks','seek','time','prevFrame','nextFrame','count','error','assess','pairs','fov','fovRange','fx','fy','cx','cy','solve','undo','clear','close','flyTo','note3d','sub','frameInfo','resetK','steptip']) {
    if (!S.pnpPanel.dom[k]) out.missingDomKeys.push(k);
  }

  // 底部画面卡片上应当有「PnP 标定」按钮。
  // 注意：卡片条是 openProject → renderCameras 渲染的，若此时还没渲染，
  // 主动触发一次再检查。
  if (!document.querySelector('.cam-card')) {
    const overlay = S.state.cameraOverlay ?? null;
    if (S.ui?.renderCameras) {
      S.ui.renderCameras(st.project.annotations.filter(a => a.kind === 'camera'), overlay);
    }
  }
  const cards = document.querySelectorAll('.cam-card');
  out.camCardCount = cards.length;
  out.hasCalibButton = [...cards].some(c => c.querySelector('button[data-act="calib"]'));

  // 画面是否真的画出了像素（非全黑）
  if (frameCanvas) {
    const g = frameCanvas.getContext('2d');
    const d = g.getImageData(0, 0, frameCanvas.width, frameCanvas.height).data;
    let nonBlack = 0;
    for (let i = 0; i < d.length; i += 4 * 97) {
      if (d[i] > 12 || d[i+1] > 12 || d[i+2] > 12) nonBlack++;
    }
    out.frameNonBlack = nonBlack;
  }

  // ---- 点击画面记录像素点 ----
  const clickCanvas = (x, y) => {
    const rect = frameCanvas.getBoundingClientRect();
    const ev = new MouseEvent('click', {
      clientX: rect.left + (x / frameCanvas.width) * rect.width,
      clientY: rect.top + (y / frameCanvas.height) * rect.height,
      bubbles: true,
    });
    frameCanvas.dispatchEvent(ev);
  };
  clickCanvas(frameCanvas.width * 0.25, frameCanvas.height * 0.3);
  out.pendingMark = panel.querySelectorAll('.pnp-mark.pending').length;
  // T2A/T2B：等待三维点击期间，必须画出相机视锥（虚线）并高亮可选点
  await new Promise((r) => setTimeout(r, 120));
  out.frustumWhileArmed = S.structureLayer.stats();
  out.frustumSvgDashed = document.querySelectorAll('#structureLayer line[stroke-dasharray]').length;
  out.candidateSvgDots = document.querySelectorAll('#structureLayer circle[fill="#00e5ff"]').length;
  // 描边光晕（深色 #0b1620）：压在亮线/亮点下面，保证在任意背景上都能看清
  out.haloLineCount = document.querySelectorAll('#structureLayer line[stroke="#0b1620"]').length;
  out.haloDotCount = document.querySelectorAll('#structureLayer circle[fill="#0b1620"]').length;
  out.structureZIndex = getComputedStyle(document.getElementById('structureLayer')).zIndex;
  out.hasFitViewBtn = !!document.getElementById('btnFitView');

  // ---- 在三维视口点对应点（用真实 picker，取一个已知可见的高斯点） ----
  const THREE = await import('/vendor/three/build/three.module.js');
  const cam = st.camera;
  cam.updateMatrixWorld(true);
  const raw = st.renderer.raw;
  const wm = S.WORLD_MATRIX;

  // 收集一组可见且分散的高斯点（把它们的局部坐标当作「真值」，
  // 再用一个虚拟相机位姿算出对应像素，从而构造一组自洽的对应关系）
  const W = frameCanvas.width, H = frameCanvas.height;
  const focal = 0.5 * H / Math.tan(0.5 * 60 * Math.PI / 180);
  // 虚拟相机：放在场景外侧看向包围盒中心
  const b = st.bounds;
  const center = new THREE.Vector3(...(b ? b.center : [0,0,0]));
  const radius = b ? b.radius : 3;
  const eyeLocal = center.clone().add(new THREE.Vector3(radius*1.1, -radius*0.5, radius*1.1).applyQuaternion(S.WORLD_QUAT.clone().invert()).normalize().multiplyScalar(radius*1.2));
  const fwd = center.clone().sub(eyeLocal).normalize();
  const upLocal = new THREE.Vector3(0, 0, 1);
  const right = new THREE.Vector3().crossVectors(fwd, upLocal).normalize();
  const up = new THREE.Vector3().crossVectors(right, fwd).normalize();
  const R = [right.x, up.x, -fwd.x, right.y, up.y, -fwd.y, right.z, up.z, -fwd.z];
  const t = [
    -(R[0]*eyeLocal.x + R[1]*eyeLocal.y + R[2]*eyeLocal.z),
    -(R[3]*eyeLocal.x + R[4]*eyeLocal.y + R[5]*eyeLocal.z),
    -(R[6]*eyeLocal.x + R[7]*eyeLocal.y + R[8]*eyeLocal.z),
  ];
  const project = (P) => {
    const xc = R[0]*P[0]+R[1]*P[1]+R[2]*P[2]+t[0];
    const yc = R[3]*P[0]+R[4]*P[1]+R[5]*P[2]+t[1];
    const zc = R[6]*P[0]+R[7]*P[1]+R[8]*P[2]+t[2];
    if (!(zc > 0.3)) return null;
    return [focal*(xc/zc) + W/2, focal*(yc/zc) + H/2];
  };

  // 从真实高斯基数据里挑可见、分散、能被 picker 拾取的点。
  // 深度范围要**自适应场景尺度**（原来写死 zc > 0.3，对某些模型几乎筛不出点）。
  const collect = (minDepth, maxDepth) => {
    const out2 = [];
    const step2 = Math.max(1, Math.floor(raw.count / 6000));
    for (let i = 0; i < raw.count && out2.length < 1500; i += step2) {
      const P = [raw.position[i*3], raw.position[i*3+1], raw.position[i*3+2]];
      const xc = R[0]*P[0]+R[1]*P[1]+R[2]*P[2]+t[0];
      const yc = R[3]*P[0]+R[4]*P[1]+R[5]*P[2]+t[1];
      const zc = R[6]*P[0]+R[7]*P[1]+R[8]*P[2]+t[2];
      if (!(zc > minDepth) || zc > maxDepth) continue;
      const px = [focal*(xc/zc) + W/2, focal*(yc/zc) + H/2];
      if (px[0] < 60 || px[0] > W-60 || px[1] < 60 || px[1] > H-60) continue;
      out2.push({ i, P, px });
    }
    return out2;
  };
  // 从「远于场景中心」开始逐档放宽，直到有足够候选
  let candidates = [];
  let depthLo = Math.max(radius * 0.1, 0.05);
  for (const maxMul of [3, 5, 10, 25]) {
    candidates = collect(depthLo, radius * maxMul);
    if (candidates.length >= 60) break;
  }
  out.depthLoUsed = +depthLo.toFixed(3);
  // 挑 6 个尽量分散的点（分散度要求随候选数量自适应，避免小场景挑不满）
  const minSep = Math.min(W, H) * (candidates.length > 200 ? 0.18 : 0.08);
  const picked = [];
  for (const c of candidates) {
    if (picked.length >= 6) break;
    if (picked.every(q => Math.hypot(q.px[0]-c.px[0], q.px[1]-c.px[1]) > minSep)) picked.push(c);
  }
  out.candidateCount = candidates.length;
  out.pickedCount = picked.length;
  out.minSep = +minSep.toFixed(1);
  out.eyeLocal = eyeLocal.toArray().map(v => +v.toFixed(3));
  out.trueCenter = center.toArray().map(v => +v.toFixed(3));

  // ---- 真实点击链路：画面点像素 → 三维视口点对应点 ----
  // 关键：这里**必须对 canvas 派发真实 DOM 事件**，而不是直接调 handleSceneClick，
  // 否则测不出「标定面板挡住了三维视口」这类问题（那正是之前标定做不下去的原因）。
  const P = S.pnpPanel;
  const W0 = canvas.width, H0 = canvas.height;
  P.clearPairs();
  const mainCam = st.camera;
  mainCam.updateMatrixWorld(true);
  const v3 = new THREE.Vector3();
  const ndcCandidates = [];
  const stride = Math.max(1, Math.floor(raw.count / 4000));
  for (let i = 0; i < raw.count && ndcCandidates.length < 40; i += stride) {
    v3.set(raw.position[i*3], raw.position[i*3+1], raw.position[i*3+2]).applyMatrix4(wm);
    const p = v3.clone().project(mainCam);
    if (p.z < -1 || p.z > 1) continue;
    if (Math.abs(p.x) > 0.8 || Math.abs(p.y) > 0.8) continue;
    ndcCandidates.push({ x: p.x, y: p.y });
  }
  out.ndcCandidateCount = ndcCandidates.length;

  // 面板打开着，遮罩层是否还能挡住画布？
  const viewport = document.getElementById('viewport');
  const panelRect = panel.getBoundingClientRect();
  const dialogEl = panel.querySelector('.pnp-dialog');
  const dialogRect = dialogEl.getBoundingClientRect();
  // 等一帧再量，避免读到布局尚未稳定的中间状态
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  const dialogRect2 = dialogEl.getBoundingClientRect();
  const rect = viewport.getBoundingClientRect();
  const ps = getComputedStyle(panel);
  const ds = getComputedStyle(panel.querySelector('.pnp-backdrop'));
  out.css = {
    panelPointerEvents: ps.pointerEvents,
    backdropPointerEvents: ds.pointerEvents,
    dialogWidthComputed: getComputedStyle(dialogEl).width,
    loadedCssHasNewRules: null,
  };
  // 直接取一次磁盘上的 CSS，确认浏览器加载的是新版
  const cssText = await (await fetch('/css/app.css', { cache: 'no-store' })).text();
  out.css.loadedCssHasNewRules = cssText.includes('.pnp-panel') && cssText.includes('pointer-events: none') && cssText.includes('pnp-tabs');
  out.panelRect = { left: Math.round(panelRect.left), right: Math.round(panelRect.right), width: Math.round(panelRect.width) };
  out.dialogRect = { left: Math.round(dialogRect.left), right: Math.round(dialogRect.right), width: Math.round(dialogRect.width) };
  out.dialogRectAfterFrame = { width: Math.round(dialogRect2.width) };
  out.viewportRect = { left: Math.round(rect.left), right: Math.round(rect.right), width: Math.round(rect.width) };
  out.viewportUsable = rect.right - Math.max(rect.left, dialogRect.right);
  // 面板打开后，视口里**真正能被点到**的区域：必须避开左侧面板。
  // 之前测试没排除这一点，候选点落在面板底下，点击永远到不了三维视口。
  out.clickableRect = {
    left: Math.max(rect.left, Math.ceil(dialogRect.right)) + 4,
    right: Math.ceil(rect.right) - 4,
    top: Math.ceil(rect.top) + 4,
    bottom: Math.ceil(rect.bottom) - 4,
  };
  out.clickableWidth = out.clickableRect.right - out.clickableRect.left;
  // 画布在 DOM 里的真实几何（用于判断是不是被别的东西压小了）
  out.canvasRectNow = { l: Math.round(rect.left), r: Math.round(rect.right), t: Math.round(rect.top), b: Math.round(rect.bottom), w: Math.round(rect.width), h: Math.round(rect.height) };
  out.canvasAttrs = { w: canvas.width, h: canvas.height };
  out.innerSize = { w: window.innerWidth, h: window.innerHeight };
  out.canvasStyle = { position: getComputedStyle(canvas).position, width: getComputedStyle(canvas).width, height: getComputedStyle(canvas).height };
  out.canvasParent = (() => {
    const p = canvas.parentElement;
    const s = getComputedStyle(p);
    const r = p.getBoundingClientRect();
    return { tag: p.tagName + (p.id ? '#' + p.id : ''), w: Math.round(r.width), h: Math.round(r.height), position: s.position, display: s.display };
  })();
  out.isPanelCanvas = canvas === frameCanvas;
  out.frameCanvasIsDifferent = frameCanvas !== canvas;

  // 模拟真实点击：用 elementFromPoint 检查那个位置到底是谁在接收事件
  const probeX = Math.round(rect.left + rect.width * 0.75);
  const probeY = Math.round(rect.top + rect.height * 0.5);
  const topEl = document.elementFromPoint(probeX, probeY);
  out.topElementAtViewport = topEl ? (topEl.id || topEl.className || topEl.tagName) : 'none';
  out.viewportReceivesEvents = topEl === viewport;

  // 选中画面上的一个像素点。
  // 必须先切回步骤 1 —— 步骤 2 时画面 section 被隐藏，画布尺寸变成 0，
  // 点击坐标会全部无效（这是两步骤布局带来的真实约束，测试也要遵守）。
  const armFromFrame = (fx = 0.4, fy = 0.35) => {
    panel.querySelector('[data-tab="1"]').click();
    clickCanvas(frameCanvas.width * fx, frameCanvas.height * fy);
  };
  // 在面板画面上点一个像素 → 应当自动切到步骤 2 并进入「等待三维点」
  armFromFrame();
  out.pendingArmed = !!P._pendingImage;
  out.activeTabAfterClick = document.querySelector('.pnp-tab.is-active')?.dataset.tab ?? null;
  out.stepAfterClick = P.step;
  out.armedClass = viewport.classList.contains('pnp-armed');
  out.pendingBadge = [...panel.querySelectorAll('.pnp-tab .badge')].map((b) => b.textContent);
  out.pendingBadgeRaw = P._pendingImage ? 'yes' : 'no';
  // T2A/T2B：选完点（_pendingImage 清空 → 面板解除 armed）后，视锥与候选点必须自动撤掉
  out.transientWhenArmed = S.structureLayer.stats().transientElements;
  P._pendingImage = null;
  P._updateStepTip();
  out.transientAfterPick = S.structureLayer.stats().transientElements;
  // 复原：继续后面的真实点击链路
  armFromFrame();
  out.frameGeom = (() => {
    const r = frameCanvas.getBoundingClientRect();
    return { w: Math.round(r.width), h: Math.round(r.height), l: Math.round(r.left), t: Math.round(r.top) };
  })();
  out.imageSize = { ...P.imageSize };
  out.clickableWidthNow = out.clickableRect.right - out.clickableRect.left;
  // 判定浏览器里跑的模块身份：动态 import 的模块类应当与实例同类（同名），
  // 并且实例上应当已经挂上 app.js 传入的 onArmedChange
  try {
    const mod = await import('/js/pnp-panel.js');
    out.dynImportHasArmed = typeof mod.PnpPanel === 'function';
    out.instanceCtorName = P.constructor?.name ?? '?';
    out.sameClass = P instanceof mod.PnpPanel;
  } catch (e) {
    out.dynImportError = String(e);
  }
  out.hasArmedHook = typeof P.onArmedChange === 'function';
  out.panelProtoHasHookRef = String(P._updateStepTip).includes('onArmedChange');
  out.ownHookKeys = Object.keys(P).filter((k) => /^on[A-Z]/.test(k));
  out.panelKeysSample = Object.keys(P).slice(0, 30);

  // 对 canvas 派发真实 click，逐个候选位置试
  let domClickCount = 0;
  const spy = () => { domClickCount++; };
  canvas.addEventListener('click', spy, true);
  // 直接给面板打桩，记录 handleSceneClick 是否被调用、结果如何
  const origHandle = P.handleSceneClick.bind(P);
  let handleCalls = 0;
  let handleNull = 0;
  P.handleSceneClick = (ndc) => {
    handleCalls++;
    const before = P.pairs.length;
    const r = origHandle(ndc);
    if (P.pairs.length === before) handleNull++;
    return r;
  };
  let lastClickState = null;
  // 记录点击那一刻各条件为何不满足
  let rejectReason = null;
  canvas.addEventListener(
    'click',
    (ev) => {
      if (rejectReason) return;
      const r = canvas.getBoundingClientRect();
      const inside =
        ev.clientX >= r.left && ev.clientX <= r.right && ev.clientY >= r.top && ev.clientY <= r.bottom;
      rejectReason = {
        targetIsCanvas: ev.target === canvas,
        targetTag: ev.target?.tagName ?? String(ev.target),
        button: ev.button,
        inside,
        clientX: Math.round(ev.clientX),
        clientY: Math.round(ev.clientY),
        rect: { l: Math.round(r.left), r: Math.round(r.right), t: Math.round(r.top), b: Math.round(r.bottom) },
        pnpActive: S.pnpPanel.active,
        hadPending: !!S.pnpPanel._pendingImage,
      };
    },
    true
  );

  const domClickAtNdc = (ndc) => {
    // 目标屏幕位置（按画布几何换算）
    const targetX = rect.left + ((ndc.x + 1) / 2) * rect.width;
    const targetY = rect.top + ((1 - ndc.y) / 2) * rect.height;
    // 关键：必须夹到「画布可见区域 ∩ 面板右侧」之内。
    // 左侧停靠的面板会盖住画布一部分，落在面板底下的点击永远到不了三维视口
    // —— 这正是之前标定做不下去的真实原因，测试也必须避开这块。
    const visLeft = Math.max(rect.left + 2, dialogRect.right + 6);
    const x = Math.min(Math.max(targetX, visLeft), rect.right - 2);
    const y = Math.min(Math.max(targetY, rect.top + 2), rect.bottom - 2);
    if (rect.right - visLeft < 20 || x < visLeft) return false;
    const ev = (type) => new MouseEvent(type, { clientX: x, clientY: y, button: 0, buttons: 1, bubbles: true, cancelable: true });
    canvas.dispatchEvent(ev('pointerdown'));
    canvas.dispatchEvent(ev('pointerup'));
    canvas.dispatchEvent(ev('click'));
    lastClickState = {
      active: S.pnpPanel.active,
      hadPending: !!S.pnpPanel._pendingImage,
      pairs: S.pnpPanel.pairs.length,
      mode: st.mode,
      x: Math.round(x),
      y: Math.round(y),
    };
    return true;
  };

  // 只保留夹紧后仍在可见区域内的候选
  const visibleCands = ndcCandidates.filter((p) => {
    const targetX = rect.left + ((p.x + 1) / 2) * rect.width;
    return targetX >= Math.max(rect.left + 2, dialogRect.right + 6);
  });
  out.visibleCandidates = visibleCands.length;

  let hitAt = null;
  for (const ndc of visibleCands) {
    if (!domClickAtNdc(ndc)) continue;
    if (P.pairs.length > 0) { hitAt = ndc; break; }
    armFromFrame();
  }
  out.pendingAfterLoop = !!P._pendingImage;
  out.armedClassAfterLoop = viewport.classList.contains('pnp-armed');
  canvas.removeEventListener('click', spy, true);
  P.handleSceneClick = origHandle;
  out.domClickCount = domClickCount;
  out.handleCalls = handleCalls;
  out.handleNull = handleNull;
  out.lastClickState = lastClickState;
  out.rejectReason = rejectReason;
  // 拾取器状态：直接调用一次 picker.pick 看能不能命中
  try {
    const picker = st.picker;
    out.pickerInfo = {
      hasPicker: !!picker,
      count: picker?.count ?? null,
      hasBounds: !!picker?.bounds,
      boundsRadius: picker?.bounds?.radius ?? null,
    };
    if (picker && ndcCandidates.length) {
      const n0 = ndcCandidates[0];
      const hit0 = picker.pick({ x: n0.x, y: n0.y }, st.camera, wm, { screenRadius: 0.05 });
      out.firstCandidatePick = hit0 ? 'hit' : 'null';
    }
  } catch (e) {
    out.pickerError = String(e);
  }
  out.hitAt = hitAt ? [+hitAt.x.toFixed(3), +hitAt.y.toFixed(3)] : null;
  out.triedCount = ndcCandidates.length;
  out.pairsAfterSceneClick = P.pairs.length;
  out.marksAfterSceneClick = panel.querySelectorAll('.pnp-mark.done').length;
  out.pendingCleared = !P._pendingImage;

  P.clearPairs();

  // 用面板公开接口注入对应点（像素来自上面的虚拟相机，三维点用真实高斯坐标）
  const pairs = picked.map(p => ({ image: [p.px[0], p.px[1]], object: p.P }));
  S.pnpPanel.setIntrinsics({ fx: focal, fy: focal, cx: W/2, cy: H/2, dist: [0,0,0,0,0] }, { width: W, height: H });
  S.pnpPanel.setPairs(pairs);
  out.pairsInjected = S.pnpPanel.pairs.length;
  out.solveEnabledAfterInject = !panel.querySelector('[data-solve]').disabled;

  // 解算
  const solved = P.solve();
  out.solved = !!solved;
  out.meanError = solved ? solved.result.reprojectionError : null;
  out.poseSource = solved ? solved.pose.source : null;
  out.reprojMarks = panel.querySelectorAll('.pnp-mark.reproj').length;
  out.errorText = panel.querySelector('[data-error]').textContent;

  // 解算出的相机中心应与虚拟相机位置一致
  if (solved) {
    const d = Math.hypot(
      solved.pose.position[0] - eyeLocal.x,
      solved.pose.position[1] - eyeLocal.y,
      solved.pose.position[2] - eyeLocal.z
    );
    out.centerError = d;
  }

  // ---- 内参输入框：改动必须真的影响解算 ----
  const setInput = (sel, v) => {
    const el = panel.querySelector(sel);
    el.value = String(v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return el.value;
  };
  const fxInput = panel.querySelector('[data-fx]');
  const cyInput = panel.querySelector('[data-cy]');
  out.fxShown = Number(fxInput.value);
  out.cyShown = Number(cyInput.value);

  // 故意把 fx 翻倍 → 解算出的相机距离应当明显变化
  const posBefore = solved ? [...solved.pose.position] : null;
  setInput('[data-fx]', out.fxShown * 2);
  setInput('[data-fy]', out.fxShown * 2);
  out.panelFxAfterEdit = P.intrinsics.fx;
  const solved2 = P.solve();
  out.secondSolveOk = !!solved2;
  if (solved2 && posBefore) {
    out.centerShiftFromFx = Math.hypot(
      solved2.pose.position[0] - posBefore[0],
      solved2.pose.position[1] - posBefore[1],
      solved2.pose.position[2] - posBefore[2]
    );
  }

  // 按视场角重算内参，应回到与视场角一致的值
  panel.querySelector('[data-reset-k]').click();
  out.fxAfterReset = P.intrinsics.fx;
  out.fxResetMatchesFov = Math.abs(P.intrinsics.fx - 0.5 * H / Math.tan(0.5 * Number(panel.querySelector('[data-fov]').value) * Math.PI / 180)) < 0.5;

  out.consoleClean = true;
  return JSON.stringify(out);
})()
`;

// 需要把若干内部函数暴露给测试
const expose = `
(() => {
  const S = window.__caseSystem;
  if (!S) return 'no app';
  return Object.keys(S).join(',');
})()
`;
const ex = await send('Runtime.evaluate', { expression: expose, returnByValue: true });
console.log('__caseSystem 暴露:', ex.result?.result?.value);

const res = await send('Runtime.evaluate', { expression: script, returnByValue: true, awaitPromise: true });
if (res.result?.exceptionDetails) {
  console.log('页面内异常:', res.result.exceptionDetails.exception?.description);
}
let data = {};
try {
  data = JSON.parse(res.result?.result?.value ?? '{}');
} catch {
  console.log('无法解析返回:', String(res.result?.result?.value).slice(0, 500));
}

if (data.error) {
  console.log('  [FAIL] 页面内流程失败：' + data.error);
  fail++;
} else {
  check('三维现场已载入', !!data.model, data.model ?? '');
  check('PnP 面板已打开', data.panelVisible === true);
  check('面板所有 data-* 元素都已缓存（无缺失）', (data.missingDomKeys ?? []).length === 0,
    (data.missingDomKeys ?? []).join(', ') || '全部就位');
  check('底部画面卡片上存在「PnP 标定」入口', data.hasCalibButton === true, `${data.camCardCount} 张卡片`);
  check('面板画布尺寸有效', data.frameCanvas && data.frameCanvas.w > 100 && data.frameCanvas.h > 100,
    data.frameCanvas ? `${data.frameCanvas.w}x${data.frameCanvas.h}` : '无');
  check('画面确实抽出了内容（非全黑）', (data.frameNonBlack ?? 0) > 20, `${data.frameNonBlack} 个非黑采样点`);
  check('初始状态解算按钮**不禁用**（禁用会让点击毫无反馈，这是曾经的 bug）',
    data.solveDisabledAtStart === false, `disabled=${data.solveDisabledAtStart}`);
  check('未采够点时按钮文案提示还差几组', /还需\s*\d+\s*组/.test(data.solveTextAtStart ?? ''),
    data.solveTextAtStart ?? '');
  check('未采够点时点解算会给出明确提示（不再静默无反应）',
    /还需\s*\d+\s*组/.test(data.afterSolveToast ?? '') && data.afterSolveToast !== data.beforeSolveToast,
    (data.afterSolveToast ?? '').slice(0, 70));
  check('点击画面后出现待配对标记', data.pendingMark === 1, `${data.pendingMark} 个`);
  // T2A：视锥线框（虚线）
  check('等待点三维对应点时画出了相机视锥（虚线）',
    (data.frustumWhileArmed?.transientSegments ?? 0) >= 16 && (data.frustumSvgDashed ?? 0) >= 16,
    `图层 ${data.frustumWhileArmed?.transientSegments} 段 / SVG 虚线 ${data.frustumSvgDashed} 条`);
  // T2B：视锥内可选点高亮（抽稀）
  check('等待点选时高亮了视锥内的可选点',
    (data.frustumWhileArmed?.transientDots ?? 0) > 0 && (data.candidateSvgDots ?? 0) > 0,
    `候选点 ${data.frustumWhileArmed?.transientDots} 个`);
  check('候选点数量被抽稀控制（不会拖死浏览器）',
    (data.candidateSvgDots ?? 99999) <= 300,
    `${data.candidateSvgDots} 个亮点（图层里另有同数量的描边光晕，属预期）`);
  check('选完对应点后自动撤掉高亮（视锥与候选点都清空）',
    (data.transientWhenArmed ?? 0) > 0 && (data.transientAfterPick ?? -1) === 0,
    `选中前 ${data.transientWhenArmed} 个元素 → 选完后 ${data.transientAfterPick} 个`);
  // 视锥/候选点必须**看得见**：之前只用 1.8px 亮线压在花花绿绿的高斯上，用户反馈"不明显"。
  // 现在每条线/每个点都垫一层深色描边光晕（halo），并显式抬到 canvas 之上。
  check('视锥/候选点带深色描边光晕（提高对比度）',
    (data.haloLineCount ?? 0) >= 16 && (data.haloDotCount ?? 0) >= 1,
    `光晕线 ${data.haloLineCount} 条 / 光晕点 ${data.haloDotCount} 个`);
  check('结构覆盖层显式抬到 canvas 之上（z-index）',
    Number(data.structureZIndex ?? 0) >= 50, `z-index=${data.structureZIndex}`);
  check('「回到全景」按钮存在（画面飞走时的一键自救）', data.hasFitViewBtn === true);
  check('能收集到可供点击的三维屏幕位置', (data.ndcCandidateCount ?? 0) > 5, `${data.ndcCandidateCount} 个`);
  check('面板没有占满屏幕，三维视口仍可用', (data.viewportUsable ?? 0) > 300,
    `视口可用宽度 ${data.viewportUsable}px（对话框宽 ${data.dialogRect?.width}px，重测 ${data.dialogRectAfterFrame?.width}px）`);
  check('浏览器加载的是新版 CSS（面板已改为左侧停靠）', data.css?.loadedCssHasNewRules === true,
    `panel.pointer-events=${data.css?.panelPointerEvents} backdrop.pointer-events=${data.css?.backdropPointerEvents} dialog.width=${data.css?.dialogWidthComputed}`);
  check('三维视口在最上层、能收到鼠标事件', data.viewportReceivesEvents === true,
    `该位置最上层元素 = ${data.topElementAtViewport}`);
  check('画面点击后进入「等待三维点」状态', data.pendingArmed === true,
    `pending=${data.pendingBadgeRaw} 步骤=${data.stepAfterClick} 徽标=${JSON.stringify(data.pendingBadge)} `
    + `画面几何=${JSON.stringify(data.frameGeom)} 图像尺寸=${JSON.stringify(data.imageSize)}`);
  check('选中像素后自动切到「② 点三维对应点」子选项',
    data.activeTabAfterClick === '2' && data.stepAfterClick === 2,
    `高亮=${data.activeTabAfterClick} step=${data.stepAfterClick}`);
  check('等待三维点击时视口有提示描边', data.armedClass === true,
    `点击后 armed=${data.armedClass}，循环后 armed=${data.armedClassAfterLoop}`);
  check('标定面板实例已挂上 app.js 的 onArmedChange 钩子', data.hasArmedHook === true,
    data.dynImportError ? `动态 import 失败：${data.dynImportError}` : `原型方法含钩子引用=${data.panelProtoHasHookRef}`);
  check('真实 DOM 点击三维视口能形成 1 组对应点', data.pairsAfterSceneClick === 1,
    `命中 NDC=${JSON.stringify(data.hitAt)}；handleSceneClick 调用 ${data.handleCalls} 次；`
    + `点击时 target=canvas:${data.rejectReason?.targetIsCanvas} inside:${data.rejectReason?.inside} `
    + `待选像素:${data.rejectReason?.hadPending}；可见候选 ${data.visibleCandidates} 个`);
  check('画面上出现了已完成标记', data.marksAfterSceneClick === 1, `${data.marksAfterSceneClick} 个`);
  check('配对完成后清空待选状态', data.pendingCleared === true);
  check('能从高斯基数据里找到可见候选点', (data.candidateCount ?? 0) > 50, `${data.candidateCount} 个`);
  check('成功挑出 6 个分散点', data.pickedCount === 6, `${data.pickedCount} 个`);
  check('注入 6 组对应点后列表非空', data.pairsInjected === 6);
  check('采够点后解算按钮启用', data.solveEnabledAfterInject === true);
  check('解算成功', data.solved === true);
  check('重投影误差显示在面板上', /px/.test(data.errorText ?? ''), data.errorText ?? '');
  check('解算出的相机位置与真值一致（< 0.05m）', (data.centerError ?? 1e9) < 0.05,
    `误差 ${(data.centerError ?? 0).toFixed(4)} m`);
  check('画面上画出了重投影标记', (data.reprojMarks ?? 0) >= 6, `${data.reprojMarks} 个`);
  check('pose.source 标记为 pnp', data.poseSource === 'pnp', String(data.poseSource));
  check('内参输入框已按画面尺寸回填 fx', (data.fxShown ?? 0) > 0, `fx=${data.fxShown}`);
  check('内参输入框已回填 cy（画面中心）', Math.abs((data.cyShown ?? 0) - 180) < 1, `cy=${data.cyShown}`);
  check('改动 fx 输入框会更新面板内参', Math.abs((data.panelFxAfterEdit ?? 0) - (data.fxShown ?? 0) * 2) < 1,
    `${data.fxShown} → ${data.panelFxAfterEdit}`);
  check('改动内参后仍能解算', data.secondSolveOk === true);
  check('内参变化确实改变解算结果', (data.centerShiftFromFx ?? 0) > 0.5, `相机位置移动 ${(data.centerShiftFromFx ?? 0).toFixed(2)} m`);
  check('「按视场角重算内参」能恢复内参', data.fxResetMatchesFov === true, `fx=${(data.fxAfterReset ?? 0).toFixed(1)}`);
}

const noise = consoleErrors.filter((m) => !/DevTools|Download the React/.test(m));
check('控制台无 error 输出', noise.length === 0, noise.slice(0, 3).join(' | '));
if (pageExceptions.length) {
  console.log('\n--- 页面未捕获异常 ---');
  for (const e of pageExceptions.slice(0, 6)) console.log('  ' + String(e).split('\n').slice(0, 4).join('\n  '));
}

// 截图留证
const shot = await send('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) {
  const out = path.join(process.cwd(), 'output', 'verify', 'pnp-panel.png');
  mkdirSync(path.dirname(out), { recursive: true });
  writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log('\n截图:', out);
}

// 清理
await fetch(`${BASE}/api/projects/${pid}`, { method: 'DELETE' });
rmSync(VIDEO, { force: true });

console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
ws.close();
edge.kill();
process.exit(fail === 0 ? 0 : 1);
