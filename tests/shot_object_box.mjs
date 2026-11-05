/**
 * 截图：物体模式**可信**时的样子（包围盒 + 水平层 + 尺寸/层高标注）。
 *
 * ⚠️ 必须说清楚：这张图用的是**合成几何**。
 *    工作区里**没有任何一份可信的真实物体重建** —— 柜子那份素材算出来的盒子是
 *    16.8 m 长（trust=false），所以真实数据下界面是"什么都不画 + 说明原因"。
 *    这张图只是让人看清"判据通过时前端会画成什么样"（配色/层/标注），
 *    不是在宣称真实柜子已经能画对了。
 *
 * 用法: node tests/shot_object_box.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9285;
const OUT = path.join(process.cwd(), 'output', 'verify');
mkdirSync(OUT, { recursive: true });

const proj = await (
  await fetch(`${BASE}/api/projects`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'object-box-shot', operator: 'test' }),
  })
).json();
const pid = proj.project.id;
const cleanup = async () => {
  try {
    await fetch(`${BASE}/api/projects/${pid}`, { method: 'DELETE' });
  } catch {
    /* 忽略 */
  }
};

const PROFILE = path.join(process.cwd(), '.browsertmp', 'object-box');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });
const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1440,1000', BASE,
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let page = null;
for (let i = 0; i < 40 && !page; i++) {
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
  } catch {
    /* retry */
  }
  if (!page) await sleep(400);
}
if (!page) {
  console.log('浏览器没起来');
  edge.kill();
  await cleanup();
  process.exit(1);
}

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  }
};
const send = (method, params = {}) =>
  new Promise((res) => {
    pending.set(++id, res);
    ws.send(JSON.stringify({ id, method, params }));
  });
const evaluate = async (e, a = false) => {
  const r = await send('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: a });
  return r.result?.result?.value;
};

await send('Runtime.enable');
for (let i = 0; i < 40; i++) {
  if ((await evaluate('String(!!(window.__caseSystem && window.__caseSystem.structureLayer))')) === 'true') break;
  await sleep(500);
}
await evaluate(`(async () => {
  const list = await (await fetch('/api/projects')).json();
  const p = (list.projects ?? []).find(x => x.id === ${JSON.stringify(pid)});
  const sel = document.getElementById('projectSelect');
  if (p && sel) { sel.value = p.id; sel.dispatchEvent(new Event('change', { bubbles: true })); }
  return 'ok';
})()`, true);
await sleep(1200);

// 载入柜子的真实三维内容（这样看到的是"真实柜子上叠加了一个**合成**盒子"）
const info = await evaluate(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const m = j.models.find(x => String(x.name).includes('柜子现场-致密'));
  if (!m) return 'no-model';
  const st = await (await fetch(m.structureUrl, { cache: 'no-store' })).json();
  await S.loadModel(m, { announce: false });
  await new Promise(r => setTimeout(r, 4500));
  // 合成一个"可信"的物体记录：1.2 × 0.6 × 1.9 m，放在相机轨迹质心上，
  // 三块水平层（底/搁板/顶）。仅用于展示渲染效果。
  const traj = st.trajectory?.points ?? [];
  const c = traj.length
    ? traj.reduce((acc, p) => [acc[0] + p[0] / traj.length, acc[1] + p[1] / traj.length, acc[2] + p[2] / traj.length], [0, 0, 0])
    : [0, 0, 0];
  const [w, d, h] = [1.2, 0.6, 1.9];
  const corners = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [0, 1]) {
    corners.push([c[0] + sx * w / 2, c[1] + sy * d / 2, c[2] + (sz - 0.5) * h]);
  }
  const obj = {
    mode: 'object', center: c, size: [w, d, h], corners,
    axes: { u: [1, 0, 0], v: [0, 1, 0], up: [0, 0, 1] },
    anchor: { method: 'camera-orbit', center: c },
    layers: [
      { height: -h / 2, index: 0, prominence: 0.96 },
      { height: 0, index: 1, prominence: 0.91 },
      { height: h / 2, index: 2, prominence: 0.95 },
    ],
    trust: { ok: true, reasons: [], metrics: { note: '合成几何，仅用于展示渲染效果' } },
  };
  S.structureLayer.set({ ...st, sceneMode: 'object', sceneModeSource: 'auto', object: obj, sane: true });
  await new Promise(r => setTimeout(r, 800));
  return JSON.stringify({ model: m.name, seg: S.structureLayer.segments.length,
                          labels: S.structureLayer.labels.length, drawn: S.structureLayer.objectDrawn });
})()`, true);
console.log('  合成物体盒:', info);
const s = await send('Page.captureScreenshot', { format: 'png' });
if (s.result?.data) writeFileSync(path.join(OUT, 'object-box-synthetic.png'), Buffer.from(s.result.data, 'base64'));
console.log('  已写出 object-box-synthetic.png');
ws.close();
edge.kill();
await cleanup();
console.log('（合成几何演示；改判与案件均已复位）');
