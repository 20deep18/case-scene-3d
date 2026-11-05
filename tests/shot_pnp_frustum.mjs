/**
 * 截图：PnP 标定「等待点选三维对应点」时的相机视锥（黄色虚线）+ 候选点（亮青）。
 * 用法: node tests/shot_pnp_frustum.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9281;
const OUT = path.join(process.cwd(), 'output', 'verify');
mkdirSync(OUT, { recursive: true });

const PROFILE = path.join(process.cwd(), '.browsertmp', 'shot-frustum');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// 前置：案件里必须有一个挂了视频的相机标注
const projects = await (await fetch(`${BASE}/api/projects`)).json();
const pid = projects.projects?.[0]?.id;
if (!pid) { console.log('没有案件'); process.exit(1); }
const proj = await (await fetch(`${BASE}/api/projects/${encodeURIComponent(pid)}`)).json();
const cam = (proj.project.annotations ?? []).find((a) => a.kind === 'camera' && a.media?.[0]?.url);
if (!cam) { console.log('案件里没有「挂了视频的相机标注」，请先导入素材'); process.exit(1); }
console.log(`用案件 ${pid} 的相机标注「${cam.label ?? cam.id}」`);

const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1440,1000', BASE,
], { stdio: 'ignore' });

let page = null;
for (let i = 0; i < 40 && !page; i++) {
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
  } catch { /* retry */ }
  if (!page) await sleep(400);
}
if (!page) { console.log('浏览器没起来'); edge.kill(); process.exit(1); }

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
};
const send = (method, params = {}) =>
  new Promise((res) => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (e, a = false) => {
  const r = await send('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: a });
  if (r.result?.exceptionDetails) return `EXC: ${r.result.exceptionDetails.exception?.description}`;
  return r.result?.result?.value;
};
await send('Runtime.enable');
for (let i = 0; i < 40; i++) {
  if ((await evaluate('String(!!(window.__caseSystem && window.__caseSystem.loadModel))')) === 'true') break;
  await sleep(500);
}
await sleep(1500);

// 载入模型 + 打开 PnP 标定面板 + 在画面上点一个像素（进入等待三维点选）
const out = await evaluate(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const m = j.models.find(x => /room-real|现场模型|sample-room/.test(x.name)) ?? j.models[0];
  await S.loadModel(m, {});
  await new Promise(r => setTimeout(r, 3500));
  const ann = S.state.project.annotations.find(a => a.kind === 'camera' && a.media?.[0]?.url)
           ?? S.state.project.annotations.find(a => a.kind === 'camera');
  if (!ann) return 'no-annotation';
  await S.openPnpCalibration(ann);
  await new Promise(r => setTimeout(r, 1200));
  const fc = document.querySelector('[data-framecanvas]');
  // 直接置入一个待配对像素点并刷新提示状态 —— 等价于「在画面上点了一个像素」，
  // 目的只是把界面推进到「等待点选三维对应点」这个状态来截图。
  if (fc) {
    // 注意：_pendingImage 是**数组** [x, y]（面板内部就是按数组存的），
    // 传对象会让 _updateStepTip 里的 [0].toFixed 抛错。
    S.pnpPanel._pendingImage = [fc.clientWidth * 0.4, fc.clientHeight * 0.4];
    S.pnpPanel._updateStepTip();
  }
  await new Promise(r => setTimeout(r, 800));
  return JSON.stringify({
    armed: !!S.pnpPanel._pendingImage,
    stats: S.structureLayer.stats(),
  });
})()`, true);
console.log('状态:', out);
await sleep(900);

const shot = await send('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) {
  const p = path.join(OUT, 'pnp-frustum.png');
  writeFileSync(p, Buffer.from(shot.result.data, 'base64'));
  console.log('截图:', p);
}
ws.close();
edge.kill();
process.exit(0);
