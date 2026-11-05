/**
 * 截图 + 取景数据：验证「浮尘撑爆取景」修好之后，房间能不能占满画面。
 *
 * 用法: node tests/shot_model_framing.mjs [模型名关键字]
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const KEY = process.argv[2] ?? '现场模型';
const PORT = 9273;
const OUT = path.join(process.cwd(), 'output', 'verify');
mkdirSync(OUT, { recursive: true });

const PROFILE = path.join(process.cwd(), '.browsertmp', 'shot-frame');
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
const evaluate = async (expression, awaitPromise = false) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise });
  if (r.result?.exceptionDetails) return `EXC: ${r.result.exceptionDetails.exception?.description}`;
  return r.result?.result?.value;
};
const shot = async (file) => {
  const s = await send('Page.captureScreenshot', { format: 'png' });
  if (s.result?.data) {
    const p = path.join(OUT, file);
    writeFileSync(p, Buffer.from(s.result.data, 'base64'));
    console.log('截图:', p);
  }
};
await send('Runtime.enable');
for (let i = 0; i < 40; i++) {
  if ((await evaluate('String(!!(window.__caseSystem && window.__caseSystem.loadModel))')) === 'true') break;
  await sleep(500);
}
await sleep(1500);

const info = await evaluate(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const m = j.models.find(x => String(x.name).includes(${JSON.stringify(KEY)})) ?? j.models[0];
  if (!m) return 'no-model';
  await S.loadModel(m);
  await new Promise(r => setTimeout(r, 3500));
  const b = S.state.bounds;
  const cam = S.state.camera;
  const tgt = S.state.controls?.target;
  return JSON.stringify({
    model: m.name,
    radius: b?.radius ?? null,
    exactRadius: b?.exact?.radius ?? null,
    robustUsed: b?.robust ?? null,
    size: b?.size?.map(v => Math.round(v * 100) / 100) ?? null,
    cameraPos: cam ? [cam.position.x, cam.position.y, cam.position.z].map(v => Math.round(v * 10) / 10) : null,
    target: tgt ? [tgt.x, tgt.y, tgt.z].map(v => Math.round(v * 10) / 10) : null,
    distToTarget: cam && tgt ? Math.round(cam.position.distanceTo(tgt) * 10) / 10 : null,
    gauss: S.state.gaussians?.count ?? null,
  }, null, 1);
})()`, true);
console.log('取景数据:', info);
await sleep(600);
await shot('model-framing-front.png');

// 切到俯视态势，看整体布局
await evaluate(`(() => {
  const btns = [...document.querySelectorAll('#modeSwitch button')];
  const top = btns.find(b => b.dataset.mode === 'top');
  if (top) top.click();
  return 'ok';
})()`);
await sleep(2500);
await shot('model-framing-top.png');

// 站到房间内部平视：用来区分「模型本身糊」和「观察者渲染 bug」
// （默认环绕视角是从 13m 高处往下看，很容易只看到天花/地板那一大块白）
const inside = await evaluate(`(() => {
  const S = window.__caseSystem;
  const b = S.state.bounds;
  const c = b.center;
  const r = b.radius;
  const pos = [c[0] + r * 0.35, c[1] + 0.15 * r, c[2] + r * 0.35];
  S.state.camera.position.set(pos[0], pos[1], pos[2]);
  S.state.controls.target.set(c[0], c[1], c[2]);
  S.state.controls.update();
  if (S.ui?.setMode) S.ui.setMode('orbit');
  return JSON.stringify({ pos: pos.map(v => Math.round(v * 10) / 10), target: c.map(v => Math.round(v * 10) / 10) });
})()`);
console.log('室内视角:', inside);
await sleep(2500);
await shot('model-framing-inside.png');

ws.close();
edge.kill();
process.exit(0);
