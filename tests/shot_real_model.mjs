/**
 * 截图：在系统里加载训练出的真实场景模型 room-real.splat。
 * 用法: node tests/shot_real_model.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9241;
const PROFILE = path.join(process.cwd(), '.browsertmp', 'realmodel');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1200,1400', BASE,
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
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
};
const send = (method, params = {}) => new Promise((res) => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
await send('Runtime.enable');

let booted = false;
for (let i = 0; i < 40 && !booted; i++) {
  const r = await send('Runtime.evaluate', { expression: 'String(!!(window.__caseSystem && window.__caseSystem.state))', returnByValue: true });
  booted = r.result?.result?.value === 'true';
  if (!booted) await sleep(500);
}
await sleep(2000);

const load = `
(async () => {
  const S = window.__caseSystem, st = S.state;
  const models = await (await fetch('/models/index.json')).json();
  const m = models.models.find(x => /room-real/.test(x.name));
  if (!m) return 'no-model:' + models.models.map(x=>x.name).join(',');
  await S.loadModel(m);
  await new Promise(r => setTimeout(r, 3500));
  const gl = document.getElementById('viewport').getContext('webgl2');
  const W = document.getElementById('viewport').width, H = document.getElementById('viewport').height;
  const b = new Uint8Array(W*H*4);
  gl.readPixels(0,0,W,H,gl.RGBA,gl.UNSIGNED_BYTE,b);
  let nz=0; for (let i=0;i<b.length;i+=4) if (b[i]||b[i+1]||b[i+2]||b[i+3]) nz++;
  return JSON.stringify({ model: m.name, gaussians: st.renderer?.stats?.drawn ?? null, nonZero: nz, canvas: W+'x'+H, radius: st.bounds?.radius ?? null });
})()
`;
const r = await send('Runtime.evaluate', { expression: load, returnByValue: true, awaitPromise: true });
console.log('加载结果:', r.result?.result?.value ?? JSON.stringify(r).slice(0, 300));

await sleep(1500);
const shot = await send('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) {
  const out = path.join(process.cwd(), 'output', 'verify', 'room-real-app.png');
  writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log('截图:', out);
}
ws.close();
edge.kill();
process.exit(0);
