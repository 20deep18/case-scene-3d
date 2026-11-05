/**
 * 截图：左下图层面板展开后的样子（含具体条目与显隐开关）。
 * 用法: node tests/shot_layers.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9232;
const PROFILE = path.join(process.cwd(), '.browsertmp', 'layers');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

const proj = await (
  await fetch(`${BASE}/api/projects`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'layers-shot', operator: 'test' }),
  })
).json();
const pid = proj.project.id;
const mk = (annotation) =>
  fetch(`${BASE}/api/projects/${pid}/annotations`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ annotation, operator: 'test' }),
  });

await mk({ kind: 'point', points: [[0.6, 0.2, 0.4]], label: '物证·玻璃杯' });
await mk({ kind: 'point', points: [[-0.4, 0.2, 0.9]], label: '物证·鞋印' });
await mk({ kind: 'measure', points: [[0, 0.2, 0], [1.2, 0.2, 0]], label: '门到桌距离' });
await mk({ kind: 'polyline', points: [[-1, 0.2, -1], [0, 0.2, 0], [1, 0.2, 1]], label: '嫌疑人路径' });
await mk({ kind: 'polygon', points: [[-1.5, 0.2, -1.5], [1.5, 0.2, -1.5], [0, 0.2, 1.5]], label: '中心现场' });

const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1400,1000', BASE,
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

// 载入模型并把项目切到我们的测试案件
const prep = `
(async () => {
  const S = window.__caseSystem, st = S.state;
  const models = await (await fetch('/models/index.json')).json();
  await S.loadModel(models.models.find(m => /sample-room/.test(m.name)) ?? models.models[0]);
  return 'ok';
})()
`;
await send('Runtime.evaluate', { expression: prep, returnByValue: true, awaitPromise: true });
await sleep(3000);

// 切到测试案件（通过下拉框）并展开所有图层
const pick = `
(async () => {
  const sel = document.getElementById('projectSelect');
  const opt = [...sel.options].find(o => /layers-shot/.test(o.textContent));
  if (!opt) return 'no-option:' + [...sel.options].map(o=>o.textContent).join('|');
  sel.value = opt.value;
  sel.dispatchEvent(new Event('change', { bubbles: true }));
  await new Promise(r => setTimeout(r, 2500));
  const S = window.__caseSystem;
  // 展开全部图层
  for (const key of ['point','polyline','polygon','measure','camera']) {
    const c = document.querySelector('[data-caret="' + key + '"]');
    if (c && !c.disabled && c.textContent.trim() === '▸') c.click();
  }
  await new Promise(r => setTimeout(r, 400));
  const list = document.getElementById('layerList');
  return JSON.stringify({
    project: S.state.project?.id,
    groups: list.querySelectorAll('.layer-group').length,
    subitems: list.querySelectorAll('.layer-subitem').length,
    labels: [...list.querySelectorAll('.layer-subitem .subname')].map(n => n.textContent),
    eyes: list.querySelectorAll('.layer-eye').length,
  });
})()
`;
const r = await send('Runtime.evaluate', { expression: pick, returnByValue: true, awaitPromise: true });
console.log('图层面板:', r.result?.result?.value ?? JSON.stringify(r).slice(0, 400));

await sleep(1200);
const shot = await send('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) {
  const out = path.join(process.cwd(), 'output', 'verify', 'layers-panel.png');
  writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log('截图:', out);
}

await fetch(`${BASE}/api/projects/${pid}`, { method: 'DELETE' });
ws.close();
edge.kill();
process.exit(0);
