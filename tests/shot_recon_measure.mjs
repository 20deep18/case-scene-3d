/**
 * 截图：现场建模面板 + 精确测量清单（用于人工核对界面）。
 * 用法: node tests/shot_recon_measure.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9271;
const PROFILE = path.join(process.cwd(), '.browsertmp', 'shot-rm');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

const api = async (p, opts = {}) => {
  const r = await fetch(BASE + p, {
    method: opts.method ?? 'GET',
    headers: opts.body ? { 'content-type': 'application/json' } : {},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  return r.json().catch(() => ({}));
};

const proj = (await api('/api/projects', { method: 'POST', body: { name: '现场建模演示' } })).project;
const pid = proj.id;

// 造几条待测尺寸，让界面有内容
const mk = (label, a, b, prio = 'normal', note = '') =>
  api(`/api/projects/${pid}/annotations`, {
    method: 'POST',
    body: {
      annotation: {
        kind: 'dimension', label,
        points: [a, b],
        modelDistance: Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]),
        unit: 'meter', priority: prio, note,
      },
      operator: '技术员',
    },
  });
await mk('客厅东西向净宽', [-1.5, 1.0, -2], [1.5, 1.0, -2], 'high', '沿踢脚线量');
await mk('门洞净宽', [2.0, 1.0, -0.9], [2.0, 1.0, 0], 'high');
await mk('床体长度', [-1.0, 0.5, 1.2], [1.0, 0.5, 1.2], 'normal', '床沿到床尾');
await mk('窗户高度', [2.2, 0.9, -1.8], [2.2, 2.1, -1.8], 'low');

// 造一条已实测的，展示"已测"分区
const d = await mk('房间净高', [3.2, 0.05, -2.5], [3.2, 2.85, -2.5], 'high');
await api(`/api/projects/${pid}/annotations/${d.annotation.id}`, {
  method: 'PATCH',
  body: { patch: { measured: 2.9 }, operator: '民警张三' },
});

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
  const r = await send('Runtime.evaluate', { expression: 'String(!!(window.__caseSystem && window.__caseSystem.measurePanel))', returnByValue: true });
  booted = r.result?.result?.value === 'true';
  if (!booted) await sleep(500);
}
await sleep(1500);

// 切案件 + 载入模型（让三维背景有内容）
const prep = `
(async () => {
  const S = window.__caseSystem;
  const sel = document.getElementById('projectSelect');
  const opt = [...sel.options].find(o => o.value.includes('${pid}'));
  if (!opt) return 'no-option';
  sel.value = opt.value;
  sel.dispatchEvent(new Event('change', { bubbles: true }));
  await new Promise(r => setTimeout(r, 2200));
  const models = await (await fetch('/models/index.json')).json();
  const m = models.models.find(x => /room-real/.test(x.name)) ?? models.models[0];
  if (m) { await S.loadModel(m); await new Promise(r => setTimeout(r, 3000)); }
  return 'ok';
})()
`;
const r1 = await send('Runtime.evaluate', { expression: prep, returnByValue: true, awaitPromise: true });
console.log('准备:', r1.result?.result?.value);

// 截图 1：左侧面板（含现场建模 + 精确测量）
await sleep(1200);
let shot = await send('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) {
  const out = path.join(process.cwd(), 'output', 'verify', 'ui-recon-measure.png');
  writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log('截图1:', out);
}

// 截图 2：打开待测清单对话框
const r2 = await send('Runtime.evaluate', {
  expression: `(async () => { await window.__caseSystem.measurePanel.open(); await new Promise(r=>setTimeout(r,800)); return 'ok'; })()`,
  returnByValue: true, awaitPromise: true,
});
await sleep(800);
shot = await send('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) {
  const out = path.join(process.cwd(), 'output', 'verify', 'ui-measure-dialog.png');
  writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log('截图2:', out);
}

// 截图 3：现场建模对话框
const r3 = await send('Runtime.evaluate', {
  expression: `(async () => {
    document.getElementById('dlgMeasure')?.close();
    const P = window.__caseSystem.reconPanel;
    P.openDialog();
    await new Promise(r=>setTimeout(r,600));
    return JSON.stringify({ open: document.getElementById('dlgRecon')?.open });
  })()`,
  returnByValue: true, awaitPromise: true,
});
console.log('建模对话框:', r3.result?.result?.value);
await sleep(700);
shot = await send('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) {
  const out = path.join(process.cwd(), 'output', 'verify', 'ui-recon-dialog.png');
  writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log('截图3:', out);
}

ws.close();
edge.kill();
process.exit(0);
