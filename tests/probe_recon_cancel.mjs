/**
 * 诊断探针：现场建模对话框里点「取消」为什么关不掉？
 *
 * 做法：
 *   1) 起无头 Edge，连 CDP；
 *   2) 打开 #dlgRecon；
 *   3) 分别用「程序化 .click()」和「真实鼠标事件 Input.dispatchMouseEvent」点取消；
 *   4) 每次记录 dialog.open / 表单 submit 是否触发 / 是否有异常。
 *
 * 用法: node tests/probe_recon_cancel.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9261;

const PROFILE = path.join(process.cwd(), '.browsertmp', 'probe-cancel');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

const edge = spawn(
  EDGE,
  [
    '--headless=new', '--no-sandbox', '--disable-crash-reporter',
    '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
    `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
    '--window-size=1400,900', BASE,
  ],
  { stdio: 'ignore' }
);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let page = null;
for (let i = 0; i < 40 && !page; i++) {
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
  } catch { /* retry */ }
  if (!page) await sleep(400);
}
if (!page) {
  console.log('浏览器没起来');
  edge.kill();
  process.exit(1);
}

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const errors = [];
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails?.exception?.description ?? '?');
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
    errors.push(m.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
  }
};
const send = (method, params = {}) =>
  new Promise((res) => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression, awaitPromise = false) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise });
  if (r.result?.exceptionDetails) return { __error: r.result.exceptionDetails.exception?.description };
  return r.result?.result?.value;
};
await send('Runtime.enable');

for (let i = 0; i < 40; i++) {
  const ok = await evaluate('String(!!(window.__caseSystem && window.__caseSystem.reconPanel))');
  if (ok === 'true') break;
  await sleep(500);
}
await sleep(1200);

console.log('--- 打开对话框 ---');
console.log(await evaluate(`(() => {
  const P = window.__caseSystem.reconPanel;
  P.openDialog();
  const dlg = document.getElementById('dlgRecon');
  const form = dlg.querySelector('form');
  window.__probe = { submits: 0, cancels: 0, closes: 0 };
  form.addEventListener('submit', () => { window.__probe.submits++; });
  form.addEventListener('cancel', () => { window.__probe.cancels++; });
  dlg.addEventListener('close', () => { window.__probe.closes++; });
  const btn = [...dlg.querySelectorAll('button')].map(b => ({ text: b.textContent.trim(), type: b.type, value: b.value, form: !!b.form, name: b.name }));
  return JSON.stringify({
    open: dlg.open, method: form.method, buttons: btn,
    cancelButtonMatches: [...dlg.querySelectorAll('button')].filter(b => b.textContent.trim() === '取消').length,
  }, null, 1);
})()`));

console.log('--- 方式 A：程序化 .click() ---');
console.log(await evaluate(`(() => {
  const dlg = document.getElementById('dlgRecon');
  const btn = [...dlg.querySelectorAll('button')].find(b => b.textContent.trim() === '取消');
  if (!btn) return '找不到取消按钮';
  btn.click();
  return JSON.stringify({ open: dlg.open, returnValue: dlg.returnValue, probe: window.__probe });
})()`));
await sleep(300);

console.log('--- 重新打开（若已关闭） ---');
console.log(await evaluate(`(() => {
  const dlg = document.getElementById('dlgRecon');
  if (!dlg.open) { window.__caseSystem.reconPanel.openDialog(); window.__probe = { submits: 0, cancels: 0, closes: 0 };
    const form = dlg.querySelector('form');
    form.addEventListener('submit', () => { window.__probe.submits++; });
    form.addEventListener('cancel', () => { window.__probe.cancels++; });
    dlg.addEventListener('close', () => { window.__probe.closes++; });
  }
  return JSON.stringify({ open: dlg.open });
})()`));
await sleep(300);

console.log('--- 方式 B：真实鼠标事件 ---');
const rect = await evaluate(`(() => {
  const dlg = document.getElementById('dlgRecon');
  const btn = [...dlg.querySelectorAll('button')].find(b => b.textContent.trim() === '取消');
  const r = btn.getBoundingClientRect();
  return JSON.stringify({ x: r.x + r.width / 2, y: r.y + r.height / 2, w: r.width, h: r.height, top: r.top, left: r.left });
})()`);
console.log('取消按钮位置:', rect);
const pos = JSON.parse(rect);
for (const type of ['mousePressed', 'mouseReleased']) {
  await send('Input.dispatchMouseEvent', { type, x: pos.x, y: pos.y, button: 'left', clickCount: 1 });
  await sleep(60);
}
await sleep(400);
console.log(await evaluate(`(() => {
  const dlg = document.getElementById('dlgRecon');
  return JSON.stringify({ open: dlg.open, returnValue: dlg.returnValue, probe: window.__probe,
    activeElement: document.activeElement?.id || document.activeElement?.tagName });
})()`));

console.log('--- 方式 C：直接调 form.requestSubmit(btn) ---');
console.log(await evaluate(`(() => {
  const dlg = document.getElementById('dlgRecon');
  if (!dlg.open) return JSON.stringify({ skipped: true, open: dlg.open });
  const form = dlg.querySelector('form');
  const btn = [...dlg.querySelectorAll('button')].find(b => b.textContent.trim() === '取消');
  try { form.requestSubmit(btn); } catch (e) { return 'requestSubmit 抛错: ' + e.message; }
  return JSON.stringify({ open: dlg.open, returnValue: dlg.returnValue, probe: window.__probe });
})()`));
await sleep(300);

console.log('--- 方式 D：dlg.close() 直接调用 ---');
console.log(await evaluate(`(() => {
  const dlg = document.getElementById('dlgRecon');
  dlg.close('cancel');
  return JSON.stringify({ open: dlg.open, returnValue: dlg.returnValue,
    computedDisplay: getComputedStyle(dlg).display, probe: window.__probe });
})()`));

console.log('--- 表单与按钮的完整 HTML ---');
console.log(await evaluate(`(() => {
  const dlg = document.getElementById('dlgRecon');
  return dlg.outerHTML.replace(/\\s+/g, ' ').slice(0, 800);
})()`));

console.log('--- 控制台错误 ---');
console.log(errors.slice(0, 6));

ws.close();
edge.kill();
