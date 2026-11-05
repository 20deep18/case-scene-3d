/**
 * 探针二：按「真实用户操作序列」点击，定位「取消关不掉窗口」的真实触发条件。
 *
 * 覆盖三条路径：
 *   A. 左栏「开始建模」按钮 → 打开对话框 → 点对话框里的「取消」（真实鼠标事件）
 *   B. 建模进行中（伪造 running 任务）→ 点状态区「取消」（走 window.confirm 的那条）
 *   C. 打开建模日志对话框 → 关闭
 *
 * 用法: node tests/probe_recon_cancel2.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9262;

const PROFILE = path.join(process.cwd(), '.browsertmp', 'probe-cancel2');
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
if (!page) { console.log('浏览器没起来'); edge.kill(); process.exit(1); }

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
  if (r.result?.exceptionDetails) return `EXC: ${r.result.exceptionDetails.exception?.description}`;
  return r.result?.result?.value;
};
const clickSelector = async (selector) => {
  const box = await evaluate(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return 'null';
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) return 'zero-size:' + JSON.stringify({x:r.x,y:r.y,w:r.width,h:r.height});
    return JSON.stringify({ x: r.x + r.width / 2, y: r.y + r.height / 2 });
  })()`);
  if (typeof box !== 'string' || !box.startsWith('{')) return `无法点击 ${selector}（${box}）`;
  const p = JSON.parse(box);
  for (const type of ['mouseMoved', 'mousePressed', 'mouseReleased']) {
    await send('Input.dispatchMouseEvent', { type, x: p.x, y: p.y, button: 'left', clickCount: 1 });
    await sleep(50);
  }
  return `已点击 ${selector} @ ${Math.round(p.x)},${Math.round(p.y)}`;
};
await send('Runtime.enable');
await send('Page.enable');

for (let i = 0; i < 40; i++) {
  const ok = await evaluate('String(!!(window.__caseSystem && window.__caseSystem.reconPanel))');
  if (ok === 'true') break;
  await sleep(500);
}
await sleep(1500);

console.log('=== A. 真按钮「开始建模」→ 对话框「取消」 ===');
console.log(await clickSelector('#btnRecon'));
await sleep(500);
console.log('点后:', await evaluate(`JSON.stringify({ open: document.getElementById('dlgRecon').open })`));
console.log(await clickSelector('#dlgRecon menu button[value="cancel"]'));
await sleep(400);
console.log('点取消后:', await evaluate(`JSON.stringify({ open: document.getElementById('dlgRecon').open, rv: document.getElementById('dlgRecon').returnValue })`));

console.log('\n=== A2. 再开一次，用键盘 ESC ===');
console.log(await clickSelector('#btnRecon'));
await sleep(400);
await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
await sleep(400);
console.log('ESC 后:', await evaluate(`JSON.stringify({ open: document.getElementById('dlgRecon').open })`));

console.log('\n=== B. 建模进行中：状态区「取消」按钮 ===');
console.log(await evaluate(`(() => {
  const P = window.__caseSystem.reconPanel;
  P.jobId = 'recon-fake-0001';
  document.getElementById('reconStatus').classList.remove('hidden');
  document.getElementById('reconStage').textContent = '高斯泼溅训练';
  document.getElementById('reconPct').textContent = '42%';
  document.getElementById('btnReconCancel').disabled = false;
  // 统计 window.confirm 是否被调用（无头环境下原生弹窗会被自动关闭）
  window.__confirmCalls = [];
  const orig = window.confirm;
  window.confirm = (msg) => { window.__confirmCalls.push(String(msg)); return false; };
  window.__origConfirm = orig;
  return 'prepared';
})()`));
console.log(await clickSelector('#btnReconCancel'));
await sleep(500);
console.log('点状态区取消后:', await evaluate(`JSON.stringify({
  confirmCalls: window.__confirmCalls,
  statusHidden: document.getElementById('reconStatus').classList.contains('hidden'),
  jobId: window.__caseSystem.reconPanel.jobId,
  toasts: [...document.querySelectorAll('#toast')].map(t => t.textContent),
})`));

console.log('\n=== B2. 若原生 confirm 被环境屏蔽（返回 undefined / 抛错） ===');
console.log(await evaluate(`(async () => {
  const P = window.__caseSystem.reconPanel;
  P.jobId = 'recon-fake-0002';
  window.confirm = () => { throw new Error('confirm 被环境屏蔽'); };
  let err = null;
  try { await P.cancel(); } catch (e) { err = e.message; }
  return JSON.stringify({ threw: err, statusHidden: document.getElementById('reconStatus').classList.contains('hidden'), jobId: P.jobId });
})()`, true));

console.log('\n=== C. 建模日志对话框 ===');
console.log(await evaluate(`(() => { window.__caseSystem.reconPanel.jobId = 'recon-fake-0003'; return 'jobId set'; })()`));
console.log(await clickSelector('#btnReconLog'));
await sleep(400);
console.log('日志窗:', await evaluate(`JSON.stringify({ open: document.getElementById('dlgReconLog').open })`));
console.log(await clickSelector('#dlgReconLog menu button'));
await sleep(400);
console.log('点关闭后:', await evaluate(`JSON.stringify({ open: document.getElementById('dlgReconLog').open })`));

console.log('\n=== 控制台错误 ===');
console.log(errors.slice(0, 8));

ws.close();
edge.kill();
