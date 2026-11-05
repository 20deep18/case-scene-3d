/**
 * 真机复现：启动**真实的**建模任务 → 用真实鼠标点进度条下的「取消」→ 看服务器到底收没收到。
 *
 * 背景：用户反馈「开始建模后显示的小进度条下的取消选项，点击后没有反应」，
 * 而服务端日志里**没有** `[系统] 已请求取消任务` —— 说明请求根本没发出去。
 * 这个脚本就是用来判定：到底是前端点击链路坏了，还是页面跑的是旧 JS。
 *
 * 用法: node tests/browser_recon_cancel_live.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, linkSync, existsSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9264;

let pass = 0;
let fail = 0;
function check(name, ok, detail = '') {
  if (ok) { pass++; console.log(`  [OK]   ${name}${detail ? `  ${detail}` : ''}`); }
  else { fail++; console.log(`  [FAIL] ${name}${detail ? `  ${detail}` : ''}`); }
}
const enc = encodeURIComponent;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- 用硬链接把真实大视频挂进案件 media（同盘，不占额外空间、瞬间完成）
const SRC = path.join(process.cwd(), '测试视频（较大型）.mp4');
if (!existsSync(SRC)) { console.log('缺少测试视频'); process.exit(1); }
const proj = await (await fetch(`${BASE}/api/projects`, {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ name: 'recon-cancel-live', operator: 'test' }),
})).json();
const pid = proj.project.id;
const mediaDir = path.join(process.cwd(), 'data', 'projects', pid, 'media');
mkdirSync(mediaDir, { recursive: true });
const name = '现场勘查视频.mp4';
try { linkSync(SRC, path.join(mediaDir, name)); }
catch { const { copyFileSync } = await import('node:fs'); copyFileSync(SRC, path.join(mediaDir, name)); }
await fetch(`${BASE}/api/projects/${enc(pid)}/annotations`, {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({
    annotation: {
      kind: 'camera', label: '现场视频', points: [[0, 1.5, -4]],
      pose: { position: [0, 1.5, -4], quaternion: [0, 0, 0, 1], fov: 60, source: 'default' },
      media: [{ url: `/media/${enc(pid)}/media/${enc(name)}`, name, kind: 'video' }],
    },
    operator: 'test',
  }),
});
console.log(`测试案件 ${pid}（素材 112 MB，足够过体积门槛）`);

// ---- 浏览器
const PROFILE = path.join(process.cwd(), '.browsertmp', 'cancel-live');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });
const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1400,900', BASE,
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
const consoleErrors = [];
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') consoleErrors.push(m.params.exceptionDetails?.exception?.description ?? '?');
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
    consoleErrors.push(m.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
  }
};
const send = (method, params = {}) =>
  new Promise((res) => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression, awaitPromise = false) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise });
  if (r.result?.exceptionDetails) return `EXC: ${r.result.exceptionDetails.exception?.description}`;
  return r.result?.result?.value;
};
const json = async (e, a = false) => { try { return JSON.parse(await evaluate(e, a)); } catch { return {}; } };
const clickSelector = async (selector) => {
  const raw = await evaluate(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return 'null';
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) return 'zero';
    return JSON.stringify({ x: r.x + r.width / 2, y: r.y + r.height / 2 });
  })()`);
  if (typeof raw !== 'string' || !raw.startsWith('{')) return `无法点击(${raw})`;
  const p = JSON.parse(raw);
  for (const type of ['mouseMoved', 'mousePressed', 'mouseReleased']) {
    await send('Input.dispatchMouseEvent', { type, x: p.x, y: p.y, button: 'left', clickCount: 1 });
    await sleep(50);
  }
  return 'ok';
};
await send('Runtime.enable');
for (let i = 0; i < 40; i++) {
  if ((await evaluate('String(!!(window.__caseSystem && window.__caseSystem.reconPanel))')) === 'true') break;
  await sleep(500);
}
await sleep(1500);
await evaluate(`(async () => {
  const sel = document.getElementById('projectSelect');
  const opt = [...sel.options].find(o => o.value.includes(${JSON.stringify(pid)}));
  if (opt) { sel.value = opt.value; sel.dispatchEvent(new Event('change', { bubbles: true })); }
  await new Promise(r => setTimeout(r, 2000));
  return 'ok';
})()`, true);

// 记录页面加载的是哪一版前端：新版 cancel() 不会调用 window.confirm
await evaluate(`(() => {
  window.__nativeConfirms = [];
  const orig = window.confirm.bind(window);
  window.confirm = (m) => { window.__nativeConfirms.push(String(m)); return orig(m); };
  window.__cancelPosts = [];
  const real = window.fetch.bind(window);
  window.fetch = (u, i) => {
    const s = String(u);
    if (/\\/cancel$/.test(s)) window.__cancelPosts.push({ url: s, method: String(i?.method ?? 'GET') });
    return real(u, i);
  };
  return 'wrapped';
})()`);

console.log('\n--- 1. 走真实用户流程启动建模 ---');
await clickSelector('#btnRecon');
await sleep(1500);
// 体积充足（112MB）→ 不应有确认框；若有就点「仍然开始建模」
const gate = await json(`JSON.stringify({ open: document.getElementById('dlgConfirm').open })`);
if (gate.open) { console.log('  （体积不足，替用户点「仍然开始建模」）'); await clickSelector('#cfOk'); await sleep(600); }
await clickSelector('#btnReconStart');
console.log('  已点「开始建模」，等待后端创建任务…');

let jobId = null;
for (let i = 0; i < 40; i++) {
  await sleep(500);
  const st = await json(`JSON.stringify({ jobId: window.__caseSystem.reconPanel.jobId, disabled: document.getElementById('btnReconCancel')?.disabled, hidden: document.getElementById('reconStatus')?.classList.contains('hidden'), stage: document.getElementById('reconStage')?.textContent })`);
  if (st.jobId) { jobId = st.jobId; console.log(`  任务已创建：${jobId}  阶段=${st.stage}  取消按钮disabled=${st.disabled}  进度区隐藏=${st.hidden}`); break; }
}
check('真实任务已在后端创建', typeof jobId === 'string' && jobId.length > 0, jobId ?? '');
check('进度条下的「取消」按钮可用（不是 disabled）',
  (await evaluate('String(document.getElementById("btnReconCancel").disabled)')) === 'false');

console.log('\n--- 2. 真实鼠标点「取消」 ---');
const clickResult = await clickSelector('#btnReconCancel');
console.log('  点击结果:', clickResult);
await sleep(900);
const afterClick = await json(`JSON.stringify({
  confirmOpen: document.getElementById('dlgConfirm').open,
  confirmTitle: document.getElementById('cfTitle').textContent,
  nativeConfirms: window.__nativeConfirms,
  cancelPosts: window.__cancelPosts,
  toast: document.getElementById('toast').textContent,
})`);
check('点击后有反应（弹出自定义确认框，而不是「毫无反应」）', afterClick.confirmOpen === true,
  `title=${afterClick.confirmTitle} toast=${afterClick.toast}`);
check('新版前端不再调用 window.confirm', (afterClick.nativeConfirms ?? []).length === 0,
  JSON.stringify(afterClick.nativeConfirms));

if (afterClick.confirmOpen) {
  console.log('\n--- 3. 点「确定取消任务」 ---');
  await clickSelector('#cfOk');
  await sleep(1500);
  const posted = await json(`JSON.stringify({ cancelPosts: window.__cancelPosts, toast: document.getElementById('toast').textContent })`);
  check('取消请求真的发给了服务器', (posted.cancelPosts ?? []).length >= 1, JSON.stringify(posted.cancelPosts));
  check('界面给出反馈', /取消/.test(posted.toast ?? ''), posted.toast);

  // 服务器侧确认：任务应变成「已取消」
  let serverJob = null;
  for (let i = 0; i < 20; i++) {
    await sleep(700);
    const r = await fetch(`${BASE}/api/projects/${enc(pid)}/recon/${enc(jobId)}`).then((x) => x.json()).catch(() => ({}));
    serverJob = r.job ?? null;
    if (serverJob?.finishedAt) break;
  }
  check('服务器侧任务已结束且原因是「取消」',
    Boolean(serverJob?.finishedAt) && /取消/.test(serverJob?.error ?? ''),
    `stage=${serverJob?.stage} error=${serverJob?.error}`);

  // 直接看流水线日志里有没有那句
  const log = await fetch(`${BASE}/api/projects/${enc(pid)}/recon/${enc(jobId)}/log?from=0&tail=400`).then((x) => x.json()).catch(() => ({}));
  const text = (log.lines ?? []).join('\n');
  check('流水线日志里出现「已请求取消任务」', /已请求取消任务/.test(text),
    /已请求取消任务/.test(text) ? '✓' : text.split('\n').filter(Boolean).slice(-2).join(' | '));
}

const noise = consoleErrors.filter((m) => !/Download the React|DevTools|setPointerCapture|favicon/.test(m));
check('控制台无 error 输出', noise.length === 0, noise.slice(0, 2).join(' | '));

await fetch(`${BASE}/api/projects/${enc(pid)}`, { method: 'DELETE' });
ws.close();
edge.kill();
console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail === 0 ? 0 : 1);
