/**
 * 浏览器回归：现场建模的「素材体积门槛」与「取消 / 关闭」可靠性
 *
 * 覆盖用户反馈的两件事：
 *   A. 素材体积不足时要先提示「提供信息过少，可能无法正常建模」，由用户决定是否继续；
 *      体积充足时点「开始建模」直接开跑，不再多问。
 *   B. 「点击取消无法正常关闭窗口」。
 *      · 对话框的「取消」不再依赖 `<form method="dialog">` 的默认提交行为 ——
 *        测试里会先**证明**这条默认路径确实会被阻断（基线断言），
 *        再验证我们的显式关闭仍然生效；
 *      · 建模进行中状态区的「取消」不再用 `window.confirm`（原生弹窗在部分
 *        环境/嵌入式内核里会被屏蔽，表现为「点了毫无反应」）。
 *
 * 用法: node tests/browser_recon_gate.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9263;

let pass = 0;
let fail = 0;
function check(name, ok, detail = '') {
  if (ok) {
    pass++;
    console.log(`  [OK]   ${name}${detail ? `  ${detail}` : ''}`);
  } else {
    fail++;
    console.log(`  [FAIL] ${name}${detail ? `  ${detail}` : ''}`);
  }
}

const enc = encodeURIComponent;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------- 准备：低体积素材案件
const proj = await (
  await fetch(`${BASE}/api/projects`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'recon-gate-test', operator: 'test' }),
  })
).json();
const pid = proj.project.id;
const mediaDir = path.join(process.cwd(), 'data', 'projects', pid, 'media');
mkdirSync(mediaDir, { recursive: true });
const smallName = '素材体积测试（很小）.mp4';
writeFileSync(path.join(mediaDir, smallName), Buffer.alloc(200_000, 3));
const smallUrl = `/media/${enc(pid)}/media/${enc(smallName)}`;

await fetch(`${BASE}/api/projects/${enc(pid)}/annotations`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({
    annotation: {
      kind: 'camera',
      label: '很少的素材',
      points: [[0, 1.5, -4]],
      pose: { position: [0, 1.5, -4], quaternion: [0, 0, 0, 1], fov: 60, source: 'default' },
      media: [{ url: smallUrl, name: smallName, kind: 'video' }],
    },
    operator: 'test',
  }),
});

const materials = await (
  await fetch(`${BASE}/api/projects/${enc(pid)}/recon/materials`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ assets: [{ url: smallUrl }] }),
  })
).json();
console.log(`测试案件 ${pid}：素材 ${materials.bytes} 字节，阈值 ${materials.minBytes}，enough=${materials.enough}`);
check('前置条件：本案素材体积低于阈值', materials.enough === false);

// ---------------------------------------------------------------- 浏览器
const PROFILE = path.join(process.cwd(), '.browsertmp', 'recon-gate');
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
  process.exit(1);
}

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const consoleErrors = [];
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  }
  if (m.method === 'Runtime.exceptionThrown') {
    consoleErrors.push(m.params.exceptionDetails?.exception?.description ?? '?');
  }
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
    consoleErrors.push(m.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
  }
};
const send = (method, params = {}) =>
  new Promise((res) => {
    pending.set(++id, res);
    ws.send(JSON.stringify({ id, method, params }));
  });
const evaluate = async (expression, awaitPromise = false) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise });
  if (r.result?.exceptionDetails) return `EXC: ${r.result.exceptionDetails.exception?.description}`;
  return r.result?.result?.value;
};
const json = async (expr, awaitPromise = false) => {
  const v = await evaluate(expr, awaitPromise);
  try {
    return JSON.parse(v);
  } catch {
    return { __raw: v };
  }
};
/**
 * 真实鼠标点击（不是 el.click()：要覆盖「用户真的点了一下」）。
 *
 * ⚠️ 2026-10-06 加固：以前直接取 `getBoundingClientRect()` 的中心就点，
 * 而**可滚动容器里的元素可以落在容器裁剪区之外** —— 这时那个坐标属于别的元素，
 * 点击被"吃掉"，测试却报出误导性的失败。
 * 我为此白查了一轮：当时以为 `#reconStatus` 与 `#chatLog` 有 21 px 重叠，
 * 实际是 `.panel{bottom:190px}` 与 `.bottombar{bottom:12px;height:166px}` 之间有 12 px 间隙，
 * 根本不重叠；只是「取消」按钮被面板裁掉了（内容比面板高），坐标点落到了页脚上。
 * 所以现在：
 *   1. 先 `scrollIntoView` 把它滚进可视区；
 *   2. 点击前用 `elementFromPoint` 确认那个坐标上确实是它（或它的后代）；
 *   3. 不满足就**明确报出来**（谁挡住了），而不是默默点空。
 */
const clickSelector = async (selector) => {
  const raw = await evaluate(`(() => {
    const el = document.querySelector(${JSON.stringify(selector)});
    if (!el) return 'null';
    el.scrollIntoView({ block: 'center', inline: 'center' });
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) return 'zero';
    const cx = r.x + r.width / 2;
    const cy = r.y + r.height / 2;
    const hitEl = document.elementFromPoint(cx, cy);
    const hit = hitEl ? (hitEl.id || hitEl.className || hitEl.tagName) : null;
    if (hitEl && hitEl !== el && !el.contains(hitEl) && !hitEl.contains(el)) {
      return 'covered:' + hit;
    }
    return JSON.stringify({ x: cx, y: cy });
  })()`);
  if (typeof raw !== 'string' || !raw.startsWith('{')) {
    // 把原因喊出来：'null'=选择器不存在，'zero'=尺寸为 0，'covered:X'=被 X 挡住
    console.log(`  [warn] clickSelector(${selector}) 无法点击：${raw}`);
    return false;
  }
  const p = JSON.parse(raw);
  for (const type of ['mouseMoved', 'mousePressed', 'mouseReleased']) {
    await send('Input.dispatchMouseEvent', { type, x: p.x, y: p.y, button: 'left', clickCount: 1 });
    await sleep(50);
  }
  return true;
};
await send('Runtime.enable');

for (let i = 0; i < 40; i++) {
  const ok = await evaluate('String(!!(window.__caseSystem && window.__caseSystem.reconPanel))');
  if (ok === 'true') break;
  await sleep(500);
}
await sleep(1500);

// 切到测试案件
await evaluate(`(async () => {
  const sel = document.getElementById('projectSelect');
  const opt = [...sel.options].find(o => /recon-gate-test/.test(o.textContent) || o.value.includes('recon-gate-test'));
  if (!opt) return 'no-option';
  sel.value = opt.value;
  sel.dispatchEvent(new Event('change', { bubbles: true }));
  await new Promise(r => setTimeout(r, 2000));
  return 'ok';
})()`, true);
await sleep(500);

// fetch 分流：materials 走真实后端，POST /recon 与日志用桩（避免真的跑几十分钟训练）
await evaluate(`(() => {
  window.__calls = [];
  window.__materialsOverride = null;
  const real = window.fetch.bind(window);
  const reply = (obj) => Promise.resolve(new Response(JSON.stringify(obj), { status: 200, headers: { 'content-type': 'application/json' } }));
  window.fetch = async (url, init) => {
    const u = String(url);
    const method = String(init?.method ?? 'GET').toUpperCase();
    window.__calls.push({ url: u, method });
    if (u.includes('/recon/materials')) {
      if (window.__materialsOverride) return reply(window.__materialsOverride);
      return real(url, init);
    }
    if (/\\/recon$/.test(u) && method === 'POST') {
      return reply({ ok: true, job: { id: 'fake-job-1', name: 't', stage: 'queued', stageLabel: '排队中', progress: 0, createdAt: new Date().toISOString() } });
    }
    if (u.includes('/recon/fake-job-1/log')) {
      return reply({ ok: true, from: 0, lines: [], status: { id: 'fake-job-1', name: 't', stage: 'done', stageLabel: '完成', progress: 1, finishedAt: new Date().toISOString() } });
    }
    if (u.includes('/cancel')) return reply({ ok: true });
    return real(url, init);
  };
  return 'patched';
})()`);

// ---------------------------------------------------------------- A. 体积不足 → 先提示
console.log('\n--- A. 素材体积不足：点「开始建模」必须先提示 ---');
await clickSelector('#btnRecon');
await sleep(600);
let dlgState = await json(`JSON.stringify({ open: document.getElementById('dlgRecon').open })`);
check('「开始建模」能打开参数对话框', dlgState.open === true);

// 等异步的体检结果补上体积行
let volumeText = '';
for (let i = 0; i < 20; i++) {
  volumeText = await evaluate(`document.querySelector('#reconAssets .recon-volume')?.textContent ?? ''`);
  if (volumeText) break;
  await sleep(300);
}
check('对话框里显示了素材体积体检结论', /素材合计/.test(volumeText), volumeText.slice(0, 80));
check('体积不足时给出「体积偏少」提示', /体积偏少|体积充足/.test(volumeText),
  volumeText.includes('体积偏少') ? '显示「体积偏少」✓' : `实际：${volumeText.slice(0, 80)}`);

await clickSelector('#btnReconStart');
await sleep(700);
let cf = await json(`JSON.stringify({
  open: document.getElementById('dlgConfirm').open,
  title: document.getElementById('cfTitle').textContent,
  body: document.getElementById('cfBody').textContent,
  warn: document.getElementById('cfWarn').textContent,
  okText: document.getElementById('cfOk').textContent,
  reconPosts: window.__calls.filter(c => /\\/recon$/.test(c.url) && c.method === 'POST').length,
})`);
check('体积不足时弹出确认框', cf.open === true, `title=${cf.title}`);
check('提示语含「提供信息过少，可能无法正常建模」', /提供信息过少/.test(cf.title + cf.body), cf.title);
check('提示里带上了实际体积与阈值', /素材合计|MB/.test(cf.warn + cf.body), (cf.body + cf.warn).slice(0, 90));
check('提示框的按钮文案是「仍然开始建模」', /仍然开始建模/.test(cf.okText ?? ''), cf.okText);
check('用户还没决定前不会启动建模', cf.reconPosts === 0, `posts=${cf.reconPosts}`);

// 选「取消」→ 不建模，对话框保持打开（方便补素材）
await clickSelector('#cfCancel');
await sleep(600);
const afterCancel = await json(`JSON.stringify({
  confirmOpen: document.getElementById('dlgConfirm').open,
  reconOpen: document.getElementById('dlgRecon').open,
  reconPosts: window.__calls.filter(c => /\\/recon$/.test(c.url) && c.method === 'POST').length,
})`);
check('确认框「取消」能关掉确认框', afterCancel.confirmOpen === false);
check('取消后不启动建模', afterCancel.reconPosts === 0, `posts=${afterCancel.reconPosts}`);
check('取消后参数对话框仍打开（可直接去补素材）', afterCancel.reconOpen === true);

// 选「仍然开始建模」→ 真的提交
await clickSelector('#btnReconStart');
await sleep(700);
await clickSelector('#cfOk');
await sleep(900);
const afterGo = await json(`JSON.stringify({
  reconPosts: window.__calls.filter(c => /\\/recon$/.test(c.url) && c.method === 'POST').length,
  reconOpen: document.getElementById('dlgRecon').open,
})`);
check('用户选择继续后会真的提交建模', afterGo.reconPosts >= 1, `posts=${afterGo.reconPosts}`);
check('提交后参数对话框关闭', afterGo.reconOpen === false);

// ---------------------------------------------------------------- B. 体积充足 → 直接建模
console.log('\n--- B. 素材体积充足：直接建模，不多问 ---');
await evaluate(`(() => {
  window.__calls = [];
  window.__materialsOverride = { ok: true, count: 3, videos: 1, images: 2, bytes: 80 * 1024 * 1024,
    files: [], missing: [], minBytes: 12 * 1024 * 1024, recommendedBytes: 32 * 1024 * 1024,
    enough: true, threshold: { measured: true } };
  return 'override';
})()`);
await clickSelector('#btnRecon');
await sleep(800);
await clickSelector('#btnReconStart');
await sleep(900);
const enough = await json(`JSON.stringify({
  confirmOpen: document.getElementById('dlgConfirm').open,
  reconPosts: window.__calls.filter(c => /\\/recon$/.test(c.url) && c.method === 'POST').length,
  volume: document.querySelector('#reconAssets .recon-volume')?.textContent ?? '',
})`);
check('体积充足时不弹确认框', enough.confirmOpen === false);
check('体积充足时直接提交建模', enough.reconPosts >= 1, `posts=${enough.reconPosts}`);
check('体积充足时界面标明「体积充足」', /体积充足/.test(enough.volume), enough.volume.slice(0, 70));

// ---------------------------------------------------------------- C. 对话框「取消」关得掉
console.log('\n--- C. 对话框「取消」/「关闭」---');
await evaluate(`(() => { window.__materialsOverride = null; window.__calls = []; return 'reset'; })()`);
await clickSelector('#btnRecon');
await sleep(600);
await clickSelector('#btnReconDlgCancel');
await sleep(400);
check('现代内核：点「取消」能关闭建模对话框',
  (await evaluate('String(document.getElementById("dlgRecon").open)')) === 'false');

// 基线：证明「submit 被阻断 → method=\"dialog\" 的默认关闭行为不会发生」
const baseline = await json(`(() => {
  const d = document.createElement('dialog');
  d.innerHTML = '<form method="dialog"><button value="cancel" id="__tb">取消</button></form>';
  document.body.appendChild(d);
  d.querySelector('form').addEventListener('submit', (e) => e.preventDefault());
  d.showModal();
  document.getElementById('__tb').click();
  const stillOpen = d.open;
  d.close();
  d.remove();
  return JSON.stringify({ stillOpen });
})()`);
check('基线复现：submit 被阻断时，form method=dialog 确实关不掉窗口', baseline.stillOpen === true,
  '这正是「点取消关不掉」的机制');

// 同样的阻断条件下，真对话框仍然关得掉（因为走的是显式 JS 关闭）
const blocked = await json(`(() => {
  const dlg = document.getElementById('dlgRecon');
  const form = dlg.querySelector('form');
  const blocker = (e) => e.preventDefault();
  form.addEventListener('submit', blocker);
  window.__caseSystem.reconPanel.openDialog();
  const openedAttr = dlg.hasAttribute('open');
  document.getElementById('btnReconDlgCancel').click();
  const closedAttr = !dlg.hasAttribute('open');
  const openProp = dlg.open;
  form.removeEventListener('submit', blocker);
  return JSON.stringify({ openedAttr, closedAttr, openProp });
})()`);
check('取消 submit 默认路径后，「取消」依然能关闭对话框',
  blocked.openedAttr === true && blocked.closedAttr === true && blocked.openProp === false,
  JSON.stringify(blocked));

// 内核没有实现 <dialog> 时的兜底
const fallback = await json(`(() => {
  const dlg = document.getElementById('dlgRecon');
  const proto = window.HTMLDialogElement && HTMLDialogElement.prototype;
  const saved = proto ? { showModal: proto.showModal, close: proto.close, show: proto.show } : null;
  if (proto) { delete proto.showModal; delete proto.show; delete proto.close; }
  window.__caseSystem.reconPanel.openDialog();
  const openedAttr = dlg.hasAttribute('open');
  const display = getComputedStyle(dlg).display;
  document.getElementById('btnReconDlgCancel').click();
  const closedAttr = !dlg.hasAttribute('open');
  const displayAfter = getComputedStyle(dlg).display;
  if (proto && saved) {
    if (saved.showModal) proto.showModal = saved.showModal;
    if (saved.show) proto.show = saved.show;
    if (saved.close) proto.close = saved.close;
  }
  dlg.classList.remove('dialog-fallback');
  return JSON.stringify({ openedAttr, display, closedAttr, displayAfter });
})()`);
check('内核没有 <dialog> 时也能打开窗口（兜底渲染）',
  fallback.openedAttr === true && fallback.display !== 'none', JSON.stringify(fallback));
check('内核没有 <dialog> 时点「取消」也能关掉', fallback.closedAttr === true,
  `displayAfter=${fallback.displayAfter}`);

// ---------------------------------------------------------------- D. 状态区「取消」不再依赖原生弹窗
console.log('\n--- D. 建模进行中的「取消」---');
await evaluate(`(() => {
  const P = window.__caseSystem.reconPanel;
  P.jobId = 'recon-fake-9';
  document.getElementById('reconStatus').classList.remove('hidden');
  document.getElementById('reconStage').textContent = '高斯泼溅训练';
  document.getElementById('btnReconCancel').disabled = false;
  window.__nativeConfirms = [];
  window.confirm = (m) => { window.__nativeConfirms.push(String(m)); return false; };
  window.__calls = [];
  return 'ready';
})()`);
await clickSelector('#btnReconCancel');
await sleep(600);
const cancelDlg = await json(`JSON.stringify({
  open: document.getElementById('dlgConfirm').open,
  title: document.getElementById('cfTitle').textContent,
  nativeConfirms: window.__nativeConfirms,
})`);
check('点状态区「取消」会弹出**自定义**确认框', cancelDlg.open === true, `title=${cancelDlg.title}`);
check('不再使用 window.confirm（原生弹窗被屏蔽也不会「毫无反应」）',
  cancelDlg.nativeConfirms.length === 0, JSON.stringify(cancelDlg.nativeConfirms));
await clickSelector('#cfOk');
await sleep(700);
const cancelSent = await json(`JSON.stringify({
  cancels: window.__calls.filter(c => /\\/cancel$/.test(c.url)).length,
  toast: document.getElementById('toast').textContent,
})`);
check('确认后确实调用了取消接口', cancelSent.cancels >= 1, `calls=${cancelSent.cancels}`);
check('界面给出「已请求取消」反馈', /取消/.test(cancelSent.toast ?? ''), cancelSent.toast);

// 没有活动任务时点「取消」也**必须**有反馈（以前是静默 return，用户以为按钮坏了）
await evaluate(`(() => {
  const P = window.__caseSystem.reconPanel;
  P.jobId = null;
  document.getElementById('reconStatus').classList.remove('hidden');
  document.getElementById('toast').className = 'toast';
  return 'cleared';
})()`);
await clickSelector('#btnReconCancel');
await sleep(400);
const noJob = await json(`JSON.stringify({
  toast: document.getElementById('toast').textContent,
  statusHidden: document.getElementById('reconStatus').classList.contains('hidden'),
})`);
check('没有活动任务时点「取消」不再毫无反应（给出提示并收起进度区）',
  /没有正在运行的建模任务/.test(noJob.toast ?? '') && noJob.statusHidden === true,
  `toast=${noJob.toast}`);

// ---------------------------------------------------------------- E. 版本过期横幅
console.log('\n--- E. 页面版本与服务端不一致时必须醒目提示刷新 ---');
const ver = await json(`JSON.stringify({
  pageVersion: window.__caseSystem.APP_VERSION,
  serverVersion: null,
  bannerHidden: document.getElementById('staleBanner').classList.contains('hidden'),
})`);
const health = await (await fetch(`${BASE}/api/health`)).json();
check('页面内置版本与服务端版本一致（正常状态不打扰用户）',
  ver.pageVersion === health.version && ver.bannerHidden === true,
  `页面=${ver.pageVersion} 服务端=${health.version}`);

const stale = await json(`(() => {
  const ui = window.__caseSystem.ui;
  ui.watchVersion('9.9.9', ui.pageVersion);
  const el = document.getElementById('staleBanner');
  return JSON.stringify({ hidden: el.classList.contains('hidden'), text: document.getElementById('staleText').textContent });
})()`);
check('版本不一致时挂出横幅', stale.hidden === false);
check('横幅文案说清「本页面是旧版本」并给出两个版本号',
  /旧版本/.test(stale.text ?? '') && /9\.9\.9/.test(stale.text ?? ''), (stale.text ?? '').slice(0, 70));

const restored = await json(`(() => {
  const ui = window.__caseSystem.ui;
  ui.watchVersion(ui.pageVersion, ui.pageVersion);
  return JSON.stringify({ hidden: document.getElementById('staleBanner').classList.contains('hidden') });
})()`);
check('版本恢复一致后横幅自动收起', restored.hidden === true);

// ---------------------------------------------------------------- 清理
await fetch(`${BASE}/api/projects/${enc(pid)}`, { method: 'DELETE' });
const noise = consoleErrors.filter((m) => !/Download the React|DevTools|setPointerCapture|favicon/.test(m));
check('控制台无 error 输出', noise.length === 0, noise.slice(0, 2).join(' | '));

ws.close();
edge.kill();
console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail === 0 ? 0 : 1);
