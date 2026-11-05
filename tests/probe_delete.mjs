/**
 * 诊断：为什么「删除标注 / 删除导入的视频图片」没反应？
 * 打印完整事件链路与网络请求结果。
 * 用法: node tests/probe_delete.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9281;
const PROFILE = path.join(process.cwd(), '.browsertmp', 'probe-del');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

const api = async (p, opts = {}) => {
  const r = await fetch(BASE + p, {
    method: opts.method ?? 'GET',
    headers: opts.body ? { 'content-type': 'application/json' } : {},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  return { status: r.status, data: await r.json().catch(() => ({ raw: 'non-json' })) };
};

const proj = (await api('/api/projects', { method: 'POST', body: { name: 'probe-delete' } })).data.project;
const pid = proj.id;

// 造 3 类会被用户删除的对象
const point = await api(`/api/projects/${pid}/annotations`, {
  method: 'POST',
  body: { annotation: { kind: 'point', label: '物证点A', points: [[0, 1, 0]] } },
});
const measure = await api(`/api/projects/${pid}/annotations`, {
  method: 'POST',
  body: { annotation: { kind: 'measure', label: '测距B', points: [[0, 1, 0], [1, 1, 0]] } },
});
// 一条带 media 的相机标注（模拟"导入的视频/图片"）
const camera = await api(`/api/projects/${pid}/annotations`, {
  method: 'POST',
  body: {
    annotation: {
      kind: 'camera', label: '现场照片', points: [[0, 1.5, -4]],
      pose: { position: [0, 1.5, -4], quaternion: [0, 0, 0, 1], fov: 60, source: 'default' },
      media: [{ url: `/media/${pid}/media/a.png`, kind: 'image', name: 'a.png' }],
    },
  },
});
console.log('已造 3 个标注: point=%s measure=%s camera=%s',
  point.status, measure.status, camera.status);

// 先直接测 API 能不能删
const delDirect = await api(`/api/projects/${pid}/annotations/${measure.data.annotation.id}`, { method: 'DELETE' });
console.log('直接调 DELETE API:', delDirect.status, JSON.stringify(delDirect.data).slice(0, 120));
const after = await api(`/api/projects/${pid}`);
console.log('删除后剩余标注数:', after.data.project.annotations.length);

// 重建一条用于 UI 测试
const m2 = await api(`/api/projects/${pid}/annotations`, {
  method: 'POST',
  body: { annotation: { kind: 'measure', label: '测距C', points: [[0, 1, 0], [1, 1, 0]] } },
});

// ---------------------------------------------------------------- 浏览器
const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1400,900', BASE,
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
const logs = [];
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Runtime.consoleAPICalled') {
    logs.push(`[${m.params.type}] ` + m.params.args.map((a) => a.value ?? a.description ?? '').join(' ').slice(0, 180));
  }
  if (m.method === 'Runtime.exceptionThrown') {
    logs.push('[异常] ' + (m.params.exceptionDetails?.exception?.description ?? '?').slice(0, 200));
  }
  if (m.method === 'Network.requestWillBeSent') {
    const u = m.params.request.url;
    if (u.includes('/api/')) logs.push(`[请求] ${m.params.request.method} ${u.replace(BASE, '')}`);
  }
  if (m.method === 'Network.responseReceived') {
    const u = m.params.response.url;
    if (u.includes('/api/')) logs.push(`[响应] ${m.params.response.status} ${u.replace(BASE, '')}`);
  }
};
const send = (method, params = {}) => new Promise((res) => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
await send('Runtime.enable');
await send('Network.enable');

let booted = false;
for (let i = 0; i < 40 && !booted; i++) {
  const r = await send('Runtime.evaluate', { expression: 'String(!!(window.__caseSystem && window.__caseSystem.ui))', returnByValue: true });
  booted = r.result?.result?.value === 'true';
  if (!booted) await sleep(500);
}
console.log('页面已启动:', booted);
await sleep(1500);

// 切案件
await send('Runtime.evaluate', {
  expression: `(async () => {
    const sel = document.getElementById('projectSelect');
    const opt = [...sel.options].find(o => o.value.includes('${pid}'));
    if (opt) { sel.value = opt.value; sel.dispatchEvent(new Event('change', { bubbles: true })); }
    await new Promise(r => setTimeout(r, 2200));
    return 'ok';
  })()`,
  returnByValue: true, awaitPromise: true,
});

// 检查 DOM 与事件绑定情况
const info = await send('Runtime.evaluate', {
  expression: `(async () => {
    const S = window.__caseSystem;
    const items = [...document.querySelectorAll('.anno-item')];
    const delBtns = [...document.querySelectorAll('.anno-del')];
    const ui = S.ui;
    return JSON.stringify({
      annotations: (S.state.project?.annotations ?? []).length,
      items: items.length,
      delBtns: delBtns.length,
      handlersOnUI: Object.keys(ui.handlers ?? {}),
      hasOnDelete: typeof ui.handlers?.onDeleteAnnotation,
      hasOnSelect: typeof ui.handlers?.onSelectAnnotation,
      deleteAnnotationType: typeof S.ui?.handlers?.onDeleteAnnotation,
      firstDelBtnHTML: delBtns[0]?.outerHTML ?? null,
    });
  })()`,
  returnByValue: true, awaitPromise: true,
});
console.log('\n=== DOM / 处理器情况 ===');
console.log(info.result?.result?.value);

// 模拟点击 ✕（先覆盖 window.confirm 让它返回 true）
const clickDel = await send('Runtime.evaluate', {
  expression: `(async () => {
    const before = (window.__caseSystem.state.project.annotations ?? []).length;
    window.confirm = () => true;   // 强制确认
    const btn = document.querySelector('.anno-del');
    if (!btn) return JSON.stringify({ error: '找不到 .anno-del 按钮' });
    const id = btn.dataset.del;
    btn.click();
    await new Promise(r => setTimeout(r, 2000));
    const after = (window.__caseSystem.state.project.annotations ?? []).length;
    return JSON.stringify({ id, before, after, changed: before !== after });
  })()`,
  returnByValue: true, awaitPromise: true,
});
console.log('\n=== 点击 ✕ 结果 ===');
console.log(clickDel.result?.result?.value);

// 模拟点击整个列表项（应触发选中，不是删除）
const clickItem = await send('Runtime.evaluate', {
  expression: `(async () => {
    const node = document.querySelector('.anno-item');
    if (!node) return 'no-item';
    const before = window.__caseSystem.state.selectedId;
    node.click();
    await new Promise(r => setTimeout(r, 800));
    return JSON.stringify({ before, after: window.__caseSystem.state.selectedId });
  })()`,
  returnByValue: true, awaitPromise: true,
});
console.log('\n=== 点击列表项结果（应选中）===');
console.log(clickItem.result?.result?.value);

console.log('\n=== 页面日志（含网络请求）===');
for (const l of logs.slice(0, 40)) console.log('  ' + l);

await api(`/api/projects/${pid}`, { method: 'DELETE' });
ws.close();
edge.kill();
process.exit(0);
