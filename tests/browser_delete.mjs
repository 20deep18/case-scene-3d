/**
 * 回归：删除功能（标注 + 导入的视频/图片）+ 自定义确认对话框
 *
 * 用户反馈「点叉号 / 删除按钮都没反应」，本测试确保：
 *   1) 删除前一定弹出确认对话框（不再依赖可能被屏蔽的 window.confirm）
 *   2) 点「取消」不会删除
 *   3) 点「确认删除」真的删掉（标注 + 服务器上的媒体文件）
 *   4) 「多源画面」卡片上有删除入口
 *
 * 用法: node tests/browser_delete.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9301;

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
const api = async (p, opts = {}) => {
  const r = await fetch(BASE + p, {
    method: opts.method ?? 'GET',
    headers: opts.body ? { 'content-type': 'application/json' } : {},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  return { status: r.status, data: await r.json().catch(() => ({})) };
};

const proj = (await api('/api/projects', { method: 'POST', body: { name: 'delete-test' } })).data.project;
const pid = proj.id;
console.log(`测试案件 ${pid}`);

// 上传一个真实的媒体文件（用小图），并建一条挂载它的相机标注
const srcImg = path.join(process.cwd(), 'data', 'scene-big', 'images');
let mediaFile = null;
if (existsSync(srcImg)) {
  const all = readFileSync; // 占位，下面用 fs
  const fss = await import('node:fs');
  const list = fss.readdirSync(srcImg).filter((f) => f.endsWith('.png')).sort();
  if (list.length) mediaFile = path.join(srcImg, list[0]);
}
check('找到用于测试的图片素材', !!mediaFile, mediaFile ?? '（缺）');

let mediaUrl = null;
let mediaAbs = null;
if (mediaFile) {
  const fss = await import('node:fs');
  const buf = fss.readFileSync(mediaFile);
  const up = await fetch(`${BASE}/api/projects/${pid}/media?name=待删素材.png&chunk=0&chunks=1`, {
    method: 'POST',
    headers: { 'content-type': 'application/octet-stream' },
    body: buf,
  });
  const upJson = await up.json();
  mediaUrl = upJson.url;
  console.log('媒体已上传:', mediaUrl);
}

const camera = await api(`/api/projects/${pid}/annotations`, {
  method: 'POST',
  body: {
    annotation: {
      kind: 'camera', label: '待删画面', points: [[0, 1.5, -4]],
      pose: { position: [0, 1.5, -4], quaternion: [0, 0, 0, 1], fov: 60, source: 'default' },
      media: mediaUrl ? [{ url: mediaUrl, kind: 'image', name: '待删素材.png' }] : [],
    },
  },
});
const cameraId = camera.data.annotation?.id;
// 再建一条普通标注用于测「列表里的 ✕」
const point = await api(`/api/projects/${pid}/annotations`, {
  method: 'POST',
  body: { annotation: { kind: 'point', label: '待删物证点', points: [[0, 1, 0]] } },
});
const pointId = point.data.annotation?.id;
console.log('已建相机标注与物证点:', !!cameraId, !!pointId);

// 服务器上媒体文件的绝对路径
const mediaDir = path.join(process.cwd(), 'data', 'projects', pid, 'media');
mediaAbs = path.join(mediaDir, '待删素材.png');
check('媒体文件已在服务器落盘', existsSync(mediaAbs), path.basename(mediaAbs));

// ---------------------------------------------------------------- 浏览器
const PROFILE = path.join(process.cwd(), '.browsertmp', 'del');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });
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
const send = (method, params = {}) => new Promise((res) => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
await send('Runtime.enable');

let booted = false;
for (let i = 0; i < 40 && !booted; i++) {
  const r = await send('Runtime.evaluate', { expression: 'String(!!(window.__caseSystem && window.__caseSystem.ui))', returnByValue: true });
  booted = r.result?.result?.value === 'true';
  if (!booted) await sleep(500);
}
check('页面已启动', booted);
await sleep(1500);

await send('Runtime.evaluate', {
  expression: `(async () => {
    const sel = document.getElementById('projectSelect');
    const opt = [...sel.options].find(o => o.value.includes('${pid}'));
    if (opt) { sel.value = opt.value; sel.dispatchEvent(new Event('change', { bubbles: true })); }
    await new Promise(r => setTimeout(r, 2400));
    return 'ok';
  })()`,
  returnByValue: true, awaitPromise: true,
});

// ---- 1) 确认对话框存在
const dlgInfo = JSON.parse((await send('Runtime.evaluate', {
  expression: `JSON.stringify({
    dlg: !!document.getElementById('dlgConfirm'),
    ok: !!document.getElementById('cfOk'),
    cancel: !!document.getElementById('cfCancel'),
  })`,
  returnByValue: true,
})).result?.result?.value ?? '{}');
check('存在自定义确认对话框', dlgInfo.dlg && dlgInfo.ok && dlgInfo.cancel);

// ---- 2) 「多源画面」卡片有删除入口
// 注意：卡片由 ui.renderCameras() 渲染，正常流程是在「导入/载入模型」时触发。
// 测试里直接调它，避免依赖完整启动时序（截图/CSS 与删除逻辑本身无关）。
const cardInfo = JSON.parse((await send('Runtime.evaluate', {
  expression: `(async () => {
    const S = window.__caseSystem;
    const cams = (S.state.project?.annotations ?? []).filter(a => a.kind === 'camera');
    S.ui.renderCameras(cams, null);
    await new Promise(r => setTimeout(r, 300));
    return JSON.stringify({
      cams: cams.length,
      cards: document.querySelectorAll('.cam-card').length,
      delBtn: !!document.querySelector('.cam-card [data-act="delete"]'),
      delText: document.querySelector('.cam-card [data-act="delete"]')?.textContent?.trim() ?? null,
    });
  })()`,
  returnByValue: true, awaitPromise: true,
}))?.result?.result?.value ?? '{}');
check('画面卡片上有「删除」按钮', cardInfo.delBtn === true, `${cardInfo.delText}（相机 ${cardInfo.cams} 个，卡片 ${cardInfo.cards} 张）`);

// ---- 3) 点「取消」不应删除
const cancelTest = await send('Runtime.evaluate', {
  expression: `(async () => {
    const S = window.__caseSystem;
    // 卡片正常由「导入/载入模型」流程渲染；这里手动渲染，避免依赖完整启动时序
    S.ui.renderCameras((S.state.project?.annotations ?? []).filter(a => a.kind === 'camera'), null);
    await new Promise(r => setTimeout(r, 200));
    const before = (S.state.project.annotations ?? []).length;
    const btn = document.querySelector('.cam-card [data-act="delete"]');
    if (!btn) return JSON.stringify({ error: 'no-delete-btn' });
    btn.click();
    await new Promise(r => setTimeout(r, 600));
    const dlg = document.getElementById('dlgConfirm');
    const shown = dlg?.open === true;
    const bodyText = document.getElementById('cfBody')?.textContent ?? '';
    const warnText = document.getElementById('cfWarn')?.textContent ?? '';
    document.getElementById('cfCancel').click();
    await new Promise(r => setTimeout(r, 1200));
    const after = (S.state.project.annotations ?? []).length;
    return JSON.stringify({ before, after, shown, bodyText: bodyText.slice(0,40), warnText: warnText.slice(0,50) });
  })()`,
  returnByValue: true, awaitPromise: true,
});
const ct = JSON.parse(cancelTest.result?.result?.value ?? '{}');
console.log('取消测试:', JSON.stringify(ct));
check('点删除弹出了确认对话框', ct.shown === true);
check('对话框说明了要删除的对象', (ct.bodyText ?? '').length > 0, ct.bodyText ?? '');
check('对话框提示了媒体文件也会被删', /媒体文件|文件/.test(ct.warnText ?? ''), ct.warnText ?? '');
check('点「取消」不会删除', ct.after === ct.before, `${ct.before} -> ${ct.after}`);

// ---- 4) 点「确认删除」应删掉画面 + 媒体文件
const confirmTest = await send('Runtime.evaluate', {
  expression: `(async () => {
    const S = window.__caseSystem;
    S.ui.renderCameras((S.state.project?.annotations ?? []).filter(a => a.kind === 'camera'), null);
    await new Promise(r => setTimeout(r, 200));
    const before = (S.state.project.annotations ?? []).length;
    const btn = document.querySelector('.cam-card [data-act="delete"]');
    if (!btn) return JSON.stringify({ error: 'no-delete-btn' });
    btn.click();
    await new Promise(r => setTimeout(r, 600));
    document.getElementById('cfOk').click();
    await new Promise(r => setTimeout(r, 2500));
    const after = (S.state.project.annotations ?? []).length;
    const cards = document.querySelectorAll('.cam-card').length;
    return JSON.stringify({ before, after, cards });
  })()`,
  returnByValue: true, awaitPromise: true,
});
const ft = JSON.parse(confirmTest.result?.result?.value ?? '{}');
console.log('确认删除测试:', JSON.stringify(ft));
check('点「确认删除」后标注数减少', (ft.after ?? 0) < (ft.before ?? 0), `${ft.before} -> ${ft.after}`);
check('媒体文件已从服务器删除', !existsSync(mediaAbs), mediaAbs ? path.basename(mediaAbs) : '');

// ---- 5) 标注列表里的 ✕ 也要走确认框并能真删
const listDel = await send('Runtime.evaluate', {
  expression: `(async () => {
    const before = (window.__caseSystem.state.project.annotations ?? []).length;
    const btn = document.querySelector('.anno-del');
    if (!btn) return JSON.stringify({ error: 'no-anno-del' });
    btn.click();
    await new Promise(r => setTimeout(r, 600));
    const shown = document.getElementById('dlgConfirm')?.open === true;
    document.getElementById('cfOk').click();
    await new Promise(r => setTimeout(r, 2000));
    const after = (window.__caseSystem.state.project.annotations ?? []).length;
    return JSON.stringify({ before, after, shown });
  })()`,
  returnByValue: true, awaitPromise: true,
});
const lt = JSON.parse(listDel.result?.result?.value ?? '{}');
console.log('列表删除测试:', JSON.stringify(lt));
check('列表里的 ✕ 也弹确认框', lt.shown === true);
check('列表里的 ✕ 确认后真的删除', (lt.after ?? 0) < (lt.before ?? 0), `${lt.before} -> ${lt.after}`);

const noise = consoleErrors.filter((m) => !/Download the React|DevTools|setPointerCapture/.test(m));
check('控制台无 error 输出', noise.length === 0, noise.slice(0, 2).join(' | '));

await api(`/api/projects/${pid}`, { method: 'DELETE' });
ws.close();
edge.kill();
console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail === 0 ? 0 : 1);
