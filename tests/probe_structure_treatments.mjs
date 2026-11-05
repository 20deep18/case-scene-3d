/**
 * 探针：**每个模型 × 三种场景判定，界面上到底画了什么**。
 *
 * 起因（用户反馈）：「房间处理和物体处理没区别，可能是没显示出来」。
 * 与其猜，不如把 模型 × {自动, 强制房间, 强制物体} 全部跑一遍，
 * 记录线段数 / 是否画了物体盒 / 提示语 —— 用数据回答"有没有区别、区别在哪"。
 *
 * 用法: node tests/probe_structure_treatments.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9287;
const PROFILE = path.join(process.cwd(), '.browsertmp', 'probe-treatments');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

const proj = await (
  await fetch(`${BASE}/api/projects`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'probe-treatments', operator: 'probe' }),
  })
).json();
const pid = proj.project.id;
const cleanup = async () => {
  try {
    await fetch(`${BASE}/api/projects/${pid}`, { method: 'DELETE' });
  } catch {
    /* 忽略 */
  }
};

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
  } catch {
    /* retry */
  }
  if (!page) await sleep(400);
}
if (!page) {
  console.log('浏览器没起来');
  edge.kill();
  await cleanup();
  process.exit(1);
}

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  }
};
const send = (method, params = {}) =>
  new Promise((res) => {
    pending.set(++id, res);
    ws.send(JSON.stringify({ id, method, params }));
  });
const evaluate = async (e, a = false) => {
  const r = await send('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: a });
  if (r.result?.exceptionDetails) return 'EXC: ' + r.result.exceptionDetails.exception?.description;
  return r.result?.result?.value;
};

await send('Runtime.enable');
for (let i = 0; i < 40; i++) {
  if ((await evaluate('String(!!(window.__caseSystem && window.__caseSystem.structureLayer))')) === 'true') break;
  await sleep(500);
}
await evaluate(`(async () => {
  const list = await (await fetch('/api/projects')).json();
  const p = (list.projects ?? []).find(x => x.id === ${JSON.stringify(pid)});
  const sel = document.getElementById('projectSelect');
  if (p && sel) { sel.value = p.id; sel.dispatchEvent(new Event('change', { bubbles: true })); }
  return 'ok';
})()`, true);
await sleep(1000);

const out = await evaluate(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const withStruct = j.models.filter(m => m.structureUrl);
  const rows = [];
  for (const m of withStruct.slice(0, 12)) {
    await S.loadModel(m, { announce: false });
    await new Promise(r => setTimeout(r, 2500));
    const row = { name: m.name };
    for (const want of ['auto', 'room', 'object']) {
      await S.setSceneMode(want, { silent: true });
      await new Promise(r => setTimeout(r, 1200));
      const L = S.structureLayer;
      row[want] = {
        mode: L.mode, source: L.modeSource, seg: L.segments.length,
        polylines: L.polylines.length, labels: L.labels.length,
        objDrawn: L.objectDrawn === true, trust: L.objectTrust?.ok ?? null,
        visible: L.visible, svg: !!L._svg && L._svg.style.display !== 'none',
        badge: document.getElementById('structBadge')?.textContent ?? '(无)',
        badgeCls: document.getElementById('structBadge')?.className ?? '',
      };
      await S.setSceneMode('auto', { silent: true });
    }
    rows.push(row);
  }
  return JSON.stringify(rows);
})()`, true);

let rows;
try {
  rows = JSON.parse(out);
} catch {
  console.log('解析失败：', out);
  ws.close();
  edge.kill();
  await cleanup();
  process.exit(1);
}

const sig = (t) => `${t.mode}/${t.objDrawn ? '物体盒' : '无盒'}/${t.seg}线`;
console.log('模型 × 三种判定（线段数 / 是否画物体盒 / 图层可见）');
console.log('-'.repeat(104));
for (const r of rows) {
  const a = r.auto, ro = r.room, ob = r.object;
  const same = a.seg === ro.seg && ro.seg === ob.seg;
  console.log(`${r.name.slice(0, 26).padEnd(28)} 自动=${sig(a).padEnd(18)} 强制房间=${sig(ro).padEnd(18)} 强制物体=${sig(ob).padEnd(18)}`);
  const badges = [...new Set([a.badge, ro.badge, ob.badge])];
  console.log(
    `${''.padEnd(28)} trust(自动)=${a.trust}  三种判定线段是否完全相同: ${same ? '❌ 完全相同' : '✅ 有区别'}`
  );
  console.log(`${''.padEnd(28)} 状态牌: 自动「${a.badge}」/ 房间「${ro.badge}」/ 物体「${ob.badge}」`);
  console.log(
    `${''.padEnd(28)} → 状态牌是否给出区别: ` +
      (badges.length >= 2 ? `✅ 是（${badges.length} 种不同文字）` : '❌ 仍然一样')
  );
}
// 再截一张图：柜子 + 强制房间（三种判定都画 0 条线的那种情况），
// 用来看清"状态牌"到底有没有把状态说清楚。
const shot = await evaluate(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const cab = j.models.find(x => String(x.name).includes('柜子现场-致密'));
  if (!cab) return 'no-cabinet';
  await S.loadModel(cab, { announce: false });
  await new Promise(r => setTimeout(r, 3500));
  await S.setSceneMode('room', { silent: true });
  await new Promise(r => setTimeout(r, 1500));
  return document.getElementById('structBadge')?.textContent ?? '(无)';
})()`, true);
console.log(`\n截图前状态牌：「${shot}」`);
const png = await send('Page.captureScreenshot', { format: 'png' });
if (png.result?.data) {
  const fs = await import('node:fs');
  fs.mkdirSync(path.join(process.cwd(), 'output', 'verify'), { recursive: true });
  fs.writeFileSync(
    path.join(process.cwd(), 'output', 'verify', 'struct-badge-cabinet.png'),
    Buffer.from(png.result.data, 'base64')
  );
  console.log('已写出 output/verify/struct-badge-cabinet.png');
}
// 收尾：把柜子的改判清掉，别留 sidecar
await evaluate(`(async () => {
  const j = await (await fetch('/models/index.json')).json();
  const cab = j.models.find(x => String(x.name).includes('柜子现场-致密'));
  if (cab) await fetch('/api/models/scene-mode', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: decodeURIComponent(cab.url.replace('/models/', '')), mode: 'auto' }),
  });
  return 'ok';
})()`, true);

ws.close();
edge.kill();
await cleanup();
