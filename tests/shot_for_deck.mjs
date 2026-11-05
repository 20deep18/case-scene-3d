/**
 * 作品讲解用的配图截屏。
 *
 * 一次跑完，产出：
 *   01-overview.png   房间现场（三维漫游 + 界面全貌）
 *   02-annotate.png   同一现场 + 真实标注（打点 / 测距 / 尺寸标注）——展示"模型有什么用"
 *   03-cabinet.png    另一个现场（柜子/狭小空间）
 *   04-recon.png      现场建模面板（视频 → 抽帧 → 位姿 → 三维模型 的入口）
 *
 * 做法：临时建一个演示案件 → 载入模型 → 截图 → 用真实接口打标注 → 再截图 → 清理案件。
 * 用法: node tests/shot_for_deck.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9291;
const OUT = path.join(process.cwd(), 'output', 'deck');
mkdirSync(OUT, { recursive: true });

const proj = await (
  await fetch(`${BASE}/api/projects`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: '现场演示', description: '作品讲解配图', operator: '演示' }),
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

const PROFILE = path.join(process.cwd(), '.browsertmp', 'deck-shots');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });
const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1500,1000', BASE,
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
const shoot = async (name) => {
  const s = await send('Page.captureScreenshot', { format: 'png' });
  if (s.result?.data) {
    writeFileSync(path.join(OUT, name), Buffer.from(s.result.data, 'base64'));
    console.log(`  已写出 ${name}`);
  }
};

await send('Runtime.enable');
for (let i = 0; i < 40; i++) {
  if ((await evaluate('String(!!(window.__caseSystem && window.__caseSystem.structureLayer))')) === 'true') break;
  await sleep(500);
}
// 切到演示案件
await evaluate(`(async () => {
  const list = await (await fetch('/api/projects')).json();
  const p = (list.projects ?? []).find(x => x.id === ${JSON.stringify(pid)});
  const sel = document.getElementById('projectSelect');
  if (p && sel) { sel.value = p.id; sel.dispatchEvent(new Event('change', { bubbles: true })); }
  return 'ok';
})()`, true);
await sleep(1200);

// ---- 01/02：房间现场 -------------------------------------------------------
const loadRoom = await evaluate(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const m = j.models.find(x => String(x.name).includes('room-real'));
  if (!m) return 'no-model';
  await S.loadModel(m, { announce: false });
  await new Promise(r => setTimeout(r, 5000));
  // 用结构图层里的真实房间盒把相机放到"站在房间里看"的位置
  const L = S.structureLayer;
  const room = L?.data?.room ?? null;
  if (room?.corners?.length === 8) {
    const c = room.corners;
    const mid = (a, b) => c[a].map((v, i) => (v + c[b][i]) / 2);
    const center = c.reduce((acc, p) => acc.map((v, i) => v + p[i] / 8), [0, 0, 0]);
    return JSON.stringify({ center, size: room.size });
  }
  return JSON.stringify({ center: null });
})()`, true);
console.log('  房间盒:', loadRoom);
await sleep(800);
await shoot('01-overview.png');

// ---- 用真实接口打标注（打点 / 测距 / 尺寸标注）---------------------------
const annotate = await evaluate(`(async () => {
  const S = window.__caseSystem;
  const L = S.structureLayer;
  const data = L?.data ?? null;
  const room = data?.room ?? null;
  if (!room?.corners?.length) return 'no-room';
  const c = room.corners;
  const center = c.reduce((acc, p) => acc.map((v, i) => v + p[i] / 8), [0, 0, 0]);
  const u = room.axisU, v = room.axisV;
  const add = (p, i) => p.map((x, k) => x + (u[k] * i[0] + v[k] * i[1]));
  const post = (annotation) => fetch('/api/projects/${pid}/annotations', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ annotation, operator: '演示' }),
  }).then(r => r.json());
  // 打点两处（物证位置）+ 一条测距 + 一条尺寸标注（带实测值）
  const p1 = add(center, [2.2, 1.4]);
  const p2 = add(center, [-1.8, -1.2]);
  const p3 = add(center, [1.0, -2.0]);
  await post({ kind: 'point', points: [p1], label: '物证 1 · 桌面', color: '#ff8a80' });
  await post({ kind: 'point', points: [p2], label: '物证 2 · 地面', color: '#ffd54f' });
  await post({ kind: 'measure', points: [p1, p2], label: '空间距离' });
  await post({ kind: 'dimension', points: [p2, p3], measured: 2.4, label: '实测标注' });
  await new Promise(r => setTimeout(r, 1500));
  const detail = await (await fetch('/api/projects/${pid}')).json();
  return JSON.stringify({ count: detail.project.annotations.length });
})()`, true);
console.log('  标注:', annotate);
await sleep(1200);
await shoot('02-annotate.png');

// ---- 03：柜子现场 ---------------------------------------------------------
await evaluate(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const m = j.models.find(x => String(x.name).includes('柜子现场-致密'));
  if (m) { await S.loadModel(m, { announce: false }); }
  await new Promise(r => setTimeout(r, 5000));
  return 'ok';
})()`, true);
await sleep(600);
await shoot('03-cabinet.png');

// ---- 04：现场建模面板（把左栏滚到"现场建模"）-----------------------------
await evaluate(`(() => {
  const h = [...document.querySelectorAll('h3')].find(x => x.textContent.includes('现场建模'));
  if (h) h.scrollIntoView({ block: 'start' });
  return 'ok';
})()`, true);
await sleep(700);
await shoot('04-recon.png');

ws.close();
edge.kill();
await cleanup();
console.log(`\n配图已写出到 ${OUT}（演示案件已清理）`);
