/**
 * 截图：**同一台机器上，柜子现场 vs 真实房间**，结构图层分别画了什么。
 *
 * 为什么值得留证据：2026-10-06 之前 `structure-layer.set()` 无条件画房间盒，
 * 于是柜子那份素材在界面上被糊了一个假的「6.0 × 3.7 × 3.7 m 房间盒」。
 * 修完之后应该是：
 *   · 柜子现场（sceneMode=object）  → 只有相机轨迹（橙红），没有任何户型线
 *   · room-real（sceneMode=room）   → 房间盒（青）+ 地面网格 + 墙面（黄）+ 轨迹
 *
 * 用法: node tests/shot_scene_mode.mjs [柜子模型关键字] [房间模型关键字]
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9281;
const OUT = path.join(process.cwd(), 'output', 'verify');
mkdirSync(OUT, { recursive: true });

const OBJ_KEY = process.argv[2] ?? '柜子现场-致密';
const ROOM_KEY = process.argv[3] ?? 'room-real';

// 建一个临时案件（载入模型需要有当前案件）
const proj = await (
  await fetch(`${BASE}/api/projects`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'scene-mode-shot', operator: 'test' }),
  })
).json();
const pid = proj.project.id;

/**
 * ⚠️ 一定要删掉这个临时案件 —— 2026-10-06 我忘了删，结果它成了"最近更新的案件"，
 * 而 `browser_pnp_panel.mjs` 依赖"应用会自动选中最新案件"（它是在页面里按
 * `st.project.annotations` 找自己的相机标注的）。多出来的案件一插队，
 * 那个测试就报「找不到测试相机标注」而失败 —— **测试污染，不是功能坏了**。
 * 实测：删掉遗留案件后 browser_pnp_panel 立刻回到 44 通过 / 0 失败。
 */
const cleanup = async () => {
  try {
    await fetch(`${BASE}/api/projects/${pid}`, { method: 'DELETE' });
    console.log(`已清理临时案件 ${pid}`);
  } catch (err) {
    console.log(`清理临时案件失败（请手动删除）：${err.message}`);
  }
};

const PROFILE = path.join(process.cwd(), '.browsertmp', 'scenemode');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

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
if (!page) { console.log('浏览器没起来'); edge.kill(); process.exit(1); }

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
};
const send = (method, params = {}) =>
  new Promise((res) => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (e, a = false) => {
  const r = await send('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: a });
  return r.result?.result?.value;
};

await send('Runtime.enable');
for (let i = 0; i < 40; i++) {
  if ((await evaluate('String(!!(window.__caseSystem && window.__caseSystem.structureLayer))')) === 'true') break;
  await sleep(500);
}
// 选中刚建的案件
await evaluate(`(async () => {
  const S = window.__caseSystem;
  const list = await (await fetch('/api/projects')).json();
  const p = (list.projects ?? []).find(x => x.id === ${JSON.stringify(pid)});
  if (p && S.selectProject) await S.selectProject(p);
  return 'ok';
})()`, true);

const shoot = async (key, file) => {
  const info = await evaluate(`(async () => {
    const S = window.__caseSystem;
    const j = await (await fetch('/models/index.json')).json();
    const m = j.models.find(x => String(x.name).includes(${JSON.stringify(key)}));
    if (!m) return JSON.stringify({ error: '找不到模型 ' + ${JSON.stringify(key)} });
    await S.loadModel(m, { announce: false });
    await new Promise(r => setTimeout(r, 4500));
    const L = S.structureLayer;
    return JSON.stringify({ name: m.name, mode: L.mode, notice: L.notice, segments: L.segments.length, polylines: L.polylines.length });
  })()`, true);
  const s = await send('Page.captureScreenshot', { format: 'png' });
  const p = path.join(OUT, file);
  if (s.result?.data) writeFileSync(p, Buffer.from(s.result.data, 'base64'));
  console.log(`  ${file}: ${info}`);
  return p;
};

await shoot(OBJ_KEY, 'scene-mode-object.png');
await shoot(ROOM_KEY, 'scene-mode-room.png');

ws.close();
edge.kill();
await cleanup();
console.log(`截图已写出到 ${OUT}`);
