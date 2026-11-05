/**
 * 截图：场景判定的人工改判入口（左侧「模型文件」区）。
 *
 * 拍两张：自动判定（画房间盒）→ 人工改判为「物体」（只剩轨迹）。
 * 放在一起就能看出"改判真的生效了、而且界面说清了是人工改的"。
 *
 * 用法: node tests/shot_scene_mode_override.mjs [模型关键字]
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9283;
const OUT = path.join(process.cwd(), 'output', 'verify');
mkdirSync(OUT, { recursive: true });
const KEY = process.argv[2] ?? 'room-real';

const proj = await (
  await fetch(`${BASE}/api/projects`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'scene-mode-override-shot', operator: 'test' }),
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

const PROFILE = path.join(process.cwd(), '.browsertmp', 'scenemode-override');
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
await sleep(1200);

const shoot = async (file) => {
  const s = await send('Page.captureScreenshot', { format: 'png' });
  if (s.result?.data) writeFileSync(path.join(OUT, file), Buffer.from(s.result.data, 'base64'));
  console.log(`  已写出 ${file}`);
};

// 载入模型（自动判定）
const info = await evaluate(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const m = j.models.find(x => String(x.name).includes(${JSON.stringify(KEY)}));
  if (!m) return 'no-model';
  await S.setSceneMode('auto', { silent: true });   // 先确保是自动判定
  await S.loadModel(m, { announce: false });
  await new Promise(r => setTimeout(r, 4500));
  const sel = document.getElementById('sceneModeSelect');
  return JSON.stringify({ model: m.name, mode: S.structureLayer.mode, seg: S.structureLayer.segments.length,
                          hint: document.getElementById('sceneModeHint').textContent, selValue: sel.value });
})()`, true);
console.log('  自动判定:', info);
await shoot('scene-mode-override-auto.png');

// 改判为「物体」
const forced = await evaluate(`(async () => {
  const sel = document.getElementById('sceneModeSelect');
  sel.value = 'object';
  sel.dispatchEvent(new Event('change', { bubbles: true }));
  await new Promise(r => setTimeout(r, 2000));
  const S = window.__caseSystem;
  return JSON.stringify({ mode: S.structureLayer.mode, source: S.structureLayer.modeSource,
                          seg: S.structureLayer.segments.length,
                          hint: document.getElementById('sceneModeHint').textContent });
})()`, true);
console.log('  改判为物体:', forced);
await shoot('scene-mode-override-object.png');

// 收尾：恢复自动判定，别把改判留在模型上
await evaluate(`window.__caseSystem.setSceneMode('auto', { silent: true })`, true);
ws.close();
edge.kill();
await cleanup();
console.log(`截图已写出到 ${OUT}（改判已复位）`);
