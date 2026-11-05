/**
 * 截图：素材体积门槛的两态（体积偏少 → 提示确认框 / 体积充足 → 直接建模）。
 * 用法: node tests/shot_recon_gate.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9272;
const PROFILE = path.join(process.cwd(), '.browsertmp', 'shot-gate');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });
const OUT = path.join(process.cwd(), 'output', 'verify');
mkdirSync(OUT, { recursive: true });

const enc = encodeURIComponent;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// 造一个「素材体积只有 195 KB」的案件，正好落进「偏少」分支
const proj = await (
  await fetch(`${BASE}/api/projects`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: '体积门槛演示', description: '素材体积不足的场景' }),
  })
).json();
const pid = proj.project.id;
const mediaDir = path.join(process.cwd(), 'data', 'projects', pid, 'media');
mkdirSync(mediaDir, { recursive: true });
const name = '现场勘查（东侧）.mp4';
writeFileSync(path.join(mediaDir, name), Buffer.alloc(200_000, 3));
// 再放几个小图片，让清单看着像真实使用
const smalls = ['现场照片 01.png', '现场照片 02.png', '现场照片 03.png'];
for (const s of smalls) writeFileSync(path.join(mediaDir, s), Buffer.alloc(30_000, 4));

await fetch(`${BASE}/api/projects/${enc(pid)}/annotations`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({
    annotation: {
      kind: 'camera',
      label: '现场素材组',
      points: [[0, 1.5, -4]],
      pose: { position: [0, 1.5, -4], quaternion: [0, 0, 0, 1], fov: 60, source: 'default' },
      media: [
        { url: `/media/${enc(pid)}/media/${enc(name)}`, name, kind: 'video' },
        ...smalls.map((s) => ({ url: `/media/${enc(pid)}/media/${enc(s)}`, name: s, kind: 'image' })),
      ],
    },
    operator: '技术员',
  }),
});

const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1440,1000', BASE,
], { stdio: 'ignore' });

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
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
};
const send = (method, params = {}) =>
  new Promise((res) => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression, awaitPromise = false) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise });
  return r.result?.result?.value;
};
const shot = async (file) => {
  const s = await send('Page.captureScreenshot', { format: 'png' });
  if (s.result?.data) {
    const p = path.join(OUT, file);
    writeFileSync(p, Buffer.from(s.result.data, 'base64'));
    console.log('截图:', p);
  }
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
  await new Promise(r => setTimeout(r, 2200));
  const models = await (await fetch('/models/index.json')).json();
  const m = models.models.find(x => /room-real/.test(x.name)) ?? models.models[0];
  if (m) { await window.__caseSystem.loadModel(m); await new Promise(r => setTimeout(r, 2500)); }
  return 'ok';
})()`, true);

// 1) 对话框：显示「素材合计 / 实测建模下限 / ⚠ 体积偏少」
await evaluate(`(() => { window.__caseSystem.reconPanel.openDialog(); return 'ok'; })()`);
await sleep(1200);
await shot('ui-recon-volume-warn.png');

// 2) 点「开始建模」→ 应弹「提供信息过少，可能无法正常建模」
await evaluate(`(() => { document.getElementById('btnReconStart').click(); return 'ok'; })()`);
await sleep(1000);
await shot('ui-recon-volume-confirm.png');

// 3) 体积充足时：把体检结果换成 80 MB，界面显示「体积充足」
await evaluate(`(() => {
  const dlg = document.getElementById('dlgConfirm');
  if (dlg.open) dlg.close('cancel');
  const P = window.__caseSystem.reconPanel;
  P.materials = { bytes: 80 * 1024 * 1024, minBytes: 12 * 1024 * 1024, enough: true, videos: 1, images: 3, missing: [] };
  P._renderAssets();
  return 'ok';
})()`);
await sleep(500);
await shot('ui-recon-volume-ok.png');

// 清理演示案件
await fetch(`${BASE}/api/projects/${enc(pid)}`, { method: 'DELETE' });
ws.close();
edge.kill();
process.exit(0);
