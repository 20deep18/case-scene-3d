/**
 * 诊断：环绕漫游 / 第一人称 / 俯视态势 切换时「视角乱飞」。
 * 记录每次切换前后的相机位置、朝向、up、target，定位跳变来源。
 * 用法: node tests/probe_mode_switch.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9291;
const PROFILE = path.join(process.cwd(), '.browsertmp', 'probe-mode');
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
const logs = [];
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') logs.push('[异常] ' + (m.params.exceptionDetails?.exception?.description ?? '?').slice(0, 160));
};
const send = (method, params = {}) => new Promise((res) => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
await send('Runtime.enable');

let booted = false;
for (let i = 0; i < 40 && !booted; i++) {
  const r = await send('Runtime.evaluate', { expression: 'String(!!(window.__caseSystem && window.__caseSystem.state))', returnByValue: true });
  booted = r.result?.result?.value === 'true';
  if (!booted) await sleep(500);
}
await sleep(2500);

const script = `
(async () => {
  const S = window.__caseSystem, st = S.state, THREE = await import('/vendor/three/build/three.module.js');
  const out = [];
  const snap = (label) => {
    const fwd = new THREE.Vector3(0,0,-1).applyQuaternion(st.camera.quaternion);
    const pos = st.camera.position, tgt = st.controls.target;
    out.push({
      label,
      mode: st.mode,
      pos: pos.toArray().map(v => Number.isFinite(v) ? +v.toFixed(3) : String(v)),
      fwd: fwd.toArray().map(v => Number.isFinite(v) ? +v.toFixed(3) : String(v)),
      up: st.camera.up.toArray().map(v => +v.toFixed(3)),
      target: tgt.toArray().map(v => Number.isFinite(v) ? +v.toFixed(3) : String(v)),
      controlsEnabled: st.controls.enabled,
      distToTarget: +pos.distanceTo(tgt).toFixed(3),
      fov: st.camera.fov,
    });
  };
  snap('初始(orbit)');
  const modes = ['fly','top','orbit','top','fly','orbit'];
  for (const m of modes) {
    S.ui.handlers.onMode?.(m);
    await new Promise(r => setTimeout(r, 260));
    snap('切到 ' + m);
  }
  return JSON.stringify(out);
})()
`;
const r = await send('Runtime.evaluate', { expression: script, returnByValue: true, awaitPromise: true });
if (r.result?.exceptionDetails) console.log('异常:', r.result.exceptionDetails.exception?.description?.slice(0, 300));
const rows = JSON.parse(r.result?.result?.value ?? '[]');
console.log('=== 模式切换时的相机状态 ===');
for (const s of rows) {
  console.log(
    `${s.label.padEnd(14)} mode=${String(s.mode).padEnd(5)} pos=${JSON.stringify(s.pos).padEnd(26)} ` +
    `fwd=${JSON.stringify(s.fwd).padEnd(26)} up=${JSON.stringify(s.up).padEnd(16)} target=${JSON.stringify(s.target).padEnd(24)} ` +
    `dist=${String(s.distToTarget).padEnd(8)} ctrl=${s.controlsEnabled}`
  );
}
console.log('\n=== 跳变分析 ===');
for (let i = 1; i < rows.length; i++) {
  const a = rows[i - 1], b = rows[i];
  const dPos = Math.hypot(...b.pos.map((v, k) => v - a.pos[k]));
  const dFwd = Math.hypot(...b.fwd.map((v, k) => v - a.fwd[k]));
  const upChanged = JSON.stringify(a.up) !== JSON.stringify(b.up);
  const flag = dPos > 2 || dFwd > 0.5 || upChanged ? '  <== 跳变' : '';
  console.log(`  ${a.label} -> ${b.label}: Δpos=${dPos.toFixed(3)} Δfwd=${dFwd.toFixed(3)} up变化=${upChanged}${flag}`);
}
if (logs.length) { console.log('\n日志:'); logs.slice(0, 8).forEach(l => console.log('  ' + l)); }
ws.close();
edge.kill();
process.exit(0);
