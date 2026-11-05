/**
 * 「渲染参数」三个滑杆的回归测试。
 *
 * 背景（真 bug）：index.html 里一直有「高斯尺寸 / 不透明度 / 曝光」三个滑杆，
 * 但 app.js **没有给它们绑定任何事件**，值只写进 state.filters 没有任何人读 ——
 * 拖动滑杆画面纹丝不动。同时渲染器的着色器里连 opacity/exposure 的 uniform 都没有。
 *
 * 这个测试锁住三件事：
 *   1. 三个滑杆存在，且初值与 state.filters 一致
 *   2. 拖动（dispatch input）之后，**渲染器 options 真的变了**
 *   3. 渲染器构造时就带上了当前值（否则"加载模型前调过的值"会丢）
 *
 * 用法：node tests/browser_render_sliders.mjs [http://127.0.0.1:8787/]
 */
import { spawn } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';

const BASE = process.argv[2] || 'http://127.0.0.1:8787/';
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const PORT = 9431;
const PROFILE = path.join(process.cwd(), '.browsertmp', 'sliders-' + Date.now());
fs.mkdirSync(PROFILE, { recursive: true });

const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader', `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${PROFILE}`, '--window-size=1400,900', BASE,
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let page = null;
for (let i = 0; i < 40 && !page; i++) {
  try {
    const l = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    page = l.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
  } catch { /* retry */ }
  if (!page) await sleep(500);
}
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pend = new Map();
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); }
};
const send = (method, params = {}) =>
  new Promise((res) => { pend.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
const ev = async (expr, aw = false) => {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: aw });
  const ex = r.result?.exceptionDetails;
  if (ex) throw new Error('页面异常: ' + (ex.exception?.description || ex.text));
  return r.result?.result?.value;
};

let pass = 0, fail = 0;
const check = (name, ok, extra = '') => {
  if (ok) { pass++; console.log(`  [PASS] ${name}`); }
  else { fail++; console.log(`  [FAIL] ${name} ${extra}`); }
};

await send('Runtime.enable');
for (let i = 0; i < 40; i++) {
  if ((await ev('String(!!window.__caseSystem?.structureLayer)').catch(() => 'false')) === 'true') break;
  await sleep(500);
}

// 1) 三个滑杆存在且初值同步
const meta = JSON.parse(await ev(`JSON.stringify({
  ids: ['rGaussScale','rOpacity','rExposure'].map(id => !!document.getElementById(id)),
  values: ['rGaussScale','rOpacity','rExposure'].map(id => document.getElementById(id)?.value),
  filters: window.__caseSystem?.state?.filters,
  labels: ['vScale','vOpacity','vExposure'].map(id => document.getElementById(id)?.textContent),
})`));
console.log('   滑杆:', meta);
check('三个滑杆都存在', meta.ids.every(Boolean), JSON.stringify(meta.ids));
check('滑杆初值与 state.filters 一致',
  Math.abs(Number(meta.values[0]) - Number(meta.filters.gaussScale)) < 1e-6 &&
  Math.abs(Number(meta.values[1]) - Number(meta.filters.opacity)) < 1e-6 &&
  Math.abs(Number(meta.values[2]) - Number(meta.filters.exposure)) < 1e-6,
  `sliders=${meta.values} filters=${JSON.stringify(meta.filters)}`);

// 2) 加载一个模型，让渲染器存在
await ev(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const m = j.models.find(x => /room-real/.test(x.name || x.url)) || j.models[0];
  await S.loadModel(m, { announce: false });
  return 'ok';
})()`, true).catch(() => 'load failed');
check('渲染器已创建', (await ev('String(!!window.__caseSystem.state.renderer)')) === 'true');

// 3) 拖动滑杆 → 渲染器 options 真的变化
const after = JSON.parse(await ev(`(() => {
  const S = window.__caseSystem;
  const set = (id, v) => {
    const el = document.getElementById(id);
    el.value = String(v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  };
  set('rGaussScale', 1.5);
  set('rOpacity', 0.6);
  set('rExposure', 1.6);
  const o = S.state.renderer?.options ?? {};
  return JSON.stringify({ scale: o.gaussScale, opacity: o.opacityGain, exposure: o.exposure,
                          filters: S.state.filters });
})()`));
console.log('   拖动后:', after);
check('高斯尺寸滑杆生效', Math.abs(Number(after.scale) - 1.5) < 1e-6, `scale=${after.scale}`);
check('不透明度滑杆生效', Math.abs(Number(after.opacity) - 0.6) < 1e-6, `opacityGain=${after.opacity}`);
check('曝光滑杆生效', Math.abs(Number(after.exposure) - 1.6) < 1e-6, `exposure=${after.exposure}`);

// 4) 渲染器构造时带上了当前值（把滑杆调到新值后重新载入模型，看新渲染器是否继承）
const rebuilt = JSON.parse(await ev(`(async () => {
  const S = window.__caseSystem;
  const set = (id, v) => {
    const el = document.getElementById(id);
    el.value = String(v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  };
  set('rGaussScale', 1.25);
  S.state.renderer = null;                       // 强制下次加载重建渲染器
  const j = await (await fetch('/models/index.json')).json();
  const m = j.models.find(x => /room-real/.test(x.name || x.url)) || j.models[0];
  await S.loadModel(m, { announce: false });
  return JSON.stringify({ scale: S.state.renderer?.options?.gaussScale });
})()`, true).catch((e) => JSON.stringify({ err: String(e) })));
check('渲染器重建后继承滑杆当前值', Math.abs(Number(rebuilt.scale) - 1.25) < 1e-6, JSON.stringify(rebuilt));

ws.close();
edge.kill();
console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail ? 1 : 0);
