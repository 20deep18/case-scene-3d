/**
 * 「模型下拉框」回归测试 —— 锁住一个会直接毁掉交付的 bug。
 *
 * 症状：在下拉框里选 `room-real`，模型确实换了，但**结构线一条都没有**；
 *       而同一份模型通过启动时自动加载 / 直接调 loadModel 就有结构线。
 *
 * 根因（两个叠加）：
 *   1. 下拉框**没有 change 监听**，只有"双击=点载入按钮"（用户根本不知道要再点一下）；
 *   2. 更关键：`onLoadModel(url)` 只用 URL 拼了 `{name, url}`，
 *      **把清单里的 structureUrl 丢了** → applyStructure 直接 early-return。
 *
 * 用法：node tests/browser_model_dropdown.mjs [base]
 */
import { spawn } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';

const BASE = process.argv[2] || 'http://127.0.0.1:8787/';
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const PORT = 9501;
const PROFILE = path.join(process.cwd(), '.browsertmp', 'dropdown-' + Date.now());
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
const send = (method, params = {}) => new Promise((res) => { pend.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
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
// 等页面自己的初始化跑完（它会自动恢复上次模型，太早操作会被覆盖）
for (let i = 0; i < 60; i++) {
  if ((await ev('String(!!window.__caseSystem?.structureLayer)')) === 'true') break;
  await sleep(700);
}
await sleep(2500);

const opts = JSON.parse(await ev(
  `JSON.stringify([...document.querySelectorAll('#modelSelect option')].map(o => o.textContent.trim()))`));
console.log('   下拉框:', opts);
const idx = opts.findIndex((t) => /room-real/.test(t));
check('下拉框里有 room-real', idx >= 0, JSON.stringify(opts));

// 模拟用户在下拉框里选择（只 dispatch change，**不点任何按钮**）
const dispatched = await ev(`(() => {
  const sel = document.getElementById('modelSelect');
  const opt = [...sel.options].find(o => /room-real/.test(o.textContent));
  if (!opt) return 'no option';
  sel.value = opt.value;
  sel.dispatchEvent(new Event('change', { bubbles: true }));
  return opt.value;
})()`);
check('下拉框 change 已触发', String(dispatched).includes('room-real'), String(dispatched));

// 等模型**和结构都**就绪（最多 25 秒）。
// ⚠️ 只等"模型 URL 变了"不够：结构文件是模型载入**之后**才 fetch + 应用的，
//    太早断言会误报"结构线没画"（实测：静态版比本地慢一拍，测试因此假失败）。
let st = null;
for (let i = 0; i < 50; i++) {
  st = JSON.parse(await ev(`(() => {
    const S = window.__caseSystem, s = S.state, L = S.structureLayer;
    return JSON.stringify({
      model: (s.model?.url || ''),
      structureUrl: s.model?.structureUrl ?? null,
      gauss: (s.gaussianPositions?.length || 0) / 3,
      hasData: !!L.data, mode: L.mode, segments: L.segments.length,
    });
  })()`));
  if (/room-real/.test(st.model) && st.hasData && st.segments > 0) break;
  await sleep(500);
}
console.log('   切换后:', st);
check('选中即载入（不需要再点按钮）', /room-real/.test(st.model), `model=${st.model}`);
check('模型确实换了（高斯数变化）', st.gauss > 0 && st.gauss < 100000, `gauss=${st.gauss}`);
check('结构文件被加载（关键：structureUrl 没丢）', st.hasData === true, `hasData=${st.hasData}`);
check('结构线真的画出来了', st.segments > 0, `segments=${st.segments}`);
check('场景模式判定为房间', st.mode === 'room', `mode=${st.mode}`);

ws.close();
edge.kill();
console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail ? 1 : 0);
