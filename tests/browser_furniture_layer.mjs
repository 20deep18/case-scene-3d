/**
 * 家具盒图层的浏览器回归测试。
 *
 * 验的是四件容易做错的事：
 *   1. 有 `*.furniture.json` 的模型 → 盒子真的被画出来（线段数 = 盒数×12）
 *   2. 开关能把它关掉/打开（关掉后线段数为 0）
 *   3. **换到没有家具文件的模型时必须清空**（否则上一个模型的橙框会留在屏幕上）
 *   4. 提示条要报出"家具 N 件"，别让用户猜
 *
 * 用法: node tests/browser_furniture_layer.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const BASE = process.env.BASE || 'http://127.0.0.1:8787/';
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const PORT = 9333;
const OUT = path.join(process.cwd(), 'output', 'furniture-ui');
mkdirSync(OUT, { recursive: true });
const PROFILE = path.join(process.cwd(), '.browsertmp', 'furniture-' + Date.now());
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
  if (!page) await sleep(500);
}
if (!page) {
  console.log('浏览器没起来');
  edge.kill();
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
const ev = async (expr, awaitPromise = false) => {
  const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise });
  const e = r.result?.exceptionDetails;
  if (e) throw new Error('页面异常: ' + (e.exception?.description || e.text));
  return r.result?.result?.value;
};

let pass = 0;
let fail = 0;
const check = (name, ok, extra = '') => {
  if (ok) {
    pass++;
    console.log(`  [PASS] ${name}`);
  } else {
    fail++;
    console.log(`  [FAIL] ${name} ${extra}`);
  }
};

await send('Runtime.enable');
for (let i = 0; i < 60; i++) {
  if ((await ev('String(!!(window.__caseSystem && window.__caseSystem.structureLayer))')) === 'true') break;
  await sleep(700);
}

// ---- 1. 载入"有家具数据"的模型 ----
const r1 = await ev(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const m = j.models.find(x => x.name.includes('VGGT+3DGS'));
  if (!m) return JSON.stringify({ err: '找不到测试模型 VGGT+3DGS厕所' });
  await S.loadModel(m, { announce: false });
  await new Promise(r => setTimeout(r, 9000));
  const L = S.structureLayer;
  return JSON.stringify({
    loaded: !!L.furniture,
    boxes: L.furniture?.objects?.length ?? 0,
    drawn: L.furnitureDrawn,
    segs: L.segments.length,
    hud: document.getElementById('modelHint')?.textContent ?? '',
    cb: document.getElementById('cFurniture')?.checked,
  });
})()`, true);
const d1 = JSON.parse(r1);
console.log('\n载入有家具的模型：', d1);
check('家具数据已加载', d1.loaded === true, `loaded=${d1.loaded}`);
check('盒子数量 > 0', d1.boxes > 0, `boxes=${d1.boxes}`);
check('画的线段数 = 盒数 × 12', d1.drawn === d1.boxes * 12, `drawn=${d1.drawn} boxes=${d1.boxes}`);
check('提示条报出家具件数', /家具 \d+ 件/.test(d1.hud || ''), `hud=${d1.hud}`);
check('开关默认勾选', d1.cb === true, `cb=${d1.cb}`);

const png1 = await send('Page.captureScreenshot', { format: 'png' });
if (png1.result?.data) {
  writeFileSync(path.join(OUT, 'with-furniture.png'), Buffer.from(png1.result.data, 'base64'));
}

// ---- 2. 关掉家具盒 ----
const r2 = await ev(`(() => {
  const cb = document.getElementById('cFurniture');
  cb.checked = false;
  cb.dispatchEvent(new Event('change'));
  const L = window.__caseSystem.structureLayer;
  return JSON.stringify({ drawn: L.furnitureDrawn, boxes: L.furniture?.objects?.length ?? 0 });
})()`);
const d2 = JSON.parse(r2);
console.log('关掉家具盒：', d2);
check('关掉后不再画家具线段', d2.drawn === 0, `drawn=${d2.drawn}`);
check('关掉只是隐藏，数据仍在（可再打开）', d2.boxes === d1.boxes, `boxes=${d2.boxes}`);

// ---- 3. 再打开 ----
const d3 = JSON.parse(await ev(`(() => {
  const cb = document.getElementById('cFurniture');
  cb.checked = true;
  cb.dispatchEvent(new Event('change'));
  const L = window.__caseSystem.structureLayer;
  return JSON.stringify({ drawn: L.furnitureDrawn });
})()`));
console.log('再打开：', d3);
check('再打开后恢复线段', d3.drawn === d1.drawn, `drawn=${d3.drawn}`);

// ---- 4. 换到没有家具数据的模型 → 必须清空 ----
const r4 = await ev(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const m = j.models.find(x => /room-real/.test(x.name));
  await S.loadModel(m, { announce: false });
  await new Promise(r => setTimeout(r, 8000));
  const L = S.structureLayer;
  return JSON.stringify({ furniture: !!L.furniture, drawn: L.furnitureDrawn,
    hud: (document.getElementById('modelHint')?.textContent ?? '').slice(0, 120) });
})()`, true);
const d4 = JSON.parse(r4);
console.log('换到无家具模型：', d4);
check('换模型后家具数据被清空', d4.furniture === false, `furniture=${d4.furniture}`);
check('换模型后不再画家具线段', d4.drawn === 0, `drawn=${d4.drawn}`);
check('换模型后提示条不再提家具', !/家具 \d+ 件/.test(d4.hud || ''), `hud=${d4.hud}`);

ws.close();
edge.kill();
console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail ? 1 : 0);
