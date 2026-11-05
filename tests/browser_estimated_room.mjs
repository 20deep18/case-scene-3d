/**
 * 「推测房间盒」的浏览器回归测试（2026-11-02）。
 *
 * 背景：只有 1 面可信竖墙时，系统用曼哈顿正交补全给出房间盒 ——
 * 实测尺寸误差 10~19%（5 面墙全拟合才 ~1%），所以**必须**在界面上
 * 和"已核实"区分开：虚线 + 状态牌写（推测）+ 说明里给误差范围。
 *
 * 断言：
 *   1. structureLayer.estimated === true
 *   2. 房间盒线段带 dash（虚线），且确实渲染出 stroke-dasharray
 *   3. 状态牌文案含「推测」
 *   4. notice 里解释了"只有 1 面墙"和误差范围
 *   5. 对照：同一模型若把 estimated 改成 false（模拟已核实），不再有虚线
 *
 * 用法: node tests/browser_estimated_room.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const BASE = process.env.BASE || 'http://127.0.0.1:8787/';
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const PORT = 9361;
const OUT = path.join(process.cwd(), 'output', 'estimated-room');
mkdirSync(OUT, { recursive: true });
const PROFILE = path.join(process.cwd(), '.browsertmp', 'estimated-' + Date.now());
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

const r = await ev(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const m = j.models.find(x => x.name.includes('手机拍摄'));
  if (!m) return JSON.stringify({ err: '找不到测试模型（厕所现场-手机拍摄）' });
  await S.loadModel(m, { announce: false });
  // ⚠️ 不能固定 sleep：结构文件是 loadModel 之后异步加载的，
  //    实测固定等 9 秒仍可能读在"家具已画、结构未应用"的中间态
  //    （那次断言拿到 roomSegs=0 / 状态牌"未载入"，但再等一会儿就是对的）。
  //    改成轮询"结构已应用"再断言。
  const L = S.structureLayer;
  for (let i = 0; i < 40; i++) {
    if (L.data && L.roomDrawn && L.segments.some(x => x.color === '#4dd0e1')) break;
    await new Promise(r => setTimeout(r, 500));
  }
  const roomSegs = L.segments.filter(s => s.color === '#4dd0e1');
  const svgLines = Array.from(document.querySelectorAll('#structureLayer line'));
  const dashed = svgLines.filter(el => el.getAttribute('stroke-dasharray'));
  return JSON.stringify({
    estimated: L.estimated,
    roomDrawn: L.roomDrawn,
    mode: L.mode,
    roomSegs: roomSegs.length,
    dashedOnBox: roomSegs.filter(s => !!s.dash).length,
    dashedInSvg: dashed.length,
    badge: document.getElementById('structBadge')?.textContent ?? '',
    notice: L.notice ?? '',
    sizeLabels: L.labels.map(l => l.text).join(' | '),
  });
})()`, true);
const d = JSON.parse(r);
console.log('\n载入（推测房间盒）:', d);

check('estimated 标记为 true', d.estimated === true, `estimated=${d.estimated}`);
check('房间盒已画', d.roomDrawn === true, `roomDrawn=${d.roomDrawn}`);
check('房间盒线段 12 条', d.roomSegs === 12, `roomSegs=${d.roomSegs}`);
check('房间盒线段都带 dash', d.dashedOnBox === 12, `dash=${d.dashedOnBox}`);
check('SVG 里真的渲染了虚线', d.dashedInSvg >= 12, `svg=${d.dashedInSvg}`);
check('状态牌含「推测」', /推测/.test(d.badge || ''), `badge=${d.badge}`);
check('说明解释了"只有 1 面墙"', /1 面|一面/.test(d.notice || ''), `notice=${d.notice}`);
check('说明给了误差范围', /误差/.test(d.notice || ''), `notice=${d.notice}`);
check('尺寸标签不再谎称"m"（未校正时标"单位"）',
  /单位/.test(d.sizeLabels || '') && !/\d+\.\d+ m/.test(d.sizeLabels || ''), `labels=${d.sizeLabels}`);

await send('Page.captureScreenshot', { format: 'png' }).then((p) => {
  if (p.result?.data) {
    writeFileSync(path.join(OUT, 'estimated-room.png'), Buffer.from(p.result.data, 'base64'));
  }
});

// 对照：把 estimated 人为改成 false → 不该再有虚线
const r2 = await ev(`(() => {
  const S = window.__caseSystem;
  const L = S.structureLayer;
  const data = JSON.parse(JSON.stringify(L.data));
  data.estimated = false;
  L.set(data);
  const roomSegs = L.segments.filter(s => s.color === '#4dd0e1');
  return JSON.stringify({ dashed: roomSegs.filter(s => !!s.dash).length,
    badge: document.getElementById('structBadge')?.textContent ?? '' });
})()`);
const d2 = JSON.parse(r2);
check('对照：estimated=false 时无虚线', d2.dashed === 0, `dash=${d2.dashed}`);

ws.close();
edge.kill();
console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail ? 1 : 0);
