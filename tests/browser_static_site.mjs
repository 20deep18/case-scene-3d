/**
 * 静态托管版（只读）的浏览器回归测试。
 *
 * 目标：在没有 `/api/*` 的**纯静态服务器**上（模拟 Cloudflare Pages），确认：
 *   1. 页面能起、`__DSH_STATIC__` 生效
 *   2. 模型清单来自静态 `models/index.json`（不需要服务端扫描）
 *   3. 能载入模型并画出结构线（房间盒 + 墙 + 网格）与家具盒
 *   4. **写操作会明确报"只读演示版"**，而不是假装成功
 *   5. 控制台没有未捕获异常（静态模式下最容易犯的错是某个 api 调用炸掉整页）
 *
 * 用法: node tests/browser_static_site.mjs [http://127.0.0.1:8899/]
 */
import { spawn } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';

const BASE = process.argv[2] || 'http://127.0.0.1:8899/';
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const PORT = 9391;
const OUT = path.join(process.cwd(), 'output', 'static-site');
fs.mkdirSync(OUT, { recursive: true });
const PROFILE = path.join(process.cwd(), '.browsertmp', 'static-' + Date.now());
fs.mkdirSync(PROFILE, { recursive: true });

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
const errors = [];
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  } else if (m.method === 'Runtime.exceptionThrown') {
    errors.push(m.params?.exceptionDetails?.exception?.description ?? 'unknown');
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
// ⚠️ 必须等页面加载完再求值：CDP 连接可能早于 head 里的内联脚本执行，
//    否则会读到 undefined/false，报"静态模式标记失效"（实测偶发）。
for (let i = 0; i < 40; i++) {
  const rs = await ev('String(document.readyState)').catch(() => '');
  if (rs === 'complete') break;
  await sleep(250);
}

// 静态模式标记：**轮询**等它出现。
// ⚠️ 只等 document.readyState 不够：CDP 连上的瞬间可能还停在 about:blank，
//    readyState 是上一个文档的 'complete'，会读到 undefined/false（实测偶发）。
let mode = 'false';
for (let i = 0; i < 40; i++) {
  const url = await ev('String(location.href)').catch(() => '');
  mode = await ev('String(window.__DSH_STATIC__ === true)').catch(() => 'false');
  if (url.includes('8899') && mode === 'true') break;
  await sleep(250);
}
check('静态模式标记生效', mode === 'true', `__DSH_STATIC__=${mode}`);

// 等 app 就绪
let ready = false;
for (let i = 0; i < 60; i++) {
  if ((await ev('String(!!(window.__caseSystem && window.__caseSystem.structureLayer))')) === 'true') {
    ready = true;
    break;
  }
  await sleep(700);
}
check('应用在纯静态服务器上启动成功', ready);

const models = JSON.parse(
  await ev(`(async () => {
    const S = window.__caseSystem;
    const list = await S.api?.models?.() ?? [];
    return JSON.stringify({ n: list.length, names: list.slice(0, 4).map(m => m.name) });
  })()`, true).catch(() => '{"n":-1}')
);
console.log('   模型清单:', models);
// 静态包对模型做了**策展**（只打演示用的几个，避免评审点错），所以这里要求 ≥3 而不是 ≥10。
check('模型清单来自静态 index.json（策展后 ≥3 个）', (models.n ?? 0) >= 3, `n=${models.n}`);

// 载入 room-real 并等结构
const loaded = JSON.parse(
  await ev(`(async () => {
    const S = window.__caseSystem;
    const j = await (await fetch('/models/index.json')).json();
    const m = j.models.find(x => /room-real/.test(x.name || x.url));
    if (!m) return JSON.stringify({ err: 'no room-real' });
    await S.loadModel(m, { announce: false });
    const L = S.structureLayer;
    for (let i = 0; i < 40; i++) {
      if (L.data && L.segments.length > 0) break;
      await new Promise(r => setTimeout(r, 500));
    }
    return JSON.stringify({
      name: m.name, gauss: (S.state.gaussianPositions?.length ?? 0) / 3,
      segs: L.segments.length, cyan: L.segments.filter(s => s.color === '#4dd0e1').length,
      badge: document.getElementById('structBadge')?.textContent ?? '',
      hint: (document.getElementById('modelHint')?.textContent ?? '').slice(0, 90),
      structureUrl: m.structureUrl ?? null,
      hasData: !!L.data,
      dataMode: L.data?.sceneMode ?? null,
      dataSane: L.data?.sane ?? null,
      notice: (L.notice ?? '').slice(0, 100),
    });
  })()`, true)
);
console.log('   载入结果:', loaded);
check('模型载入成功（有高斯）', (loaded.gauss ?? 0) > 1000, `gauss=${loaded.gauss}`);
check('结构线已画出（房间盒 12 条）', (loaded.cyan ?? 0) === 12, `cyan=${loaded.cyan}`);
check('状态牌显示结构状态', /结构：/.test(loaded.badge || ''), `badge=${loaded.badge}`);

// 写操作必须明确拒绝
const writeCheck = await ev(`(async () => {
  const S = window.__caseSystem;
  try {
    await S.api.addAnnotation('x', { kind: 'point' });
    return 'NO-ERROR';
  } catch (e) {
    return String(e.message || e);
  }
})()`, true).catch((e) => 'THROW:' + e.message);
check('写操作明确拒绝并说明原因',
  /只读演示版/.test(String(writeCheck)) || /不可用/.test(String(writeCheck)), `resp=${writeCheck}`);

check('没有未捕获异常', errors.length === 0, `errors=${errors.slice(0, 2).join(' | ')}`);

const png = await send('Page.captureScreenshot', { format: 'png' });
if (png.result?.data) {
  fs.writeFileSync(path.join(OUT, 'static-viewer.png'), Buffer.from(png.result.data, 'base64'));
}

ws.close();
edge.kill();
console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail ? 1 : 0);
