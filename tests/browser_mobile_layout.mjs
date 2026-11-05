/**
 * 移动端布局回归测试（2026-11-02）。
 *
 * 验的是"手机上真的能用"这件事，而不是"页面上有这个元素"：
 *   1. 390×844 竖屏下，左右两栏**默认在屏幕外**（画布不被挤成一条缝）
 *   2. 抽屉按钮可见且触摸区 ≥ 44px
 *   3. 点按钮 → 对应面板滑入；点遮罩 → 收起
 *   4. 画布宽度 ≈ 视口宽度（不是被两栏挤掉的残余宽度）
 *   5. 顶栏在窄屏瘦身（FPS/高斯/在线人数收起）
 *   6. 桌面 1280×800 下**恢复原状**：两栏常驻、抽屉按钮隐藏
 *
 * 用法: node tests/browser_mobile_layout.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const BASE = process.env.BASE || 'http://127.0.0.1:8787/';
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const PORT = 9341;
const OUT = path.join(process.cwd(), 'output', 'mobile-ui');
mkdirSync(OUT, { recursive: true });
const PROFILE = path.join(process.cwd(), '.browsertmp', 'mobile-' + Date.now());
mkdirSync(PROFILE, { recursive: true });

const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1280,800', 'about:blank',
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
await send('Page.enable');

// 手机视口：**先设视口再导航**。第一版是先导航再改视口，页面从 1280 布局调到 390，
// 抽屉的 transform 过渡还在跑，测量结果每次都不一样（同一份代码一次"没关"一次"没开"）。
await send('Emulation.setDeviceMetricsOverride', {
  width: 390, height: 844, deviceScaleFactor: 3, mobile: true,
});
await send('Page.navigate', { url: BASE });
await sleep(1500);
// 关掉过渡动画：测的是"最终布局"，不是"动画过程"
await ev(`(() => {
  const s = document.createElement('style');
  s.id = '__notransition';
  s.textContent = '*{transition:none!important;animation:none!important}';
  document.head.appendChild(s);
})()`);
await sleep(300);

const state = async () =>
  JSON.parse(
    await ev(`(() => {
      const r = (el) => { if (!el) return null; const b = el.getBoundingClientRect();
        return { l: Math.round(b.left), r: Math.round(b.right), w: Math.round(b.width), h: Math.round(b.height) }; };
      const L = document.getElementById('panelLeft');
      const R = document.getElementById('panelRight');
      const bl = document.getElementById('btnDrawerLeft');
      const br = document.getElementById('btnDrawerRight');
      const backdrop = document.getElementById('drawerBackdrop');
      const canvas = document.querySelector('canvas');
      const vis = (el) => !!el && getComputedStyle(el).display !== 'none';
      return JSON.stringify({
        vw: innerWidth, vh: innerHeight,
        left: r(L), right: r(R),
        btnL: r(bl), btnR: r(br),
        btnLVisible: vis(bl), btnRVisible: vis(br),
        backdropVisible: vis(backdrop),
        canvasW: canvas ? Math.round(canvas.getBoundingClientRect().width) : 0,
        fpsVisible: vis(document.getElementById('statFps')),
        gaussVisible: vis(document.getElementById('statGauss')),
        cls: document.body.className,
      });
    })()`)
  );

const s1 = await state();
console.log('\n手机 390×844 初始：', { vw: s1.vw, left: s1.left, right: s1.right, canvasW: s1.canvasW });
check('画布宽度 ≈ 视口宽度（未被两栏挤压）', Math.abs(s1.canvasW - s1.vw) <= 2, `canvasW=${s1.canvasW} vw=${s1.vw}`);
check('左栏默认在屏幕外', s1.left.r <= 0, `left.r=${s1.left.r}`);
check('右栏默认在屏幕外', s1.right.l >= s1.vw, `right.l=${s1.right.l} vw=${s1.vw}`);
check('抽屉按钮可见', s1.btnLVisible && s1.btnRVisible, `L=${s1.btnLVisible} R=${s1.btnRVisible}`);
check('左按钮触摸区 ≥44px', s1.btnL.w >= 44 && s1.btnL.h >= 44, `w=${s1.btnL.w} h=${s1.btnL.h}`);
check('右按钮触摸区 ≥44px', s1.btnR.w >= 44 && s1.btnR.h >= 44, `w=${s1.btnR.w} h=${s1.btnR.h}`);
check('顶栏瘦身：FPS 隐藏', s1.fpsVisible === false, `fpsVisible=${s1.fpsVisible}`);
check('顶栏瘦身：高斯数隐藏', s1.gaussVisible === false, `gaussVisible=${s1.gaussVisible}`);
check('遮罩默认不可见', s1.backdropVisible === false, `backdrop=${s1.backdropVisible}`);

await send('Page.captureScreenshot', { format: 'png' }).then((p) => {
  if (p.result?.data) writeFileSync(path.join(OUT, 'mobile-collapsed.png'), Buffer.from(p.result.data, 'base64'));
});

// 打开左抽屉
await ev(`document.getElementById('btnDrawerLeft').click()`);
await sleep(500);
const s2 = await state();
console.log('点左按钮后：', { cls: s2.cls, left: s2.left });
check('左抽屉已滑入（class 生效）', /drawer-open-left/.test(s2.cls || ''), `cls=${s2.cls}`);
check('左栏进入视口', s2.left.l >= 0 && s2.left.l < 80, `left.l=${s2.left.l}`);
check('打开时遮罩可见', s2.backdropVisible === true, `backdrop=${s2.backdropVisible}`);

await send('Page.captureScreenshot', { format: 'png' }).then((p) => {
  if (p.result?.data) writeFileSync(path.join(OUT, 'mobile-drawer-left.png'), Buffer.from(p.result.data, 'base64'));
});

// 点遮罩收起
await ev(`document.getElementById('drawerBackdrop').click()`);
await sleep(400);
const s3 = await state();
check('点遮罩后收起', !/drawer-open/.test(s3.cls || ''), `cls=${s3.cls}`);
check('收起后左栏又移到屏幕外', s3.left.r <= 0, `left.r=${s3.left.r}`);

// 打开右抽屉（同时应关掉左边，保证同时只有一个抽屉）
await ev(`document.getElementById('btnDrawerRight').click()`);
await sleep(500);
const s4 = await state();
check('右抽屉打开时左抽屉自动关闭', /drawer-open-right/.test(s4.cls || '') && !/drawer-open-left/.test(s4.cls || ''), `cls=${s4.cls}`);
check('右栏进入视口', s4.right.r <= s4.vw + 1 && s4.right.r > s4.vw - 80, `right.r=${s4.right.r} vw=${s4.vw}`);

await send('Page.captureScreenshot', { format: 'png' }).then((p) => {
  if (p.result?.data) writeFileSync(path.join(OUT, 'mobile-drawer-right.png'), Buffer.from(p.result.data, 'base64'));
});

// 回到桌面视口 → 必须恢复常驻布局
await send('Emulation.setDeviceMetricsOverride', {
  width: 1280, height: 800, deviceScaleFactor: 1, mobile: false,
});
await sleep(800);
const s5 = await state();
console.log('\n桌面 1280×800：', { vw: s5.vw, left: s5.left, right: s5.right });
check('桌面：左栏常驻可视', s5.left.l >= 0 && s5.left.r < s5.vw / 2, `left=${s5.left}`);
check('桌面：右栏常驻可视', s5.right.r <= s5.vw + 1 && s5.right.l > s5.vw / 2, `right=${s5.right}`);
check('桌面：抽屉按钮隐藏', !s5.btnLVisible && !s5.btnRVisible, `L=${s5.btnLVisible} R=${s5.btnRVisible}`);
check('桌面：FPS 恢复显示', s5.fpsVisible === true, `fpsVisible=${s5.fpsVisible}`);
check('桌面：body 上不留抽屉状态', !/drawer-open/.test(s5.cls || ''), `cls=${s5.cls}`);

ws.close();
edge.kill();
console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail ? 1 : 0);
