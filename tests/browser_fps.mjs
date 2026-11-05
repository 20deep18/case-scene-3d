/**
 * 在真实浏览器里测量渲染循环的帧率，验证 60FPS 上限是否生效、FPS 显示是否合理。
 * 用法: node tests/browser_fps.mjs [url]
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const URL_TARGET = process.argv[2] ?? 'http://127.0.0.1:8787/';
const PORT = 9227;
const PROFILE = path.join(process.cwd(), '.browsertmp', 'fps-profile');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1280,800', URL_TARGET,
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
if (!page) { console.error('无法连接浏览器'); edge.kill(); process.exit(1); }

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
};
const send = (method, params = {}) => new Promise((res) => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
await send('Runtime.enable');
await sleep(11000); // 等模型载入

// 统计 3 秒内 requestAnimationFrame 的回调次数（与页面渲染循环同源）
const measure = `
(async () => {
  const t0 = performance.now();
  let count = 0;
  await new Promise((resolve) => {
    const tick = () => {
      count++;
      if (performance.now() - t0 >= 3000) resolve();
      else requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
  const dt = performance.now() - t0;
  const hudFps = document.querySelector('#hudFps')?.textContent ?? null;
  return JSON.stringify({
    rafFps: +(count / (dt / 1000)).toFixed(1),
    frames: count,
    ms: +dt.toFixed(0),
    rendererFpsText: hudFps,
  });
})()
`;

// 先记录 HUD 上的 FPS 文本，再测 rAF 频率
const hud = await send('Runtime.evaluate', {
  expression: `(() => {
    const el = [...document.querySelectorAll('*')].find(n => /FPS/i.test(n.textContent||'') && n.children.length === 0);
    return el ? el.textContent.trim() : '(未找到 FPS 文本)';
  })()`,
  returnByValue: true,
});
console.log('HUD 上的 FPS 文本:', hud.result?.result?.value);

const res = await send('Runtime.evaluate', { expression: measure, returnByValue: true, awaitPromise: true });
if (res.result?.exceptionDetails) console.log('异常:', res.result.exceptionDetails.exception?.description);
const val = res.result?.result?.value;
console.log('测量结果:', val);

// 再读一次 HUD，确认显示值来自 500ms 时间窗（应当接近 rAF 频率且不超过 60 太多）
await sleep(1500);
const hud2 = await send('Runtime.evaluate', {
  expression: `(() => {
    const el = [...document.querySelectorAll('*')].find(n => /FPS/i.test(n.textContent||'') && n.children.length === 0);
    return el ? el.textContent.trim() : '(未找到)';
  })()`,
  returnByValue: true,
});
console.log('稳定后 HUD FPS 文本:', hud2.result?.result?.value);

ws.close();
edge.kill();
process.exit(0);
