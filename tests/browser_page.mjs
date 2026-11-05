/**
 * 打开一个页面、等待 window.__done，然后把页面文本/日志抓回来。
 * 用法: node tests/browser_page.mjs <url> [waitMs]
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const URL_TARGET = process.argv[2];
const WAIT_MS = Number(process.argv[3] ?? 12000);
const PORT = 9223;
const PROFILE = path.join(process.cwd(), '.browsertmp', 'page-profile');

rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

const edge = spawn(
  EDGE,
  [
    '--headless=new',
    '--no-sandbox',
    '--disable-crash-reporter',
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${PROFILE}`,
    '--window-size=1280,900',
    URL_TARGET,
  ],
  { stdio: 'ignore' }
);

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
  console.error('无法连接浏览器');
  edge.kill();
  process.exit(1);
}

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const logs = [];
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  }
  if (m.method === 'Runtime.consoleAPICalled') {
    logs.push(`[${m.params.type}] ${m.params.args.map((a) => a.value ?? a.description ?? '').join(' ')}`);
  }
  if (m.method === 'Runtime.exceptionThrown') {
    logs.push('[异常] ' + (m.params.exceptionDetails?.exception?.description ?? ''));
  }
};
const send = (method, params = {}) =>
  new Promise((res) => {
    pending.set(++id, res);
    ws.send(JSON.stringify({ id, method, params }));
  });

await send('Runtime.enable');
await sleep(WAIT_MS);

const text = await send('Runtime.evaluate', {
  expression: 'document.body.innerText',
  returnByValue: true,
});
console.log('=== 页面文本 ===');
console.log(text.result?.result?.value ?? '(空)');

const done = await send('Runtime.evaluate', { expression: 'String(window.__done)', returnByValue: true });
console.log('window.__done =', done.result?.result?.value);

if (logs.length) {
  console.log('\n=== 控制台 ===');
  for (const l of logs.slice(-30)) console.log('  ' + l);
}

const shot = await send('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) {
  const out = path.join(process.cwd(), 'output', 'verify', 'page.png');
  mkdirSync(path.dirname(out), { recursive: true });
  writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log('\n截图:', out);
}

ws.close();
edge.kill();
process.exit(0);
