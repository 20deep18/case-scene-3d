// REST + WebSocket 冒烟测试（使用 Node 24 内置 fetch 与 WebSocket）
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
let pass = 0;
let fail = 0;
function check(name, cond, extra = '') {
  if (cond) {
    pass++;
    console.log(`  \u2713 ${name} ${extra}`);
  } else {
    fail++;
    console.log(`  \u2717 ${name} ${extra}`);
  }
}

const health = await (await fetch(`${BASE}/api/health`)).json();
check('health', health.ok === true, JSON.stringify(health).slice(0, 120));

const created = await (
  await fetch(`${BASE}/api/projects`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: '冒烟测试案件', description: '自动化测试', operator: 'tester' }),
  })
).json();
check('create project', created.ok === true && !!created.project?.id, created.project?.id);
const pid = created.project.id;

const annRes = await fetch(`${BASE}/api/projects/${pid}/annotations`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ kind: 'point', label: '血迹', points: [[1, 2, 3]], color: '#ff0000', operator: 'tester' }),
});
const ann = await annRes.json();
check('add annotation', ann.ok === true && ann.annotation?.kind === 'point', `status=${annRes.status} ${ann.annotation?.id ?? ann.error ?? ''}`);
if (!ann.annotation) {
  console.log(JSON.stringify(ann));
  process.exit(1);
}

const patched = await (
  await fetch(`${BASE}/api/projects/${pid}/annotations/${ann.annotation.id}`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ patch: { label: '血迹(已确认)', note: '直径约4cm' }, operator: 'tester' }),
  })
).json();
check('update annotation', patched.ok === true && patched.annotation?.label === '血迹(已确认)');

const measure = await (
  await fetch(`${BASE}/api/projects/${pid}/annotations`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ kind: 'measure', points: [[0, 0, 0], [3, 4, 0]], operator: 'tester' }),
  })
).json();
check('add measure', measure.ok === true);

const listed = await (await fetch(`${BASE}/api/projects/${pid}`)).json();
check('get project', listed.ok === true && listed.project.annotations.length === 2, `annotations=${listed.project.annotations.length}`);

// ---- WebSocket 协同
const url = `${BASE.replace('http', 'ws')}/ws?project=${encodeURIComponent(pid)}&name=甲&color=%23ff0000`;
const url2 = `${BASE.replace('http', 'ws')}/ws?project=${encodeURIComponent(pid)}&name=乙&color=%2300ff00`;

const received = [];
const ws1 = new WebSocket(url);
const ws2 = new WebSocket(url2);
const msgs2 = [];
const msgs1 = [];
const wsErrors = [];
ws1.addEventListener('error', (e) => wsErrors.push(String(e.message ?? e)));
ws2.addEventListener('error', (e) => wsErrors.push(String(e.message ?? e)));

await new Promise((resolve, reject) => {
  let n = 0;
  const done = () => (++n === 2 ? resolve() : null);
  ws1.onopen = done;
  ws2.onopen = done;
  setTimeout(() => reject(new Error(`ws open timeout; errors=${JSON.stringify(wsErrors)}`)), 5000);
});
ws1.onerror = (e) => wsErrors.push(String(e.message ?? e));
ws2.onerror = (e) => wsErrors.push(String(e.message ?? e));
check('ws connect x2', ws1.readyState === 1 && ws2.readyState === 1);

ws2.onmessage = (e) => msgs2.push(JSON.parse(e.data));
ws1.onmessage = (e) => msgs1.push(JSON.parse(e.data));

const waitFor = (arr, pred, ms = 4000) =>
  new Promise((resolve) => {
    const t0 = Date.now();
    const tick = () => {
      const hit = arr.find(pred);
      if (hit) return resolve(hit);
      if (Date.now() - t0 > ms) return resolve(null);
      setTimeout(tick, 40);
    };
    tick();
  });

await new Promise((r) => setTimeout(r, 300));
check('welcome received', msgs2.some((m) => m.type === 'welcome'));
check('roster has 2', msgs2.some((m) => m.type === 'roster' && m.roster.length === 2));

ws1.send(JSON.stringify({ type: 'pose', p: [1, 2, 3], q: [0, 0, 0, 1], fov: 55, mode: 'orbit' }));
const pose = await waitFor(msgs2, (m) => m.type === 'pose');
check('pose broadcast', !!pose && pose.p[0] === 1);

ws2.send(JSON.stringify({ type: 'annotation.add', annotation: { kind: 'polyline', label: '路线', points: [[0, 0, 0], [1, 1, 0]] } }));
const addMsg = await waitFor(msgs1, (m) => m.type === 'annotation.add');
check('annotation.add broadcast', !!addMsg && addMsg.annotation.label === '路线');

ws1.send(JSON.stringify({ type: 'chat', text: '发现可疑脚印' }));
const chat = await waitFor(msgs2, (m) => m.type === 'chat');
check('chat broadcast', !!chat && chat.text === '发现可疑脚印');

// 恶意路径防护（用原始 socket 发送，避免 fetch 预先规范化 URL）
import http from 'node:http';
const rawStatus = await new Promise((resolve) => {
  const req = http.request(
    {
      host: '127.0.0.1',
      port: Number(new URL(BASE).port || 80),
      method: 'GET',
      path: `/media/${pid}/%2e%2e%2f%2e%2e%2f%2e%2e%2f%2e%2e%2fWindows/win.ini`,
    },
    (res) => {
      res.resume();
      resolve(res.statusCode);
    }
  );
  req.on('error', () => resolve(0));
  req.end();
});
check('path traversal blocked', rawStatus === 400 || rawStatus === 404, `status=${rawStatus}`);

// 清理
await fetch(`${BASE}/api/projects/${pid}`, { method: 'DELETE' });
const after = await (await fetch(`${BASE}/api/projects/${pid}`)).json();
check('delete project', after.ok === false);

ws1.close();
ws2.close();

console.log(`\n结果: ${pass} 通过 / ${fail} 失败`);
process.exit(fail ? 1 : 0);
