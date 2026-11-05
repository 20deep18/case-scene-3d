/**
 * 端到端验收：案件 → 相机标注（带已存在的真实视频）→ 媒体可回放 → 列表只有 1 个相机
 * 用法: node tests/verify_user_case.mjs
 */
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';

let pass = 0, fail = 0;
const check = (n, ok, d = '') => {
  if (ok) { pass++; console.log(`  [OK]   ${n}${d ? '  ' + d : ''}`); }
  else { fail++; console.log(`  [FAIL] ${n}${d ? '  ' + d : ''}`); }
};

console.log('用户案件端到端验收\n');

const list = await (await fetch(`${BASE}/api/projects`)).json();
check('案件列表可读取', Array.isArray(list.projects), `${list.projects.length} 个案件`);

const target = list.projects.find((p) => p.id === 'case-20261003') ?? list.projects[0];
if (!target) {
  console.log('没有案件可验证');
  process.exit(1);
}
const pid = target.id;
console.log(`  使用案件：${pid}（${target.name}）\n`);

// 已有媒体文件
const detail = await (await fetch(`${BASE}/api/projects/${pid}`)).json();
const cams = detail.project.annotations.filter((a) => a.kind === 'camera');
console.log(`  该案件现有相机 ${cams.length} 个`);

for (const c of cams) {
  for (const m of c.media ?? []) {
    const r = await fetch(BASE + m.url, { headers: { Range: 'bytes=0-2047' } });
    const buf = new Uint8Array(await r.arrayBuffer());
    const mp4 = buf[4] === 0x66 && buf[5] === 0x74 && buf[6] === 0x79 && buf[7] === 0x70;
    check(
      `媒体可回放 ${m.url.split('/').pop()}`,
      r.status === 206 && mp4,
      `status=${r.status} type=${r.headers.get('content-type')} ${mp4 ? 'MP4(ftyp) ✔' : '非 MP4'}`
    );
  }
}

// 幂等：再提交一次同一个相机，数量不应变化
if (cams.length > 0) {
  const before = cams.length;
  const res = await fetch(`${BASE}/api/projects/${pid}/annotations/${cams[0].id}`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ patch: { note: '验收测试写入的备注' }, operator: '验收' }),
  });
  check('相机标注可更新（PATCH）', res.ok, `status=${res.status}`);
  const after = (await (await fetch(`${BASE}/api/projects/${pid}`)).json()).project.annotations.filter((a) => a.kind === 'camera');
  check('更新后相机数量不变（不产生镜像对象）', after.length === before, `${before} → ${after.length}`);

  // 还原备注
  await fetch(`${BASE}/api/projects/${pid}/annotations/${cams[0].id}`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ patch: { note: cams[0].note ?? '' }, operator: '验收' }),
  });
}

// 前端资源与关键脚本
const assets = ['/', '/js/splat-renderer.js', '/js/camera-overlay.js', '/js/ui.js', '/js/app.js', '/css/app.css', '/gltest.html'];
const gone = [];
for (const a of assets.slice(0, 6)) {
  const r = await fetch(BASE + a, { method: 'HEAD' });
  if (!r.ok) gone.push(a);
}
check('前端关键资源全部可访问', gone.length === 0, gone.length ? `失败：${gone.join(', ')}` : '6/6');

// 调试用的临时页面应已从磁盘删除。
// 注意：服务端对未知路径有 SPA 兜底（返回 index.html），所以不能只看状态码，
// 必须检查返回内容不是那个调试页面本身。
for (const p of ['/gltest.html', '/glreal.html']) {
  const r = await fetch(BASE + p);
  const body = await r.text();
  const isDebugPage = /真实 3DGS 着色器最小验证|WebGL2 点精灵最小验证/.test(body);
  check(`调试页面 ${p} 已清理`, !isDebugPage, isDebugPage ? '仍在返回调试页内容' : `返回 SPA 兜底（${r.status}）`);
}

console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail === 0 ? 0 : 1);
