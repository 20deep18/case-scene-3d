/**
 * 用用户案件里**已存在的真实视频文件**走一遍「导入视频 → 相机标注 → 贴回现场」链路。
 * 会在用户的 case-20261003 里创建一个临时相机标注，验证完自动删除。
 * 用法: node tests/verify_real_video.mjs
 */
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PID = process.env.PID ?? 'case-20261003';

let pass = 0, fail = 0;
const check = (n, ok, d = '') => {
  if (ok) { pass++; console.log(`  [OK]   ${n}${d ? '  ' + d : ''}`); }
  else { fail++; console.log(`  [FAIL] ${n}${d ? '  ' + d : ''}`); }
};

console.log(`真实视频端到端验证（案件 ${PID}）\n`);

// 1) 找出磁盘上已有的媒体文件
const detail = await (await fetch(`${BASE}/api/projects/${PID}`)).json();
const existing = [];
for (const a of detail.project.annotations) {
  for (const m of a.media ?? []) existing.push(m);
}
// 磁盘上直接找（用户可能上传过但标注被删了）
const fs = await import('node:fs');
const path = await import('node:path');
const mediaDir = path.join(process.cwd(), 'data', 'projects', PID, 'media');
const files = fs.existsSync(mediaDir) ? fs.readdirSync(mediaDir).filter((f) => /\.(mp4|webm|mov|mkv|m4v)$/i.test(f)) : [];
check('案件目录里存在视频文件', files.length > 0, files.join(', ') || '（无）');
if (files.length === 0) { console.log('\n没有视频文件，跳过'); process.exit(1); }

const fileName = files[0];
const url = `/media/${PID}/media/${encodeURIComponent(fileName)}`;

// 2) 媒体可回放（完整 + Range）
const full = await fetch(BASE + url);
check('媒体完整请求返回 200', full.status === 200, `type=${full.headers.get('content-type')} 长度=${full.headers.get('content-length')}`);
check('支持 Range（浏览器播放/拖动必需）', full.headers.get('accept-ranges') === 'bytes');
const ranged = await fetch(BASE + url, { headers: { Range: 'bytes=0-4095' } });
const rbuf = new Uint8Array(await ranged.arrayBuffer());
check('Range 请求返回 206', ranged.status === 206, ranged.headers.get('content-range'));
const mp4 = rbuf[4] === 0x66 && rbuf[5] === 0x74 && rbuf[6] === 0x79 && rbuf[7] === 0x70;
const webm = rbuf[0] === 0x1a && rbuf[1] === 0x45 && rbuf[2] === 0xdf && rbuf[3] === 0xa3;
check('文件头是浏览器可识别的视频容器', mp4 || webm, mp4 ? 'MP4(ftyp) ✔' : webm ? 'WebM(EBML) ✔' : '未知容器');

// 3) 建一个临时相机标注（模拟前端导入后的状态）
const create = await (
  await fetch(`${BASE}/api/projects/${PID}/annotations`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      annotation: {
        kind: 'camera',
        points: [[1.5, -0.8, 1.5]],
        pose: {
          source: 'manual',
          frame: 'gaussian',
          position: [1.5, -0.8, 1.5],
          target: [0, 0, 0],
          up: [0, 0, 1],
          fovY: 45,
          aspect: 1.777,
        },
        media: [{ url, kind: 'video', name: fileName }],
        label: '验收临时相机',
      },
      operator: '验收',
    }),
  })
).json();
const camId = create.annotation?.id;
check('成功创建带视频的相机标注', !!camId, camId ? camId.slice(0, 8) : '');
check('标注类型为 camera', create.annotation?.kind === 'camera');
check('媒体已关联到标注', create.annotation?.media?.[0]?.url === url);

// 4) 幂等：再次提交同一 id
await fetch(`${BASE}/api/projects/${PID}/annotations`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ annotation: { ...create.annotation, label: '验收临时相机（重复）' }, operator: '验收' }),
});
const after = (await (await fetch(`${BASE}/api/projects/${PID}`)).json()).project.annotations.filter(
  (a) => a.kind === 'camera' && a.label.startsWith('验收临时相机')
);
check('重复提交没有产生第二个相机（无镜像对象）', after.length === 1, `匹配到 ${after.length} 个`);

// 5) 清理
if (camId) {
  const del = await fetch(`${BASE}/api/projects/${PID}/annotations/${camId}`, { method: 'DELETE' });
  check('临时标注已清理', del.ok, `status=${del.status}`);
}
const left = (await (await fetch(`${BASE}/api/projects/${PID}`)).json()).project.annotations.filter(
  (a) => a.kind === 'camera' && a.label.startsWith('验收临时相机')
);
check('清理后不再残留', left.length === 0);

console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail === 0 ? 0 : 1);
