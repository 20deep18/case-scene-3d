/**
 * 回归：建模素材体积体检 + 素材路径（百分号编码）解析
 *
 * 覆盖两个真实缺陷：
 *   1) **「开始建模」点了没反应**：上传中文名素材后，媒体 URL 是
 *      `/media/<pid>/media/%E7%8E%B0%E5%9C%BA.mp4`。服务端此前直接拿这串字符当磁盘路径，
 *      得到的是字面量 `%E7%8E%B0...` 这种不存在的文件，于是流水线报
 *      「视频不存在 → 可用图片 0 张 → 素材不足」，用户看到的却是「素材不够」。
 *   2) **没有体积门槛**：素材体积远低于实测下限时也直接开跑，白等几十分钟。
 *
 * 用法: node tests/test_recon_materials.mjs
 */
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';

let pass = 0;
let fail = 0;
function check(name, ok, detail = '') {
  if (ok) {
    pass++;
    console.log(`  [OK]   ${name}${detail ? `  ${detail}` : ''}`);
  } else {
    fail++;
    console.log(`  [FAIL] ${name}${detail ? `  ${detail}` : ''}`);
  }
}

const fs = await import('node:fs');
const path = await import('node:path');

const enc = encodeURIComponent;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------- 建案件 + 造素材
const proj = await (
  await fetch(`${BASE}/api/projects`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'recon-materials-test', operator: 'test' }),
  })
).json();
const pid = proj.project.id;
const mediaDir = path.join(process.cwd(), 'data', 'projects', pid, 'media');
fs.mkdirSync(mediaDir, { recursive: true });
console.log(`测试案件 ${pid}\n素材目录 ${mediaDir}`);

// 中文名 + 空格 + 括号：把 URL 编码的坑一次踩全
const videoName = '现场勘查（东侧）视频 01.mp4';
const imageName = '现场照片 01.png';
const videoBytes = 1_200_000;
const imageBytes = 300_000;
fs.writeFileSync(path.join(mediaDir, videoName), Buffer.alloc(videoBytes, 7));
fs.writeFileSync(path.join(mediaDir, imageName), Buffer.alloc(imageBytes, 9));

const assets = [
  { url: `/media/${enc(pid)}/media/${enc(videoName)}`, name: videoName, kind: 'video' },
  { url: `/media/${enc(pid)}/media/${enc(imageName)}`, name: imageName, kind: 'image' },
];

// 挂到「相机」标注上（模拟用户在多源画面里导入）
const annoRes = await fetch(`${BASE}/api/projects/${pid}/annotations`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({
    annotation: {
      kind: 'camera',
      label: '现场素材组',
      points: [[0, 1.5, -4]],
      pose: { position: [0, 1.5, -4], quaternion: [0, 0, 0, 1], fov: 60, source: 'default' },
      media: assets.map((a) => ({ url: a.url, name: a.name, kind: a.kind })),
    },
    operator: 'test',
  }),
});
check('相机标注建立成功（中文名素材挂得上）', annoRes.ok, `HTTP ${annoRes.status}`);

const postMaterials = (body) =>
  fetch(`${BASE}/api/projects/${enc(pid)}/recon/materials`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  }).then((r) => r.json());

// ---------------------------------------------------------------- 1. 体积体检
console.log('\n--- 1. 素材体积体检（URL 是百分号编码的中文名）---');
const m1 = await postMaterials({ assets });
check('materials 接口返回 ok', m1.ok === true);
check('编码过的中文名素材能被定位到', m1.count === 2 && (m1.missing ?? []).length === 0,
  `count=${m1.count} missing=${JSON.stringify(m1.missing)}`);
check('素材总体积 = 各文件大小之和', m1.bytes === videoBytes + imageBytes,
  `got ${m1.bytes} want ${videoBytes + imageBytes}`);
check('按类型统计正确', m1.videos === 1 && m1.images === 1, `videos=${m1.videos} images=${m1.images}`);
check('返回实测最小体积阈值（>0）', Number(m1.minBytes) > 0, `minBytes=${m1.minBytes}`);
check('1.5 MB 素材被判为「体积不足」', m1.enough === false,
  `bytes=${m1.bytes} minBytes=${m1.minBytes}`);
check('阈值来源可自证（带 measured/report 字段）', 'measured' in (m1.threshold ?? {}),
  JSON.stringify(m1.threshold));

// ---------------------------------------------------------------- 2. 文件名的两种写法
console.log('\n--- 2. 兼容：解码后的名字 / 只给文件名 / 直接给字符串 ---');
const m2 = await postMaterials({ assets: [videoName] });
check('直接给「解码后的文件名」也能定位', m2.bytes === videoBytes, `bytes=${m2.bytes}`);
const m3 = await postMaterials({ assets: [{ url: `/media/${enc(pid)}/media/${enc(videoName)}` }] });
check('只给 url（不带 name/kind）也能定位', m3.bytes === videoBytes, `bytes=${m3.bytes}`);

// ---------------------------------------------------------------- 3. 缺失文件要报出来
console.log('\n--- 3. 素材文件不存在时要说清楚（而不是笼统「素材不足」）---');
const m4 = await postMaterials({ assets: [{ url: `/media/${enc(pid)}/media/${enc('不存在的视频.mp4')}` }] });
check('缺失文件进入 missing 列表', (m4.missing ?? []).length === 1 && m4.bytes === 0,
  `missing=${JSON.stringify(m4.missing)} bytes=${m4.bytes}`);

console.log('\n--- 4. 不传 assets 时扫描本案 media 目录 ---');
const m5 = await postMaterials({});
check('扫描到 media 目录下全部素材', m5.count === 2 && m5.bytes === videoBytes + imageBytes,
  `count=${m5.count} bytes=${m5.bytes}`);

console.log('\n--- 4b. 阈值口径与 pipeline/recon_threshold.json 一致 ---');
const thresholdFile = JSON.parse(fs.readFileSync(path.join(process.cwd(), 'pipeline', 'recon_threshold.json'), 'utf8'));
check('接口返回的 minBytes 就是阈值文件里的值', m5.minBytes === thresholdFile.minBytes,
  `api=${m5.minBytes} file=${thresholdFile.minBytes}`);
check('阈值文件标记为已实测', thresholdFile.measured === true, `measuredAt=${thresholdFile.measuredAt}`);

console.log('\n--- 4c. 体积达标时 enough=true（真的写一个超过下限的文件）---');
const bigName = '足够大的素材.mp4';
fs.writeFileSync(path.join(mediaDir, bigName), Buffer.alloc(thresholdFile.minBytes + 1024 * 1024, 5));
const m6 = await postMaterials({ assets: [{ url: `/media/${enc(pid)}/media/${enc(bigName)}` }] });
check('超过下限的素材被判为「体积充足」', m6.enough === true, `bytes=${m6.bytes} minBytes=${m6.minBytes}`);
fs.rmSync(path.join(mediaDir, bigName), { force: true });

// ---------------------------------------------------------------- 5. 真的启动一次建模
console.log('\n--- 5. POST /recon：素材必须被解析成「存在的文件」---');
const startRes = await fetch(`${BASE}/api/projects/${enc(pid)}/recon`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ name: 'materials-test', params: { iterations: 100 }, assets }),
});
const startJson = await startRes.json().catch(() => ({}));
check('启动建模返回 202（旧实现在这里会 400/素材 0 个）', startRes.status === 202,
  `HTTP ${startRes.status} ${startJson.error ?? ''}`);
check('后端登记了 2 个素材', startJson.job?.input?.count === 2,
  `count=${startJson.job?.input?.count} videos=${startJson.job?.input?.videos} images=${startJson.job?.input?.images}`);

// 等日志出来：必须出现「视频不存在」以外的信息
const jobId = startJson.job?.id ?? null;
let logText = '';
for (let i = 0; i < 40 && jobId; i++) {
  const r = await fetch(`${BASE}/api/projects/${enc(pid)}/recon/${enc(jobId)}/log?from=0&tail=200`).then((x) => x.json());
  logText = (r.lines ?? []).join('\n');
  if (/可用于重建的图片总数|视频不存在|抽帧 现场/.test(logText)) break;
  await sleep(500);
}
check('流水线日志里没有「视频不存在」', jobId ? !/视频不存在/.test(logText) : false,
  logText ? '' : '（没拿到日志）');
// 日志里出现**解码后**的中文名，说明从 URL 到磁盘路径整条链都还原正确了
check('流水线拿到的素材名是解码后的可读文本',
  logText.includes(videoName), logText.includes(videoName) ? videoName : '（日志里没找到解码后的文件名）');

// 收尾：取消任务（假 mp4 会很快失败，取消应当被优雅处理）
if (jobId) {
  const c = await fetch(`${BASE}/api/projects/${enc(pid)}/recon/${enc(jobId)}/cancel`, { method: 'POST' });
  check('取消接口可用（任务可能已自行结束，409 也算正常）', c.status === 200 || c.status === 409, `HTTP ${c.status}`);
}

await fetch(`${BASE}/api/projects/${enc(pid)}`, { method: 'DELETE' });
console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail === 0 ? 0 : 1);
