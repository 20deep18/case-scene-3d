/**
 * 验证视频上传与回放链路：上传→Range 请求→Content-Type→首字节
 * 用法: node tests/repro_video_upload.mjs "D:\path\video.mp4"
 */
import { readFileSync, statSync } from 'node:fs';
import path from 'node:path';

const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const file = process.argv[2];
if (!file) {
  console.error('用法: node tests/repro_video_upload.mjs <视频文件>');
  process.exit(2);
}
const buf = readFileSync(file);
console.log(`本地文件: ${path.basename(file)}  ${(buf.length / 1048576).toFixed(2)}MB`);

const proj = await (
  await fetch(`${BASE}/api/projects`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'repro-video', operator: 'test' }),
  })
).json();
const pid = proj.project.id;

// 分片上传（与前端 api.js 一致）
const CHUNK = 8 * 1024 * 1024;
const chunks = Math.max(1, Math.ceil(buf.length / CHUNK));
const token = `t${Date.now().toString(36)}`;
let res = null;
for (let i = 0; i < chunks; i++) {
  const slice = buf.subarray(i * CHUNK, Math.min((i + 1) * CHUNK, buf.length));
  const url = `${BASE}/api/projects/${pid}/media?name=${encodeURIComponent(path.basename(file))}&chunk=${i}&chunks=${chunks}&upload=${token}`;
  const r = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/octet-stream' }, body: slice });
  res = await r.json();
  console.log(`  分片 ${i + 1}/${chunks} 上传返回: status=${r.status}`, JSON.stringify(res).slice(0, 120));
}

console.log('\n上传结果:', JSON.stringify(res));
const mediaUrl = res.url;
console.log('媒体 URL:', mediaUrl);

// 完整 GET
const full = await fetch(BASE + mediaUrl);
console.log(`\n完整 GET: status=${full.status}  content-type=${full.headers.get('content-type')}  content-length=${full.headers.get('content-length')}  accept-ranges=${full.headers.get('accept-ranges')}`);

// Range 请求（浏览器播放必需）
const ranged = await fetch(BASE + mediaUrl, { headers: { Range: 'bytes=0-1023' } });
const rbuf = new Uint8Array(await ranged.arrayBuffer());
console.log(`Range GET: status=${ranged.status}  content-range=${ranged.headers.get('content-range')}  收到字节=${rbuf.length}`);
const isMp4 = rbuf[4] === 0x66 && rbuf[5] === 0x74 && rbuf[6] === 0x79 && rbuf[7] === 0x70;
console.log(`首 8 字节: ${Array.from(rbuf.slice(0, 8)).map((b) => b.toString(16).padStart(2, '0')).join(' ')}  → ${isMp4 ? '✔ 是 MP4 (ftyp)' : '✘ 不是 MP4'}`);

// 校验字节完整性与分片是否按序拼接
const dl = new Uint8Array(await (await fetch(BASE + mediaUrl)).arrayBuffer());
console.log(`\n下载完整性: 本地 ${buf.length} 字节 / 服务端 ${dl.length} 字节  ${buf.length === dl.length ? '✔' : '✘ 大小不符'}`);
let firstDiff = -1;
for (let i = 0; i < Math.min(buf.length, dl.length); i++) {
  if (buf[i] !== dl[i]) {
    firstDiff = i;
    break;
  }
}
console.log(`字节逐位比对: ${firstDiff === -1 ? '✔ 完全一致' : `✘ 第 ${firstDiff} 字节起不同`}`);

await fetch(`${BASE}/api/projects/${pid}`, { method: 'DELETE' });
console.log('\n已清理测试案件');
