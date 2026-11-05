/**
 * 诊断：browser_delete 在 F 盘工作区为何报「媒体文件已从服务器删除」前置断言失败。
 * 复刻测试里的相对路径运算。
 */
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import path from 'node:path';

const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
console.log('process.cwd() =', process.cwd());

const srcDir = path.join(process.cwd(), 'data', 'scene-big', 'images');
console.log('素材目录 =', srcDir, '存在=', existsSync(srcDir));
const list = existsSync(srcDir) ? readdirSync(srcDir).filter((f) => f.endsWith('.png')).sort() : [];
console.log('素材数量 =', list.length);
if (!list.length) {
  console.log('!! 找不到素材，测试会在这一步就出问题');
  process.exit(1);
}

const proj = await (
  await fetch(`${BASE}/api/projects`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'diag-media' }),
  })
).json();
const pid = proj.project.id;
console.log('案件 =', pid);

const buf = readFileSync(path.join(srcDir, list[0]));
const upRes = await fetch(`${BASE}/api/projects/${pid}/media?name=待删素材.png&chunk=0&chunks=1`, {
  method: 'POST',
  headers: { 'content-type': 'application/octet-stream' },
  body: buf,
});
const upJson = await upRes.json();
console.log('上传状态 =', upRes.status, 'url =', upJson.url);

const mediaDir = path.join(process.cwd(), 'data', 'projects', pid, 'media');
const mediaAbs = path.join(mediaDir, '待删素材.png');
console.log('mediaDir  =', mediaDir, '存在=', existsSync(mediaDir));
console.log('mediaAbs  =', mediaAbs, '存在=', existsSync(mediaAbs));
if (existsSync(mediaDir)) console.log('目录内容 =', readdirSync(mediaDir));

// 也可通过 HTTP 看看服务端自己认不认这个文件
const viaHttp = await fetch(`${BASE}${upJson.url}`);
console.log('HTTP 取文件状态 =', viaHttp.status);

await fetch(`${BASE}/api/projects/${pid}`, { method: 'DELETE' });
console.log('--- 诊断结束 ---');
