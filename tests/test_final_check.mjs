/** 最终验收：确认后端在线、模型清单正确、前端首个模型是高质量演示场景 */
const base = process.argv[2] ?? 'http://127.0.0.1:8787';

const health = await (await fetch(`${base}/api/health`)).json();
console.log(`后端：${health.name} ${health.version}`);
console.log(`      Node ${health.node}，已运行 ${Math.round(health.uptime)} 秒，在线协同客户端 ${health.clients} 个`);

const idx = await (await fetch(`${base}/models/index.json`)).json();
console.log(`\n可用三维现场：${idx.models.length} 个（列表按修改时间降序，前端默认选第一个）`);
for (const [i, m] of idx.models.entries()) {
  const mark = i === 0 ? '★ 默认' : '     ';
  const mb = (m.bytes / 1048576).toFixed(2).padStart(6);
  const gc = m.gaussianCount ? `${m.gaussianCount} 高斯` : '未知数量';
  console.log(`  ${mark} ${m.name.padEnd(26)} ${mb} MB  ${gc}`);
}

// 首页与关键资源可达性
const assets = ['/', '/css/app.css', '/js/app.js', '/js/splat-renderer.js', '/js/pnp.js', '/vendor/three/build/three.module.js'];
console.log('\n关键资源：');
let ok = 0;
for (const a of assets) {
  const r = await fetch(base + a, { method: 'HEAD' });
  const type = r.headers.get('content-type') ?? '';
  const good = r.ok;
  if (good) ok++;
  console.log(`  ${good ? '✓' : '✗'} ${a.padEnd(42)} ${r.status}  ${type.split(';')[0]}`);
}

console.log(`\n结果：${ok}/${assets.length} 个资源可访问`);
process.exit(ok === assets.length ? 0 : 1);
