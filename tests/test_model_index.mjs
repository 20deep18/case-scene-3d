/**
 * 校验后端模型索引与 .splat 二进制格式是否符合前端预期。
 * 用法: node tests/test_model_index.mjs [baseUrl]
 */
const base = process.argv[2] ?? 'http://127.0.0.1:8787';
let pass = 0;
let fail = 0;
const check = (name, cond, extra = '') => {
  if (cond) {
    pass++;
    console.log(`  \u2713 ${name} ${extra}`);
  } else {
    fail++;
    console.log(`  \u2717 ${name} ${extra}`);
  }
};

const res = await fetch(`${base}/models/index.json`);
check('模型索引可访问', res.ok, `status=${res.status}`);
const idx = await res.json();
check('索引含 models 数组', Array.isArray(idx.models));
check('至少有一个模型', idx.models.length > 0, `${idx.models.length} 个`);

for (const m of idx.models) {
  console.log(`\n  [${m.name}] ${(m.bytes / 1048576).toFixed(2)}MB  format=${m.format}  gaussians=${m.gaussianCount ?? '?'}`);

  // Range 请求（前端流式下载依赖它）
  const range = await fetch(base + m.url, { headers: { Range: 'bytes=0-63' } });
  check('支持 Range 请求', range.status === 206, `status=${range.status}`);
  const head = await fetch(base + m.url, { method: 'HEAD' });
  check('模型可访问', head.ok, `status=${head.status}`);
  check('Content-Length 与索引一致', Number(head.headers.get('content-length')) === m.bytes);

  if (m.format !== 'splat') {
    console.log('    （非 .splat 格式，跳过二进制布局校验）');
    continue;
  }

  // 取前两个高斯，按 32 字节格式解析
  const buf = await (await fetch(base + m.url, { headers: { Range: 'bytes=0-63' } })).arrayBuffer();
  const dv = new DataView(buf);
  const u8 = new Uint8Array(buf);
  const g0 = {
    pos: [dv.getFloat32(0, true), dv.getFloat32(4, true), dv.getFloat32(8, true)],
    scale: [dv.getFloat32(12, true), dv.getFloat32(16, true), dv.getFloat32(20, true)],
    rgba: [u8[24], u8[25], u8[26], u8[27]],
    quat: [u8[28], u8[29], u8[30], u8[31]],
  };
  const finite = [...g0.pos, ...g0.scale].every(Number.isFinite);
  const inRange = [...g0.pos, ...g0.scale].every((v) => Math.abs(v) < 1e5);
  check('首个高斯数值有限且量级正常', finite && inRange, `pos=[${g0.pos.map((v) => v.toFixed(3))}]`);
  check('颜色通道在 0..255', g0.rgba.every((v) => v >= 0 && v <= 255), `rgba=[${g0.rgba}]`);
  check('四元数字节在 0..255', g0.quat.every((v) => v >= 0 && v <= 255), `quat=[${g0.quat}]`);

  // 四元数解码后应接近单位长度
  const q = g0.quat.map((v) => (v - 128) / 128);
  const norm = Math.hypot(...q);
  check('四元数解码后接近单位长度', Math.abs(norm - 1) < 0.08, `|q|=${norm.toFixed(4)}`);

  // 包围盒应与元数据一致（抽查若干高斯）
  if (m.bounds) {
    const sample = await (await fetch(base + m.url, { headers: { Range: `bytes=0-${Math.min(m.bytes, 32 * 400) - 1}` } })).arrayBuffer();
    const sdv = new DataView(sample);
    let inside = 0;
    const n = Math.floor(sample.byteLength / 32);
    for (let i = 0; i < n; i++) {
      const o = i * 32;
      const x = sdv.getFloat32(o, true);
      const y = sdv.getFloat32(o + 4, true);
      const z = sdv.getFloat32(o + 8, true);
      const mn = m.bounds.min;
      const mx = m.bounds.max;
      const pad = 0.5;
      if (x >= mn[0] - pad && x <= mx[0] + pad && y >= mn[1] - pad && y <= mx[1] + pad && z >= mn[2] - pad && z <= mx[2] + pad) inside++;
    }
    const ratio = n ? inside / n : 0;
    check('抽样高斯落在元数据包围盒内', ratio > 0.98, `${inside}/${n} (${(ratio * 100).toFixed(1)}%)`);
  }
}

console.log(`\n结果: ${pass} 通过 / ${fail} 失败`);
process.exit(fail ? 1 : 0);
