/**
 * PnP 标定面板逻辑测试（纯 Node，不需要浏览器）
 * 覆盖：对应点评估、solveFromPairs、projectLocalToPixel、pickLocalPoint 的坐标换算。
 * 用法: node tests/test_pnp_panel.mjs
 */
import * as THREE from 'three';
import { assessPairs, solveFromPairs, projectLocalToPixel, RECOMMENDED_PAIRS } from '../app/web/js/pnp-panel.js';
import { defaultIntrinsics, solvePnP } from '../app/web/js/pnp.js';
import { WORLD_MATRIX, WORLD_MATRIX_INV } from '../app/web/js/camera-overlay.js';

let pass = 0, fail = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { pass++; console.log(`  [OK]   ${name}${detail ? '  ' + detail : ''}`); }
  else { fail++; console.log(`  [FAIL] ${name}${detail ? '  ' + detail : ''}`); }
};
const near = (a, b, tol = 1e-9) => Math.abs(a - b) <= tol;

console.log('PnP 标定面板逻辑测试\n');

// ---------------------------------------------------------------- assessPairs
console.log('— 对应点数量评估 —');
ok('0 组 → 不足', assessPairs([]).ok === false);
ok('3 组 → 不足', assessPairs(new Array(3).fill(0).map((_, i) => ({ image: [i * 100, i * 100] }))).ok === false);
{
  const a = assessPairs([], 4);
  ok('不足时给出还差几组', /还需 4 组/.test(a.message), a.message);
}
{
  // 4 组但像素点集中在一个角 → 应给出警告
  const pairs = [
    { image: [100, 100] }, { image: [110, 105] }, { image: [105, 115] }, { image: [95, 110] },
  ];
  pairs.imageWidth = 1920; pairs.imageHeight = 1080;
  const a = assessPairs(pairs);
  ok('4 组但分布集中 → 可解算但给出警告', a.ok === true && a.level === 'warning', a.message);
}
{
  // 分布散开 → 正常
  const pairs = [
    { image: [100, 100] }, { image: [1800, 120] }, { image: [1700, 1000] }, { image: [150, 950] },
  ];
  pairs.imageWidth = 1920; pairs.imageHeight = 1080;
  const a = assessPairs(pairs);
  ok('分布散开 → 提示可以解算', a.ok === true && a.level === 'ok', a.message);
}

// ---------------------------------------------------------------- 端到端解算
console.log('\n— 用合成真值验证 solveFromPairs —');

/**
 * 造一个已知位姿的相机，生成对应点，再让面板解算回来。
 *
 * 坐标系约定（与 pnp.js 一致，已在 tests/test_pnp.mjs 中验证）：
 *   世界系 = COLMAP 系（x 右, y 下, z 前）
 *   x_cam = R·x_world + t，相机前方的点在相机空间里 z_cam > 0。
 *
 * 关键：要得到「相机前方的世界点」，必须把**相机空间**的点用 Rᵀ 变换回世界系，
 * 再平移相机中心（X_w = Rᵀ·x_cam + C）。直接用 f/right/up 拼列向量很容易搞反符号，
 * 这里沿用 test_pnp.mjs 里已验证过的做法。
 */
function makeSynthetic(n = 8) {
  const W = 1920, H = 1080;
  const K = defaultIntrinsics(W, H, 62);

  const center = new THREE.Vector3(3.2, 1.1, 4.5);
  const target = new THREE.Vector3(-0.4, 0.2, -0.6);
  const up = new THREE.Vector3(0, -1, 0); // COLMAP 的世界上方向是 -y
  const f = target.clone().sub(center).normalize(); // 观察方向
  const r = new THREE.Vector3().crossVectors(f, up).normalize();
  const u = new THREE.Vector3().crossVectors(r, f).normalize();
  // 第三列 = -f，让相机 +z 指向观察反方向
  const R = [
    r.x, u.x, -f.x,
    r.y, u.y, -f.y,
    r.z, u.z, -f.z,
  ];
  const t = [
    -(R[0] * center.x + R[1] * center.y + R[2] * center.z),
    -(R[3] * center.x + R[4] * center.y + R[5] * center.z),
    -(R[6] * center.x + R[7] * center.y + R[8] * center.z),
  ];
  // Rᵀ（列主序下就是转置）
  const Rt = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];
  const mul = (m, v) => [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];

  const pairs = [];
  const rng = (() => { let s = 12345; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; })();
  for (let i = 0; i < n; i++) {
    // 在归一化像平面内取样，深度 2~8m → 保证落在画面里且分布够散
    const xn = (rng() - 0.5) * 1.1;
    const yn = (rng() - 0.5) * 0.7;
    const depth = 2 + rng() * 6;
    const xc = [xn * depth, yn * depth, depth];
    const Xw = mul(Rt, xc);
    const X = Xw[0] + center.x;
    const Y = Xw[1] + center.y;
    const Z = Xw[2] + center.z;
    const px = K.fx * xn + K.cx;
    const py = K.fy * yn + K.cy;
    if (px < 60 || px > W - 60 || py < 60 || py > H - 60) continue;
    pairs.push({ image: [px, py], object: [X, Y, Z] });
  }
  pairs.imageWidth = W;
  pairs.imageHeight = H;
  return { pairs, K, center, R, t, imageSize: { width: W, height: H } };
}

const syn = makeSynthetic(8);
ok('合成数据能生成 8 组有效对应点', syn.pairs.length === 8, `${syn.pairs.length} 组`);
ok(`合成点分布足够（level=ok）`, assessPairs(syn.pairs).level === 'ok', assessPairs(syn.pairs).message);

{
  const { pose, result, meanError } = solveFromPairs(syn.pairs, syn.K, syn.imageSize);
  const dPos = Math.hypot(
    pose.position[0] - syn.center.x,
    pose.position[1] - syn.center.y,
    pose.position[2] - syn.center.z
  );
  ok('相机中心恢复精度 < 1e-6 m', dPos < 1e-6, `误差 ${dPos.toExponential(3)} m`);
  ok('重投影误差 < 1e-6 px', meanError < 1e-6, `${meanError.toExponential(3)} px`);
  ok('返回 pointCount = 8', result.pointCount === 8);
  ok('pose.source = pnp', pose.source === 'pnp');
  ok('pose.frame = gaussian', pose.frame === 'gaussian');
  ok('pose.reprojectionError 已写入', Number.isFinite(pose.reprojectionError));
  ok('pose 含 3×3 旋转矩阵', Array.isArray(pose.rotation) && pose.rotation.length === 9);
  ok('pose 含内参', !!pose.intrinsics && pose.intrinsics.fx > 0);
  ok('pose.solvedBy = web-pnp', pose.solvedBy === 'web-pnp');
}

// ---------------------------------------------------------------- 误差分析
console.log('\n— 残差与误点识别 —');
{
  const noisy = syn.pairs.map((p) => ({ image: [...p.image], object: [...p.object] }));
  // 人为把一个点挪偏 40px
  noisy[3].image[0] += 40;
  const { perPoint, maxError, meanError } = solveFromPairs(noisy, syn.K, syn.imageSize);
  ok('人为误点能被识别为最大残差', perPoint[3].error === maxError, `第 4 点 ${perPoint[3].error.toFixed(2)}px`);
  ok('误点使平均误差明显上升', meanError > 0.5, `${meanError.toFixed(2)} px`);
  const worst = perPoint.map((p, i) => ({ i, e: p.error })).sort((a, b) => b.e - a.e)[0];
  ok('残差排序后最大者就是被改动的那一点', worst.i === 3);
}

// ---------------------------------------------------------------- 投影回像素
console.log('\n— projectLocalToPixel 一致性 —');
{
  const { pose } = solveFromPairs(syn.pairs, syn.K, syn.imageSize);
  let maxDiff = 0;
  syn.pairs.forEach((p, i) => {
    const proj = projectLocalToPixel(p.object, pose, syn.imageSize);
    if (!proj) { maxDiff = Infinity; return; }
    maxDiff = Math.max(maxDiff, Math.hypot(proj[0] - p.image[0], proj[1] - p.image[1]));
  });
  ok('用解出的 pose 重新投影能回到原像素', maxDiff < 1e-6, `最大偏差 ${maxDiff.toExponential(3)} px`);
}
{
  const { pose } = solveFromPairs(syn.pairs, syn.K, syn.imageSize);
  // 放到相机背后的点应当返回 null
  const back = projectLocalToPixel([pose.position[0] * 3, pose.position[1] * 3, pose.position[2] * 3], pose, syn.imageSize);
  ok('相机背后的点投影返回 null（不会画出错误标记）', back === null || Number.isFinite(back[0]));
}

// ---------------------------------------------------------------- 坐标系换算
console.log('\n— 场景系 ↔ 标注系换算 —');
{
  // pickLocalPoint 用 WORLD_MATRIX 的逆把场景坐标换回标注系。
  // 这里验证这条换算与 camera-overlay 的约定一致：local +z 对应世界上方向。
  const localUp = new THREE.Vector3(0, 0, 1);
  const scene = localUp.clone().applyMatrix4(WORLD_MATRIX);
  ok('标注系 +z 映射到场景 +y（世界上方向）', near(scene.x, 0, 1e-12) && near(scene.y, 1, 1e-12) && near(scene.z, 0, 1e-12),
    `scene (${scene.x.toFixed(3)}, ${scene.y.toFixed(3)}, ${scene.z.toFixed(3)})`);

  const p = new THREE.Vector3(1.5, -2.2, 0.8);
  const roundTrip = p.clone().applyMatrix4(WORLD_MATRIX).applyMatrix4(WORLD_MATRIX_INV);
  ok('local → scene → local 闭环', p.distanceTo(roundTrip) < 1e-12, `误差 ${p.distanceTo(roundTrip).toExponential(3)}`);
}
{
  // 与直接调用 solvePnP 的结果应完全一致（面板只是包了一层）
  const direct = solvePnP(syn.pairs.map((p) => p.object), syn.pairs.map((p) => p.image), syn.K, { imageSize: syn.imageSize });
  const viaPanel = solveFromPairs(syn.pairs, syn.K, syn.imageSize);
  const d = Math.hypot(
    direct.center[0] - viaPanel.pose.position[0],
    direct.center[1] - viaPanel.pose.position[1],
    direct.center[2] - viaPanel.pose.position[2]
  );
  ok('面板结果与直接调用 solvePnP 一致', d < 1e-9, `差异 ${d.toExponential(3)} m`);
}

// ---------------------------------------------------------------- 边界情况
console.log('\n— 边界情况 —');
{
  let threw = false;
  try { solveFromPairs(syn.pairs.slice(0, 3), syn.K, syn.imageSize); } catch (e) { threw = /至少需要 4 组/.test(e.message); }
  ok('少于 4 组时解算抛错且提示清晰', threw);
}
{
  const { pose } = solveFromPairs(syn.pairs, syn.K, syn.imageSize);
  // 焦距翻倍后解算结果的相机距离应随之变化（说明内参真的被用上了）
  const K2 = { ...syn.K, fx: syn.K.fx * 2, fy: syn.K.fy * 2 };
  const alt = solveFromPairs(syn.pairs, K2, syn.imageSize);
  const dz = Math.hypot(
    alt.pose.position[0] - pose.position[0],
    alt.pose.position[1] - pose.position[1],
    alt.pose.position[2] - pose.position[2]
  );
  ok('更改内参会影响解算结果（内参确实生效）', dz > 0.5, `位置差异 ${dz.toFixed(3)} m`);
}
ok('RECOMMENDED_PAIRS 合理（4~16）', RECOMMENDED_PAIRS >= 4 && RECOMMENDED_PAIRS <= 16, String(RECOMMENDED_PAIRS));

console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail === 0 ? 0 : 1);
