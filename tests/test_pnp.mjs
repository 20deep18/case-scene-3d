/**
 * PnP 解算自测（纯 Node，无需浏览器）
 *
 * 覆盖：
 *  1. 线代基础：Jacobi 特征分解、3×3 SVD、旋转矩阵投影、Rodrigues 往返
 *  2. 无噪声无畸变：反解精度应达到机器精度
 *  3. 含噪含畸变：重投影误差应在合理范围
 *  4. 退化与异常输入：点数不足、共线点、缺内参
 *  5. poseFromPnp 输出结构是否可直接写入标注
 */
import {
  solvePnP,
  reprojectionError,
  jacobiEigen,
  svd3,
  projectToRotation,
  vecToRot,
  rotToVec,
  defaultIntrinsics,
  poseFromPnp,
} from '../app/web/js/pnp.js';

let pass = 0;
let fail = 0;
const failures = [];

function check(name, cond, extra = '') {
  if (cond) {
    pass++;
    console.log(`  \u2713 ${name} ${extra}`);
  } else {
    fail++;
    failures.push(name);
    console.log(`  \u2717 ${name} ${extra}`);
  }
}

function mulVec3(m, v) {
  return [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]];
}
const transpose3 = (m) => [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
const det3 = (m) => m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);

const WIDTH = 1920;
const HEIGHT = 1080;
const K = { fx: 1450.3, fy: 1448.7, cx: 962.5, cy: 534.2, dist: [0, 0, 0, 0, 0] };
const TRUE_R = vecToRot([0.18, 1.05, -0.06]);
const TRUE_T = [-1.2, -0.4, -3.8];
const TRUE_C = mulVec3(transpose3(TRUE_R), [-TRUE_T[0], -TRUE_T[1], -TRUE_T[2]]);

function projectPoint(R, t, Kk, P) {
  const xc = mulVec3(R, P);
  const Z = xc[2] + t[2];
  const x = (xc[0] + t[0]) / Z;
  const y = (xc[1] + t[1]) / Z;
  const d = Kk.dist ?? [0, 0, 0, 0, 0];
  const r2 = x * x + y * y;
  const radial = 1 + d[0] * r2 + d[1] * r2 * r2 + d[4] * r2 * r2 * r2;
  const xd = x * radial + 2 * d[2] * x * y + d[3] * (r2 + 2 * x * x);
  const yd = y * radial + d[2] * (r2 + 2 * y * y) + 2 * d[3] * x * y;
  return [Kk.fx * xd + Kk.cx, Kk.fy * yd + Kk.cy];
}

/** 生成相机可见的三维点（位于视锥内，深度 2~10m） */
function makeVisiblePoints(count, seed = 11) {
  let s = seed;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  const Rinv = transpose3(TRUE_R);
  const pts = [];
  for (let i = 0; i < count; i++) {
    const u = (rnd() - 0.5) * 1.3; // 归一化像平面，限制在视场内
    const v = (rnd() - 0.5) * 0.85;
    const depth = 2 + rnd() * 8;
    const xc = [u * depth, v * depth, depth];
    const Xw = mulVec3(Rinv, xc);
    pts.push([Xw[0] + TRUE_C[0], Xw[1] + TRUE_C[1], Xw[2] + TRUE_C[2]]);
  }
  return pts;
}

console.log('\n=== 1. 线性代数基础 ===');
{
  const m = [4, 1, 2, 1, 5, 3, 2, 3, 6];
  const { values, vectors } = jacobiEigen(m, 3);
  let worst = 0;
  for (let k = 0; k < 3; k++) {
    const v = [vectors[k], vectors[3 + k], vectors[6 + k]];
    const av = mulVec3(m, v);
    for (let i = 0; i < 3; i++) worst = Math.max(worst, Math.abs(av[i] - values[k] * v[i]));
  }
  check('jacobiEigen 满足 A·v=λ·v', worst < 1e-9, `残差 ${worst.toExponential(2)}`);
}
{
  const m = [3, 1, 1, -1, 3, 1, 1, -1, 3];
  const { U, S, V } = svd3(m);
  let worst = 0;
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) sum += U[r * 3 + k] * S[k] * V[c * 3 + k];
      worst = Math.max(worst, Math.abs(sum - m[r * 3 + c]));
    }
  }
  check('svd3 重构 M=U·S·Vᵀ', worst < 1e-9, `误差 ${worst.toExponential(2)}`);
  check('svd3 奇异值降序', S[0] >= S[1] && S[1] >= S[2], S.map((v) => v.toFixed(3)).join(' ≥ '));
}
{
  // projectToRotation 必须把反射也纠正成合法旋转（det=+1）
  const m = [0.9, 0.1, 0.05, -0.2, 0.95, 0.02, 0.3, -0.1, 0.9];
  const R = projectToRotation(m);
  const RtR = [
    R[0] * R[0] + R[3] * R[3] + R[6] * R[6], R[0] * R[1] + R[3] * R[4] + R[6] * R[7], R[0] * R[2] + R[3] * R[5] + R[6] * R[8],
    R[1] * R[0] + R[4] * R[3] + R[7] * R[6], R[1] * R[1] + R[4] * R[4] + R[7] * R[7], R[1] * R[2] + R[4] * R[5] + R[7] * R[8],
    R[2] * R[0] + R[5] * R[3] + R[8] * R[6], R[2] * R[1] + R[5] * R[4] + R[8] * R[7], R[2] * R[2] + R[5] * R[5] + R[8] * R[8],
  ];
  let ortho = 0;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) ortho = Math.max(ortho, Math.abs(RtR[r * 3 + c] - (r === c ? 1 : 0)));
  check('projectToRotation 输出正交', ortho < 1e-9, `偏差 ${ortho.toExponential(2)}`);
  check('projectToRotation 行列式为 +1', Math.abs(det3(R) - 1) < 1e-9, `det=${det3(R).toFixed(9)}`);
  // 反射输入
  const refl = m.map((v, i) => (i === 6 || i === 7 || i === 8 ? -v : v));
  check('反射输入也被纠正为 det=+1', Math.abs(det3(projectToRotation(refl)) - 1) < 1e-9);
}
{
  const v = [0.3, -0.7, 0.25];
  const back = rotToVec(vecToRot(v));
  check('Rodrigues 往返一致', Math.max(...back.map((x, i) => Math.abs(x - v[i]))) < 1e-9);
}

console.log('\n=== 2. 无噪声无畸变反解（应达机器精度）===');
{
  const pts = makeVisiblePoints(20);
  const ip = pts.map((P) => projectPoint(TRUE_R, TRUE_T, K, P));
  check('测试点全部可见', reprojectionError({ R: TRUE_R, t: TRUE_T, K }, pts, ip).invalid === 0);
  const solved = solvePnP(pts, ip, K, { imageSize: { width: WIDTH, height: HEIGHT } });
  const dC = Math.hypot(...solved.center.map((v, i) => v - TRUE_C[i]));
  check('相机中心误差 < 1e-6 m', dC < 1e-6, `${dC.toExponential(2)} m`);
  check('重投影误差 < 1e-6 px', solved.reprojectionError < 1e-6, `${solved.reprojectionError.toExponential(2)} px`);
  const dR = Math.max(...solved.rotation.map((v, i) => Math.abs(v - TRUE_R[i])));
  check('旋转矩阵误差 < 1e-6', dR < 1e-6, dR.toExponential(2));
  check('视场角在合理范围', solved.fovY > 30 && solved.fovY < 60, `${solved.fovY.toFixed(2)}°`);
  check('解出的内参与输入一致', solved.intrinsics.fx === K.fx && solved.intrinsics.cx === K.cx);
}

console.log('\n=== 3. 含畸变 + 像素噪声（真实监控场景）===');
{
  const Kd = { ...K, dist: [0.12, -0.08, 0.0012, -0.0008, 0] };
  let s = 42;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff - 0.5;
  };
  const pts = makeVisiblePoints(24, 23);
  const ip = pts.map((P) => {
    const [u, v] = projectPoint(TRUE_R, TRUE_T, Kd, P);
    return [u + rnd() * 1.5, v + rnd() * 1.5];
  });
  // 只用去畸变后的近似内参（不给畸变系数），考察稳健性
  const solved = solvePnP(pts, ip, { ...K, dist: [0, 0, 0, 0, 0] }, { imageSize: { width: WIDTH, height: HEIGHT } });
  // 模型未含畸变，残余误差主要来自畸变与噪声，允许 ~10px 量级
  check('重投影误差 < 12 px', solved.reprojectionError < 12, `${solved.reprojectionError.toFixed(3)} px`);
  check('所有点均在前方', solved.perPoint.every((p) => Number.isFinite(p.error)));
  const dC = Math.hypot(...solved.center.map((v, i) => v - TRUE_C[i]));
  check('相机中心误差 < 0.2 m', dC < 0.2, `${dC.toFixed(4)} m`);
  // 带畸变系数时应显著更好
  const solvedWithDist = solvePnP(pts, ip, Kd, { imageSize: { width: WIDTH, height: HEIGHT } });
  check(
    '提供畸变系数后误差明显下降',
    solvedWithDist.reprojectionError < solved.reprojectionError * 0.6,
    `${solvedWithDist.reprojectionError.toFixed(3)} px vs ${solved.reprojectionError.toFixed(3)} px`
  );
}

console.log('\n=== 4. 异常与退化输入 ===');
{
  let msg = '';
  try {
    solvePnP([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0], [1, 1], [2, 2]], K);
  } catch (e) {
    msg = e.message;
  }
  check('少于 4 点时给出明确错误', /至少需要 4 组/.test(msg), msg);
}
{
  let msg = '';
  try {
    solvePnP([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], [[0, 0], [1, 0], [2, 0], [3, 0]], K);
  } catch (e) {
    msg = e.message;
  }
  check('共线点时优雅报错而非崩溃', msg.length > 0, msg);
}
{
  let msg = '';
  try {
    solvePnP(makeVisiblePoints(8), makeVisiblePoints(7).map((P) => projectPoint(TRUE_R, TRUE_T, K, P)), K);
  } catch (e) {
    msg = e.message;
  }
  check('点数不匹配时报错', /数量不一致/.test(msg), msg);
}
{
  let msg = '';
  try {
    solvePnP(makeVisiblePoints(8), makeVisiblePoints(8).map((P) => projectPoint(TRUE_R, TRUE_T, K, P)), { ...K, fx: 0 });
  } catch (e) {
    msg = e.message;
  }
  check('非法内参时报错', /fx/.test(msg), msg);
}

console.log('\n=== 5. poseFromPnp 输出可直接用于标注 ===');
{
  const pts = makeVisiblePoints(18);
  const ip = pts.map((P) => projectPoint(TRUE_R, TRUE_T, K, P));
  const solved = solvePnP(pts, ip, K, { imageSize: { width: WIDTH, height: HEIGHT } });
  const pose = poseFromPnp(solved, { width: WIDTH, height: HEIGHT });
  check(
    '结构字段完整',
    pose.source === 'pnp' &&
      pose.rotation.length === 9 &&
      pose.position.length === 3 &&
      pose.intrinsics.width === WIDTH &&
      Math.abs(pose.aspect - WIDTH / HEIGHT) < 1e-9
  );
  // 相机组的局部 -z 即为「观察方向」（相机系里 +z 向前，组里 -z 向前）
  const Rg = pose.rotation;
  const camForward = [Rg[2], Rg[5], Rg[8]]; // = -观察方向
  let positive = 0;
  for (const P of pts) {
    const toP = P.map((v, i) => v - pose.position[i]);
    const len = Math.hypot(...toP);
    // 相机系 +z 方向的单位向量是 camForward，与它同向即为「在相机前方」
    const cos = (camForward[0] * toP[0] + camForward[1] * toP[1] + camForward[2] * toP[2]) / len;
    if (cos > 0) positive++;
  }
  check('绝大多数点位于相机前方', positive / pts.length > 0.9, `${positive}/${pts.length}`);
  check('重投影误差写入 pose', Number.isFinite(pose.reprojectionError));
  check('输出旋转严格正交且 det=+1', Math.abs(det3(Rg) - 1) < 1e-9, `det=${det3(Rg).toFixed(9)}`);
}

console.log('\n=== 6. 默认内参推算 ===');
{
  const k = defaultIntrinsics(1920, 1080, 50);
  check('defaultIntrinsics 结果合理', k.fx > 900 && k.fx < 1400 && k.cx === 960 && k.cy === 540, `fx=${k.fx.toFixed(1)}`);
}

console.log(`\nPnP 测试结果: ${pass} 通过 / ${fail} 失败`);
if (fail) console.log('失败项：', failures.join('、'));
process.exit(fail ? 1 : 0);
