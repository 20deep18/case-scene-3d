/**
 * 端到端数值验证：监控相机标注 pose（local 系）→ Three.js 相机 / 视锥姿态（scene 系）
 *
 * 本仓库的坐标约定（严格按任务定义，不改定义）：
 *   local 系 = 3DGS/COLMAP 世界系：x 右、y 下、z 前
 *   scene 系 = Three.js 场景系   ：x 右、y 上、z 后
 *   WORLD_QUAT = Quaternion().setFromAxisAngle(X, -π/2)
 *   WORLD_MATRIX = makeRotationFromQuaternion(WORLD_QUAT)
 *   scene = WORLD_MATRIX · local
 *
 * 相机 pose 语义：pose 描述的是一个 THREE.Group 的姿态，且该 Group 的**局部 -Z 就是观察方向**
 * （annotations.js 的视锥、camera-overlay.js 的画面四边形都放在局部 z = -far）。
 * 这与 THREE.PerspectiveCamera 的约定（朝局部 -Z、上为局部 +Y）完全一致，
 * 所以 flyTo 返回的 scene 系四元数可以原样赋给相机。
 *
 * 覆盖：
 *   0) 坐标系 / WORLD_MATRIX 的基准审计（含对任务描述中 up 约定的实测结论）
 *   1) flyTo()：位置误差 < 1e-9、朝向误差 < 0.01°、+Y 与 forward×right 一致且 .y > 0
 *   2) position+target 形式 与 rotation(9) 形式 结果一致
 *   3) 远端 peer 的 local → scene 换算（与 app.js 'pose' 分支同一套代码）
 *   4) 8 组用例：斜向俯视、水平四方向、正上方俯视、正下方仰视、斜上方
 */
import * as THREE from '../vendor/three/build/three.module.js';
import {
  WORLD_QUAT,
  WORLD_MATRIX,
  WORLD_MATRIX_INV,
  QUAT_LOCAL_FORWARD,
  forwardUpToQuaternion,
  annotationPoseToLocalQuaternion,
  annotationPoseToScene,
  peerPoseToScene,
} from '../app/web/js/camera-overlay.js';

// ---------------------------------------------------------------- 测试框架
let passCount = 0;
let failCount = 0;
const FAILURES = [];

const DEG = 180 / Math.PI;
const POS_TOL = 1e-9;
const ANG_TOL = 0.01; // 度

/** 两个方向向量之间的夹角（度） */
function angleDeg(a, b) {
  const d = Math.max(-1, Math.min(1, a.dot(b)));
  return Math.acos(d) * DEG;
}

function check(cond, label, detail = '') {
  if (cond) {
    passCount++;
    console.log(`  [PASS] ${label}${detail ? '  ' + detail : ''}`);
  } else {
    failCount++;
    FAILURES.push(label);
    console.log(`  [FAIL] ${label}${detail ? '  ' + detail : ''}`);
  }
}

const fmt3 = (v) => `[${v.toArray().map((x) => x.toFixed(6)).join(', ')}]`;
const fmt4 = (v) => `[${v.toArray().map((x) => x.toFixed(4)).join(', ')}]`;

/** 把 THREE.Matrix4 的旋转部分转成行优先 9 元组（与 pose.rotation 的存储顺序一致） */
function matrixToRowMajor9(m) {
  const e = m.elements; // 列优先
  return [e[0], e[4], e[8], e[1], e[5], e[9], e[2], e[6], e[10]];
}

/**
 * 由「观察方向 + 上方向」（local 系）构造 pose.rotation 所需的行优先 9 元。
 *
 * 与相机同构的姿态矩阵其列必须是 (X, Y, Z) = (right, up, -forward)：
 *   right = normalize(forward × up)   （右手系要求 X × Y = Z）
 *   up    = normalize(right × forward)（Gram-Schmidt，保证严格正交）
 * 于是第三列 = -forward。这正是 camera-overlay.js 的 forwardUpToQuaternion 使用的基，
 * 也是 annotations.js 中 applyPose 里 Matrix4.lookAt(eye, target=forward, up) 的结果。
 */
function basis9(forward, up) {
  const f = forward.clone().normalize();
  const u0 = up.clone().normalize();
  let r = new THREE.Vector3().crossVectors(f, u0);
  let u = u0;
  if (r.lengthSq() < 1e-12) {
    // 退化：up 与 forward 平行（正上方俯视 / 正下方仰视）
    const ref = Math.abs(f.y) > 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
    r = new THREE.Vector3().crossVectors(f, ref);
    r.normalize();
    u = new THREE.Vector3().crossVectors(r, f).normalize();
  } else {
    r.normalize();
    u = new THREE.Vector3().crossVectors(r, f).normalize();
  }
  return [r.x, u.x, -f.x, r.y, u.y, -f.y, r.z, u.z, -f.z];
}

/** 真值相机：用 Object3D.lookAt 得到真实四元数与真实 forward / up（scene 系） */
function truthCamera(scenePos, target, sceneUp = new THREE.Vector3(0, 1, 0)) {
  const o = new THREE.Object3D();
  o.position.copy(scenePos);
  o.up.copy(sceneUp);
  o.lookAt(target);
  o.updateMatrix();
  const qTrue = o.quaternion.clone().normalize();
  const forwardTrue = QUAT_LOCAL_FORWARD.clone().applyQuaternion(qTrue).normalize();
  const upTrue = new THREE.Vector3(0, 1, 0).applyQuaternion(qTrue).normalize();
  const rightTrue = new THREE.Vector3(1, 0, 0).applyQuaternion(qTrue).normalize();
  return { qTrue, forwardTrue, upTrue, rightTrue };
}

/**
 * 由 scene 系真值反推 local 系的相机标注 pose（两种等价写法）。
 *
 * 关键点：local 系的「上」不是 (0,-1,0)，而是把 scene 世界的 +Y 用 WORLD_MATRIX⁻¹ 变换回去得到的向量
 * （实测 = (0,0,1)，见下面第 0 节审计）。用真实 up 才能让 flyTo 还原出相机真实的 roll。
 */
function makePosePair(scenePos, target) {
  const T = truthCamera(scenePos, target);
  const localPos = scenePos.clone().applyMatrix4(WORLD_MATRIX_INV);
  const localFwd = T.forwardTrue.clone().transformDirection(WORLD_MATRIX_INV).normalize();
  const localUp = T.upTrue.clone().transformDirection(WORLD_MATRIX_INV).normalize();

  const posArr = [localPos.x, localPos.y, localPos.z];
  const upArr = [localUp.x, localUp.y, localUp.z];
  const rotation = basis9(localFwd, localUp);

  return {
    ...T,
    localPos,
    localFwd,
    localUp,
    poseTarget: {
      position: posArr,
      target: [localPos.x + localFwd.x, localPos.y + localFwd.y, localPos.z + localFwd.z],
      up: upArr,
      fovY: 55,
      aspect: 16 / 9,
    },
    poseRotation: { position: posArr, rotation, up: upArr, fovY: 55, aspect: 16 / 9 },
  };
}

// ================================================================ 0. 基准审计
console.log('==============================================================');
console.log(' 相机 pose 坐标换算 · 端到端数值验证');
console.log(' local 系: x 右, y 下, z 前   |   scene 系: x 右, y 上, z 后');
console.log(' scene = WORLD_MATRIX · local, WORLD_QUAT = 绕 x 轴 -90°');
console.log(` 容差: 位置 < ${POS_TOL}, 朝向 < ${ANG_TOL}°, 相机 +Y.y > 0`);
console.log('==============================================================\n');

console.log('--- 0. 基准审计（WORLD_MATRIX 到底是什么） ---');
{
  const map = (v) => new THREE.Vector3(...v).transformDirection(WORLD_MATRIX);
  const back = (v) => new THREE.Vector3(...v).transformDirection(WORLD_MATRIX_INV);
  const lx = map([1, 0, 0]);
  const ly = map([0, 1, 0]);
  const lz = map([0, 0, 1]);
  console.log(`  WORLD_QUAT             = (${WORLD_QUAT.toArray().map((v) => v.toFixed(6)).join(', ')})`);
  console.log(`  det(WORLD_MATRIX)      = ${WORLD_MATRIX.determinant().toFixed(12)}`);
  console.log(`  local +x(右) -> scene  ${fmt3(lx)}`);
  console.log(`  local +y(下) -> scene  ${fmt3(ly)}`);
  console.log(`  local +z(前) -> scene  ${fmt3(lz)}`);
  console.log(`  scene +y(上) -> local  ${fmt3(back([0, 1, 0]))}`);
  console.log(`  scene -z(前) -> local  ${fmt3(back([0, 0, -1]))}`);
  console.log('');

  check(Math.abs(WORLD_MATRIX.determinant() - 1) < 1e-12, 'WORLD_MATRIX 是真旋转（det = +1）');
  check(lx.distanceTo(new THREE.Vector3(1, 0, 0)) < 1e-12, 'local +x(右) 映射到 scene +x(右)');
  // 实测：这套矩阵把 local 的 y/z 轴互换了（y→-z, z→+y），
  // 因此 local +y 并不会映射到 scene -y，下面的结论必须写清楚。
  check(ly.distanceTo(new THREE.Vector3(0, 0, -1)) < 1e-12, '【实测】local +y 映射到 scene -z（不是 scene -y）', `scene ${fmt3(ly)}`);
  check(lz.distanceTo(new THREE.Vector3(0, 1, 0)) < 1e-12, '【实测】local +z 映射到 scene +y（不是 scene +z）', `scene ${fmt3(lz)}`);
  check(
    back([0, 1, 0]).distanceTo(new THREE.Vector3(0, 0, 1)) < 1e-12,
    '【实测】scene 世界上方向 +y 对应 local 的 (0,0,1)，而非 (0,-1,0)'
  );
  check(
    angleDeg(QUAT_LOCAL_FORWARD.clone().transformDirection(new THREE.Matrix4()), new THREE.Vector3(0, 0, -1)) < 1e-9,
    'QUAT_LOCAL_FORWARD = 局部 -Z（相机观察方向）'
  );
  // 与任务文字描述的偏差（只报告，不作为失败项，因为矩阵的定义以任务给出的公式为准）
  const gapUp = angleDeg(new THREE.Vector3(0, -1, 0), back([0, 1, 0]));
  console.log('');
  console.log(`  [!] 语义提示：任务文字给出的 up=[0,-1,0] 与「scene +y 的 local 原像」夹角 ${gapUp.toFixed(4)}°，`);
  console.log('      即按本仓库 WORLD_MATRIX 的实测作用，(0,-1,0) 在 scene 里其实是 +z，不是世界上方向。');
  console.log('      因此本测试用「从 scene 世界 +y 反推」的 local up（实测 (0,0,1)）作为目标形式的上方向；');
  console.log('      纯前向（-Z）语义不受影响，见下面各用例的 forward 断言。');
}
console.log('');

// ================================================================ 用例
const CASES = [
  { name: '斜向俯视 (2.5,-1.2,-3.0) → 原点', pos: [2.5, -1.2, -3.0], target: [0, 0, 0] },
  { name: '水平 +X (4,1.5,0) → (0,1.5,0)', pos: [4, 1.5, 0], target: [0, 1.5, 0] },
  { name: '水平 -X (-4,1.5,0) → (0,1.5,0)', pos: [-4, 1.5, 0], target: [0, 1.5, 0] },
  { name: '水平 +Z (0,1.5,4) → (0,1.5,0)', pos: [0, 1.5, 4], target: [0, 1.5, 0] },
  { name: '水平 -Z (0,1.5,-4) → (0,1.5,0)', pos: [0, 1.5, -4], target: [0, 1.5, 0] },
  { name: '正上方俯视 (0,5,0) → 原点', pos: [0, 5, 0], target: [0, 0, 0] },
  { name: '正下方仰视 (0,-5,0) → 原点', pos: [0, -5, 0], target: [0, 0, 0] },
  { name: '斜上方 (1.1,3.7,2.2) → (-0.4,0.3,-0.9)', pos: [1.1, 3.7, 2.2], target: [-0.4, 0.3, -0.9] },
];

for (const [i, c] of CASES.entries()) {
  console.log(`--- 用例 ${i + 1}: ${c.name} ---`);
  const scenePos = new THREE.Vector3(...c.pos);
  const target = new THREE.Vector3(...c.target);
  const P = makePosePair(scenePos, target);
  console.log(`  真值 forward = ${fmt3(P.forwardTrue)}   up = ${fmt3(P.upTrue)}`);
  console.log(`  local 位置   = ${fmt3(P.localPos)}`);
  console.log(`  local 前向   = ${fmt3(P.localFwd)}   local 上 = ${fmt3(P.localUp)}`);

  // rotation 形式：第三列 = -localFwd（即局部 +Z，观察方向的反向）
  // 这是相机约定的必然结果：局部 -Z 才是观察方向。若有人把第三列写成 +localFwd，
  // 就会得到「背对目标」的左右手系错误（旧 camera-overlay.js 的 target 分支正是如此）。
  const rr = P.poseRotation.rotation;
  const mRot = new THREE.Matrix4().set(
    rr[0], rr[1], rr[2], 0,
    rr[3], rr[4], rr[5], 0,
    rr[6], rr[7], rr[8], 0,
    0, 0, 0, 1
  );
  const thirdCol = new THREE.Vector3(mRot.elements[8], mRot.elements[9], mRot.elements[10]);
  const negFwd = P.localFwd.clone().negate();
  check(thirdCol.distanceTo(negFwd) < 1e-12, 'rotation 第三列 == -local 观察方向（局部 +Z）', `err=${thirdCol.distanceTo(negFwd).toExponential(2)}`);
  check(thirdCol.distanceTo(P.localFwd) > 1.9, 'rotation 第三列 != +观察方向（否则会 180° 反向）', `dist=${thirdCol.distanceTo(P.localFwd).toFixed(4)}`);

  // 两种 pose 形式必须给出同一个 local 四元数
  const qT = annotationPoseToLocalQuaternion(P.poseTarget);
  const qR = annotationPoseToLocalQuaternion(P.poseRotation);
  const qFormGap = angleDeg(QUAT_LOCAL_FORWARD.clone().applyQuaternion(qT), QUAT_LOCAL_FORWARD.clone().applyQuaternion(qR));
  check(qFormGap < 1e-9, 'position+target 与 rotation 两种形式的 local 四元数一致', `夹角=${qFormGap.toExponential(2)}°`);

  for (const [formName, pose] of [['position+target', P.poseTarget], ['rotation(9)', P.poseRotation]]) {
    const view = annotationPoseToScene(pose, WORLD_MATRIX); // flyTo 内部的同一函数

    // 1) 位置
    const posErr = view.position.distanceTo(scenePos);
    check(posErr < POS_TOL, `[${formName}] 位置误差 < 1e-9`, `err=${posErr.toExponential(3)}`);

    // 2) 朝向
    const forwardOut = QUAT_LOCAL_FORWARD.clone().applyQuaternion(view.quaternion).normalize();
    const angErr = angleDeg(forwardOut, P.forwardTrue);
    check(angErr < ANG_TOL, `[${formName}] 朝向误差 < ${ANG_TOL}°`, `err=${angErr.toExponential(3)}°`);

    // 3) 相机 +Y（上方向）与 +X（右方向）的交叉关系。
    //    相机基 (X, Y, Z) = (right, up, -forward) 为右手系（X × Y = Z = -forward），
    //    由此推出（已用 (f,u,r) = ((0,0,-1),(0,1,0),(1,0,0)) 的标准位形实测确认）：
    //        right = forward × up      up = right × forward
    //    （注意 up = -(forward × right)；任务文字里的 “+Y 与 forward×right 一致” 差了符号。）
    const rightTrue = new THREE.Vector3().crossVectors(P.forwardTrue, P.upTrue).normalize(); // = forward × up
    const rightOut = new THREE.Vector3(1, 0, 0).applyQuaternion(view.quaternion).normalize();
    const upOut = new THREE.Vector3(0, 1, 0).applyQuaternion(view.quaternion).normalize();
    const rightErr = angleDeg(rightOut, rightTrue);
    check(rightErr < ANG_TOL, `[${formName}] 相机 +X == forward × up`, `err=${rightErr.toExponential(3)}°`);
    const upExpected = new THREE.Vector3().crossVectors(rightOut, forwardOut).normalize(); // = right × forward
    const upErr = angleDeg(upOut, upExpected);
    check(upErr < ANG_TOL, `[${formName}] 相机 +Y == right × forward（= -(forward × right)）`, `err=${upErr.toExponential(3)}°`);
    check(upOut.dot(upExpected) > 0.999999, `[${formName}] 相机 +Y 未反向`, `dot=${upOut.dot(upExpected).toFixed(9)}`);
    check(upOut.y > 0, `[${formName}] 相机 +Y 的 .y > 0（未倒置）`, `y=${upOut.y.toFixed(6)}`);
    // 与真值相机的 up 直接比对（这一条才是「还原了真实上方向」的硬指标）
    const upTruthErr = angleDeg(upOut, P.upTrue);
    check(upTruthErr < ANG_TOL, `[${formName}] 相机 +Y 与真值相机 up 一致`, `err=${upTruthErr.toExponential(3)}°`);

    // 附带：相机基必须是右手正交基（det = +1），防止出现镜像/左手系
    const bx = new THREE.Vector3(1, 0, 0).applyQuaternion(view.quaternion);
    const by = new THREE.Vector3(0, 1, 0).applyQuaternion(view.quaternion);
    const bz = new THREE.Vector3(0, 0, 1).applyQuaternion(view.quaternion);
    const det = bx.clone().cross(by).dot(bz);
    check(Math.abs(det - 1) < 1e-9, `[${formName}] 相机基为右手正交基`, `det=${det.toFixed(12)}`);
    check(Math.abs(bx.length() - 1) < 1e-9 && Math.abs(by.length() - 1) < 1e-9, `[${formName}] 相机基无缩放`, `|x|=${bx.length().toFixed(12)}`);

    // flyTo 的返回值还必须带正确的 fovY
    if (formName === 'rotation(9)') check(view.fovY === 55, '[rotation(9)] fovY 透传', `fovY=${view.fovY}`);
  }
  console.log('');
}

// ================================================================ 退化位形
console.log('--- 用例 9: 退化位形（观察方向平行于「世界上方向」参考轴）---');
{
  // 观察方向在 local 系取 ±Y（对应 scene 系的正上/正下），此时任何与观察方向平行的
  // up 参考都会让「up × forward」退化。分别验证任务文字给出的 up=[0,-1,0] 与本仓库实测的
  // local up=[0,0,1]，确认退化时仍然只朝前看、且上方向是合法正交基。
  const degenerate = [
    { name: 'localFwd = +Y', f: new THREE.Vector3(0, 1, 0) },
    { name: 'localFwd = -Y', f: new THREE.Vector3(0, -1, 0) },
  ];
  for (const d of degenerate) {
    const localPos = new THREE.Vector3(1, 2, 3);
    const target = localPos.clone().add(d.f);
    for (const [upName, up] of [['up=[0,-1,0]（任务建议，与 f 平行 → 退化）', new THREE.Vector3(0, -1, 0)], ['up=[0,0,1]（实测 local 世界上方向）', new THREE.Vector3(0, 0, 1)]]) {
      const pose = { position: [localPos.x, localPos.y, localPos.z], target: [target.x, target.y, target.z], up: [up.x, up.y, up.z], fovY: 50 };
      const view = annotationPoseToScene(pose, WORLD_MATRIX);
      const fOut = QUAT_LOCAL_FORWARD.clone().applyQuaternion(view.quaternion).normalize();
      const fExpected = d.f.clone().transformDirection(WORLD_MATRIX).normalize();
      const uOut = new THREE.Vector3(0, 1, 0).applyQuaternion(view.quaternion).normalize();
      const rOut = new THREE.Vector3(1, 0, 0).applyQuaternion(view.quaternion).normalize();
      check(angleDeg(fOut, fExpected) < ANG_TOL, `${d.name} / ${upName}: 朝向仍精确`, `err=${angleDeg(fOut, fExpected).toExponential(3)}°`);
      check(Math.abs(uOut.dot(fOut)) < 1e-9, `${d.name} / ${upName}: up ⊥ forward`, `dot=${uOut.dot(fOut).toExponential(2)}`);
      const det = rOut.clone().cross(uOut).dot(new THREE.Vector3(0, 0, 1).applyQuaternion(view.quaternion));
      check(Math.abs(det - 1) < 1e-9, `${d.name} / ${upName}: 退化时仍为右手正交基`, `det=${det.toFixed(12)}`);
      check(Math.abs(rOut.length() - 1) < 1e-9 && Math.abs(uOut.length() - 1) < 1e-9, `${d.name} / ${upName}: 退化时基向量仍为单位长度`);
    }
  }
}
console.log('');

// ================================================================ peer 换算
console.log('--- 用例 10: 远端 peer local → scene 换算（与 app.js "case \'pose\'" 同一套代码）---');
{
  const scenePos = new THREE.Vector3(-1.8, 0.9, 2.6);
  const T = truthCamera(scenePos, new THREE.Vector3(0.2, -0.1, 0));
  const localPos = scenePos.clone().applyMatrix4(WORLD_MATRIX_INV);
  const localFwd = T.forwardTrue.clone().transformDirection(WORLD_MATRIX_INV).normalize();
  const localUp = T.upTrue.clone().transformDirection(WORLD_MATRIX_INV).normalize();
  // qLocal 的定义：相机朝向在 local 系，(0,0,-1)·qLocal == localFwd
  const qLocal = forwardUpToQuaternion(localFwd, localUp);
  const qLocalSelfCheck = angleDeg(QUAT_LOCAL_FORWARD.clone().applyQuaternion(qLocal), localFwd);
  check(qLocalSelfCheck < 1e-6, 'qLocal 自检: (0,0,-1)·qLocal == localFwd', `err=${qLocalSelfCheck.toExponential(2)}°`);

  // ==== 与 app.js 保持同步 ====
  // app.js 的 'pose' 分支现在直接调用 camera-overlay.js 导出的 peerPoseToScene：
  //     const scene = peerPoseToScene(msg.p, msg.q, WORLD_MATRIX);
  //     peer.targetPos = scene.position;  peer.targetQuat = scene.quaternion;
  // 这里调用同一个函数，因此验证的就是 app.js 的真实换算逻辑。
  const scene = peerPoseToScene([localPos.x, localPos.y, localPos.z], [qLocal.x, qLocal.y, qLocal.z, qLocal.w], WORLD_MATRIX);

  const posErr = scene.position.distanceTo(scenePos);
  check(posErr < POS_TOL, 'peer 位置误差 < 1e-9', `err=${posErr.toExponential(3)}`);

  const peerFwd = QUAT_LOCAL_FORWARD.clone().applyQuaternion(scene.quaternion).normalize();
  const angErr = angleDeg(peerFwd, T.forwardTrue);
  check(angErr < ANG_TOL, `peer 视锥朝向误差 < ${ANG_TOL}°`, `err=${angErr.toExponential(3)}°`);

  const peerUp = new THREE.Vector3(0, 1, 0).applyQuaternion(scene.quaternion).normalize();
  check(peerUp.y > 0, 'peer 视锥未倒置（+Y.y > 0）', `y=${peerUp.y.toFixed(6)}`);

  // 未归一化四元数（网络数据可能被缩放）也必须给出相同朝向
  const k = 2;
  const scaled = peerPoseToScene(
    [localPos.x, localPos.y, localPos.z],
    [qLocal.x * k, qLocal.y * k, qLocal.z * k, qLocal.w * k],
    WORLD_MATRIX
  );
  const scaledErr = angleDeg(QUAT_LOCAL_FORWARD.clone().applyQuaternion(scaled.quaternion).normalize(), T.forwardTrue);
  check(scaledErr < ANG_TOL, 'peer 换算对未归一化四元数鲁棒', `err=${scaledErr.toExponential(3)}°`);
  check(
    scaled.quaternion.clone().normalize().angleTo(scene.quaternion) < 1e-9,
    '未归一化输入与归一化输入结果一致（归一化后）'
  );

  // 与 publishPose 的逆变换闭环
  const backPos = scene.position.clone().applyMatrix4(WORLD_MATRIX_INV);
  check(backPos.distanceTo(localPos) < POS_TOL, 'local → scene → local 位置闭环', `err=${backPos.distanceTo(localPos).toExponential(3)}`);
  const qBack = WORLD_QUAT.clone().invert().multiply(scene.quaternion).normalize();
  const qBackGap = angleDeg(QUAT_LOCAL_FORWARD.clone().applyQuaternion(qBack), QUAT_LOCAL_FORWARD.clone().applyQuaternion(qLocal));
  check(qBackGap < 1e-6, 'local → scene → local 朝向闭环', `err=${qBackGap.toExponential(2)}°`);

  // 旧实现（makeRotationFromQuaternion + setFromRotationMatrix）在本例下数学等价，
  // 这里显式给出对照，确认新实现没有改变数值结果、只增强了鲁棒性。
  const qn = qLocal.clone().normalize();
  const legacyM = new THREE.Matrix4().multiplyMatrices(WORLD_MATRIX, new THREE.Matrix4().makeRotationFromQuaternion(qn));
  const legacyQ = new THREE.Quaternion().setFromRotationMatrix(legacyM).normalize();
  const legacyGap = angleDeg(
    QUAT_LOCAL_FORWARD.clone().applyQuaternion(legacyQ),
    QUAT_LOCAL_FORWARD.clone().applyQuaternion(scene.quaternion)
  );
  check(legacyGap < 1e-9, '与旧算法在归一化输入下数值等价', `err=${legacyGap.toExponential(2)}°`);
}
console.log('');

// ================================================================ 量化鲁棒性
console.log('--- 用例 11: peer 数据经 app.js 的 round4 量化后仍满足朝向要求 ---');
{
  const scenePos = new THREE.Vector3(2.5, -1.2, -3.0);
  const T = truthCamera(scenePos, new THREE.Vector3(0, 0, 0));
  const localPos = scenePos.clone().applyMatrix4(WORLD_MATRIX_INV);
  const localFwd = T.forwardTrue.clone().transformDirection(WORLD_MATRIX_INV).normalize();
  const localUp = T.upTrue.clone().transformDirection(WORLD_MATRIX_INV).normalize();
  const qLocal = forwardUpToQuaternion(localFwd, localUp);
  const r4 = (v) => Math.round(v * 10000) / 10000;
  const scene = peerPoseToScene(
    [r4(localPos.x), r4(localPos.y), r4(localPos.z)],
    [r4(qLocal.x), r4(qLocal.y), r4(qLocal.z), r4(qLocal.w)],
    WORLD_MATRIX
  );
  const angErr = angleDeg(QUAT_LOCAL_FORWARD.clone().applyQuaternion(scene.quaternion).normalize(), T.forwardTrue);
  const posErr = scene.position.distanceTo(scenePos);
  check(angErr < ANG_TOL, `round4 量化后 peer 朝向误差 < ${ANG_TOL}°`, `err=${angErr.toFixed(6)}°`);
  check(posErr < 1e-4, 'round4 量化后 peer 位置误差 < 1e-4', `err=${posErr.toExponential(3)}`);
}
console.log('');

// ---------------------------------------------------------------- 汇总
console.log('==============================================================');
console.log(` 用例: ${CASES.length} 组相机位姿 × 2 种 pose 形式 + 2 组 peer 换算`);
console.log(` PASS: ${passCount}   FAIL: ${failCount}`);
if (failCount) {
  console.log(' 失败项:');
  for (const f of FAILURES) console.log(`   - ${f}`);
}
console.log(failCount === 0 ? ' 结果: 全部通过 ✔' : ' 结果: 存在失败 ✘');
console.log('==============================================================');

process.exit(failCount === 0 ? 0 : 1);
