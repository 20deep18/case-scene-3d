/**
 * PnP（Perspective-n-Point）相机位姿解算 —— 纯 JavaScript 实现，无第三方依赖。
 *
 * 用途：把监控/勘查视频画面与 3DGS 重建场景对齐。
 * 给定 n≥4 组「三维点 ↔ 画面像素」对应关系，解出相机位姿，
 * 使监控画面能以正确透视贴回三维现场。
 *
 * 坐标系约定与全系统一致（COLMAP / OpenCV 相机系）：
 *   x_cam = R · x_world + t     相机朝向 +z，x 向右，y 向下
 *   相机中心 C = -Rᵀ t
 *
 * 解算流程：
 *   1. 畸变矫正到归一化像平面
 *   2. DLT：对 A（2n×12）作 AᵀA 的 Jacobi 特征分解，取最小特征值对应向量得到 P
 *   3. RQ 分解（通过 3×3 SVD）：P = K[R|t]
 *   4. Levenberg–Marquardt 最小化稳健重投影误差（可同时优化焦距与主点）
 */

// ---------------------------------------------------------------- 基础线代

function mulVec3(m, v) {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

function transpose3(m) {
  return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
}

function mul3(a, b) {
  const out = new Array(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
    }
  }
  return out;
}

function det3(m) {
  return (
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
  );
}

function invert3(m) {
  const d = det3(m);
  if (Math.abs(d) < 1e-15) return null;
  const id = 1 / d;
  return [
    (m[4] * m[8] - m[5] * m[7]) * id, (m[2] * m[7] - m[1] * m[8]) * id, (m[1] * m[5] - m[2] * m[4]) * id,
    (m[5] * m[6] - m[3] * m[8]) * id, (m[0] * m[8] - m[2] * m[6]) * id, (m[2] * m[3] - m[0] * m[5]) * id,
    (m[3] * m[7] - m[4] * m[6]) * id, (m[1] * m[6] - m[0] * m[7]) * id, (m[0] * m[4] - m[1] * m[3]) * id,
  ];
}

/**
 * 对称矩阵 Jacobi 特征分解 A = V diag(w) Vᵀ
 * @param {ArrayLike<number>} a 行优先 n×n
 * @returns {{values: Float64Array, vectors: Float64Array}} 第 k 个特征向量为 vectors[:,k]
 */
export function jacobiEigen(a, n) {
  const A = Float64Array.from(a);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;

  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
    if (off < 1e-26) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-20) continue;
        const app = A[p * n + p];
        const aqq = A[q * n + q];
        const theta = (aqq - app) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p];
          const akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k];
          const aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p];
          const vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = A[i * n + i];
  return { values, vectors: V };
}

/**
 * 3×3 奇异值分解 M = U diag(S) Vᵀ（单边 Jacobi）
 * @param {number[]} m 行优先
 * @returns {{U:number[], S:number[], V:number[]}}
 */
export function svd3(m) {
  const U = [
    m[0], m[3], m[6],
    m[1], m[4], m[7],
    m[2], m[5], m[8],
  ]; // 按列存放原始列向量
  let V = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const col = (i) => [U[i * 3], U[i * 3 + 1], U[i * 3 + 2]];
  const setCol = (i, v) => {
    U[i * 3] = v[0];
    U[i * 3 + 1] = v[1];
    U[i * 3 + 2] = v[2];
  };

  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < 2; p++) {
      for (let q = p + 1; q < 3; q++) {
        const cp = col(p);
        const cq = col(q);
        const alpha = cp[0] * cp[0] + cp[1] * cp[1] + cp[2] * cp[2];
        const beta = cq[0] * cq[0] + cq[1] * cq[1] + cq[2] * cq[2];
        const gamma = cp[0] * cq[0] + cp[1] * cq[1] + cp[2] * cq[2];
        off += gamma * gamma;
        if (Math.abs(gamma) < 1e-18 * Math.max(1, Math.sqrt(alpha * beta))) continue;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(zeta * zeta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        const np = [c * cp[0] - s * cq[0], c * cp[1] - s * cq[1], c * cp[2] - s * cq[2]];
        const nq = [s * cp[0] + c * cq[0], s * cp[1] + c * cq[1], s * cp[2] + c * cq[2]];
        setCol(p, np);
        setCol(q, nq);
        const vp = [V[0 * 3 + p], V[1 * 3 + p], V[2 * 3 + p]];
        const vq = [V[0 * 3 + q], V[1 * 3 + q], V[2 * 3 + q]];
        for (let r = 0; r < 3; r++) {
          V[r * 3 + p] = c * vp[r] - s * vq[r];
          V[r * 3 + q] = s * vp[r] + c * vq[r];
        }
      }
    }
    if (off < 1e-30) break;
  }

  const S = [0, 0, 0];
  const Ucols = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) {
    const c = col(i);
    const len = Math.hypot(c[0], c[1], c[2]);
    S[i] = len;
    Ucols[i] = len > 1e-18 ? [c[0] / len, c[1] / len, c[2] / len] : [i === 0 ? 1 : 0, i === 1 ? 1 : 0, i === 2 ? 1 : 0];
  }
  // 按奇异值降序排列
  const order = [0, 1, 2].sort((a, b) => S[b] - S[a]);
  const Uo = new Array(9);
  const Vo = new Array(9);
  const So = order.map((i) => S[i]);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      Uo[r * 3 + c] = Ucols[order[c]][r];
      Vo[r * 3 + c] = V[r * 3 + order[c]];
    }
  }
  return { U: Uo, S: So, V: Vo };
}

/**
 * 求对称半正定矩阵 A（n×n，行优先）最小特征值对应的特征向量。
 * 采用「正则化 + 反迭代」：既避开大矩阵 Jacobi 的收敛问题，也不受条件数影响。
 * 这里 A 是 AᵀA（DLT 的法方程矩阵）。
 */
function nullVector(A, n) {
  // 缩放，避免数量级过大导致数值问题
  let maxDiag = 0;
  for (let i = 0; i < n; i++) maxDiag = Math.max(maxDiag, Math.abs(A[i * n + i]));
  if (!(maxDiag > 0)) throw new Error('DLT 法方程矩阵为零');
  const reg = maxDiag * 1e-10;
  const M = Float64Array.from(A);
  for (let i = 0; i < n; i++) M[i * n + i] += reg;

  let v = new Float64Array(n);
  for (let i = 0; i < n; i++) v[i] = 1 / Math.sqrt(n);
  for (let iter = 0; iter < 80; iter++) {
    const H = Float64Array.from(M);
    const x = solveLinear(H, v, n);
    if (!x) break;
    let norm = 0;
    for (let i = 0; i < n; i++) norm += x[i] * x[i];
    norm = Math.sqrt(norm);
    if (!(norm > 1e-300)) break;
    let delta = 0;
    for (let i = 0; i < n; i++) {
      const nv = x[i] / norm;
      delta += Math.abs(nv - v[i]);
      v[i] = nv;
    }
    if (delta < 1e-15) break;
  }
  return Array.from(v);
}

/**
 * 直接线性法（DLT）求解投影矩阵 P，满足 [u,v,1]ᵀ ≅ P·[X,Y,Z,1]ᵀ
 *
 * 使用 Hartley 归一化：先把三维点平移到质心并缩放到平均距离 √3，
 * 把二维点平移到质心并缩放到平均距离 √2，解完再反归一化。
 * 这一步对数值稳定性至关重要（否则法方程矩阵条件数会到 1e8 以上）。
 *
 * @param {Array<[number,number]>} pix 去畸变后的像素坐标
 * @param {Array<[number,number,number]>} world 世界坐标
 * @returns {number[]} 12 元素、行优先的 P
 */
function dlt(pix, world) {
  const n = world.length;

  // --- 三维归一化：质心 + 平均距离 √3
  const c3 = [0, 0, 0];
  for (const P of world) for (let k = 0; k < 3; k++) c3[k] += P[k] / n;
  let meanD3 = 0;
  for (const P of world) meanD3 += Math.hypot(P[0] - c3[0], P[1] - c3[1], P[2] - c3[2]) / n;
  const s3 = meanD3 > 1e-12 ? Math.sqrt(3) / meanD3 : 1;
  const W = world.map((P) => [(P[0] - c3[0]) * s3, (P[1] - c3[1]) * s3, (P[2] - c3[2]) * s3]);

  // --- 二维归一化：质心 + 平均距离 √2
  const c2 = [0, 0];
  for (const q of pix) for (let k = 0; k < 2; k++) c2[k] += q[k] / n;
  let meanD2 = 0;
  for (const q of pix) meanD2 += Math.hypot(q[0] - c2[0], q[1] - c2[1]) / n;
  const s2 = meanD2 > 1e-12 ? Math.sqrt(2) / meanD2 : 1;
  const Q = pix.map((q) => [(q[0] - c2[0]) * s2, (q[1] - c2[1]) * s2]);

  // --- 法方程 A = Σ (r1 r1ᵀ + r2 r2ᵀ)
  const A = new Float64Array(144);
  for (let i = 0; i < n; i++) {
    const [X, Y, Z] = W[i];
    const [u, v] = Q[i];
    const r1 = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u];
    const r2 = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v];
    for (let a = 0; a < 12; a++) {
      for (let b = 0; b < 12; b++) A[a * 12 + b] += r1[a] * r1[b] + r2[a] * r2[b];
    }
  }

  const pn = nullVector(A, 12);
  // 反迭代得到的向量范数无意义，归一化后再反归一化
  const len = Math.hypot(...pn);
  if (!(len > 1e-300)) throw new Error('DLT 解退化，请检查对应点是否共线或共面');
  const phat = pn.map((v) => v / len);

  // --- 反归一化：P = T2⁻¹ · P̂ · T1
  // T2 = [[s2,0,-s2*c2x],[0,s2,-s2*c2y],[0,0,1]]，T1 = [[s3,0,0,-s3*c3x],[0,s3,0,-s3*c3y],[0,0,s3,-s3*c3z],[0,0,0,1]]
  const Phat = [
    [phat[0], phat[1], phat[2], phat[3]],
    [phat[4], phat[5], phat[6], phat[7]],
    [phat[8], phat[9], phat[10], phat[11]],
  ];
  // Phat · T1
  const tmp = Phat.map((row) => [
    row[0] * s3,
    row[1] * s3,
    row[2] * s3,
    -s3 * (row[0] * c3[0] + row[1] * c3[1] + row[2] * c3[2]) + row[3],
  ]);
  // T2⁻¹ = [[1/s2,0,c2x],[0,1/s2,c2y],[0,0,1]]
  const out = new Array(12);
  for (let r = 0; r < 3; r++) {
    const scale = r < 2 ? 1 / s2 : 1;
    const off = r < 2 ? c2[r] : 0;
    for (let c = 0; c < 4; c++) {
      out[r * 4 + c] = tmp[r][c] * scale + (r < 2 ? off * tmp[2][c] : 0);
    }
  }
  return out;
}

/**
 * 由投影矩阵 P（已反归一化，像素约定）恢复 R、t（COLMAP/OpenCV 相机系）。
 * P ≅ λ·K[R|t] 中的 λ 用迭代法确定：λ 需满足 K⁻¹M / λ 是正交矩阵。
 */
function poseFromProjectionMatrix(P, objectPoints, imagePoints, K) {
  const { fx, fy, cx, cy, dist } = K;
  const Kl = [fx, 0, cx, 0, fy, cy, 0, 0, 1];
  const Kinv = invert3(Kl);
  if (!Kinv) throw new Error('内参矩阵不可逆，请检查 fx / fy');

  const M = [P[0], P[1], P[2], P[4], P[5], P[6], P[8], P[9], P[10]];
  const p4 = [P[3], P[7], P[11]];

  // 相机中心与 λ 无关：C = -M⁻¹p₄
  const Minv = invert3(M);
  if (!Minv) throw new Error('DLT 解退化（投影矩阵前 3×3 不可逆）');
  const center = mulVec3(Minv, [-p4[0], -p4[1], -p4[2]]);

  const T = mul3(Kinv, M); // = λ·R
  // |λ| 用 T 的三个奇异值均值估计：因为 R 正交，T 的奇异值恒等于 |λ|。
  // 这是与符号无关的稳健估计（相比之下 tr(T)/3 在 R 与 T 反号时会退化，
  // 定点迭代 λ ← tr(RᵀT)/3 也会在正负分支间振荡）。
  const svT = svd3(T);
  const absLambda = (svT.S[0] + svT.S[1] + svT.S[2]) / 3;
  if (!(absLambda > 1e-300)) throw new Error('DLT 尺度恢复失败（奇异值全为 0）');
  if (globalThis.__PNP_DEBUG__) {
    console.log('[pnp] T 奇异值 =', svT.S.map((v) => v.toExponential(5)).join(', '), ' => |λ| =', absLambda.toExponential(6));
  }

  // ± λ 各试一次，用重投影误差 + 前方性联合判定。
  // 关键：必须用 projectToRotation 保证候选是真正的旋转矩阵（det=+1），
  // 否则 -R 会混进来（det=-1，不是合法相机姿态）。
  const cam = { fx, fy, cx, cy, dist };
  let best = null;
  for (const sign of [1, -1]) {
    const R = projectToRotation(T.map((v) => v / (sign * absLambda)));
    const t = mulVec3(R, center).map((v) => -v);
    const behind = countBehind(R, t, objectPoints);
    const err = behind === 0 ? reprojectionError({ R, t, K: cam }, objectPoints, imagePoints) : null;
    const mean = err && Number.isFinite(err.mean) ? err.mean : Infinity;
    const score = behind * 1e9 + (Number.isFinite(mean) ? mean : 1e12);
    if (globalThis.__PNP_DEBUG__) {
      console.log(`[pnp] sign=${sign} |λ|=${absLambda.toExponential(5)} behind=${behind} mean=${mean}`);
    }
    if (!best || score < best.score) best = { R, t, behind, score, mean, sign };
  }

  return { R: best.R, t: best.t, center, err: best.mean, behind: best.behind, lambda: absLambda, sign: best.sign };
}

/** 投影到最近的旋转矩阵（det=+1）：R = U·diag(1,1,det(UVᵀ))·Vᵀ */
export function projectToRotation(m) {
  const { U, S, V } = svd3(m);
  let R = mul3(U, transpose3(V));
  if (det3(R) < 0) {
    // 必须翻转「最小奇异值」对应的方向，才能得到 Frobenius 范数意义下最近的旋转矩阵。
    // svd3 已按奇异值降序排列，因此最小奇异值在最后一列（索引 2）。
    let minIdx = 0;
    for (let i = 1; i < 3; i++) if (S[i] < S[minIdx]) minIdx = i;
    const U2 = U.slice();
    for (let r = 0; r < 3; r++) U2[r * 3 + minIdx] *= -1;
    R = mul3(U2, transpose3(V));
  }
  return R;
}

/** 解线性方程组 Ax=b（部分主元高斯消元），A 会被修改 */
function solveLinear(A, b, n) {
  const x = Float64Array.from(b);
  for (let c = 0; c < n; c++) {
    let piv = c;
    let best = Math.abs(A[c * n + c]);
    for (let r = c + 1; r < n; r++) {
      const v = Math.abs(A[r * n + c]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best < 1e-14) return null;
    if (piv !== c) {
      for (let k = 0; k < n; k++) {
        const t = A[c * n + k];
        A[c * n + k] = A[piv * n + k];
        A[piv * n + k] = t;
      }
      const t = x[c];
      x[c] = x[piv];
      x[piv] = t;
    }
    const d = A[c * n + c];
    for (let r = c + 1; r < n; r++) {
      const f = A[r * n + c] / d;
      if (f === 0) continue;
      for (let k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k];
      x[r] -= f * x[c];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = x[r];
    for (let k = r + 1; k < n; k++) s -= A[r * n + k] * x[k];
    x[r] = s / A[r * n + r];
  }
  return x;
}

// ---------------------------------------------------------------- 投影模型

function distortNormalized(x, y, dist) {
  const k1 = dist?.[0] ?? 0;
  const k2 = dist?.[1] ?? 0;
  const p1 = dist?.[2] ?? 0;
  const p2 = dist?.[3] ?? 0;
  const k3 = dist?.[4] ?? 0;
  const r2 = x * x + y * y;
  const radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
  return [x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y];
}

/** 迭代法反畸变（把有畸变的像素坐标转成理想归一化坐标） */
function undistortNormalized(xd, yd, dist) {
  if (!dist || dist.every((v) => !v)) return [xd, yd];
  let x = xd;
  let y = yd;
  for (let i = 0; i < 12; i++) {
    const [px, py] = distortNormalized(x, y, dist);
    const dx = px - xd;
    const dy = py - yd;
    x -= dx;
    y -= dy;
    if (Math.hypot(dx, dy) < 1e-12) break;
  }
  return [x, y];
}

function projectPoint(R, t, K, P) {
  const xc = mulVec3(R, P);
  const X = xc[0] + t[0];
  const Y = xc[1] + t[1];
  const Z = xc[2] + t[2];
  // 相机背后（z<=0）的点没有有效投影，返回 NaN 以便上层识别
  if (!(Z > 1e-9)) return [NaN, NaN];
  const [xd, yd] = distortNormalized(X / Z, Y / Z, K.dist);
  return [K.fx * xd + K.cx, K.fy * yd + K.cy];
}

// ---------------------------------------------------------------- PnP 主流程

/**
 * @param {Array<[number,number,number]>} objectPoints 世界坐标（COLMAP 系，米）
 * @param {Array<[number,number]>} imagePoints 像素坐标
 * @param {{fx:number, fy:number, cx:number, cy:number, dist?:number[]}} K 内参
 */
export function solvePnP(objectPoints, imagePoints, K, opts = {}) {
  const n = objectPoints?.length ?? 0;
  if (n < 4) throw new Error(`PnP 至少需要 4 组对应点，当前 ${n} 组`);
  if (imagePoints.length !== n) throw new Error('三维点与像素点数量不一致');
  const { fx, fy, cx, cy } = K;
  if (!(fx > 0) || !(fy > 0)) throw new Error('内参 fx / fy 必须为正数');

  const dist = K.dist ?? [0, 0, 0, 0, 0];

  // ---- 1) 去畸变：把像素坐标转成理想针孔成像下的像素坐标
  const pix = objectPoints.map((_, i) => {
    const [xd, yd] = undistortNormalized((imagePoints[i][0] - cx) / fx, (imagePoints[i][1] - cy) / fy, dist);
    return [fx * xd + cx, fy * yd + cy];
  });

  // ---- 2) DLT（Hartley 归一化 + 反迭代求零空间）
  const p = dlt(pix, objectPoints);

  if (globalThis.__PNP_DEBUG__) {
    // 校验 P 是否满足 [u,v,1]ᵀ ≅ P·[X,Y,Z,1]ᵀ
    let maxResid = 0;
    for (let i = 0; i < n; i++) {
      const [X, Y, Z] = objectPoints[i];
      const h = [
        p[0] * X + p[1] * Y + p[2] * Z + p[3],
        p[4] * X + p[5] * Y + p[6] * Z + p[7],
        p[8] * X + p[9] * Y + p[10] * Z + p[11],
      ];
      if (Math.abs(h[2]) < 1e-12) continue;
      maxResid = Math.max(maxResid, Math.abs(h[0] / h[2] - pix[i][0]), Math.abs(h[1] / h[2] - pix[i][1]));
    }
    console.log('[pnp] P 投影残差（像素） =', maxResid.toExponential(4));
    console.log('[pnp] P =', p.map((v) => v.toExponential(5)).join(', '));
  }

  // ---- 3) 从 P 恢复位姿
  const pose = poseFromProjectionMatrix(p, objectPoints, imagePoints, K);
  let Rinit = pose.R;
  let tVec = pose.t;

  if (globalThis.__PNP_DEBUG__) {
    console.log('[pnp] 恢复：C =', pose.center.map((v) => v.toFixed(6)).join(','), ' 精度 =', pose.err);
  }

  const camera = { R: Rinit, t: tVec, K: { fx, fy, cx, cy, dist } };

  // ---- 4) LM 精化
  if (opts.refine !== false) {
    refinePose(camera, objectPoints, imagePoints, {
      maxIter: opts.maxIter ?? 80,
      loss: opts.loss ?? 6,
      fixIntrinsics: opts.fixIntrinsics !== false,
    });
  }
  if (countBehind(camera.R, camera.t, objectPoints) > n / 2) {
    camera.R = camera.R.map((v) => -v);
    camera.t = camera.t.map((v) => -v);
  }

  const err = reprojectionError(camera, objectPoints, imagePoints);
  const height = opts.imageSize?.height ?? 1080;
  const width = opts.imageSize?.width ?? Math.round((height * 16) / 9);
  const center = mulVec3(transpose3(camera.R), [-camera.t[0], -camera.t[1], -camera.t[2]]);

  return {
    rotation: camera.R.slice(),
    translation: camera.t.slice(),
    center,
    intrinsics: { ...camera.K },
    fovY: (2 * Math.atan(0.5 * height / camera.K.fy) * 180) / Math.PI,
    aspect: width / height,
    imageSize: { width, height },
    reprojectionError: err.mean,
    maxError: err.max,
    perPoint: err.perPoint,
    pointCount: n,
  };
}

function countBehind(R, t, pts) {
  let behind = 0;
  for (const P of pts) {
    const xc = mulVec3(R, P);
    if (xc[2] + t[2] <= 0) behind++;
  }
  return behind;
}

/** M = K·R 的 RQ 分解：借用 SVD 完成（Mᵀ = Rᵀ Kᵀ 的 QR） */
function rqDecompose(M) {
  const Mt = transpose3(M);
  const { U, V } = svd3(Mt);
  // Mt = U S Vᵀ  ->  取正交因子 Q = U Vᵀ，上三角 Rk = V S Vᵀ = Qᵀ Mt
  const Q = mul3(U, transpose3(V));
  const Rk = mul3(transpose3(Q), Mt);
  // M = Mtᵀ = Rkᵀ Qᵀ  => K = Rkᵀ（上三角），R = Qᵀ
  let K = transpose3(Rk);
  let R = transpose3(Q);
  for (let i = 0; i < 3; i++) {
    if (K[i * 3 + i] < 0) {
      for (let k = 0; k < 3; k++) K[i * 3 + k] *= -1;
      for (let k = 0; k < 3; k++) R[k * 3 + i] *= -1;
    }
  }
  return { K, R };
}

/** 重投影误差统计 */
export function reprojectionError(camera, objectPoints, imagePoints) {
  let sum = 0;
  let max = 0;
  let invalid = 0;
  const perPoint = [];
  for (let i = 0; i < objectPoints.length; i++) {
    const proj = projectPoint(camera.R, camera.t, camera.K, objectPoints[i]);
    const e = Math.hypot(proj[0] - imagePoints[i][0], proj[1] - imagePoints[i][1]);
    if (!Number.isFinite(e)) {
      invalid++;
      perPoint.push({ index: i, error: Infinity, projected: proj });
      max = Infinity;
      continue;
    }
    sum += e;
    max = Math.max(max, e);
    perPoint.push({ index: i, error: e, projected: proj });
  }
  const valid = objectPoints.length - invalid;
  return {
    mean: invalid ? Infinity : sum / objectPoints.length,
    max,
    invalid,
    perPoint,
    validCount: valid,
  };
}

/** Levenberg–Marquardt 优化（6 参数：Rodrigues 旋转 + 平移；可选 4 个内参） */
export function refinePose(camera, objectPoints, imagePoints, { maxIter = 80, loss = 6, fixIntrinsics = true } = {}) {
  const np = fixIntrinsics ? 6 : 10;
  let params = [...rotToVec(camera.R), camera.t[0], camera.t[1], camera.t[2]];
  if (!fixIntrinsics) params = [...params, camera.K.fx, camera.K.fy, camera.K.cx, camera.K.cy];

  const n = objectPoints.length;

  const evaluate = (p) => ({
    R: vecToRot(p.slice(0, 3)),
    t: [p[3], p[4], p[5]],
    K: fixIntrinsics ? camera.K : { ...camera.K, fx: p[6], fy: p[7], cx: p[8], cy: p[9] },
  });

  const residualList = (p) => {
    const { R, t, K } = evaluate(p);
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      const proj = projectPoint(R, t, K, objectPoints[i]);
      out[i] = [proj[0] - imagePoints[i][0], proj[1] - imagePoints[i][1]];
    }
    return out;
  };

  const costOf = (p) => {
    const res = residualList(p);
    let s = 0;
    for (const r of res) {
      const d = Math.hypot(r[0], r[1]);
      s += loss > 0 ? loss * loss * Math.log1p((d / loss) * (d / loss)) : d * d;
    }
    return s;
  };

  let lambda = 1e-3;
  let cost = costOf(params);

  for (let iter = 0; iter < maxIter; iter++) {
    const res = residualList(params);
    const base = evaluate(params);
    const bp = objectPoints.map((P) => projectPoint(base.R, base.t, base.K, P));
    const JtJ = new Float64Array(np * np);
    const Jtr = new Float64Array(np);

    // 数值雅可比：每个参数一步
    const step = new Float64Array(np);
    for (let a = 0; a < np; a++) step[a] = 1e-7 * Math.max(1, Math.abs(params[a]));
    const cols = [];
    for (let a = 0; a < np; a++) {
      const q = params.slice();
      q[a] += step[a];
      const e = evaluate(q);
      const col = new Float64Array(2 * n);
      for (let i = 0; i < n; i++) {
        const pr = projectPoint(e.R, e.t, e.K, objectPoints[i]);
        col[2 * i] = (pr[0] - bp[i][0]) / step[a];
        col[2 * i + 1] = (pr[1] - bp[i][1]) / step[a];
      }
      cols.push(col);
    }

    for (let i = 0; i < n; i++) {
      const ri = res[i];
      const d = Math.hypot(ri[0], ri[1]);
      const w = loss > 0 ? 1 / (1 + (d / loss) ** 2) : 1;
      for (let a = 0; a < np; a++) {
        const ja0 = cols[a][2 * i];
        const ja1 = cols[a][2 * i + 1];
        Jtr[a] += w * (ja0 * ri[0] + ja1 * ri[1]);
        for (let b = a; b < np; b++) {
          const jb0 = cols[b][2 * i];
          const jb1 = cols[b][2 * i + 1];
          JtJ[a * np + b] += w * (ja0 * jb0 + ja1 * jb1);
        }
      }
    }
    // 对称补全
    for (let a = 0; a < np; a++) for (let b = 0; b < a; b++) JtJ[a * np + b] = JtJ[b * np + a];

    let improved = false;
    for (let attempt = 0; attempt < 8; attempt++) {
      const H = Float64Array.from(JtJ);
      for (let d = 0; d < np; d++) H[d * np + d] = H[d * np + d] * (1 + lambda) + 1e-12;
      const delta = solveLinear(H, Jtr, np);
      if (!delta) {
        lambda *= 10;
        continue;
      }
      const trial = params.map((v, i) => v - delta[i]);
      const trialCost = costOf(trial);
      if (Number.isFinite(trialCost) && trialCost < cost) {
        const improvement = cost - trialCost;
        params = trial;
        cost = trialCost;
        lambda = Math.max(lambda * 0.35, 1e-10);
        improved = true;
        if (improvement < 1e-12) iter = maxIter;
        break;
      }
      lambda *= 8;
    }
    if (!improved && lambda > 1e8) break;
  }

  camera.R = vecToRot(params.slice(0, 3));
  camera.t = [params[3], params[4], params[5]];
  if (!fixIntrinsics) camera.K = { ...camera.K, fx: params[6], fy: params[7], cx: params[8], cy: params[9] };
  return camera;
}

/** 旋转矩阵 -> Rodrigues 向量 */
export function rotToVec(R) {
  const trace = R[0] + R[4] + R[8];
  const cos = Math.max(-1, Math.min(1, (trace - 1) / 2));
  const angle = Math.acos(cos);
  if (angle < 1e-9) return [0, 0, 0];
  if (angle > Math.PI - 1e-6) {
    const d = [(R[0] + 1) / 2, (R[4] + 1) / 2, (R[8] + 1) / 2];
    const axis = [Math.sqrt(Math.max(d[0], 0)), Math.sqrt(Math.max(d[1], 0)), Math.sqrt(Math.max(d[2], 0))];
    if (R[1] + R[3] < 0) axis[1] = -axis[1];
    if (R[2] + R[6] < 0) axis[2] = -axis[2];
    const len = Math.hypot(...axis) || 1;
    return axis.map((v) => (v / len) * Math.PI);
  }
  const s = 2 * Math.sin(angle);
  return [((R[7] - R[5]) / s) * angle, ((R[2] - R[6]) / s) * angle, ((R[3] - R[1]) / s) * angle];
}

/** Rodrigues 向量 -> 旋转矩阵 */
export function vecToRot(v) {
  const theta = Math.hypot(v[0], v[1], v[2]);
  if (theta < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const x = v[0] / theta;
  const y = v[1] / theta;
  const z = v[2] / theta;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const C = 1 - c;
  return [
    c + x * x * C, x * y * C - z * s, x * z * C + y * s,
    y * x * C + z * s, c + y * y * C, y * z * C - x * s,
    z * x * C - y * s, z * y * C + x * s, c + z * z * C,
  ];
}

/**
 * 把 PnP 解算结果转成三维标注 camera.pose 的结构（COLMAP 系）
 * 相机组的局部 +z 为「观察反方向」，因此 R_group = R_cv · diag(1,-1,-1)
 */
export function poseFromPnp(result, imageSize) {
  // 保证输出严格是旋转矩阵（DLT 解带轻微斜切时会破坏正交性）
  const R = projectToRotation(result.rotation);
  const Rgroup = [
    R[0], -R[1], -R[2],
    R[3], -R[4], -R[5],
    R[6], -R[7], -R[8],
  ];
  const width = imageSize?.width ?? result.imageSize?.width ?? 1920;
  const height = imageSize?.height ?? result.imageSize?.height ?? 1080;
  return {
    source: 'pnp',
    frame: 'gaussian',
    position: [...result.center],
    rotation: Rgroup,
    up: [0, -1, 0],
    fovY: (2 * Math.atan(0.5 * height / result.intrinsics.fy) * 180) / Math.PI,
    aspect: width / height,
    intrinsics: { ...result.intrinsics, width, height },
    reprojectionError: result.reprojectionError,
    solvedAt: new Date().toISOString(),
    solvedBy: 'web-pnp',
  };
}

/** 用内参与图像尺寸推算默认内参（未知焦距时用 50mm 等效估算） */
export function defaultIntrinsics(width, height, fovYDeg = 50) {
  const fy = 0.5 * height / Math.tan(((fovYDeg * Math.PI) / 180) / 2);
  return { fx: fy, fy, cx: width / 2, cy: height / 2, dist: [0, 0, 0, 0, 0] };
}
