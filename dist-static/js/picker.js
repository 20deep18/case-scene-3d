/**
 * 三维空间拾取器：把高斯点云组织成均匀网格，支持屏幕空间最近点拾取与局部法向估计。
 * 3DGS 场景没有三角面片，因此不能用常规 Raycaster；这里用「射线最近高斯点 + 邻域 PCA」估计表面点。
 */
import * as THREE from 'three';

export class SplatPicker {
  /**
   * @param {Float32Array} position 平坦化的 xyz
   * @param {number} count
   * @param {{cellSize?: number, maxNeighbors?: number}} [opts]
   */
  constructor(position, count, opts = {}) {
    this.position = position;
    this.count = count;

    // 估计范围与网格尺寸
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    const stride = Math.max(1, Math.floor(count / 400000));
    for (let i = 0; i < count; i += stride) {
      const x = position[i * 3], y = position[i * 3 + 1], z = position[i * 3 + 2];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }
    this.min = new THREE.Vector3(minX, minY, minZ);
    this.max = new THREE.Vector3(maxX, maxY, maxZ);
    const diag = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) || 1;
    this.cell = opts.cellSize ?? Math.max(diag / 160, 0.02);
    this.maxNeighbors = opts.maxNeighbors ?? 48;
    this.grid = new Map();
    this.bounds = null;
    this._build();
  }

  /** 设置点云包围盒（用于限定射线采样范围），传入 computeBounds 的结果 */
  set bounds(b) {
    this._bounds = b
      ? {
          min: b.min ?? [this.min.x, this.min.y, this.min.z],
          max: b.max ?? [this.max.x, this.max.y, this.max.z],
          diagonal: b.diagonal ?? Math.hypot(
            (b.max?.[0] ?? this.max.x) - (b.min?.[0] ?? this.min.x),
            (b.max?.[1] ?? this.max.y) - (b.min?.[1] ?? this.min.y),
            (b.max?.[2] ?? this.max.z) - (b.min?.[2] ?? this.min.z)
          ),
        }
      : null;
  }

  get bounds() {
    return this._bounds ?? null;
  }

  _key(ix, iy, iz) {
    // 每个轴 21 位，可覆盖 ±100 万格
    return `${ix},${iy},${iz}`;
  }

  _build() {
    const c = this.cell;
    const { position, count } = this;
    // 为大场景降采样建格，避免内存爆炸（拾取精度足够）
    const sample = count > 2500000 ? 2 : 1;
    this.sample = sample;
    for (let i = 0; i < count; i += sample) {
      const ix = Math.floor((position[i * 3] - this.min.x) / c);
      const iy = Math.floor((position[i * 3 + 1] - this.min.y) / c);
      const iz = Math.floor((position[i * 3 + 2] - this.min.z) / c);
      const k = this._key(ix, iy, iz);
      let arr = this.grid.get(k);
      if (!arr) {
        arr = [];
        this.grid.set(k, arr);
      }
      arr.push(i);
    }
  }

  /** 网格邻域内的候选点索引 */
  _candidates(p, radiusCells) {
    const c = this.cell;
    const bx = Math.floor((p.x - this.min.x) / c);
    const by = Math.floor((p.y - this.min.y) / c);
    const bz = Math.floor((p.z - this.min.z) / c);
    const out = [];
    for (let dx = -radiusCells; dx <= radiusCells; dx++) {
      for (let dy = -radiusCells; dy <= radiusCells; dy++) {
        for (let dz = -radiusCells; dz <= radiusCells; dz++) {
          const arr = this.grid.get(this._key(bx + dx, by + dy, bz + dz));
          if (arr) out.push(...arr);
        }
      }
    }
    return out;
  }

  /**
   * 屏幕空间拾取：在候选点中挑选「距射线近 且 屏幕距离小」的点。
   * @param {{x:number,y:number}} ndc 归一化设备坐标 (-1..1)
   * @param {THREE.Camera} camera
   * @param {THREE.Matrix4} [worldMatrix]  点云自身的世界变换
   * @param {{screenRadius?: number}} [opts]
   * @returns {{position: THREE.Vector3, normal: THREE.Vector3|null, screenDistance: number}|null}
   */
  pick(ndc, camera, worldMatrix = null, opts = {}) {
    const screenRadius = opts.screenRadius ?? 0.035;
    const ray = new THREE.Ray();
    const origin = camera.position.clone();
    const dir = new THREE.Vector3(ndc.x, ndc.y, 0.5).unproject(camera).sub(origin).normalize();
    ray.set(origin, dir);

    // 先沿射线在若干深度上采样，找到大致的命中位置
    const bounds = this.bounds ?? null;
    let tStart = 0.01;
    let tEnd = 200;
    if (bounds) {
      const box = new THREE.Box3(new THREE.Vector3(...bounds.min), new THREE.Vector3(...bounds.max));
      const hit = ray.intersectBox(box, new THREE.Vector3());
      if (!hit) return null;
      tStart = Math.max(tStart, origin.distanceTo(hit) - bounds.diagonal * 0.05);
      tEnd = tStart + bounds.diagonal * 1.2;
    }

    const probe = new THREE.Vector3();
    const candidates = new Set();
    const steps = 48;
    for (let s = 0; s <= steps; s++) {
      const t = tStart + ((tEnd - tStart) * s) / steps;
      probe.copy(ray.direction).multiplyScalar(t).add(ray.origin);
      for (const idx of this._candidates(probe, 2)) candidates.add(idx);
      if (candidates.size > 60000) break;
    }
    if (!candidates.size) return null;

    const pos = this.position;
    const v = new THREE.Vector3();
    const localOrigin = worldMatrix ? origin.clone().applyMatrix4(new THREE.Matrix4().copy(worldMatrix).invert()) : origin;
    const localDir = dir.clone();
    if (worldMatrix) {
      const inv = new THREE.Matrix4().copy(worldMatrix).invert();
      localDir.transformDirection(inv).normalize();
    }

    let best = null;
    let bestScore = Infinity;
    const ndcV = new THREE.Vector3();
    for (const i of candidates) {
      v.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
      const toPoint = v.clone().sub(localOrigin);
      const t = toPoint.dot(localDir);
      if (t <= 0) continue;
      const closest = localOrigin.clone().addScaledVector(localDir, t);
      const perp = closest.distanceTo(v);
      // 屏幕空间检查：投影到 NDC，与鼠标位置比较
      ndcV.copy(v).project(camera);
      if (ndcV.z < -1 || ndcV.z > 1) continue;
      const sd = Math.hypot(ndcV.x - ndc.x, ndcV.y - ndc.y);
      if (sd > screenRadius * 3) continue;
      const score = sd / screenRadius + perp * 12;
      if (score < bestScore) {
        bestScore = score;
        best = { index: i, local: v.clone(), perp, screenDistance: sd };
      }
    }
    if (!best) return null;

    const world = worldMatrix ? best.local.clone().applyMatrix4(worldMatrix) : best.local.clone();
    const normal = this._estimateNormal(best.index, worldMatrix);
    return { position: world, normal, screenDistance: best.screenDistance, index: best.index };
  }

  /** 用 K 近邻 PCA 估计表面法向（最小特征值方向） */
  _estimateNormal(index, worldMatrix) {
    const pos = this.position;
    const p = new THREE.Vector3(pos[index * 3], pos[index * 3 + 1], pos[index * 3 + 2]);
    const cands = this._candidates(p, 2);
    if (cands.length < 4) return null;
    const dists = [];
    for (const i of cands) {
      if (i === index) continue;
      const dx = pos[i * 3] - p.x;
      const dy = pos[i * 3 + 1] - p.y;
      const dz = pos[i * 3 + 2] - p.z;
      dists.push({ i, d: dx * dx + dy * dy + dz * dz });
    }
    dists.sort((a, b) => a.d - b.d);
    const k = Math.min(this.maxNeighbors, dists.length);
    if (k < 4) return null;

    let mx = 0, my = 0, mz = 0;
    for (let j = 0; j < k; j++) {
      const i = dists[j].i;
      mx += pos[i * 3];
      my += pos[i * 3 + 1];
      mz += pos[i * 3 + 2];
    }
    mx /= k;
    my /= k;
    mz /= k;

    let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
    for (let j = 0; j < k; j++) {
      const i = dists[j].i;
      const ax = pos[i * 3] - mx;
      const ay = pos[i * 3 + 1] - my;
      const az = pos[i * 3 + 2] - mz;
      xx += ax * ax;
      xy += ax * ay;
      xz += ax * az;
      yy += ay * ay;
      yz += ay * az;
      zz += az * az;
    }
    // 3x3 对称阵最小特征向量：用叉积法求两个最大特征向量再叉积
    const cov = [xx, xy, xz, xy, yy, yz, xz, yz, zz].map((n) => n / k);
    const e1 = powerIteration(cov, [1, 0, 0], 32);
    const deflated = deflate(cov, e1);
    const e2 = powerIteration(deflated, [0, 1, 0], 32);
    const n = [
      e1[1] * e2[2] - e1[2] * e2[1],
      e1[2] * e2[0] - e1[0] * e2[2],
      e1[0] * e2[1] - e1[1] * e2[0],
    ];
    const len = Math.hypot(n[0], n[1], n[2]);
    if (!len || !Number.isFinite(len)) return null;
    const normal = new THREE.Vector3(n[0] / len, n[1] / len, n[2] / len);
    if (worldMatrix) normal.transformDirection(new THREE.Matrix4().copy(worldMatrix).invert().transpose());
    return normal.normalize();
  }
}

function powerIteration(m, init, iters) {
  let v = init.slice();
  let nv = norm(v);
  if (!nv) v = [1, 0, 0];
  for (let k = 0; k < iters; k++) {
    const w = [
      m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
      m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
      m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ];
    nv = norm(w);
    if (nv < 1e-12) return v;
    v = w.map((x) => x / nv);
  }
  return v;
}

function deflate(m, e) {
  // λ 用 Rayleigh 商
  const lam =
    e[0] * (m[0] * e[0] + m[1] * e[1] + m[2] * e[2]) +
    e[1] * (m[3] * e[0] + m[4] * e[1] + m[5] * e[2]) +
    e[2] * (m[6] * e[0] + m[7] * e[1] + m[8] * e[2]);
  const out = m.slice();
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) out[i * 3 + j] -= lam * e[i] * e[j];
  return out;
}

function norm(v) {
  return Math.hypot(v[0], v[1], v[2]);
}
