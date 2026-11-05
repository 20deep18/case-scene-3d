/**
 * 多源画面：把监控/勘查视频按其标定好的相机位姿贴回三维现场。
 *  - 每个 camera 标注在三维空间中有：机位、朝向、视场角（由内参换算）
 *  - 视频在「相机位置后方 far 距离」处绘制为一个与视锥截面等大的四边形
 *  - 当观察者接近该相机时，视频画面即与真实拍摄视角重合，实现「画面贴回现场」
 */
import * as THREE from 'three';

const PIXELS = 1080;

export function cameraOverlayHasVideo(overlay, id) {
  const entry = overlay?.entries?.get(id);
  return !!entry?.video && entry.video.readyState >= 2;
}

/**
 * COLMAP(3DGS) 世界系 -> Three.js 场景系的旋转。
 *   世界系：x 右、y 下、z 前；Three.js 场景系：x 右、y 上、z 后。
 *   注意：本矩阵的实测作用是 (x, y, z)_local -> (x, z, -y)_scene（即把 y/z 两轴互换），
 *   而不是「y 取反、z 取反」。仓库内所有 local↔scene 换算都统一用它，
 *   因此位置/朝向的前向语义仍然自洽（publishPose 与 peerPoseToScene 互为逆变换）。
 */
export const WORLD_QUAT = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
export const WORLD_MATRIX = new THREE.Matrix4().makeRotationFromQuaternion(WORLD_QUAT);
/** WORLD_MATRIX 的逆：scene 系 -> local 系（用于把场景里的量写回标注数据） */
export const WORLD_MATRIX_INV = new THREE.Matrix4().copy(WORLD_MATRIX).invert();

/**
 * 相机 / 视锥的观察方向在自身局部坐标系里的常向量：局部 -Z。
 * annotations.js 的视锥顶点、本文件的画面四边形都放在局部 z = -far，
 * 因此「对象局部 -Z 经过它的旋转四元数作用后」就是相机实际观察方向。
 */
export const QUAT_LOCAL_FORWARD = new THREE.Vector3(0, 0, -1);

/**
 * 由「观察方向 + 上方向」（同一坐标系下的单位向量）构造该对象的三维朝向四元数。
 *
 * 为什么这样换算：
 *   Three.js 相机的基是 +X=right、+Y=up、+Z=-forward（相机朝局部 -Z），
 *   即旋转矩阵的三列必须是 (right, up, -forward)。
 *   该基是右手系（X × Y = Z = -forward），由它可解出：
 *       right = normalize(forward × up)
 *       up    = normalize(right × forward)     // 等价于 -(forward × right)
 *       col3  = -forward
 *   用 (col1, col2, col3) 拼出的矩阵行列式为 +1，是合法旋转矩阵。
 *   （上面两条已用标准位形 f=(0,0,-1), u=(0,1,0), r=(1,0,0) 实测确认。）
 *
 * 注意：旧实现在 target 分支里把观察方向直接放进第三列（col3 = +f），
 * 得到的是 (r, u, f) 左手系 → 实际朝向与观察方向正好差 180°，画面/视锥会背对目标。
 */
export function forwardUpToQuaternion(forward, up) {
  const f = forward.clone().normalize();
  const u0 = up.clone().normalize();
  let right = new THREE.Vector3().crossVectors(f, u0);
  if (right.lengthSq() < 1e-12) {
    // 退化：up 与 forward 平行（正上方俯视 / 正下方仰视时，世界上方向就与视线平行）。
    // 任取一个与 f 垂直的参考向量重建右方向，保证仍是右手正交基，不会倒置。
    const ref = Math.abs(f.y) > 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
    right = new THREE.Vector3().crossVectors(f, ref);
    if (right.lengthSq() < 1e-12) right = new THREE.Vector3(1, 0, 0);
  }
  right.normalize();
  // 用 Gram-Schmidt 由 right 与 f 重建 up：既保证严格正交（up 与 f 略有偏差时不会产生剪切/缩放），
  // 也统一了退化与非退化两条路径的结果。
  const u = new THREE.Vector3().crossVectors(right, f).normalize();
  const b = new THREE.Matrix4().set(
    right.x, u.x, -f.x, 0,
    right.y, u.y, -f.y, 0,
    right.z, u.z, -f.z, 0,
    0, 0, 0, 1
  );
  return new THREE.Quaternion().setFromRotationMatrix(b).normalize();
}

/**
 * 读取标注 pose 在 local 系下描述的 Group 姿态，返回 { position, quaternion }（local 系）。
 *  - rotation（行优先 9 元）：按 annotations.js applyPose 的方式解读，
 *    第三列即「局部 +Z」，也就是观察方向的反向（局部 -Z 才是观察方向）。
 *  - position + target + up：target - position 是观察方向，up 是上方向。
 *
 * 这是 pose 语义的**唯一实现**，attach / refresh / flyTo / 协同换算都走它，
 * 从而保证「标注渲染出来的视锥」和「flyTo 得到的相机」朝的是同一方向。
 */
export function annotationPoseToLocalQuaternion(pose) {
  if (!pose) return new THREE.Quaternion();
  if (Array.isArray(pose.rotation) && pose.rotation.length === 9) {
    const r = pose.rotation;
    const m = new THREE.Matrix4().set(
      r[0], r[1], r[2], 0,
      r[3], r[4], r[5], 0,
      r[6], r[7], r[8], 0,
      0, 0, 0, 1
    );
    return new THREE.Quaternion().setFromRotationMatrix(m).normalize();
  }
  if (Array.isArray(pose.target)) {
    const p = new THREE.Vector3(...(pose.position ?? [0, 0, 0]));
    const t = new THREE.Vector3(pose.target[0], pose.target[1], pose.target[2]);
    const f = t.clone().sub(p);
    if (f.lengthSq() > 1e-18) {
      const up = new THREE.Vector3(...(pose.up ?? [0, -1, 0]));
      return forwardUpToQuaternion(f, up);
    }
  }
  return new THREE.Quaternion();
}

/** 标注 pose（local 系）-> scene 系下的 { position, quaternion, fovY, localQuaternion } */
export function annotationPoseToScene(pose, worldMatrix) {
  const wm = worldMatrix ?? WORLD_MATRIX;
  const position = new THREE.Vector3(...(pose?.position ?? [0, 0, 0]));
  const localQuaternion = annotationPoseToLocalQuaternion(pose);
  position.applyMatrix4(wm);
  // scene 姿态 = 世界变换 ∘ 局部姿态（世界变换是纯旋转，直接乘四元数即可，
  // 不需要额外取逆，见 WORLD_QUAT 的推导）
  const wq = new THREE.Quaternion().setFromRotationMatrix(wm);
  const quaternion = wq.multiply(localQuaternion).normalize();
  return { position, quaternion, fovY: pose?.fovY ?? 45, localQuaternion };
}

/**
 * 协同视角 {p:[x,y,z], q:[x,y,z,w]}（local 系下的相机位置与朝向）-> scene 系。
 *
 * 为什么这样换算：
 *   对端相机的朝向 q 是「相对 local 系」的，而 Three.js 里父级是场景，
 *   所以先要把 q 用 WORLD_QUAT 左乘（world = worldQuat · local），
 *   位置则直接乘 WORLD_MATRIX；因为制作视锥的 makePeerFrustum 同样以局部 -Z 为观察方向，
 *   这样算出来的四元数作用在视锥上就与对方真实观察方向一致。
 *
 * 注意：不能用 q.toMatrix4 之外的、只取部分的换算（例如只交换分量），
 * 那等价于把旋转作用在位置轴上，会得到镜像/错向。
 */
export function peerPoseToScene(p, q, worldMatrix) {
  const wm = worldMatrix ?? WORLD_MATRIX;
  const position = new THREE.Vector3(p?.[0] ?? 0, p?.[1] ?? 0, p?.[2] ?? 0).applyMatrix4(wm);
  // 先归一化：四元数可能来自网络（未归一化会导致 setFromRotationMatrix 产生缩放/剪切）
  const local = new THREE.Quaternion(q?.[0] ?? 0, q?.[1] ?? 0, q?.[2] ?? 0, q?.[3] ?? 1).normalize();
  const wq = new THREE.Quaternion().setFromRotationMatrix(wm);
  const quaternion = wq.multiply(local).normalize();
  return { position, quaternion };
}

/**
 * 把标注 pose 直接写到 Object3D 上（local 系，position + quaternion）。
 * 语义与 annotations.js 的 applyPose 一致：对象局部 -Z 指向观察方向。
 * 现在类内代码都改用 annotationPoseToLocalQuaternion / annotationPoseToScene，
 * 这里保留并导出一份薄封装，供外部按旧签名调用。
 */
export function applyAnnotationPose(obj, pose) {
  if (!pose) return;
  obj.position.set(...(pose.position ?? [0, 0, 0]));
  obj.quaternion.copy(annotationPoseToLocalQuaternion(pose));
}

export class CameraOverlay {
  constructor(scene, { worldMatrix, far = 1.5 } = {}) {
    this.scene = scene;
    this.worldMatrix = worldMatrix ?? new THREE.Matrix4();
    this.far = far;
    this.entries = new Map(); // annotationId -> {annotation, video, texture, quad}
    this.selectedId = null;
    this.visible = true;
    this.group = new THREE.Group();
    this.group.name = 'camera-overlay';
    scene.add(this.group);
  }

  setVisible(v) {
    this.visible = v;
    for (const e of this.entries.values()) if (e.quad) e.quad.visible = v && e.playing !== false;
  }

  select(id) {
    this.selectedId = id;
    for (const [key, e] of this.entries) {
      if (e.quad) e.quad.material.opacity = key === id ? 1 : 0.55;
    }
  }

  /** 依据标注信息挂载视频 */
  async attach(annotation, url) {
    this.detach(annotation.id);
    const video = document.createElement('video');
    video.src = url;
    // 只有跨域资源才需要 crossOrigin；同源视频设了它反而会多一次 CORS 校验，
    // 响应缺少 Access-Control-Allow-Origin 时整段播放失败（表现为黑屏）。
    const sameOrigin = (() => {
      try {
        return new URL(url, location.href).origin === location.origin;
      } catch {
        return true;
      }
    })();
    if (!sameOrigin) video.crossOrigin = 'anonymous';
    video.muted = true;
    video.playsInline = true;
    video.loop = true;
    video.preload = 'auto';
    // 播放失败要能定位原因，否则现场只看到「黑屏」
    video.addEventListener('error', () => {
      console.error('[camera-overlay] 视频加载失败', url, video.error?.message ?? video.error?.code);
    });
    const texture = new THREE.VideoTexture(video);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.generateMipmaps = false;

    const aspect = annotation.pose?.aspect || 16 / 9;
    const fovY = ((annotation.pose?.fovY ?? 45) * Math.PI) / 180;
    const hh = Math.tan(fovY / 2) * this.far;
    const hw = hh * aspect;
    const quad = new THREE.Mesh(
      new THREE.PlaneGeometry(hw * 2, hh * 2),
      new THREE.MeshBasicMaterial({
        map: texture,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 1,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
      })
    );
    quad.renderOrder = 40;
    quad.position.set(0, 0, -this.far);
    quad.visible = this.visible;

    const holder = new THREE.Group();
    holder.add(quad);
    const scenePose = annotationPoseToScene(annotation.pose, this.worldMatrix);
    holder.position.copy(scenePose.position);
    holder.quaternion.copy(scenePose.quaternion);
    this.group.add(holder);

    const entry = { annotation, video, texture, quad, holder, playing: true };
    this.entries.set(annotation.id, entry);

    // 等元数据/首帧就绪再播放，否则部分浏览器直接抛 AbortError
    const tryPlay = () =>
      video.play().catch((err) => {
        console.warn('[camera-overlay] 自动播放被拒绝，将在用户点击后播放：', err?.name);
        entry.playing = false;
      });
    if (video.readyState >= 2) tryPlay();
    else video.addEventListener('loadeddata', tryPlay, { once: true });
    return entry;
  }

  detach(id) {
    const e = this.entries.get(id);
    if (!e) return;
    e.video.pause();
    e.video.src = '';
    e.texture.dispose();
    e.quad.geometry.dispose();
    e.quad.material.dispose();
    this.group.remove(e.holder);
    this.entries.delete(id);
  }

  togglePlay(id) {
    const e = this.entries.get(id);
    if (!e) return;
    if (e.video.paused) {
      e.video.play().catch(() => {});
      e.playing = true;
    } else {
      e.video.pause();
      e.playing = false;
    }
    if (e.quad) e.quad.visible = this.visible && e.playing;
  }

  /** 重建某个相机的几何（位姿被 PnP 更新后调用） */
  refresh(annotation) {
    const e = this.entries.get(annotation.id);
    if (!e) return;
    const scenePose = annotationPoseToScene(annotation.pose, this.worldMatrix);
    e.holder.position.copy(scenePose.position);
    e.holder.quaternion.copy(scenePose.quaternion);
    const aspect = annotation.pose?.aspect || 16 / 9;
    const fovY = ((annotation.pose?.fovY ?? 45) * Math.PI) / 180;
    const hh = Math.tan(fovY / 2) * this.far;
    e.quad.geometry.dispose();
    e.quad.geometry = new THREE.PlaneGeometry(hh * aspect * 2, hh * 2);
  }

  /**
   * 计算「站在该相机位置、朝同样方向看」时的场景系相机参数，用于一键贴回现场。
   *
   * 返回的 { position, quaternion, fovY } 可以直接赋给 THREE.PerspectiveCamera：
   * 因为 pose 描述的就是 Group 姿态（局部 -Z = 观察方向，局部 +Y = 上方向），
   * 与 Three.js 相机的约定完全一致，所以 scene 系的姿态四元数可以原样交给相机。
   *
   * 旧实现用「世界上方向 (0,1,0) × forward」重算 up/right：
   *   1) 丢掉了标注里记录的 up（roll），与 attach() 渲染出来的画面不再重合；
   *   2) forward 与世界上方向平行时（正上方俯视/正下方仰视）叉积退化，
   *      会退化成任意基准，导致画面上/下颠倒；
   *   3) right 与 forward 不严格垂直时还会引入轻微剪切。
   */
  flyTo(annotation) {
    const view = annotationPoseToScene(annotation.pose, this.worldMatrix);
    return {
      position: view.position.clone(),
      quaternion: view.quaternion.clone(),
      fovY: annotation.pose?.fovY ?? 45,
    };
  }

  clear() {
    for (const id of [...this.entries.keys()]) this.detach(id);
  }

  update() {
    /* VideoTexture 自动更新 */
  }

  /** 生成其他研判人员的视锥指示器 */
  makePeerFrustum(color, label) {
    const c = new THREE.Color(color || '#4dd0e1');
    const g = new THREE.Group();
    const fovY = (50 * Math.PI) / 180;
    const hh = Math.tan(fovY / 2) * 0.5;
    const hw = hh * (16 / 9);
    const verts = [
      [0, 0, 0],
      [-hw, -hh, -0.5],
      [hw, -hh, -0.5],
      [hw, hh, -0.5],
      [-hw, hh, -0.5],
    ];
    const edges = [
      [0, 1], [0, 2], [0, 3], [0, 4],
      [1, 2], [2, 3], [3, 4], [4, 1],
    ];
    const positions = [];
    for (const [a, b] of edges) positions.push(...verts[a], ...verts[b]);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const lines = new THREE.LineSegments(
      geom,
      new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: 0.9, depthTest: false })
    );
    lines.renderOrder = 50;
    g.add(lines);
    const head = new THREE.Mesh(
      new THREE.SphereGeometry(0.02, 10, 8),
      new THREE.MeshBasicMaterial({ color: c, depthTest: false, transparent: true, opacity: 0.95 })
    );
    head.renderOrder = 51;
    g.add(head);
    g.userData.label = label;
    return g;
  }
}

export { PIXELS };
