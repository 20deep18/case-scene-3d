/**
 * 三维标注渲染：打点物证、行动轨迹、区域、测距、监控相机锥体、现场照片。
 * 所有几何体都挂在 annotationRoot 下，坐标使用 COLMAP/3DGS 世界系，
 * 通过 root.matrix 统一转换到渲染世界系。
 */
import * as THREE from 'three';

const KIND_COLORS = {
  point: '#ffcc33',
  polyline: '#4dd0e1',
  polygon: '#7ef0a0',
  measure: '#ff9f43',
  camera: '#ff5f6d',
  image: '#c792ea',
  // 尺寸标注：待测用醒目红色提示民警去量，已测用绿色表示已完成
  dimension: '#ff4d6d',
};

const KIND_NAMES = {
  point: '物证点',
  polyline: '行动轨迹',
  polygon: '区域',
  measure: '测距',
  camera: '监控/勘查相机',
  image: '现场照片',
  dimension: '尺寸标注',
};

export function kindName(kind, ann = null) {
  // 尺寸标注的名称要看它有没有实测值：待测 / 已实测 是两种完全不同的状态，
  // 列表里必须一眼能分辨（否则已测完的项看起来还像没做）。
  if (kind === 'dimension') {
    const has = ann && Number.isFinite(Number(ann.measured)) && Number(ann.measured) > 0;
    return has ? '尺寸标注（已实测）' : '尺寸标注（待实测）';
  }
  return KIND_NAMES[kind] ?? kind;
}

export function defaultColor(kind) {
  return KIND_COLORS[kind] ?? '#ffcc33';
}

/** 把标注数据转成可编辑的 THREE 对象集合 */
export class AnnotationLayer {
  constructor(scene, { onLabelsChanged } = {}) {
    this.scene = scene;
    this.root = new THREE.Group();
    this.root.name = 'annotations';
    scene.add(this.root);
    this.items = new Map(); // id -> {data, group, handles, labelEls}
    this.visible = true;
    this.labelHost = null; // 由 app 注入的 HTML 覆盖层
    this.onLabelsChanged = onLabelsChanged ?? (() => {});
  }

  setLabelHost(host) {
    this.labelHost = host;
  }

  setWorldTransform(matrix) {
    this.root.matrixAutoUpdate = false;
    this.root.matrix.copy(matrix);
    this.root.matrixWorldNeedsUpdate = true;
  }

  setVisible(v) {
    this.visible = v;
    this.root.visible = v;
  }

  /**
   * 按标注类型（kind）控制显隐 —— 供「图层」面板的每一类开关使用。
   * 记录到 kindVisible，新建/更新的标注也会遵循当前设置。
   */
  setKindVisible(kind, visible) {
    if (!this.kindVisible) this.kindVisible = new Map();
    this.kindVisible.set(kind, !!visible);
    for (const { data, group } of this.items.values()) {
      if (!data || data.kind !== kind) continue;
      group.visible = !!visible && this.visible;
    }
    this.onLabelsChanged();
  }

  isKindVisible(kind) {
    return this.kindVisible ? this.kindVisible.get(kind) !== false : true;
  }

  clear() {
    for (const id of [...this.items.keys()]) this.remove(id);
  }

  /** 全量重建 */
  setAll(annotations) {
    const seen = new Set();
    for (const ann of annotations) {
      seen.add(ann.id);
      this.upsert(ann);
    }
    for (const id of [...this.items.keys()]) {
      if (!seen.has(id)) this.remove(id);
    }
    this.onLabelsChanged();
  }

  upsert(data) {
    const existing = this.items.get(data.id);
    if (existing) {
      if (existing.sig === this._signature(data)) {
        existing.data = data;
        return existing;
      }
      this.remove(data.id);
    }
    const item = this._build(data);
    // 新建的标注要遵循「图层」面板当前的显隐设置
    if (item?.group && !this.isKindVisible(data?.kind)) item.group.visible = false;
    this.items.set(data.id, item);
    this.onLabelsChanged();
    return item;
  }

  _signature(d) {
    return JSON.stringify([
      d.kind, d.points, d.color, d.width, d.opacity, d.label, d.visible, d.pose, d.media,
      // 尺寸标注的实测值变化后必须重画（颜色从红变绿、标签显示实测值）
      d.measured, d.modelDistance, d.priority,
    ]);
  }

  remove(id) {
    const item = this.items.get(id);
    if (!item) return;
    this.root.remove(item.group);
    disposeGroup(item.group);
    if (item.video) {
      item.video.pause();
      item.video.src = '';
      item.video.load?.();
    }
    this.items.delete(id);
    this.onLabelsChanged();
  }

  get(id) {
    return this.items.get(id) ?? null;
  }

  list() {
    return [...this.items.values()];
  }

  _build(data) {
    const group = new THREE.Group();
    group.name = `ann:${data.id}`;
    group.userData.annotationId = data.id;
    const color = data.color ?? defaultColor(data.kind);
    const baseColor = new THREE.Color(color);
    const opacity = data.opacity ?? 1;

    const item = { data, group, handles: [], labelEls: [], sig: this._signature(data) };

    const mat = (opts = {}) =>
      new THREE.MeshBasicMaterial({
        color: baseColor,
        transparent: true,
        opacity: opacity * (opts.opacityScale ?? 1),
        depthTest: false,
        depthWrite: false,
        side: THREE.DoubleSide,
        toneMapped: false,
        ...opts,
      });

    switch (data.kind) {
      case 'point': {
        const p = data.points[0];
        const radius = 0.045;
        const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 20, 14), mat());
        mesh.position.set(p[0], p[1], p[2]);
        mesh.renderOrder = 12;
        group.add(mesh);
        // 光晕
        const halo = new THREE.Mesh(
          new THREE.SphereGeometry(radius * 2.1, 16, 12),
          new THREE.MeshBasicMaterial({ color: baseColor, transparent: true, opacity: 0.16 * opacity, depthTest: false, depthWrite: false })
        );
        halo.position.copy(mesh.position);
        halo.renderOrder = 11;
        group.add(halo);
        const stem = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(p[0], p[1], p[2]),
            new THREE.Vector3(p[0], p[1] - 0.28, p[2]),
          ]),
          new THREE.LineBasicMaterial({ color: baseColor, transparent: true, opacity: opacity * 0.8, depthTest: false })
        );
        stem.renderOrder = 11;
        group.add(stem);
        item.handles.push({ type: 'vertex', index: 0, position: mesh.position, mesh });
        item.labelAnchor = new THREE.Vector3(p[0], p[1] - 0.3, p[2]);
        break;
      }
      case 'polyline':
      case 'measure':
      case 'dimension':
      case 'polygon': {
        const pts = data.points.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
        const closed = data.kind === 'polygon' && data.closed !== false;
        const list = closed ? [...pts, pts[0]] : pts;
        if (list.length >= 2) {
          const curve = new THREE.CatmullRomCurve3(list, false, 'catmullrom', 0.0);
          const tubular = data.kind === 'polygon' ? 16 : 64;
          const tube = new THREE.Mesh(
            new THREE.TubeGeometry(curve, tubular, (data.width ?? 1) * 0.012, 8, false),
            mat({ depthTest: true, depthWrite: false })
          );
          tube.renderOrder = 10;
          group.add(tube);
        }
        if (data.kind === 'polygon' && pts.length >= 3) {
          const geom = new THREE.BufferGeometry().setFromPoints(pts);
          const idx = THREE.ShapeUtils.triangulateShape(
            pts.map((p) => new THREE.Vector2(p.x, p.z)),
            []
          );
          const flat = [];
          for (const tri of idx) flat.push(tri[0], tri[1], tri[2]);
          geom.setIndex(flat);
          geom.computeVertexNormals();
          const mesh = new THREE.Mesh(geom, mat({ opacity: 0.18 * opacity, depthTest: false }));
          mesh.renderOrder = 9;
          group.add(mesh);
        }
        for (let i = 0; i < pts.length; i++) {
          const handle = new THREE.Mesh(
            new THREE.SphereGeometry(0.03, 12, 8),
            new THREE.MeshBasicMaterial({ color: baseColor, transparent: true, opacity, depthTest: false })
          );
          handle.position.copy(pts[i]);
          handle.renderOrder = 13;
          group.add(handle);
          item.handles.push({ type: 'vertex', index: i, position: handle.position, mesh: handle });
        }
        if (data.kind === 'dimension') {
          // 尺寸标注额外画一对「大一点的端点球 + 一条醒目直线」，
          // 让民警一眼看出「这两点之间需要实测」。
          // 待测=红色，已测=绿色（颜色由数据里的 measured 决定）。
          const done = Number.isFinite(Number(data.measured)) && Number(data.measured) > 0;
          const markColor = new THREE.Color(done ? '#3ddc84' : '#ff4d6d');
          for (const p of pts) {
            const ring = new THREE.Mesh(
              new THREE.SphereGeometry(0.075, 16, 12),
              new THREE.MeshBasicMaterial({ color: markColor, transparent: true, opacity: 0.85, depthTest: false })
            );
            ring.position.copy(p);
            ring.renderOrder = 14;
            group.add(ring);
          }
          if (pts.length >= 2) {
            const line = new THREE.Line(
              new THREE.BufferGeometry().setFromPoints(pts),
              new THREE.LineBasicMaterial({ color: markColor, transparent: true, opacity: 0.95, depthTest: false })
            );
            line.renderOrder = 14;
            group.add(line);
          }
          // 标签挂在两点中点上方
          if (pts.length >= 2) {
            const mid = pts[0].clone().add(pts[1]).multiplyScalar(0.5);
            item.labelAnchor = mid.setY(mid.y + 0.18);
          }
        }
        item.labelAnchor = pts[0]?.clone() ?? new THREE.Vector3();
        break;
      }
      case 'image': {
        const p = data.points[0];
        const media = data.media?.[0];
        const w = 0.9;
        const h = media?.height && media?.width ? (w * media.height) / media.width : 0.6;
        const mesh = new THREE.Mesh(
          new THREE.PlaneGeometry(w, h),
          mat({ opacity: 1, side: THREE.DoubleSide, depthTest: false })
        );
        mesh.position.set(p[0], p[1], p[2]);
        mesh.renderOrder = 12;
        if (media?.url) {
          const tex = new THREE.TextureLoader().load(media.url);
          tex.colorSpace = THREE.SRGBColorSpace;
          mesh.material.map = tex;
          mesh.material.needsUpdate = true;
        }
        group.add(mesh);
        objectLookAt(mesh, data.meta?.target ?? [p[0], p[1], p[2] + 1]);
        item.handles.push({ type: 'vertex', index: 0, position: mesh.position, mesh, rotateOnly: true });
        item.labelAnchor = mesh.position.clone();
        break;
      }
      case 'camera': {
        const pose = data.pose ?? {};
        const pos = new THREE.Vector3(...(pose.position ?? [0, 0, 0]));
        const fovY = ((pose.fovY ?? 45) * Math.PI) / 180;
        const aspect = pose.aspect ?? 16 / 9;
        const near = 0.15;
        const far = 1.5;
        const hh = Math.tan(fovY / 2);
        const hw = hh * aspect;
        const verts = [
          [0, 0, 0],
          [-hw * near, -hh * near, -near],
          [hw * near, -hh * near, -near],
          [hw * near, hh * near, -near],
          [-hw * near, hh * near, -near],
          [-hw * far, -hh * far, -far],
          [hw * far, -hh * far, -far],
          [hw * far, hh * far, -far],
          [-hw * far, hh * far, -far],
        ];
        const edges = [
          [0, 1], [0, 2], [0, 3], [0, 4],
          [1, 2], [2, 3], [3, 4], [4, 1],
          [1, 5], [2, 6], [3, 7], [4, 8],
          [5, 6], [6, 7], [7, 8], [8, 5],
        ];
        const lineGeom = new THREE.BufferGeometry();
        const positions = [];
        for (const [a, b] of edges) {
          positions.push(...verts[a], ...verts[b]);
        }
        lineGeom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        const lines = new THREE.LineSegments(
          lineGeom,
          new THREE.LineBasicMaterial({ color: baseColor, transparent: true, opacity: opacity * 0.95, depthTest: false })
        );
        lines.renderOrder = 14;
        const camGroup = new THREE.Group();
        camGroup.add(lines);

        // 画面四边形（用于贴视频）
        const quad = new THREE.Mesh(
          new THREE.PlaneGeometry(hw * 2 * far, hh * 2 * far),
          new THREE.MeshBasicMaterial({
            color: 0xffffff,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.98,
            depthTest: false,
            toneMapped: false,
          })
        );
        quad.position.set(0, 0, -far);
        quad.renderOrder = 16;
        camGroup.add(quad);
        camGroup.position.copy(pos);
        applyPose(camGroup, pose);
        group.add(camGroup);

        item.videoQuad = quad;
        item.camGroup = camGroup;
        item.video = null;
        item.handles.push({ type: 'pose', object: camGroup, position: camGroup.position, data });
        item.labelAnchor = pos.clone();
        break;
      }
      default:
        break;
    }

    group.visible = data.visible !== false;
    this.root.add(group);
    return item;
  }

  /** 由 app 调用：更新相机贴图 */
  attachVideo(id, url, { muted = true, loop = true, live = false } = {}) {
    const item = this.get(id);
    if (!item?.videoQuad) return null;
    if (item.video) {
      item.video.pause();
      item.video.src = '';
    }
    const video = document.createElement('video');
    video.src = url;
    video.crossOrigin = 'anonymous';
    video.muted = muted;
    video.loop = loop && !live;
    video.playsInline = true;
    video.preload = 'auto';
    const tex = new THREE.VideoTexture(video);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    item.videoQuad.material.map = tex;
    item.videoQuad.material.color.set(0xffffff);
    item.videoQuad.material.needsUpdate = true;
    item.video = video;
    video.play().catch(() => {});
    return video;
  }

  setVideoVisible(id, visible) {
    const item = this.get(id);
    if (!item?.videoQuad) return;
    item.videoQuad.visible = visible;
  }

  /** 命中测试：返回被点中的标注 id 与手柄索引 */
  hitTest(raycaster) {
    const hits = raycaster.intersectObjects(this.root.children, true);
    for (const h of hits) {
      let obj = h.object;
      while (obj && !obj.userData.annotationId) obj = obj.parent;
      if (!obj) continue;
      const id = obj.userData.annotationId;
      const item = this.items.get(id);
      let handleIndex = -1;
      for (let i = 0; i < (item?.handles?.length ?? 0); i++) {
        if (item.handles[i].mesh === h.object) handleIndex = i;
      }
      return { id, handleIndex, point: h.point, distance: h.distance, kind: item?.data.kind };
    }
    return null;
  }
}

function disposeGroup(group) {
  group.traverse((obj) => {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const m of mats) {
        if (m.map && m.map.isVideoTexture) m.map.dispose();
        m.dispose();
      }
    }
  });
}

function objectLookAt(obj, target) {
  obj.lookAt(new THREE.Vector3(target[0], target[1], target[2]));
}

/** 依据 pose（position/target 或 rotation 矩阵）设置对象姿态 */
export function applyPose(obj, pose) {
  if (pose.rotation && pose.rotation.length === 9) {
    const m = new THREE.Matrix4().set(
      pose.rotation[0], pose.rotation[1], pose.rotation[2], 0,
      pose.rotation[3], pose.rotation[4], pose.rotation[5], 0,
      pose.rotation[6], pose.rotation[7], pose.rotation[8], 0,
      0, 0, 0, 1
    );
    obj.quaternion.setFromRotationMatrix(m);
  } else if (pose.target) {
    const t = new THREE.Vector3(pose.target[0], pose.target[1], pose.target[2]);
    const up = new THREE.Vector3(...(pose.up ?? [0, 1, 0]));
    const dir = t.clone().sub(obj.position);
    if (dir.lengthSq() > 1e-9) {
      const m = new THREE.Matrix4();
      m.lookAt(new THREE.Vector3(0, 0, 0), dir.clone().normalize().negate(), up);
      obj.quaternion.setFromRotationMatrix(m);
    }
  }
}
