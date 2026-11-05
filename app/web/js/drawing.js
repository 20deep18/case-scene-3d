/**
 * 研判绘图工具：把用户在三维现场中的点击转换成标注草稿。
 * 坐标一律使用 COLMAP/3DGS 世界系（y 向下）。
 */
import * as THREE from 'three';
import { defaultColor } from './annotations.js';

const DRAFT_COLOR = '#ffffff';

export class Drawing {
  constructor(scene, opts) {
    this.scene = scene;
    this.opts = opts;
    this.worldMatrix = opts.getWorldMatrix?.() ?? new THREE.Matrix4();
    this.tool = 'select';
    this.points = []; // COLMAP 系下的点
    this.normals = [];
    this.active = false;
    this.cameraStage = 0;
    this.cameraPosition = null;
    this.cameraNormal = null;

    this.group = new THREE.Group();
    this.group.name = 'draft';
    scene.add(this.group);

    // 预览点与线（高亮白色）
    this.previewDot = new THREE.Mesh(
      new THREE.SphereGeometry(0.03, 12, 8),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(DRAFT_COLOR), depthTest: false, transparent: true, opacity: 0.9 })
    );
    this.previewDot.renderOrder = 30;
    this.previewDot.visible = false;
    scene.add(this.previewDot);

    this.previewLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.05, gapSize: 0.03, depthTest: false, transparent: true, opacity: 0.85 })
    );
    this.previewLine.renderOrder = 30;
    this.previewLine.visible = false;
    scene.add(this.previewLine);

    this._world = new THREE.Vector3();
  }

  setTool(tool) {
    this.cancel();
    this.tool = tool;
  }

  isActive() {
    return this.active || this.points.length > 0;
  }

  _toWorld(p) {
    return new THREE.Vector3(p[0], p[1], p[2]).applyMatrix4(this.worldMatrix);
  }

  _toLocal(v) {
    const inv = new THREE.Matrix4().copy(this.worldMatrix).invert();
    return v.clone().applyMatrix4(inv);
  }

  /** 鼠标移动时的虚线预览 */
  previewAt(localPoint, normal) {
    this._previewPoint = localPoint;
    if (this.points.length === 0) {
      this.previewDot.visible = true;
      this.previewDot.position.copy(this._toWorld(localPoint));
      this.previewLine.visible = false;
      return;
    }
    this.previewDot.visible = true;
    this.previewDot.position.copy(this._toWorld(localPoint));
    const last = this._toWorld(this.points.at(-1));
    const geom = new THREE.BufferGeometry().setFromPoints([last, this._toWorld(localPoint)]);
    this.previewLine.geometry.dispose();
    this.previewLine.geometry = geom;
    this.previewLine.computeLineDistances();
    this.previewLine.visible = true;
  }

  /** 添加一个绘制点 */
  click(tool, localPoint, normal) {
    if (this.tool !== tool) this.setTool(tool);
    this.active = true;
    if (tool === 'point') {
      this._commit({
        kind: 'point',
        points: [[localPoint.x, localPoint.y, localPoint.z]],
        color: defaultColor('point'),
        meta: { normal: normal ? [normal.x, normal.y, normal.z] : null },
      });
      return;
    }
    this.points.push([localPoint.x, localPoint.y, localPoint.z]);
    this.normals.push(normal ? [normal.x, normal.y, normal.z] : null);

    // 测距 / 尺寸标注：两个点后自动完成
    if ((tool === 'measure' || tool === 'dimension') && this.points.length === 2) {
      this.finish();
      return;
    }
    this.opts.onHint?.(
      tool === 'polyline'
        ? `已放置 ${this.points.length} 个轨迹点，双击或回车结束`
        : tool === 'polygon'
          ? `已放置 ${this.points.length} 个顶点，双击或回车闭合区域`
          : tool === 'dimension'
            ? `尺寸标注：再点终点完成（已选 ${this.points.length}/2）；完成后在右侧属性面板填写民警实测值`
            : `测距：再点一个点完成（已选 ${this.points.length}/2）`
    );
  }

  /** 相机工具：第一次点机位，第二次点朝向目标 */
  cameraClick(localPoint, normal) {
    this.active = true;
    if (this.cameraStage === 0) {
      this.cameraPosition = [localPoint.x, localPoint.y, localPoint.z];
      this.previewDot.visible = true;
      this.previewDot.position.copy(this._toWorld(this.cameraPosition));
      this.cameraStage = 1;
      this.opts.onHint?.('已确定机位，请点击画面中心朝向的目标点');
      return;
    }
    const target = [localPoint.x, localPoint.y, localPoint.z];
    this._commit({
      kind: 'camera',
      points: [this.cameraPosition],
      pose: {
        source: 'manual',
        position: this.cameraPosition,
        target,
        up: [0, -1, 0],
        fovY: 45,
        aspect: 16 / 9,
      },
      color: defaultColor('camera'),
    });
    this.cameraStage = 0;
    this.cameraPosition = null;
    this.opts.onHint?.('相机已放置，可在右侧属性里导入视频或做 PnP 精修');
  }

  /** 结束当前绘制（双击/回车） */
  finish() {
    if (this.tool === 'polyline' && this.points.length >= 2) {
      this._commit({ kind: 'polyline', points: [...this.points], color: defaultColor('polyline') });
    } else if (this.tool === 'polygon' && this.points.length >= 3) {
      this._commit({ kind: 'polygon', points: [...this.points], color: defaultColor('polygon'), closed: true });
    } else if (this.tool === 'measure' && this.points.length >= 2) {
      this._commit({ kind: 'measure', points: [...this.points], color: defaultColor('measure') });
    } else if (this.tool === 'dimension' && this.points.length >= 2) {
      // 尺寸标注：写成 dimension，measured 留空 = 「待民警实测」
      const pts = [...this.points];
      const [a, b] = pts;
      const modelDistance = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
      this._commit({
        kind: 'dimension',
        points: pts,
        color: defaultColor('dimension'),
        measured: null,
        modelDistance,
        unit: 'meter',
        priority: 'normal',
      });
    } else if (this.points.length > 0) {
      this.opts.onHint?.('点数不足，已取消本次绘制');
    }
    this.cancel();
  }

  cancel() {
    this.points = [];
    this.normals = [];
    this.active = false;
    this.cameraStage = 0;
    this.cameraPosition = null;
    this.previewDot.visible = false;
    this.previewLine.visible = false;
    for (const child of [...this.group.children]) {
      this.group.remove(child);
      child.geometry?.dispose();
      child.material?.dispose();
    }
  }

  _commit(draft) {
    const data = {
      ...draft,
      createdBy: undefined,
      meta: { ...(draft.meta ?? {}), normals: this.normals.length ? this.normals : undefined },
    };
    delete data.createdBy;
    this.opts.onCommit?.(data);
    this.cancel();
  }

  update() {
    /* 预留：可用于闪烁动画 */
  }
}
