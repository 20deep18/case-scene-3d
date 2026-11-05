/**
 * 案件库（Project Store）
 * 每个案件一个目录：data/projects/<id>/，包含 project.json 与 models/
 * 所有写入采用「临时文件 + 原子重命名」，避免断电/崩溃导致 JSON 损坏。
 */
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';

const ANNOTATION_KINDS = new Set(['point', 'polyline', 'polygon', 'measure', 'camera', 'image', 'dimension']);
const POLYGON_KINDS = new Set(['polyline', 'polygon', 'measure', 'dimension']);
const EVENT_LIMIT = 3000;

function nowISO() {
  return new Date().toISOString();
}

/**
 * 案件目录名（即 API 中的 projectId）统一使用 ASCII，避免 URL 编码带来的歧义；
 * 中文案件名保存在 project.json 的 name 字段中展示。
 */
function slugify(name, fallback = 'case') {
  const cleaned = String(name ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
  const ascii = cleaned.replace(/[^a-z0-9-]/g, '');
  const stamp = new Date().toISOString().slice(0, 10).replace(/-/g, '');
  return ascii ? `${ascii}-${stamp}` : `${fallback}-${stamp}`;
}

/** 校验并规整矢量：必须有 3 个有限数字 */
function vec3(input, label = 'vector') {
  if (!Array.isArray(input) || input.length < 3) throw new Error(`${label} 必须是长度 3 的数组`);
  const out = input.slice(0, 3).map((v) => Number(v));
  if (out.some((v) => !Number.isFinite(v))) throw new Error(`${label} 含非法数字`);
  return out;
}

/** 两点欧氏距离（用于 dimension 的模型侧长度） */
function distance3(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function color(input, fallback = '#ffcc33') {  if (typeof input !== 'string') return fallback;
  const s = input.trim();
  if (/^#[0-9a-fA-F]{6}$/.test(s)) return s.toLowerCase();
  if (/^#[0-9a-fA-F]{3}$/.test(s)) {
    return `#${s[1]}${s[1]}${s[2]}${s[2]}${s[3]}${s[3]}`.toLowerCase();
  }
  return fallback;
}

export class Store {
  constructor(rootDir) {
    this.root = rootDir;
    this.projectsDir = path.join(rootDir, 'projects');
    this.cache = new Map(); // id -> project object（含 _dir）
    this.listeners = new Set();
  }

  onEvent(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  _emit(event) {
    for (const fn of this.listeners) {
      try {
        fn(event);
      } catch {
        /* 监听器异常不影响主流程 */
      }
    }
  }

  async init() {
    await fs.mkdir(this.projectsDir, { recursive: true });
    await this.reload();
  }

  async reload() {
    this.cache.clear();
    let entries = [];
    try {
      entries = await fs.readdir(this.projectsDir, { withFileTypes: true });
    } catch {
      await fs.mkdir(this.projectsDir, { recursive: true });
    }
    for (const e of entries) {
      if (!e.isDirectory()) continue;
      const dir = path.join(this.projectsDir, e.name);
      try {
        const raw = await fs.readFile(path.join(dir, 'project.json'), 'utf8');
        const project = JSON.parse(raw);
        project.id = e.name;
        project._dir = dir;
        this.cache.set(project.id, project);
      } catch (err) {
        console.warn(`[store] 跳过损坏的案件目录 ${e.name}: ${err.message}`);
      }
    }
    return [...this.cache.values()];
  }

  list() {
    return [...this.cache.values()]
      .map((p) => this._summary(p))
      .sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1));
  }

  _summary(p) {
    return {
      id: p.id,
      name: p.name,
      description: p.description ?? '',
      createdAt: p.createdAt,
      updatedAt: p.updatedAt,
      revision: p.revision,
      model: p.model ?? null,
      annotationCount: p.annotations?.length ?? 0,
      cameraCount: (p.annotations ?? []).filter((a) => a.kind === 'camera').length,
      stats: p.stats ?? null,
      thumbnail: p.thumbnail ?? null,
    };
  }

  get(id) {
    return this.cache.get(id) ?? null;
  }

  async create({ name, description = '', center = [0, 0, 0], up = [0, 1, 0], unit = 'meter', operator = '未署名' }) {
    const base = slugify(name || '未命名案件');
    let id = base;
    let n = 2;
    while (this.cache.has(id)) id = `${base}_${n++}`;
    const dir = path.join(this.projectsDir, id);
    await fs.mkdir(path.join(dir, 'models'), { recursive: true });
    const project = {
      version: 1,
      id,
      name: name || '未命名案件',
      description,
      createdAt: nowISO(),
      updatedAt: nowISO(),
      revision: 1,
      world: { unit, up: vec3(up, 'up'), center: vec3(center, 'center') },
      model: null,
      presentation: {
        background: '#0b0f14',
        grid: true,
        cameraPreset: null,
        fog: 0.0,
        exposure: 1.0,
      },
      annotations: [],
      users: [],
      events: [
        { at: nowISO(), type: 'project.create', operator, detail: { name } },
      ],
      _dir: dir,
    };
    this.cache.set(id, project);
    await this._persist(project);
    return project;
  }

  async _persist(project) {
    const file = path.join(project._dir, 'project.json');
    const tmp = `${file}.${randomUUID().slice(0, 8)}.tmp`;
    const clone = { ...project };
    delete clone._dir;
    await fs.writeFile(tmp, JSON.stringify(clone, null, 2), 'utf8');
    await fs.rename(tmp, file);
  }

  async mutate(id, mutator, event = null) {
    const project = this.get(id);
    if (!project) return null;
    mutator(project);
    project.revision = (project.revision ?? 0) + 1;
    project.updatedAt = nowISO();
    if (event) this._pushEvent(project, event);
    await this._persist(project);
    return project;
  }

  _pushEvent(project, event) {
    project.events.push({ at: nowISO(), ...event });
    if (project.events.length > EVENT_LIMIT) {
      project.events.splice(0, project.events.length - EVENT_LIMIT);
    }
  }

  async logEvent(id, event) {
    const project = this.get(id);
    if (!project) return null;
    this._pushEvent(project, event);
    project.updatedAt = nowISO();
    await this._persist(project);
    return project;
  }

  // ---------- 标注 ----------

  normalizeAnnotation(input, operator = '未署名') {
    if (!input || typeof input !== 'object') throw new Error('标注必须是对象');
    const kind = String(input.kind ?? 'point');
    if (!ANNOTATION_KINDS.has(kind)) throw new Error(`不支持的标注类型：${kind}`);

    const base = {
      id: input.id && String(input.id).length > 2 ? String(input.id) : randomUUID(),
      kind,
      label: String(input.label ?? ''),
      note: String(input.note ?? ''),
      color: color(input.color),
      visible: input.visible !== false,
      locked: input.locked === true,
      createdBy: String(input.createdBy ?? operator),
      createdAt: input.createdAt ?? nowISO(),
      updatedAt: nowISO(),
      points: [],
      media: [],
      pose: null,
      width: Number.isFinite(Number(input.width)) ? Number(input.width) : 1,
      opacity: Number.isFinite(Number(input.opacity)) ? Math.min(Math.max(Number(input.opacity), 0), 1) : 1,
      meta: input.meta && typeof input.meta === 'object' ? input.meta : {},
    };

    if (POLYGON_KINDS.has(kind)) {
      const pts = input.points;
      if (!Array.isArray(pts) || pts.length < 2) throw new Error(`${kind} 至少需要 2 个点`);
      base.points = pts.map((p, i) => vec3(p, `points[${i}]`));
      if (kind === 'polygon' && base.points.length < 3) throw new Error('polygon 至少需要 3 个点');
      if (input.closed !== undefined) base.closed = input.closed !== false;
    } else {
      base.points = [vec3(input.points?.[0] ?? input.position, 'points[0]')];
      if (input.position) base.points = [vec3(input.position, 'position')];
    }

    if (kind === 'image') {
      const media = Array.isArray(input.media) ? input.media : [];
      base.media = media.map((m) => ({
        url: String(m.url),
        kind: String(m.kind ?? 'image'),
        width: Number(m.width) || 0,
        height: Number(m.height) || 0,
      }));
    }

    if (kind === 'camera') {
      if (!input.pose || typeof input.pose !== 'object') throw new Error('camera 标注必须提供 pose');
      const p = input.pose;
      base.pose = {
        source: String(p.source ?? 'manual'),
        frame: String(p.frame ?? 'gaussian'),
        position: vec3(p.position, 'pose.position'),
        /** 相机朝向：目标点 + 上方向，直观且与 Web 端一致 */
        target: p.target ? vec3(p.target, 'pose.target') : null,
        up: p.up ? vec3(p.up, 'pose.up', [0, 1, 0]) : [0, 1, 0],
        /** 旋转矩阵（行优先 9 元）优先级高于 target */
        rotation: Array.isArray(p.rotation) && p.rotation.length === 9 ? p.rotation.map(Number) : null,
        fovY: Number(p.fovY) || 45,
        aspect: Number(p.aspect) || 16 / 9,
        near: Number(p.near) || 0.05,
        far: Number(p.far) || 200,
        intrinsics: p.intrinsics
          ? {
              fx: Number(p.intrinsics.fx),
              fy: Number(p.intrinsics.fy),
              cx: Number(p.intrinsics.cx),
              cy: Number(p.intrinsics.cy),
              width: Number(p.intrinsics.width),
              height: Number(p.intrinsics.height),
              dist: Array.isArray(p.intrinsics.dist) ? p.intrinsics.dist.map(Number) : [],
            }
          : null,
        reprojectionError: Number.isFinite(Number(p.reprojectionError)) ? Number(p.reprojectionError) : null,
        solvedAt: p.solvedAt ?? null,
        solvedBy: p.solvedBy ?? null,
      };
      base.media = (Array.isArray(input.media) ? input.media : []).map((m) => ({
        url: String(m.url),
        kind: String(m.kind ?? 'video'),
        width: Number(m.width) || 0,
        height: Number(m.height) || 0,
        live: m.live === true,
      }));
      base.opacity = Number.isFinite(Number(input.opacity)) ? Number(input.opacity) : 0.6;
    }

    // 「尺寸标注」：用于「在粗模上标出需精确测量的位置 → 民警实测后填写 →
    // 反算比例尺提升整体精度」这条闭环。
    //   points[0]/points[1]：两个端点（模型局部坐标，米）
    //   modelDistance      ：两点在**当前模型**里的距离（前端算好一起存，便于对比）
    //   measuredDistance   ：民警实测值（null 表示「待测量」）
    if (kind === 'dimension') {
      if (base.points.length < 2) {
        // 带 statusCode 的错误由 API 层翻译成 400（用户输入问题，不是服务端故障）
        const err = new Error('dimension 至少需要 2 个端点（先点起点，再点终点）');
        err.statusCode = 400;
        throw err;
      }
      const mv = input.measured;
      const parsed = mv === null || mv === undefined || mv === '' ? null : Number(mv);
      base.measured = Number.isFinite(parsed) && parsed > 0 ? parsed : null;
      base.unit = String(input.unit ?? 'meter');
      // 模型侧距离：优先用前端传的，否则由两点算出来
      const md = Number(input.modelDistance);
      base.modelDistance = Number.isFinite(md) && md > 0 ? md : distance3(base.points[0], base.points[1]);
      // 这条测量的实测人和时间（用于「谁测的」追溯）
      base.measuredBy = base.measured ? String(input.measuredBy ?? operator) : null;
      base.measuredAt = base.measured ? nowISO() : null;
      // 「需要精确测量」的标记：measured 为空即待测
      base.pending = base.measured === null;
      base.priority = ['low', 'normal', 'high'].includes(String(input.priority))
        ? String(input.priority)
        : 'normal';
      base.tolerance = Number.isFinite(Number(input.tolerance)) ? Number(input.tolerance) : null;
    }

    return base;
  }

  async addAnnotation(id, input, operator) {
    const project = this.get(id);
    if (!project) return null;
    const ann = this.normalizeAnnotation(input, operator);

    // 幂等：如果调用方带了 id，且该 id 已存在，就按「更新」处理而不是再插一条。
    // 否则重复提交（网络重试、对话框二次触发、协同回显）会凭空多出一个对象——
    // 这正是「只导入一个视频却出现两个相机」的原因。
    const existingIdx = project.annotations.findIndex((a) => a.id === ann.id);
    if (existingIdx >= 0) {
      const prev = project.annotations[existingIdx];
      ann.createdAt = prev.createdAt;
      ann.createdBy = prev.createdBy;
      if (!input.createdBy) ann.createdBy = prev.createdBy;
      project.annotations[existingIdx] = ann;
      project.revision += 1;
      project.updatedAt = nowISO();
      this._pushEvent(project, {
        type: 'annotation.upsert',
        operator,
        detail: { id: ann.id, kind: ann.kind, label: ann.label },
      });
      await this._persist(project);
      return { project, annotation: ann, updated: true };
    }

    project.annotations.push(ann);
    project.revision += 1;
    project.updatedAt = nowISO();
    this._pushEvent(project, { type: 'annotation.add', operator, detail: { id: ann.id, kind: ann.kind, label: ann.label } });
    await this._persist(project);
    return { project, annotation: ann, updated: false };
  }

  async updateAnnotation(id, annId, patch, operator) {
    const project = this.get(id);
    if (!project) return null;
    const idx = project.annotations.findIndex((a) => a.id === annId);
    if (idx < 0) return null;
    const merged = { ...project.annotations[idx], ...patch, id: annId, updatedAt: nowISO() };
    const normalized = this.normalizeAnnotation(merged, operator);
    normalized.createdAt = project.annotations[idx].createdAt;
    normalized.createdBy = project.annotations[idx].createdBy;
    project.annotations[idx] = normalized;
    project.revision += 1;
    project.updatedAt = nowISO();
    this._pushEvent(project, { type: 'annotation.update', operator, detail: { id: annId } });
    await this._persist(project);
    return { project, annotation: normalized };
  }

  async deleteAnnotation(id, annId, operator) {
    const project = this.get(id);
    if (!project) return null;
    const idx = project.annotations.findIndex((a) => a.id === annId);
    if (idx < 0) return null;
    const [removed] = project.annotations.splice(idx, 1);
    project.revision += 1;
    project.updatedAt = nowISO();
    this._pushEvent(project, { type: 'annotation.delete', operator, detail: { id: annId } });
    await this._persist(project);
    return { project, annotation: removed };
  }

  async setModel(id, model, operator) {
    return this.mutate(
      id,
      (p) => {
        p.model = {
          name: String(model.name ?? 'model'),
          url: String(model.url),
          format: String(model.format ?? 'splat'),
          center: model.center ? vec3(model.center, 'model.center') : [0, 0, 0],
          scale: Number(model.scale) || 1,
          rotation: Array.isArray(model.rotation) && model.rotation.length === 9 ? model.rotation.map(Number) : null,
          bytes: Number(model.bytes) || 0,
          gaussianCount: Number(model.gaussianCount) || 0,
          shDegree: Number(model.shDegree) || 0,
          source: model.source ?? null,
        };
        if (model.thumbnail !== undefined) p.thumbnail = model.thumbnail;
      },
      { type: 'model.set', operator, detail: { name: model.name, url: model.url } }
    );
  }

  async setWorld(id, world, operator) {
    return this.mutate(
      id,
      (p) => {
        if (world.center) p.world.center = vec3(world.center, 'world.center');
        if (world.up) p.world.up = vec3(world.up, 'world.up');
        if (world.unit) p.world.unit = String(world.unit);
        if (world.scaleToMeters !== undefined) p.world.scaleToMeters = Number(world.scaleToMeters) || 1;
      },
      { type: 'world.set', operator, detail: world }
    );
  }

  /**
   * 依据「尺寸标注」里民警实测的数据，反算模型的比例尺修正系数。
   *
   * 原理：设模型单位到米的换算为 s，则对每条已实测的尺寸 i 有
   *          s · d_model,i  ≈  d_real,i
   * 最小二乘最优解（过原点）为：
   *          s* = Σ(d_model · d_real) / Σ(d_model²)
   * 这就是「用真实测量校正粗模比例尺」的核心公式。
   *
   * 注意：
   *   · 这里只解**尺度**（1 个自由度）。若测量里混入了端点拾取误差，
   *     s* 会被平均掉，所以会同时给出每条测量的偏差与整体 RMS，便于判断可信度；
   *   · 不解旋转/平移：本系统里模型与现场只差一个尺度（重建的比例是被任意定的）。
   *
   * @param {string} id 案件 id
   * @param {object} opts
   * @param {string} opts.operator 操作人
   * @param {boolean} opts.apply 是否把结果写入 project.world.scaleToMeters
   */
  async calibrateFromDimensions(id, { operator = '未署名', apply = true } = {}) {
    const project = this.get(id);
    if (!project) return null;

    const dims = (project.annotations ?? []).filter(
      (a) => a.kind === 'dimension' && Number.isFinite(Number(a.measured)) && Number(a.measured) > 0
    );
    if (!dims.length) {
      return { ok: false, reason: 'no-measurements', count: 0 };
    }

    let sumProd = 0;
    let sumSq = 0;
    const rows = [];
    for (const d of dims) {
      const dm = Number(d.modelDistance);
      const dr = Number(d.measured);
      if (!Number.isFinite(dm) || dm <= 0) continue;
      sumProd += dm * dr;
      sumSq += dm * dm;
      rows.push({ id: d.id, label: d.label, modelDistance: dm, measured: dr });
    }
    if (sumSq <= 0) return { ok: false, reason: 'degenerate', count: rows.length };

    const scaleToMeters = sumProd / sumSq;

    // 逐条残差 + RMS，用来判断这组测量是否自洽
    let sq = 0;
    for (const r of rows) {
      r.predicted = scaleToMeters * r.modelDistance;
      r.error = r.predicted - r.measured;
      r.relative = r.measured > 0 ? r.error / r.measured : 0;
      sq += r.error * r.error;
    }
    const rms = Math.sqrt(sq / rows.length);
    const meanAbsRel =
      rows.reduce((acc, r) => acc + Math.abs(r.relative), 0) / rows.length;

    const result = {
      ok: true,
      count: rows.length,
      scaleToMeters,
      rms,
      meanAbsRel,
      rows,
      // 可信度：平均相对偏差 <2% 很好，<5% 可用，否则建议检查标注点
      confidence: meanAbsRel < 0.02 ? 'high' : meanAbsRel < 0.05 ? 'medium' : 'low',
    };

    if (apply) {
      await this.mutate(
        id,
        (p) => {
          p.world = p.world ?? {};
          p.world.scaleToMeters = scaleToMeters;
          p.world.calibratedAt = nowISO();
          p.world.calibratedBy = operator;
          p.world.calibration = {
            count: rows.length,
            rms,
            meanAbsRel,
            confidence: result.confidence,
          };
        },
        { type: 'world.calibrate', operator, detail: { scaleToMeters, count: rows.length, rms } }
      );
    }
    return result;
  }

  /** 列出待实测的尺寸标注（民警要填的表） */
  pendingDimensions(id) {
    const project = this.get(id);
    if (!project) return null;
    return (project.annotations ?? [])
      .filter((a) => a.kind === 'dimension' && !a.measured)
      .map((a) => ({
        id: a.id,
        label: a.label,
        modelDistance: a.modelDistance,
        unit: a.unit ?? 'meter',
        priority: a.priority ?? 'normal',
        note: a.note ?? '',
        createdAt: a.createdAt,
        createdBy: a.createdBy,
      }));
  }

  async setPresentation(id, presentation, operator) {
    return this.mutate(
      id,
      (p) => {
        p.presentation = { ...(p.presentation ?? {}), ...presentation };
      },
      { type: 'presentation.set', operator, detail: presentation }
    );
  }

  async deleteProject(id, operator) {
    const project = this.get(id);
    if (!project) return false;
    this.cache.delete(id);
    await fs.rm(project._dir, { recursive: true, force: true });
    this._emit({ type: 'project.delete', projectId: id, operator });
    return true;
  }

  toWire(project) {
    const clone = { ...project };
    delete clone._dir;
    return clone;
  }
}

export { ANNOTATION_KINDS, vec3, color, nowISO };
