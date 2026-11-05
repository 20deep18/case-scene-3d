/**
 * 案发现场三维重建与协同研判系统 · 后端服务
 * 纯 Node.js 内置模块实现：静态资源、REST API、媒体上传（分片）、WebSocket 实时协同。
 */
import http from 'node:http';
import path from 'node:path';
import { promises as fs, createReadStream, readFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { WebSocketServer } from './ws.js';
import { Store } from './store.js';
import { ReconstructionJobs } from './recon-jobs.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..', '..');
const WEB_DIR = path.join(ROOT, 'app', 'web');
const VENDOR_DIR = path.join(ROOT, 'vendor');
const DATA_DIR = process.env.DSH_CASE_DATA ?? path.join(ROOT, 'data');
const PORT = Number(process.env.PORT ?? 8787);
const HOST = process.env.HOST ?? '0.0.0.0';
const MAX_UPLOAD = Number(process.env.MAX_UPLOAD ?? 2 * 1024 * 1024 * 1024); // 2GB

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.mkv': 'video/x-matroska',
  '.mov': 'video/quicktime',
  '.m4v': 'video/x-m4v',
  '.ts': 'video/mp2t',
  '.splat': 'application/octet-stream',
  '.ksplat': 'application/octet-stream',
  '.ply': 'application/octet-stream',
  '.bin': 'application/octet-stream',
  '.wasm': 'application/wasm',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.woff2': 'font/woff2',
};

const store = new Store(DATA_DIR);
await store.init();

// 现场建模任务管理器：把「视频/图片 → COLMAP → gsplat → .splat」跑成后台任务。
// onEvent 用回调形式注入，避免这里直接依赖 broadcast（它在下方才定义）。
const reconJobs = new ReconstructionJobs({
  root: ROOT,
  dataDir: DATA_DIR,
  modelsDir: path.join(DATA_DIR, 'models'),
  onEvent: (evt) => {
    try {
      if (evt.projectId) broadcast(evt.projectId, evt);
    } catch {
      /* 广播失败不影响任务本身 */
    }
  },
});

// ---------------------------------------------------------------- 工具函数

function json(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(payload),
    'cache-control': 'no-store',
  });
  res.end(payload);
}

function fail(res, status, message, extra = {}) {
  json(res, status, { ok: false, error: message, ...extra });
}

async function readBody(req, limit = MAX_UPLOAD) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw new Error(`请求体超过上限 ${(limit / 1048576).toFixed(0)}MB`);
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

async function readJSON(req) {
  const buf = await readBody(req, 32 * 1024 * 1024);
  if (!buf.length) return {};
  try {
    return JSON.parse(buf.toString('utf8'));
  } catch (err) {
    throw new Error(`JSON 解析失败：${err.message}`);
  }
}

function safeJoin(base, rel) {
  const target = path.resolve(base, '.' + path.posix.normalize('/' + rel.replace(/\\/g, '/')));
  if (!target.startsWith(path.resolve(base))) return null;
  return target;
}

const VIDEO_RE = /\.(mp4|webm|mkv|mov|m4v|ts)$/i;
const IMAGE_RE = /\.(jpg|jpeg|png|bmp|tif|tiff|webp)$/i;

function kindOfName(name) {
  return VIDEO_RE.test(name) ? 'video' : 'image';
}

function looksEncoded(s) {
  return /%[0-9A-Fa-f]{2}/.test(String(s ?? ''));
}

/**
 * 从媒体 URL / 相对路径解析出「案件 media 目录下的文件名」。
 *
 * ⚠️ 这里必须 `decodeURIComponent`，这是「开始建模点了没反应」的根因之一：
 *   上传接口把中文文件名编码进 URL，例如
 *     `/media/case-20261003/media/%E6%B5%8B%E8%AF%95%E8%A7%86%E9%A2%91.mp4`
 *   如果直接拿这串字符当路径去 open，拿到的是**字面量 `%E6%B5%8B…`** 这种不存在的
 *   文件，于是流水线报「视频不存在 → 可用图片 0 张 → 素材不足」，
 *   而素材其实好好躺在 `media/` 目录里。前端展示层已经解过码，服务端此前没有。
 */
function mediaFileNameFromUrl(raw) {
  let s = String(raw ?? '');
  try {
    s = decodeURIComponent(s);
  } catch {
    /* 不是合法的百分号编码（例如文件名里本来就有 %）→ 按原样处理 */
  }
  s = s.split('?')[0].split('#')[0].replace(/\\/g, '/');
  return path.posix.basename(s.replace(/\/+$/, ''));
}

/**
 * 收集一次建模要用的素材，解析成**磁盘绝对路径**并核对文件确实存在。
 *
 * @returns {Promise<{inputs: Array<{path:string,name:string,kind:string,bytes:number}>, missing: string[], bytes: number}>}
 */
async function collectReconInputs(project, assets) {
  const mediaDir = path.join(project._dir, 'media');
  /** @type {Array<{path:string,name:string,kind:string,bytes:number}>} */
  const inputs = [];
  const missing = [];
  const seen = new Set();

  const add = async (abs, name, kind) => {
    const key = path.resolve(abs).toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    let st = null;
    try {
      st = await fs.stat(abs);
    } catch {
      st = null;
    }
    if (!st || !st.isFile()) {
      missing.push(name || path.basename(abs));
      return;
    }
    inputs.push({ path: abs, name: name || path.basename(abs), kind: kind ?? kindOfName(name), bytes: st.size });
  };

  const explicit = Array.isArray(assets) ? assets.filter(Boolean) : [];
  for (const a of explicit) {
    if (typeof a === 'string') {
      const n = mediaFileNameFromUrl(a);
      await add(path.join(mediaDir, n), n, kindOfName(n));
      continue;
    }
    if (a && a.url) {
      // URL 里的中文名是 %XX 编码；a.name 偶尔为空或是编码过的，都要还原成可读文本
      const n = mediaFileNameFromUrl(a.url);
      const shown = a.name && !looksEncoded(a.name) ? a.name : n;
      await add(path.join(mediaDir, n), shown, a.kind ?? kindOfName(n));
    }
  }

  // 没给素材清单（或压根没解析出任何路径）时，退化为扫描本案 media 目录
  if (!inputs.length && !missing.length) {
    try {
      const entries = await fs.readdir(mediaDir, { withFileTypes: true });
      for (const e of entries) {
        if (!e.isFile() || e.name.startsWith('.')) continue;
        if (!VIDEO_RE.test(e.name) && !IMAGE_RE.test(e.name)) continue;
        await add(path.join(mediaDir, e.name), e.name, kindOfName(e.name));
      }
    } catch {
      /* media 目录可能还不存在 */
    }
  }

  const bytes = inputs.reduce((s, i) => s + i.bytes, 0);
  return { inputs, missing, bytes };
}

/**
 * 「建模所需最小素材体积」阈值。
 *
 * 数值来自**真实训练实测**（tools/threshold_experiment.py），
 * 单一口径放在 pipeline/recon_threshold.json，服务端与 Python 侧共用。
 */
const THRESHOLD_FILE = path.join(ROOT, 'pipeline', 'recon_threshold.json');
const FALLBACK_THRESHOLD = {
  minBytes: 12 * 1024 * 1024,
  recommendedBytes: 32 * 1024 * 1024,
  measured: false,
  note: '尚未读取到实测阈值文件，使用保守默认值',
};

function loadReconThreshold() {
  try {
    const raw = JSON.parse(readFileSync(THRESHOLD_FILE, 'utf8'));
    const min = Number(raw?.minBytes);
    const rec = Number(raw?.recommendedBytes);
    return {
      minBytes: Number.isFinite(min) && min > 0 ? min : FALLBACK_THRESHOLD.minBytes,
      recommendedBytes: Number.isFinite(rec) && rec > 0 ? rec : FALLBACK_THRESHOLD.recommendedBytes,
      measured: raw?.measured !== false,
      note: raw?.note ?? '',
      measuredAt: raw?.measuredAt ?? null,
      report: raw?.report ?? null,
    };
  } catch {
    return { ...FALLBACK_THRESHOLD };
  }
}

async function serveFile(req, res, filePath, { immutable = false } = {}) {
  let stat;
  try {
    stat = await fs.stat(filePath);
  } catch {
    return false;
  }
  if (stat.isDirectory()) {
    return serveFile(req, res, path.join(filePath, 'index.html'), { immutable });
  }
  const ext = path.extname(filePath).toLowerCase();
  const type = MIME[ext] ?? 'application/octet-stream';
  const etag = `W/"${stat.size}-${Number(stat.mtimeMs).toString(36)}"`;
  if (req.headers['if-none-match'] === etag) {
    res.writeHead(304, { etag });
    res.end();
    return true;
  }
  const range = req.headers.range;
  // 非 immutable 的资源（HTML / JS / CSS）一律 **no-store**。
  // 用 `no-cache` 只表示「需要回源校验」，但浏览器仍可能在内存里继续用旧副本，
  // 表现为「改了代码刷新却还是老行为」——排查这种问题非常费时。
  // 开发期（本系统就是本地单机部署）强制不缓存最省事、也最不容易出错。
  const baseHeaders = {
    'content-type': type,
    'accept-ranges': 'bytes',
    etag,
    'cache-control': immutable ? 'public, max-age=604800' : 'no-store, must-revalidate',
  };
  if (range) {
    const m = /^bytes=(\d*)-(\d*)$/.exec(range.trim());
    if (m) {
      const start = m[1] ? Number(m[1]) : 0;
      const end = m[2] ? Math.min(Number(m[2]), stat.size - 1) : stat.size - 1;
      if (Number.isNaN(start) || start > end || start >= stat.size) {
        res.writeHead(416, { 'content-range': `bytes */${stat.size}` });
        res.end();
        return true;
      }
      res.writeHead(206, {
        ...baseHeaders,
        'content-range': `bytes ${start}-${end}/${stat.size}`,
        'content-length': end - start + 1,
      });
      if (req.method === 'HEAD') return res.end(), true;
      createReadStream(filePath, { start, end }).pipe(res);
      return true;
    }
  }
  res.writeHead(200, { ...baseHeaders, 'content-length': stat.size });
  if (req.method === 'HEAD') return res.end(), true;
  createReadStream(filePath).pipe(res);
  return true;
}

const server = http.createServer();

// ---------------------------------------------------------------- 协同房间

/** projectId -> Map<clientId, presence> */
const rooms = new Map();

function roomOf(projectId) {
  if (!rooms.has(projectId)) rooms.set(projectId, new Map());
  return rooms.get(projectId);
}

function roster(projectId) {
  return [...roomOf(projectId).values()].map((p) => ({
    id: p.id,
    name: p.name,
    color: p.color,
    role: p.role,
    pose: p.pose ?? null,
  }));
}

const wss = new WebSocketServer(server, { path: '/ws' });

wss.on('connection', (conn, req) => {
  const url = new URL(req.url, 'http://localhost');
  const projectId = url.searchParams.get('project') ?? '';
  const name = decodeURIComponent(url.searchParams.get('name') ?? '匿名');
  const color = url.searchParams.get('color') ?? '#4dd0e1';
  const role = url.searchParams.get('role') ?? 'viewer';
  const project = store.get(projectId);

  conn.data.projectId = projectId;
  conn.data.name = name;
  conn.data.color = color;

  if (!project) {
    conn.sendJSON({ type: 'error', message: `案件 ${projectId} 不存在` });
    return conn.close(1008, 'unknown project');
  }

  const presence = { id: conn.id, name, color, role, pose: null, joinedAt: Date.now(), conn };
  roomOf(projectId).set(conn.id, presence);

  conn.sendJSON({
    type: 'welcome',
    you: { id: conn.id, name, color, role },
    project: store.toWire(project),
    roster: roster(projectId),
  });
  broadcast(projectId, { type: 'roster', roster: roster(projectId) }); // 含自己，前端自行忽略
  broadcast(projectId, { type: 'system', level: 'info', message: `${name} 加入了研判` }, conn.id);

  conn.on('message', (raw, isBinary) => {
    if (isBinary) return;
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch {
      return;
    }
    handleMessage(conn, presence, msg);
  });

  conn.on('close', () => {
    roomOf(projectId).delete(conn.id);
    broadcast(projectId, { type: 'roster', roster: roster(projectId) });
    broadcast(projectId, { type: 'system', level: 'info', message: `${name} 离开了研判` });
    store.logEvent(projectId, { type: 'user.leave', operator: name }).catch(() => {});
  });

  store.logEvent(projectId, { type: 'user.join', operator: name, detail: { role } }).catch(() => {});
});

function broadcast(projectId, payload, exceptId = null) {
  const data = typeof payload === 'string' ? payload : JSON.stringify(payload);
  for (const p of roomOf(projectId).values()) {
    if (exceptId && p.id === exceptId) continue;
    if (p.conn.readyState === 'open') p.conn.send(data);
  }
}

function handleMessage(conn, presence, msg) {
  const projectId = conn.data.projectId;
  switch (msg.type) {
    case 'ping':
      return conn.sendJSON({ type: 'pong', t: msg.t ?? Date.now() });
    case 'pose': {
      presence.pose = {
        p: msg.p,
        q: msg.q,
        fov: msg.fov,
        mode: msg.mode ?? 'orbit',
        focus: msg.focus ?? null,
        at: Date.now(),
      };
      presence.name = typeof msg.name === 'string' && msg.name.trim() ? msg.name.trim() : presence.name;
      return broadcast(projectId, { type: 'pose', id: conn.id, ...presence.pose }, conn.id);
    }
    case 'cursor':
      presence.cursor = { p: msg.p, label: msg.label ?? null, at: Date.now() };
      return broadcast(projectId, { type: 'cursor', id: conn.id, p: msg.p, label: msg.label ?? null }, conn.id);
    case 'chat':
      return broadcast(projectId, {
        type: 'chat',
        id: conn.id,
        name: presence.name,
        color: presence.color,
        text: String(msg.text ?? '').slice(0, 2000),
        at: Date.now(),
      });
    case 'select':
      presence.select = msg.annotationId ?? null;
      return broadcast(projectId, { type: 'select', id: conn.id, annotationId: msg.annotationId ?? null }, conn.id);
    case 'annotation.add':
    case 'annotation.update':
    case 'annotation.delete':
    case 'annotation.batch':
      return handleAnnotationMutation(conn, presence, msg);
    default:
      return;
  }
}

async function handleAnnotationMutation(conn, presence, msg) {
  const projectId = conn.data.projectId;
  try {
    if (msg.type === 'annotation.add') {
      const result = await store.addAnnotation(projectId, msg.annotation, presence.name);
      if (!result) throw new Error('案件不存在');
      // 幂等命中时也要广播，让所有端（含发起端）收敛到同一份数据
      broadcast(projectId, {
        type: result.updated ? 'annotation.update' : 'annotation.add',
        annotation: result.annotation,
        revision: result.project.revision,
        by: { id: conn.id, name: presence.name, color: presence.color },
      });
      return;
    }
    if (msg.type === 'annotation.update') {
      const result = await store.updateAnnotation(projectId, msg.id, msg.patch ?? {}, presence.name);
      if (!result) throw new Error('标注不存在');
      broadcast(projectId, {
        type: 'annotation.update',
        annotation: result.annotation,
        revision: result.project.revision,
        by: { id: conn.id, name: presence.name, color: presence.color },
      });
      return;
    }
    if (msg.type === 'annotation.delete') {
      const result = await store.deleteAnnotation(projectId, msg.id, presence.name);
      if (!result) throw new Error('标注不存在');
      broadcast(projectId, {
        type: 'annotation.delete',
        id: result.annotation.id,
        revision: result.project.revision,
        by: { id: conn.id, name: presence.name, color: presence.color },
      });
      return;
    }
    if (msg.type === 'annotation.batch') {
      const applied = [];
      for (const op of msg.ops ?? []) {
        if (op.op === 'add') {
          const r = await store.addAnnotation(projectId, op.annotation, presence.name);
          if (r) applied.push({ op: 'add', annotation: r.annotation });
        } else if (op.op === 'update') {
          const r = await store.updateAnnotation(projectId, op.id, op.patch ?? {}, presence.name);
          if (r) applied.push({ op: 'update', annotation: r.annotation });
        } else if (op.op === 'delete') {
          const r = await store.deleteAnnotation(projectId, op.id, presence.name);
          if (r) applied.push({ op: 'delete', id: r.annotation.id });
        }
      }
      broadcast(projectId, { type: 'annotation.batch', ops: applied, by: { id: conn.id, name: presence.name } });
    }
  } catch (err) {
    conn.sendJSON({ type: 'error', message: err.message, ref: msg.type });
  }
}

// ---------------------------------------------------------------- HTTP 路由

async function apiHandler(req, res, url) {
  const parts = url.pathname
    .replace(/^\/api\/?/, '')
    .split('/')
    .filter(Boolean)
    .map((p) => {
      try {
        return decodeURIComponent(p);
      } catch {
        return p;
      }
    });
  const method = req.method ?? 'GET';

  // GET /api/health
  if (parts[0] === 'health') {
    return json(res, 200, {
      ok: true,
      name: '案发现场三维重建与协同研判系统',
      version: '1.0.3',
      uptime: process.uptime(),
      node: process.version,
      projects: store.list().length,
      clients: wss.clients.size,
      time: new Date().toISOString(),
    });
  }

  // /api/models/scene-mode —— 场景判定的**人工改判**（用户可覆盖）
  //
  // 为什么需要它：几何本身分不开「贴着物体拍」与「以旋转为主拍的房间」
  // （实测柜子素材 0.2344 vs 某段房间素材 0.2270），所以判不准时不该硬猜，
  // 而应该**让拍摄的人直接指定**。后端 `tools/scene_structure.py --scene-mode`
  // 早就有这个能力，这里补上前端唯一的入口。
  //
  // ⚠️ 改判结果存**独立 sidecar**（`<模型名>.scenemode.json`），不写进模型元数据
  //   `<模型名>.json` —— 后者会被重建流水线重写，写进去会在下一次建模时被悄悄抹掉。
  if (parts[0] === 'models' && parts[1] === 'scene-mode') {
    const MODELS_DIR = path.join(DATA_DIR, 'models');
    const MODES = ['auto', 'room', 'object'];
    const sidecarOf = (modelFile) =>
      path.join(MODELS_DIR, `${String(modelFile).replace(/\.(splat|ply|ksplat)$/i, '')}.scenemode.json`);
    /** 只接受 data/models 下的**真实模型文件名**：挡掉路径穿越与任意文件读写 */
    const resolveModel = async (modelFile) => {
      if (typeof modelFile !== 'string' || !modelFile) return null;
      if (modelFile.includes('/') || modelFile.includes('\\') || modelFile.includes('..')) return null;
      if (!/\.(splat|ply|ksplat)$/i.test(modelFile)) return null;
      try {
        await fs.stat(path.join(MODELS_DIR, modelFile));
        return modelFile;
      } catch {
        return null;
      }
    };
    const readOverride = async (modelFile) => {
      try {
        const raw = JSON.parse(await fs.readFile(sidecarOf(modelFile), 'utf8'));
        if (raw && MODES.includes(raw.mode) && raw.mode !== 'auto') {
          return { mode: raw.mode, operator: raw.operator ?? null, updatedAt: raw.updatedAt ?? null };
        }
      } catch {
        /* 没有改判 = 跟随自动判定 */
      }
      return null;
    };

    if (method === 'GET') {
      const q = new URL(req.url, `http://${req.headers.host ?? 'localhost'}`);
      const modelFile = await resolveModel(q.searchParams.get('model') ?? '');
      if (!modelFile) return fail(res, 400, '未知的模型文件（或路径非法）');
      return json(res, 200, { ok: true, sceneModeOverride: await readOverride(modelFile) });
    }
    if (method === 'POST') {
      const body = await readJSON(req).catch(() => ({}));
      const modelFile = await resolveModel(body.model);
      if (!modelFile) return fail(res, 400, '未知的模型文件（或路径非法）');
      const mode = String(body.mode ?? '');
      if (!MODES.includes(mode)) return fail(res, 400, 'mode 只能是 auto / room / object');
      if (mode === 'auto') {
        // auto = 清除改判，回到证据自动判定
        await fs.rm(sidecarOf(modelFile), { force: true });
        return json(res, 200, { ok: true, sceneModeOverride: null });
      }
      const rec = {
        mode,
        operator: body.operator ?? '未署名',
        updatedAt: new Date().toISOString(),
        note: '人工改判：前端据此覆盖 structure.json 里的 sceneMode（不改写结构文件本身）',
      };
      await fs.mkdir(MODELS_DIR, { recursive: true });
      await fs.writeFile(sidecarOf(modelFile), JSON.stringify(rec, null, 2), 'utf8');
      return json(res, 200, { ok: true, sceneModeOverride: rec });
    }
    return fail(res, 405, '方法不支持');
  }

  // /api/projects
  if (parts[0] === 'projects' && parts.length === 1) {
    if (method === 'GET') return json(res, 200, { ok: true, projects: store.list() });
    if (method === 'POST') {
      const body = await readJSON(req);
      if (!body.name) return fail(res, 400, '缺少 name');
      const project = await store.create({
        name: body.name,
        description: body.description ?? '',
        unit: body.unit ?? 'meter',
        operator: body.operator ?? '未署名',
      });
      return json(res, 201, { ok: true, project: store.toWire(project) });
    }
    return fail(res, 405, '方法不支持');
  }

  const projectId = parts[1];
  const project = projectId ? store.get(projectId) : null;
  if (parts[0] === 'projects' && projectId && !project) return fail(res, 404, `案件 ${projectId} 不存在`);

  if (parts[0] === 'projects' && project) {
    // /api/projects/:id
    if (parts.length === 2) {
      if (method === 'GET') return json(res, 200, { ok: true, project: store.toWire(project) });
      if (method === 'PATCH') {
        const body = await readJSON(req);
        const updated = await store.mutate(
          projectId,
          (p) => {
            if (typeof body.name === 'string' && body.name.trim()) p.name = body.name.trim();
            if (typeof body.description === 'string') p.description = body.description;
          },
          { type: 'project.update', operator: body.operator ?? '未署名', detail: {} }
        );
        return json(res, 200, { ok: true, project: store.toWire(updated) });
      }
      if (method === 'DELETE') {
        const body = await readJSON(req).catch(() => ({}));
        await store.deleteProject(projectId, body.operator ?? '未署名');
        return json(res, 200, { ok: true });
      }
      return fail(res, 405, '方法不支持');
    }

    // /api/projects/:id/annotations
    if (parts[2] === 'annotations') {
      if (parts.length === 3) {
        if (method === 'GET') return json(res, 200, { ok: true, annotations: project.annotations });
        if (method === 'POST') {
          const body = await readJSON(req);
          const result = await store.addAnnotation(projectId, body.annotation ?? body, body.operator ?? '未署名');
          broadcast(projectId, {
            type: result.updated ? 'annotation.update' : 'annotation.add',
            annotation: result.annotation,
            revision: result.project.revision,
            by: { id: 'rest', name: body.operator ?? 'REST' },
          });
          return json(res, result.updated ? 200 : 201, { ok: true, annotation: result.annotation });
        }
        return fail(res, 405, '方法不支持');
      }
      const annId = parts[3];
      if (method === 'PATCH') {
        const body = await readJSON(req);
        const result = await store.updateAnnotation(projectId, annId, body.patch ?? body, body.operator ?? '未署名');
        if (!result) return fail(res, 404, '标注不存在');
        broadcast(projectId, { type: 'annotation.update', annotation: result.annotation, revision: result.project.revision, by: { id: 'rest', name: body.operator ?? 'REST' } });
        return json(res, 200, { ok: true, annotation: result.annotation });
      }
      if (method === 'DELETE') {
        const body = await readJSON(req).catch(() => ({}));
        const result = await store.deleteAnnotation(projectId, annId, body.operator ?? '未署名');
        if (!result) return fail(res, 404, '标注不存在');
        broadcast(projectId, { type: 'annotation.delete', id: annId, revision: result.project.revision, by: { id: 'rest', name: body.operator ?? 'REST' } });
        return json(res, 200, { ok: true });
      }
      return fail(res, 405, '方法不支持');
    }

    // /api/projects/:id/model
    if (parts[2] === 'model' && method === 'POST') {
      const body = await readJSON(req);
      const updated = await store.setModel(projectId, body.model ?? body, body.operator ?? '未署名');
      broadcast(projectId, { type: 'model.set', model: updated.model });
      return json(res, 200, { ok: true, model: updated.model });
    }

    // /api/projects/:id/world
    if (parts[2] === 'world' && method === 'POST') {
      const body = await readJSON(req);
      const updated = await store.setWorld(projectId, body.world ?? body, body.operator ?? '未署名');
      broadcast(projectId, { type: 'world.set', world: updated.world });
      return json(res, 200, { ok: true, world: updated.world });
    }

    // /api/projects/:id/presentation
    if (parts[2] === 'presentation' && method === 'POST') {
      const body = await readJSON(req);
      const updated = await store.setPresentation(projectId, body.presentation ?? body, body.operator ?? '未署名');
      broadcast(projectId, { type: 'presentation.set', presentation: updated.presentation });
      return json(res, 200, { ok: true, presentation: updated.presentation });
    }

    // /api/projects/:id/media  （分片上传：?name=x.mp4&chunk=0&chunks=8&upload=token）
    if (parts[2] === 'media' && method === 'POST') {
      const name = url.searchParams.get('name');
      if (!name) return fail(res, 400, '缺少 name');
      const token = url.searchParams.get('upload') ?? randomUUID();
      const chunkIndex = Number(url.searchParams.get('chunk') ?? 0);
      const chunkTotal = Number(url.searchParams.get('chunks') ?? 1);
      const cleanName = path.basename(name).replace(/[\\/:*?"<>|]/g, '_');
      const mediaDir = path.join(project._dir, 'media');
      await fs.mkdir(mediaDir, { recursive: true });
      const staging = path.join(mediaDir, `.${token}.part`);
      const buf = await readBody(req);
      if (chunkIndex === 0) await fs.writeFile(staging, buf);
      else await fs.appendFile(staging, buf);
      const stat = await fs.stat(staging);
      const done = chunkIndex + 1 >= chunkTotal;
      if (!done) return json(res, 200, { ok: true, received: chunkIndex, bytes: stat.size, next: chunkIndex + 1, upload: token });
      const finalPath = path.join(mediaDir, cleanName);
      await fs.rm(finalPath, { force: true });
      await fs.rename(staging, finalPath);
      const finalStat = await fs.stat(finalPath);
      return json(res, 201, {
        ok: true,
        // 媒体文件实际落在 data/projects/<案件ID>/media/<文件名>，
        // 因此 URL 必须带上 `media` 这一段，否则 /media 路由会找不到文件而 404
        //（前端表现为视频卡片黑屏、贴回现场看不到画面）。
        url: `/media/${encodeURIComponent(projectId)}/media/${encodeURIComponent(cleanName)}`,
        bytes: finalStat.size,
        name: cleanName,
        kind: /\.(mp4|webm|mkv|mov|m4v|ts)$/i.test(cleanName) ? 'video' : 'image',
      });
    }

    // ------------------------------------------------------------------
    // 删除已上传的媒体文件
    //
    //   DELETE /api/projects/:id/media?url=/media/<pid>/media/<file>
    //
    // 前端「多源画面」的删除按钮会调用它。
    // 只允许删本案件 media 目录下的文件（safeJoin 会把路径锁在 projects/ 之内，
    // 这里再显式校验一次前缀，避免跨案件误删）。
    // ------------------------------------------------------------------
    if (parts[2] === 'media' && method === 'DELETE') {
      const raw = url.searchParams.get('url') ?? '';
      if (!raw) return fail(res, 400, '缺少 url 参数');
      let decoded = raw;
      try {
        decoded = decodeURIComponent(raw);
      } catch {
        /* 保持原样 */
      }
      // 兼容两种写法：/media/<pid>/media/<file> 或直接 media/<file>
      const rel = decoded.replace(/^\/?media\//, '').replace(new RegExp(`^${projectId}/`), '');
      const target = safeJoin(DATA_DIR, `projects/${projectId}/${rel.startsWith('media/') ? rel : `media/${path.basename(rel)}`}`);
      if (!target) return fail(res, 400, '非法路径');
      // 再确认一次确实落在本案的 media 目录里
      const mediaRoot = path.resolve(project._dir, 'media');
      if (!path.resolve(target).startsWith(mediaRoot + path.sep)) {
        return fail(res, 400, '只能删除本案 media 目录下的文件');
      }
      try {
        await fs.rm(target, { force: true });
      } catch (err) {
        return fail(res, 500, `删除文件失败：${err.message}`);
      }
      broadcast(projectId, { type: 'media.delete', url: raw, by: { id: 'rest', name: 'REST' } });
      return json(res, 200, { ok: true, deleted: path.relative(DATA_DIR, target) });
    }

    // ------------------------------------------------------------------
    // 现场建模（依据视频/图片建立现场模型）
    //
    //   GET  /api/projects/:id/recon/materials  素材体积体检（是否够建模）
    //   POST /api/projects/:id/recon            启动建模任务
    //   GET  /api/projects/:id/recon            列出该案件的建模任务
    // ------------------------------------------------------------------

    // 素材体积体检：前端在「开始建模」对话框里用它决定
    //   · 体积 ≥ minBytes  → 直接建模
    //   · 体积 <  minBytes → 先弹「提供信息过少，可能无法正常建模」，由用户决定
    if (parts[2] === 'recon' && parts[3] === 'materials') {
      if (method !== 'GET' && method !== 'POST') return fail(res, 405, '方法不支持');
      const body = method === 'POST' ? await readJSON(req).catch(() => ({})) : {};
      const assets = body.assets ?? (url.searchParams.get('assets') ? url.searchParams.get('assets').split(',') : []);
      const { inputs, missing, bytes } = await collectReconInputs(project, assets);
      const threshold = loadReconThreshold();
      return json(res, 200, {
        ok: true,
        count: inputs.length,
        videos: inputs.filter((i) => i.kind === 'video').length,
        images: inputs.filter((i) => i.kind === 'image').length,
        bytes,
        files: inputs.map((i) => ({ name: i.name, kind: i.kind, bytes: i.bytes })),
        missing,
        minBytes: threshold.minBytes,
        recommendedBytes: threshold.recommendedBytes,
        enough: bytes >= threshold.minBytes,
        // 说明阈值从哪来（实测报告），便于界面/排查时自证
        threshold: {
          measured: threshold.measured,
          measuredAt: threshold.measuredAt,
          report: threshold.report,
          note: threshold.note,
        },
      });
    }

    if (parts[2] === 'recon' && parts.length === 3) {
      if (method === 'GET') {
        return json(res, 200, { ok: true, jobs: reconJobs.list().filter((j) => j.projectId === projectId) });
      }
      if (method === 'POST') {
        const body = await readJSON(req);
        const { inputs, missing } = await collectReconInputs(project, body.assets);
        if (!inputs.length) {
          if (missing.length) {
            return fail(
              res,
              400,
              `素材文件在服务器上找不到：${missing.slice(0, 3).join('、')}${missing.length > 3 ? ` 等 ${missing.length} 个` : ''}。请重新上传后再试。`
            );
          }
          return fail(res, 400, '该案件还没有素材。请先在下方「多源画面」里导入监控/勘查视频或图片。');
        }
        if (missing.length) {
          // 部分缺失不阻断，但要说清楚，避免「用了一半素材还以为全用上了」
          console.warn(`[recon] 有 ${missing.length} 个素材文件缺失，已跳过：${missing.join('、')}`);
        }

        const result = reconJobs.start({
          projectId,
          projectDir: project._dir,
          name: body.name,
          inputs,
          params: body.params ?? {},
        });
        if (!result.ok) return fail(res, 409, result.error);
        broadcast(projectId, { type: 'recon.start', job: result.job });
        return json(res, 202, { ok: true, job: result.job, skipped: missing });
      }
      return fail(res, 405, '方法不支持');
    }

    // /api/projects/:id/recon/:jobId       任务状态
    // /api/projects/:id/recon/:jobId/log   任务日志（轮询增量）
    // /api/projects/:id/recon/:jobId/cancel 取消任务
    if (parts[2] === 'recon' && parts[3]) {
      const jobId = parts[3];
      const job = reconJobs.jobs.get(jobId);
      if (!job || job.projectId !== projectId) return fail(res, 404, `建模任务 ${jobId} 不存在`);

      if (parts[4] === 'log' && method === 'GET') {
        const tail = Math.min(Number(url.searchParams.get('tail') ?? 200), 2000);
        const fromLine = Number(url.searchParams.get('from') ?? 0);
        const data = reconJobs.readLog(jobId, { tail, fromLine });
        reconJobs.refresh(jobId);
        return json(res, 200, { ok: true, ...data, status: reconJobs.status(jobId) });
      }
      if (parts[4] === 'cancel' && method === 'POST') {
        const r = reconJobs.cancel(jobId);
        if (!r.ok) return fail(res, 409, r.error);
        broadcast(projectId, { type: 'recon.cancel', id: jobId });
        return json(res, 200, { ok: true });
      }
      if (parts.length === 4 && method === 'GET') {
        reconJobs.refresh(jobId);
        return json(res, 200, { ok: true, job: reconJobs.status(jobId) });
      }
      return fail(res, 405, '方法不支持');
    }

    // ------------------------------------------------------------------
    // 精确测量与比例尺校正
    //
    //   GET  /api/projects/:id/measurements  待实测清单 + 已实测清单 + 当前比例尺
    //   POST /api/projects/:id/calibrate     依据实测值反算比例尺（可只算不应用）
    //
    // 业务闭环：
    //   ① 现场民警在粗模上用「尺寸标注」工具点两端 → 生成 measured=null 的待测项
    //   ② 民警拿卷尺量出真实距离 → 在「待实测」列表里填写提交
    //   ③ 系统用最小二乘反算 scaleToMeters → 整个模型/测距立刻按真实尺寸显示
    // ------------------------------------------------------------------
    if (parts[2] === 'measurements' && method === 'GET') {
      const world = project.world ?? {};
      const all = (project.annotations ?? []).filter((a) => a.kind === 'dimension');
      return json(res, 200, {
        ok: true,
        pending: all
          .filter((a) => !a.measured)
          .map((a) => ({
            id: a.id,
            label: a.label,
            modelDistance: a.modelDistance,
            unit: a.unit ?? 'meter',
            priority: a.priority ?? 'normal',
            note: a.note ?? '',
            createdBy: a.createdBy,
            createdAt: a.createdAt,
          })),
        measured: all
          .filter((a) => a.measured)
          .map((a) => ({
            id: a.id,
            label: a.label,
            modelDistance: a.modelDistance,
            measured: a.measured,
            unit: a.unit ?? 'meter',
            error: a.measured ? Number(a.measured) - Number(a.modelDistance) * (world.scaleToMeters ?? 1) : null,
            measuredBy: a.measuredBy,
            measuredAt: a.measuredAt,
          })),
        scaleToMeters: world.scaleToMeters ?? 1,
        calibratedAt: world.calibratedAt ?? null,
        calibratedBy: world.calibratedBy ?? null,
        calibration: world.calibration ?? null,
      });
    }

    if (parts[2] === 'calibrate' && method === 'POST') {
      const body = await readJSON(req).catch(() => ({}));
      const result = await store.calibrateFromDimensions(projectId, {
        operator: body.operator ?? '未署名',
        apply: body.apply !== false,
      });
      if (!result) return fail(res, 404, '案件不存在');
      if (!result.ok) {
        const msg =
          result.reason === 'no-measurements'
            ? '还没有任何已实测的尺寸。请先在粗模上用「尺寸标注」标出两点，并填写民警实测值。'
            : '测量数据不足或退化，无法反算比例尺。';
        return fail(res, 400, msg);
      }
      const fresh = store.get(projectId);
      broadcast(projectId, {
        type: 'world.calibrate',
        scaleToMeters: result.scaleToMeters,
        calibration: { count: result.count, rms: result.rms, meanAbsRel: result.meanAbsRel, confidence: result.confidence },
        by: { id: 'rest', name: body.operator ?? 'REST' },
      });
      return json(res, 200, { ok: true, ...result, world: fresh?.world ?? null });
    }

    // /api/projects/:id/events
    if (parts[2] === 'events' && method === 'GET') {
      const limit = Math.min(Number(url.searchParams.get('limit') ?? 200), 2000);
      return json(res, 200, { ok: true, events: project.events.slice(-limit) });
    }

    // /api/projects/:id/export
    if (parts[2] === 'export' && method === 'GET') {
      return json(res, 200, store.toWire(project));
    }

    // /api/projects/:id/import
    if (parts[2] === 'import' && method === 'POST') {
      const body = await readJSON(req);
      const payload = body.project ?? body;
      const updated = await store.mutate(
        projectId,
        (p) => {
          if (Array.isArray(payload.annotations)) p.annotations = payload.annotations.map((a) => store.normalizeAnnotation(a));
          if (payload.world) p.world = { ...p.world, ...payload.world };
          if (payload.presentation) p.presentation = { ...p.presentation, ...payload.presentation };
        },
        { type: 'project.import', operator: body.operator ?? '未署名' }
      );
      broadcast(projectId, { type: 'project.replace', project: store.toWire(updated) });
      return json(res, 200, { ok: true, project: store.toWire(updated) });
    }
  }

  return fail(res, 404, `未知接口 ${url.pathname}`);
}

server.on('request', async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host ?? 'localhost'}`);
  res.setHeader('access-control-allow-origin', '*');
  res.setHeader('access-control-allow-headers', 'content-type');
  res.setHeader('access-control-allow-methods', 'GET,POST,PATCH,DELETE,HEAD,OPTIONS');
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    return res.end();
  }

  try {
    if (url.pathname === '/api' || url.pathname.startsWith('/api/')) {
      return await apiHandler(req, res, url);
    }

    // 案件静态资源：/media/<projectId>/<file>
    // 注意：上传时写入的是 DATA_DIR/projects/<projectId>/media/<file>，
    // 而 URL 里的 <projectId> 就是 projects 下的目录名，因此这里**不能**再补一层 projects/，
    // 否则会拼成 data/projects/projects/... 导致所有视频与图片 404（表现为播放黑屏）。
    if (url.pathname.startsWith('/media/')) {
      const rel = decodeURIComponent(url.pathname.slice('/media/'.length));
      const target = safeJoin(DATA_DIR, `projects/${rel.replace(/\\/g, '/')}`);
      if (!target) return fail(res, 400, '非法路径');
      if (await serveFile(req, res, target)) return;
      return fail(res, 404, `资源不存在：${path.relative(DATA_DIR, target)}`);
    }

    // 共享模型：/models/<file>（把训练产出的 .splat 放到 data/models/）
    if (url.pathname.startsWith('/models/')) {
      const rel = decodeURIComponent(url.pathname.slice('/models/'.length));
      if (rel === 'index.json') {
        const dir = path.join(DATA_DIR, 'models');
        let entries = [];
        try {
          entries = await fs.readdir(dir, { withFileTypes: true });
        } catch {
          await fs.mkdir(dir, { recursive: true });
        }
        const models = [];
        for (const e of entries) {
          if (!e.isFile()) continue;
          if (!/\.(splat|ply|ksplat)$/i.test(e.name)) continue;
          const full = path.join(dir, e.name);
          const stat = await fs.stat(full);
          let meta = null;
          try {
            meta = JSON.parse(await fs.readFile(full.replace(/\.(splat|ply|ksplat)$/i, '.json'), 'utf8'));
          } catch {
            /* 元数据可选 */
          }
          // 房间结构骨架（地面/墙/房间盒/相机轨迹）：由 tools/scene_structure.py 生成。
          // 有它前端才能画出「整体布局与空间线条」；没有就只是普通泼溅。
          let hasStructure = false;
          let coverage = null;
          let sceneMode = null;
          let sceneModeSource = null;
          const structPath = full.replace(/\.(splat|ply|ksplat)$/i, '.structure.json');
          try {
            const raw = JSON.parse(await fs.readFile(structPath, 'utf8'));
            hasStructure = true;
            sceneMode = raw?.sceneMode ?? null;
            sceneModeSource = raw?.sceneModeSource ?? 'auto';
            // 只把「拍摄引导」需要的几个字段带出去：前端要据此提示
            // 「哪些方向根本没拍到，模型会在那里塌，建议补拍」。
            const cov = raw?.coverage;
            if (cov && cov.frames) {
              coverage = {
                verdict: cov.verdict ?? null,
                frames: cov.frames,
                weakFrames: cov.weakFrames ?? null,
                weakRatio: cov.weakRatio ?? null,
                worstToMedianRatio: cov.worstToMedianRatio ?? null,
                medianPointsInView: cov.medianPointsInView ?? null,
                minPointsInView: cov.minPointsInView ?? null,
                weakExamples: (cov.weakExamples ?? []).slice(0, 5),
              };
            }
          } catch {
            hasStructure = false;
          }
          // 人工改判（sidecar）：存在时**覆盖**结构文件里的自动判定。
          // 结构文件本身不改写 —— 它是那次重建的证据，改判是使用者的判断，两者分开存。
          let sceneModeOverride = null;
          try {
            const sm = JSON.parse(
              await fs.readFile(full.replace(/\.(splat|ply|ksplat)$/i, '.scenemode.json'), 'utf8')
            );
            if (sm && ['room', 'object'].includes(sm.mode)) {
              sceneModeOverride = {
                mode: sm.mode,
                operator: sm.operator ?? null,
                updatedAt: sm.updatedAt ?? null,
              };
            }
          } catch {
            /* 没有改判 */
          }
          models.push({
            name: meta?.name ?? e.name,
            url: `/models/${encodeURIComponent(e.name)}`,
            format: e.name.split('.').pop().toLowerCase(),
            bytes: stat.size,
            mtime: stat.mtimeMs,
            // 元数据里「高斯数量」的字段名在不同脚本里叫法不同：
            //   to_splat.py 写的是 gaussianCount，早期脚本写 count，训练摘要写 gaussians。
            // 只读 count 会让前端永远显示「未知」，所以这里按优先级都认一下。
            gaussianCount: meta?.gaussianCount ?? meta?.count ?? meta?.gaussians ?? null,
            bounds: meta?.bounds ?? null,
            shDegree: meta?.shDegree ?? 0,
            source: meta?.source ?? null,
            quality: meta?.quality ?? null,
            createdAt: meta?.createdAt ?? null,
            imageCount: meta?.imageCount ?? null,
            structureUrl: hasStructure
              ? `/models/${encodeURIComponent(e.name.replace(/\.(splat|ply|ksplat)$/i, '.structure.json'))}`
              : null,
            // 方向覆盖度（拍摄引导用）：哪些方向其实没被拍到
            coverage,
            // 场景判定（结构文件里的自动判定 + 可能存在的**人工改判**）
            sceneMode,
            sceneModeSource,
            sceneModeOverride,
          });
        }
        models.sort((a, b) => b.mtime - a.mtime);
        return json(res, 200, { ok: true, models });
      }
      const target = safeJoin(DATA_DIR, `models/${rel}`);
      if (!target) return fail(res, 400, '非法路径');
      if (await serveFile(req, res, target, { immutable: true })) return;
      return fail(res, 404, '模型不存在');
    }

    // 前端依赖：/vendor/*
    if (url.pathname.startsWith('/vendor/')) {
      const rel = decodeURIComponent(url.pathname.slice('/vendor/'.length));
      const target = safeJoin(VENDOR_DIR, rel);
      if (!target) return fail(res, 400, '非法路径');
      if (await serveFile(req, res, target, { immutable: true })) return;
      return fail(res, 404, 'vendor 资源不存在');
    }

    // 前端应用
    const rel = url.pathname === '/' ? 'index.html' : decodeURIComponent(url.pathname.slice(1));
    const target = safeJoin(WEB_DIR, rel);
    if (target && (await serveFile(req, res, target))) return;
    // SPA 回退
    if (await serveFile(req, res, path.join(WEB_DIR, 'index.html'))) return;
    return fail(res, 404, '未找到');
  } catch (err) {
    // 校验类错误（store.normalizeAnnotation 等）会带 statusCode=400，
    // 那是用户输入问题，不该报成 500 服务端故障。
    const code = Number(err?.statusCode) || 500;
    if (code >= 500) console.error('[http]', err);
    return fail(res, code, err.message ?? '服务端异常');
  }
});

server.on('error', (err) => {
  console.error('[server] 启动失败：', err.message);
  process.exit(1);
});

server.listen(PORT, HOST, () => {
  const shown = HOST === '0.0.0.0' ? 'localhost' : HOST;
  console.log('');
  console.log('  案发现场三维重建与协同研判系统 · MVP 后端已启动');
  console.log(`  本机访问 : http://${shown}:${PORT}/`);
  console.log(`  WebSocket: ws://${shown}:${PORT}/ws?project=<案件ID>&name=<昵称>`);
  console.log(`  数据目录 : ${DATA_DIR}`);
  console.log('');
});

process.on('SIGINT', () => {
  console.log('\n[server] 正在关闭…');
  wss.close();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 1500).unref();
});

process.on('uncaughtException', (err) => {
  console.error('[server] 未捕获异常：', err);
});
process.on('unhandledRejection', (err) => {
  console.error('[server] 未处理的 Promise 拒绝：', err);
});
