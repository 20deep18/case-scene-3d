/**
 * 后端 REST 客户端
 */
const BASE = '';

/**
 * 静态演示版（只读）：部署到 Cloudflare Pages 等静态托管时，
 * `window.__DSH_STATIC__` 由 index.html 注入为 true，这里据此降级：
 *   · 读：案件与模型从静态文件取（project.json / models/index.json）
 *   · 写：一律抛出可读错误 —— **不假装成功**（假装成功会让用户以为标注保存了）
 */
const STATIC = (() => {
  try {
    return window.__DSH_STATIC__ === true;
  } catch {
    return false;
  }
})();

const STATIC_MSG = '只读演示版（静态托管）：此处只支持查看模型与结构线条，标注/上传/建模不可用';

async function staticProject() {
  const url = (typeof window !== 'undefined' && window.__STATIC_PROJECT_URL__) || '/project.json';
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`静态案件数据缺失（${res.status}）`);
  return res.json();
}

async function request(path, { method = 'GET', body, headers = {}, raw = false } = {}) {
  if (STATIC) throw new Error(STATIC_MSG);
  const init = { method, headers: { ...headers } };
  if (body !== undefined && !(body instanceof ArrayBuffer) && !(body instanceof Blob)) {
    init.headers['content-type'] = 'application/json';
    init.body = JSON.stringify(body);
  } else if (body !== undefined) {
    init.body = body;
  }
  const res = await fetch(BASE + path, init);
  if (raw) return res;
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`服务端返回非 JSON（${res.status}）：${text.slice(0, 120)}`);
  }
  if (!res.ok || data.ok === false) {
    throw new Error(data.error ?? `请求失败 ${res.status}`);
  }
  return data;
}

export const api = {
  health: () =>
    STATIC
      ? Promise.resolve({ ok: true, name: '案发现场三维重建与协同研判系统', version: 'static',
                          projects: 1, clients: 1, static: true, time: new Date().toISOString() })
      : request('/api/health'),

  projects: async () => {
    if (!STATIC) return request('/api/projects');
    const d = await staticProject();
    const p = d.project ?? d;
    // ⚠️ 界面用的是 `p.annotationCount`（见 ui.js 拼"案件名 · N 标注"）；
    //    只给 annotations 数组而不给计数，仍会显示成 "undefined 标注"（实测踩到）。
    return { ok: true, projects: [{ id: p.id, name: p.name, status: p.status ?? 'active',
                                    createdAt: p.createdAt ?? null, static: true,
                                    annotationCount: Array.isArray(p.annotations) ? p.annotations.length : 0,
                                    annotations: Array.isArray(p.annotations) ? p.annotations : [] }] };
  },
  project: async (id) => {
    if (!STATIC) return request(`/api/projects/${encodeURIComponent(id)}`);
    return staticProject();
  },
  createProject: (payload) => request('/api/projects', { method: 'POST', body: payload }),
  deleteProject: (id) => request(`/api/projects/${encodeURIComponent(id)}`, { method: 'DELETE' }),

  annotations: (id) => request(`/api/projects/${encodeURIComponent(id)}/annotations`),

  // ---- 精确测量与比例尺校正（民警实测 → 提升模型精度）----
  measurements: (id) => request(`/api/projects/${encodeURIComponent(id)}/measurements`),
  calibrate: (id, payload = {}) =>
    request(`/api/projects/${encodeURIComponent(id)}/calibrate`, { method: 'POST', body: payload }),

  // ---- 现场建模（依据视频/图片建立现场模型）----
  reconJobs: (id) => request(`/api/projects/${encodeURIComponent(id)}/recon`),
  startRecon: (id, payload) =>
    request(`/api/projects/${encodeURIComponent(id)}/recon`, { method: 'POST', body: payload }),
  reconJob: (id, jobId) =>
    request(`/api/projects/${encodeURIComponent(id)}/recon/${encodeURIComponent(jobId)}`),
  reconLog: (id, jobId, from = 0) =>
    request(`/api/projects/${encodeURIComponent(id)}/recon/${encodeURIComponent(jobId)}/log?from=${from}&tail=400`),
  cancelRecon: (id, jobId) =>
    request(`/api/projects/${encodeURIComponent(id)}/recon/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' }),
  addAnnotation: (id, annotation) =>
    request(`/api/projects/${encodeURIComponent(id)}/annotations`, { method: 'POST', body: { annotation } }),
  updateAnnotation: (id, annId, patch) =>
    request(`/api/projects/${encodeURIComponent(id)}/annotations/${encodeURIComponent(annId)}`, {
      method: 'PATCH',
      body: { patch },
    }),
  deleteMedia: (id, mediaUrl) =>
    request(`/api/projects/${encodeURIComponent(id)}/media?url=${encodeURIComponent(mediaUrl)}`, { method: 'DELETE' }),
  deleteAnnotation: (id, annId) =>
    request(`/api/projects/${encodeURIComponent(id)}/annotations/${encodeURIComponent(annId)}`, { method: 'DELETE' }),

  setModel: (id, model) =>
    request(`/api/projects/${encodeURIComponent(id)}/model`, { method: 'POST', body: { model } }),
  setWorld: (id, world) =>
    request(`/api/projects/${encodeURIComponent(id)}/world`, { method: 'POST', body: { world } }),
  setPresentation: (id, presentation) =>
    request(`/api/projects/${encodeURIComponent(id)}/presentation`, { method: 'POST', body: { presentation } }),

  /**
   * 模型清单：直接 fetch 静态 index.json。
   *
   * ⚠️ 不能走 request()：静态演示版里 request() 会主动抛"只读"错误，
   *    而模型清单恰恰是静态版**唯一必须能读**的接口（第一版就这样把清单读成 0 个，
   *    下拉框是空的，但页面看起来正常 —— 属于"假可用"）。
   */
  async models() {
    try {
      const res = await fetch(`${BASE}/models/index.json`, { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        return data.models ?? [];
      }
    } catch {
      /* 回退到内置清单 */
    }
    const known = window.__MODEL_MANIFEST__;
    return known ?? [];
  },

  /**
   * 分片上传媒体文件（支持大视频，带进度回调）
   */
  async uploadMedia(projectId, file, onProgress) {
    const CHUNK = 8 * 1024 * 1024;
    const chunks = Math.max(1, Math.ceil(file.size / CHUNK));
    const token = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    let result = null;
    for (let i = 0; i < chunks; i++) {
      const slice = file.slice(i * CHUNK, Math.min((i + 1) * CHUNK, file.size));
      const buf = await slice.arrayBuffer();
      const url = `/api/projects/${encodeURIComponent(projectId)}/media?name=${encodeURIComponent(file.name)}&chunk=${i}&chunks=${chunks}&upload=${token}`;
      result = await request(url, { method: 'POST', body: buf, headers: { 'content-type': 'application/octet-stream' } });
      onProgress?.((i + 1) / chunks);
    }
    return result;
  },

  // 说明：PnP 相机标定**完全在浏览器里完成**（app/web/js/pnp.js），
  // 因此没有对应的后端接口。早期版本这里留了一个指向未实现路由的
  // solvePnp()，会让调用方收到 404，已移除。

  events: (projectId, limit = 200) => request(`/api/projects/${encodeURIComponent(projectId)}/events?limit=${limit}`),

  exportProject: (projectId) => request(`/api/projects/${encodeURIComponent(projectId)}/export`),
  importProject: (projectId, payload) =>
    request(`/api/projects/${encodeURIComponent(projectId)}/import`, { method: 'POST', body: payload }),
};
