/**
 * 现场建模面板控制器
 *
 * 负责「依据视频和图片建立现场模型」这条核心流程的前端部分：
 *   · 列出本案可用的素材（来自多源画面）
 *   · **素材体积体检**：体积不够时先提示「提供信息过少，可能无法正常建模」再由用户决定
 *   · 弹参数对话框并启动后端建模任务
 *   · 轮询任务状态与日志，实时显示进度
 *   · 建模完成后提示并刷新「模型文件」下拉，可直接载入三维现场
 *
 * 后端接口（见 app/server/index.js）：
 *   GET  /api/projects/:id/recon/materials 素材体积体检（字节数 / 是否达到实测最小体积）
 *   POST /api/projects/:id/recon          启动任务
 *   GET  /api/projects/:id/recon/:jobId   任务状态
 *   GET  /api/projects/:id/recon/:jobId/log?from=N   增量日志
 *   POST /api/projects/:id/recon/:jobId/cancel       取消
 */

import { closeDialogModal, ensureDialogClosable, openDialogModal } from './ui.js';

const POLL_MS = 1500;

/** 人类可读的体积文本（与阈值同一口径：1 MB = 1024×1024 字节） */
export function formatBytes(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n <= 0) return '0 MB';
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024).toFixed(0)} KB`;
}

export class ReconPanel {
  /**
   * @param {object} o
   * @param {() => (string|null)} o.currentProjectId
   * @param {() => string} o.operator
   * @param {() => Promise<void>} o.refreshModels   建模完成后刷新模型下拉
   * @param {(msg:string, opts?:object)=>void} o.toast
   * @param {(open:boolean, text?:string)=>void} [o.onBusy]  可选：全局忙碌遮罩
   * @param {(o:object)=>Promise<boolean>} [o.confirm] 可选：自定义确认框（替代 window.confirm）
   */
  constructor({ currentProjectId, getProject, operator, refreshModels, toast, onBusy, confirm } = {}) {
    this.currentProjectId = currentProjectId ?? (() => null);
    this.getProject = getProject ?? (() => null);
    this.operator = operator ?? (() => '未署名');
    this.refreshModels = refreshModels ?? (async () => {});
    this.toast = toast ?? (() => {});
    this.onBusy = onBusy ?? (() => {});
    this.confirm = confirm ?? ((o) => Promise.resolve(window.confirm(`${o?.title ?? ''}\n\n${o?.body ?? ''}`)));
    this.jobId = null;
    this.timer = null;
    this.seenLines = 0;
    this.lastStage = null;
    /** @type {Array<{name:string,url:string,kind:string}>} */
    this.assets = [];
    /** 最近一次素材体积体检结果（GET /recon/materials） */
    this.materials = null;
  }

  /** 绑定 DOM。缺元素直接报出来，避免运行到一半才发现。 */
  init() {
    const $ = (id) => {
      const el = document.getElementById(id);
      if (!el) console.error(`[recon] 缺少元素 #${id}`);
      return el;
    };
    this.el = {
      btn: $('btnRecon'),
      status: $('reconStatus'),
      stage: $('reconStage'),
      pct: $('reconPct'),
      bar: $('reconBar'),
      msg: $('reconMsg'),
      hint: $('reconHint'),
      logBtn: $('btnReconLog'),
      cancelBtn: $('btnReconCancel'),
      dlg: $('dlgRecon'),
      assets: $('reconAssets'),
      name: $('rcName'),
      fps: $('rcFps'),
      longSide: $('rcLongSide'),
      blur: $('rcBlur'),
      matcher: $('rcMatcher'),
      camera: $('rcCamera'),
      iterations: $('rcIterations'),
      downscale: $('rcDownscale'),
      shDegree: $('rcShDegree'),
      startBtn: $('btnReconStart'),
      dlgCancelBtn: $('btnReconDlgCancel'),
      logDlg: $('dlgReconLog'),
      logCloseBtn: $('btnReconLogClose'),
      logBody: $('rlBody'),
      logStage: $('rlStage'),
    };

    this.el.btn?.addEventListener('click', () => this.openDialog());
    this.el.startBtn?.addEventListener('click', (ev) => {
      ev.preventDefault();
      this.start();
    });
    this.el.logBtn?.addEventListener('click', () => this.openLog());
    this.el.cancelBtn?.addEventListener('click', () => this.cancel());
    this.el.dlg?.addEventListener('close', () => this.onBusy(false));

    // 「取消 / 关闭」按钮走**显式** JS 关闭，不再依赖 <form method="dialog">：
    // 用户反馈过「点取消关不掉窗口」，而 method="dialog" 在部分内核里不生效。
    ensureDialogClosable(this.el.dlg);
    ensureDialogClosable(this.el.logDlg);
    this.el.dlgCancelBtn?.addEventListener('click', (ev) => {
      ev.preventDefault();
      closeDialogModal(this.el.dlg, 'cancel');
    });
    this.el.logCloseBtn?.addEventListener('click', (ev) => {
      ev.preventDefault();
      closeDialogModal(this.el.logDlg, 'ok');
    });

    // 刷新页面后能接回正在跑的任务
    this.resume();
    return this;
  }

  /**
   * 收集本案可用的建模素材。
   *
   * 素材来源是「相机」标注（kind='camera'）上的 media 字段 ——
   * 也就是用户在「多源画面」里导入的监控/勘查视频与照片。
   * 直接从项目状态读，而不是从 DOM 抓，避免界面结构变化导致取不到。
   */
  _collectAssets() {
    const project = this.getProject?.();
    const out = [];
    const seen = new Set();
    for (const ann of project?.annotations ?? []) {
      for (const m of ann.media ?? []) {
        if (!m?.url || seen.has(m.url)) continue;
        seen.add(m.url);
        const kind = m.kind ?? (/\.(mp4|webm|mkv|mov|m4v|ts)$/i.test(m.url) ? 'video' : 'image');
        // media.name 可能是 URL 编码过的（中文文件名会变成 %E7%8E%B0…），
        // 展示与传给后端之前都要解码回可读文本。
        out.push({ url: m.url, name: this._decodeName(m.name ?? m.url.split('/').pop()), kind });
      }
    }
    // 视频优先（训练素材里视频信息量最大）
    out.sort((a, b) => (a.kind === b.kind ? 0 : a.kind === 'video' ? -1 : 1));
    return out;
  }

  /** 把可能是 URL 编码的文件名解码成可读文本 */
  _decodeName(name) {
    const raw = String(name ?? '');
    if (!/%[0-9A-Fa-f]{2}/.test(raw)) return raw;
    try {
      return decodeURIComponent(raw);
    } catch {
      return raw;
    }
  }

  openDialog() {
    const pid = this.currentProjectId();
    if (!pid) {
      this.toast('请先选择案件', { error: true });
      return;
    }
    this.assets = this._collectAssets();
    this.materials = null;
    this._renderAssets();

    if (this.el.name && !this.el.name.value) {
      const caseSel = document.getElementById('projectSelect');
      const caseName = caseSel?.selectedOptions?.[0]?.textContent?.replace(/\s*·.*$/, '') ?? '现场';
      this.el.name.value = `${caseName} 现场模型`;
    }
    const disabled = !this.assets.length;
    if (this.el.startBtn) this.el.startBtn.disabled = disabled;
    openDialogModal(this.el.dlg);

    // 体积体检是异步的：先让对话框立刻出来，体检结果回来再补上体积行
    if (this.assets.length) this._loadMaterials();
  }

  /**
   * 渲染素材清单（含体积体检结论）。
   *
   * 「体积」是用户最容易理解、也最容易在点开始建模之前就看到的信号，
   * 所以把它直接写进对话框，而不是等失败之后再解释。
   */
  _renderAssets() {
    const el = this.el.assets;
    if (!el) return;
    const videos = this.assets.filter((a) => a.kind === 'video');
    const images = this.assets.filter((a) => a.kind === 'image');

    if (!this.assets.length) {
      el.innerHTML =
        '<div class="recon-empty">本案还没有素材。请先点右下角的 <b>「+ 导入监控/勘查视频」</b> 上传现场视频或照片，再回来建模。</div>';
      return;
    }

    const m = this.materials;
    let volume = '';
    if (m) {
      const enough = m.enough !== false;
      volume =
        `<div class="recon-volume ${enough ? 'ok' : 'warn'}">` +
        `素材合计 <b>${this._fmt(m.bytes)}</b>` +
        (Number(m.minBytes) > 0 ? ` · 实测建模下限 <b>${this._fmt(m.minBytes)}</b>` : '') +
        (enough
          ? ' <span class="v-ok">✓ 体积充足，可直接建模</span>'
          : ' <span class="v-warn">⚠ 体积偏少，可能无法正常建模</span>') +
        `</div>`;
      if (m.missing?.length) {
        volume += `<div class="recon-volume warn">⚠ 有 ${m.missing.length} 个素材文件在服务器上找不到，将被跳过：${m.missing
          .slice(0, 3)
          .map((n) => this._esc(n))
          .join('、')}</div>`;
      }
    }

    el.innerHTML =
      `<div class="recon-assets-head">将使用 <b>${videos.length}</b> 个视频、<b>${images.length}</b> 张图片</div>` +
      volume +
      '<ul class="recon-assets-list">' +
      this.assets
        .slice(0, 12)
        .map((a) => `<li><span class="tag">${a.kind === 'video' ? '视频' : '图片'}</span>${this._esc(a.name)}</li>`)
        .join('') +
      (this.assets.length > 12 ? `<li class="more">…还有 ${this.assets.length - 12} 个</li>` : '') +
      '</ul>';
  }

  _fmt(bytes) {
    return formatBytes(bytes);
  }

  /**
   * 查素材体积体检：总体积、是否达到「实测建模最小体积」。
   * 网络异常时返回 null —— 体检失败**不应该**阻断建模。
   */
  async _loadMaterials() {
    const pid = this.currentProjectId();
    if (!pid) return null;
    try {
      const res = await fetch(`/api/projects/${encodeURIComponent(pid)}/recon/materials`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ assets: this.assets }),
      });
      if (!res.ok) return null;
      const data = await res.json();
      this.materials = data && data.ok !== false ? data : null;
    } catch {
      this.materials = null;
    }
    this._renderAssets();
    return this.materials;
  }

  _esc(s) {
    return String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  }

  _params() {
    const num = (el, dflt) => {
      const v = Number(el?.value);
      return Number.isFinite(v) ? v : dflt;
    };
    return {
      fps: num(this.el.fps, 2),
      longSide: num(this.el.longSide, 1600),
      blurThreshold: num(this.el.blur, 60),
      matcher: this.el.matcher?.value ?? 'exhaustive',
      cameraModel: this.el.camera?.value ?? 'OPENCV',
      iterations: num(this.el.iterations, 30000),
      downscale: num(this.el.downscale, 2),
      shDegree: num(this.el.shDegree, 3),
      lrScaleFactor: 0.4,
      // 尺度剪枝系数：**必须与训练器默认 0.10 一致**。
      // 这里曾写成 0.05，A/B 实测（同一素材、同一份位姿，只改这一个参数）：
      //   12MB 档（23 帧）：0.05 → PSNR 19.95 dB，0.10 → 24.73 dB（+4.78 dB）
      //   24MB 档（43 帧）：0.05 → PSNR 23.95 dB，0.10 → 26.10 dB（+2.14 dB）
      // 原因是 prune-scale 同时用作「尺度上限 clamp」：
      // 0.05 把每个高斯都压到 ≤0.5 m，铺满同样面积需要更多更小的高斯，反而留孔洞。
      pruneScale: 0.1,
      evalViews: 4,
    };
  }

  async start() {
    const pid = this.currentProjectId();
    if (!pid) return this.toast('请先选择案件', { error: true });
    const name = (this.el.name?.value ?? '').trim() || `现场模型 ${new Date().toLocaleString('zh-CN')}`;
    const params = this._params();

    // ---- 素材体积门槛（本项目实测得出，见 pipeline/recon_threshold.json）----
    //   体积 ≥ 阈值 → 直接建模；
    //   体积 <  阈值 → 明确提示「提供信息过少，可能无法正常建模」，由用户决定是否继续。
    // 体检接口拿不到时（网络抖动）不拦截，避免把「检查失败」变成「不能建模」。
    const info = await this._loadMaterials();
    if (info && info.enough === false) {
      const minText = this._fmt(info.minBytes);
      const ok = await this.confirm({
        title: '提供信息过少，可能无法正常建模',
        body:
          `当前素材合计 ${this._fmt(info.bytes)}（${info.videos} 个视频、${info.images} 张图片），` +
          `实测建模所需的最小素材体积约为 ${minText}。`,
        warn:
          '素材体积过少通常意味着可用视角不足：COLMAP 可能注册失败、点云过稀，' +
          '训练出来的现场会缺面、发糊甚至完全不可用。建议先补充视频或照片。仍要现在开始建模吗？',
        okText: '仍然开始建模',
      });
      if (!ok) {
        // 保持对话框打开，方便用户直接去补素材
        this.toast('已取消。可先补充视频 / 照片后再建模。');
        return;
      }
    }

    this.el.dlg && closeDialogModal(this.el.dlg, 'ok');
    this.onBusy(true, '正在启动建模任务…');
    try {
      const res = await fetch(`/api/projects/${encodeURIComponent(pid)}/recon`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name, params, assets: this.assets, operator: this.operator() }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.ok === false) throw new Error(data.error ?? `启动失败 ${res.status}`);
      this.jobId = data.job?.id ?? null;
      this.seenLines = 0;
      this.lastStage = null;
      this.render(data.job);
      this.toast('建模已开始，可在左侧查看进度');
      this.poll();
    } catch (err) {
      this.toast(err.message, { error: true });
    } finally {
      this.onBusy(false);
    }
  }

  /** 刷新页面后接回正在运行的任务 */
  async resume() {
    const pid = this.currentProjectId();
    if (!pid) return;
    try {
      const res = await fetch(`/api/projects/${encodeURIComponent(pid)}/recon`);
      if (!res.ok) return;
      const data = await res.json();
      const active = (data.jobs ?? []).find((j) => !j.finishedAt);
      const latest = active ?? (data.jobs ?? [])[0];
      if (latest) {
        this.jobId = latest.id;
        this.seenLines = 0;
        this.render(latest);
        if (active) this.poll();
      } else {
        this.renderIdle();
      }
    } catch {
      /* 忽略：接不回就算了 */
    }
  }

  poll() {
    this._stop();
    const tick = async () => {
      const pid = this.currentProjectId();
      if (!pid || !this.jobId) return;
      try {
        const res = await fetch(
          `/api/projects/${encodeURIComponent(pid)}/recon/${encodeURIComponent(this.jobId)}/log?from=${this.seenLines}&tail=400`
        );
        if (!res.ok) return;
        const data = await res.json();
        if (Array.isArray(data.lines) && data.lines.length) {
          this.seenLines = (data.from ?? this.seenLines) + data.lines.length;
          this.appendLog(data.lines);
        }
        const st = data.status ?? null;
        if (st) this.render(st);
        if (st && (st.stage === 'done' || st.stage === 'failed')) {
          this._stop();
          if (st.stage === 'done') {
            await this.refreshModels();
            const model = st.result?.model ?? st.name;
            this.toast(`建模完成：${model}。可在「模型文件」中选择载入`);
            this.el.hint.innerHTML = `已完成，可在上方「模型文件」里选择 <b>${this._esc(model)}</b> 并点「载入三维现场」`;
          } else {
            this.toast(`建模失败：${st.error ?? '详见日志'}`, { error: true });
          }
          return;
        }
      } catch {
        /* 网络抖动忽略，下轮再试 */
      }
      this.timer = setTimeout(tick, POLL_MS);
    };
    tick();
  }

  _stop() {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  render(job) {
    if (!job || !this.el.status) return;
    this.el.status.classList.remove('hidden');
    const pct = Math.round((job.progress ?? 0) * 100);
    this.el.stage.textContent = job.stageLabel ?? job.stage ?? '—';
    this.el.pct.textContent = `${pct}%`;
    this.el.bar.style.width = `${pct}%`;
    this.el.bar.classList.toggle('is-error', job.stage === 'failed');
    this.el.bar.classList.toggle('is-done', job.stage === 'done');
    const bits = [];
    if (job.message) bits.push(job.message);
    if (job.elapsedSec) bits.push(`已用时 ${this._dur(job.elapsedSec)}`);
    this.el.msg.textContent = bits.join(' · ');
    if (this.el.logStage) this.el.logStage.textContent = job.stageLabel ?? '';
    const running = !job.finishedAt;
    if (this.el.cancelBtn) this.el.cancelBtn.disabled = !running;
    if (this.el.btn) this.el.btn.disabled = running;
    if (job.stage === 'failed') this.el.hint.textContent = job.error ?? '建模失败，请查看日志';
  }

  renderIdle() {
    if (!this.el.status) return;
    this.el.status.classList.add('hidden');
    if (this.el.hint) {
      this.el.hint.innerHTML =
        '建模会消耗较长时间（抽帧 + 重建 + 训练，通常 5~30 分钟），期间可继续查看已有模型。';
    }
    if (this.el.btn) this.el.btn.disabled = false;
  }

  _dur(sec) {
    if (sec < 60) return `${sec} 秒`;
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m} 分 ${s} 秒`;
  }

  appendLog(lines) {
    if (!this.el.logBody) return;
    const atBottom =
      this.el.logBody.scrollTop + this.el.logBody.clientHeight >= this.el.logBody.scrollHeight - 30;
    this.el.logBody.textContent += `${lines.join('\n')}\n`;
    // 只保留最近 4000 行，避免长任务把页面拖慢
    const all = this.el.logBody.textContent.split('\n');
    if (all.length > 4000) this.el.logBody.textContent = all.slice(-4000).join('\n');
    if (atBottom) this.el.logBody.scrollTop = this.el.logBody.scrollHeight;
  }

  openLog() {
    if (!this.jobId) {
      this.toast('当前没有建模任务');
      return;
    }
    openDialogModal(this.el.logDlg);
    // 打开时补全已有日志
    if (!this.el.logBody.textContent) this.poll();
  }

  async cancel() {
    const pid = this.currentProjectId();
    if (!pid) return this.toast('请先选择案件', { error: true });
    if (!this.jobId) {
      // 这里以前是**静默 return**：用户点「取消」看不到任何反应，只会以为按钮坏了。
      // 取消是幂等且可恢复的操作，任何一条分支都必须给出可见反馈。
      this.renderIdle();
      return this.toast('当前没有正在运行的建模任务，进度显示已复位');
    }

    // 用自定义确认框，不用 window.confirm：
    // 原生弹窗在部分环境/嵌入式浏览器里会被屏蔽，一旦被屏蔽点了「取消」就**毫无反应**，
    // 用户看到的现象正是「点了取消没反应 / 任务一直在跑」。
    let ok = true;
    try {
      ok = await this.confirm({
        title: '取消当前建模任务',
        body: '确定要取消当前建模任务吗？已完成的步骤不会保留。',
        warn: '取消后需要重新开始建模（会重新抽帧 + 重建 + 训练）。',
        okText: '确定取消任务',
      });
    } catch (err) {
      // 连确认框都弹不出来时也不能静默失败：取消可恢复，直接执行并说明原因
      console.warn('[recon] 确认框不可用，直接请求取消', err);
      this.toast('确认框不可用，已直接请求取消');
      ok = true;
    }
    if (ok === false) return;

    // 请求期间给出「正在取消」的可见状态，避免用户以为没点上
    const btn = this.el.cancelBtn;
    const prevText = btn?.textContent;
    if (btn) {
      btn.disabled = true;
      btn.textContent = '取消中…';
    }
    if (this.el.msg) this.el.msg.textContent = '正在请求取消…';
    try {
      const res = await fetch(
        `/api/projects/${encodeURIComponent(pid)}/recon/${encodeURIComponent(this.jobId)}/cancel`,
        { method: 'POST' }
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.ok === false) throw new Error(data.error ?? '取消失败');
      this.toast('已请求取消');
      if (this.el.msg) this.el.msg.textContent = '已请求取消，正在结束进程…';
    } catch (err) {
      this.toast(err.message, { error: true });
      if (this.el.msg) this.el.msg.textContent = `取消失败：${err.message}`;
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = prevText || '取消';
      }
    }
  }

  /** 切案件时重置面板 */
  onProjectChanged() {
    this._stop();
    this.jobId = null;
    this.seenLines = 0;
    if (this.el.logBody) this.el.logBody.textContent = '';
    this.renderIdle();
    this.resume();
  }
}
