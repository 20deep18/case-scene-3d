/**
 * 交互式 PnP 相机标定面板
 *
 * 用法（在 app.js 里）：
 *   const panel = new PnpPanel({ onSolved: (ann, pose, result) => {...} });
 *   panel.open({ annotation, mediaUrl, scene, camera, canvas, picker, worldMatrix, overlay });
 *
 * 工作流：
 *   1. 从监控/勘查视频里抽一帧作为标定画面
 *   2. 在画面上点一个特征像素 → 在三维现场里点同一个特征点 → 形成一组对应
 *   3. 采够 ≥4 组（建议 6~8 组，尽量散开）→ 解算 → 显示重投影误差 → 写回相机位姿
 *
 *   PnP 解算在浏览器里完成（app/web/js/pnp.js），不需要后端参与。
 */

import * as THREE from 'three';
import { solvePnP, refinePose, poseFromPnp, defaultIntrinsics, reprojectionError } from './pnp.js';

const MIN_PAIRS = 4;

/** 推荐的最大点数（超过后边际收益很小，且误点影响变大） */
export const RECOMMENDED_PAIRS = 8;

/**
 * 判断当前对应点是否足以解算，并给出面向使用者的提示。
 * 抽成纯函数便于单元测试。
 */
export function assessPairs(pairs, minPairs = MIN_PAIRS) {
  const n = pairs?.length ?? 0;
  if (n < minPairs) {
    return { ok: false, level: 'insufficient', message: `还需 ${minPairs - n} 组对应点（至少 ${minPairs} 组）` };
  }
  // 像素点是否过于集中：计算归一化包围盒面积
  const xs = pairs.map((p) => p.image[0]);
  const ys = pairs.map((p) => p.image[1]);
  const w = (Math.max(...xs) - Math.min(...xs)) / Math.max(1, pairs.imageWidth ?? 1);
  const h = (Math.max(...ys) - Math.min(...ys)) / Math.max(1, pairs.imageHeight ?? 1);
  const spread = Math.max(w, h);
  if (spread < 0.25) {
    return {
      ok: true,
      level: 'warning',
      message: '可以解算，但所选像素点过于集中，建议把点分散到画面各处以提高精度',
    };
  }
  if (n > RECOMMENDED_PAIRS + 4) {
    return { ok: true, level: 'ok', message: `${n} 组对应点（偏多，注意剔除误点）` };
  }
  return { ok: true, level: 'ok', message: `${n} 组对应点，可以解算` };
}

/**
 * 把一组对应点交给解算器，返回可直接写回相机的位姿。
 * 三维点必须是**标注坐标系（COLMAP 系）**的坐标。
 */
export function solveFromPairs(pairs, intrinsics, imageSize, opts = {}) {
  if ((pairs?.length ?? 0) < MIN_PAIRS) {
    throw new Error(`至少需要 ${MIN_PAIRS} 组对应点，当前 ${pairs.length} 组`);
  }
  const objectPoints = pairs.map((p) => [...p.object]);
  const imagePoints = pairs.map((p) => [...p.image]);
  const result = solvePnP(objectPoints, imagePoints, intrinsics, {
    imageSize,
    maxIter: opts.maxIter ?? 120,
  });
  // 返回时带上每点的残差，便于在面板里标出误点
  const err = reprojectionError(
    { R: result.rotation, t: result.translation, K: result.intrinsics },
    objectPoints,
    imagePoints
  );
  const pose = poseFromPnp(result, imageSize);
  return { result, pose, perPoint: err.perPoint, meanError: err.mean, maxError: err.max };
}

/** 把一个局部（COLMAP 系）点投影到像素，用于把已有点画到画面上做核对 */
export function projectLocalToPixel(pointLocal, pose, imageSize) {
  const { intrinsics } = pose;
  if (!intrinsics) return null;
  const R = pose.rotation ? ungroupRotation(pose.rotation) : null;
  if (!R) return null;
  const t = rotationCenterToTranslation(R, pose.position);
  const [X, Y, Z] = pointLocal;
  const xc = R[0] * X + R[1] * Y + R[2] * Z + t[0];
  const yc = R[3] * X + R[4] * Y + R[5] * Z + t[1];
  const zc = R[6] * X + R[7] * Y + R[8] * Z + t[2];
  if (zc <= 1e-9) return null;
  return [intrinsics.fx * (xc / zc) + intrinsics.cx, intrinsics.fy * (yc / zc) + intrinsics.cy];
}

/** pose.rotation 是「相机组」的旋转（局部 +z 为观察反方向），换算回 CV 约定 */
function ungroupRotation(Rg) {
  return [Rg[0], -Rg[1], -Rg[2], Rg[3], -Rg[4], -Rg[5], Rg[6], -Rg[7], -Rg[8]];
}

function rotationCenterToTranslation(R, center) {
  return [
    -(R[0] * center[0] + R[1] * center[1] + R[2] * center[2]),
    -(R[3] * center[0] + R[4] * center[1] + R[5] * center[2]),
    -(R[6] * center[0] + R[7] * center[1] + R[8] * center[2]),
  ];
}

/** 用射线与三维点云求交，返回局部（COLMAP 系）坐标 */
export function pickLocalPoint(ndc, ctx) {
  const { picker, camera, canvas, worldMatrix } = ctx;
  if (!picker || !camera) return null;
  const hit = picker.pick(ndc, camera, worldMatrix, { screenRadius: 0.05 });
  if (!hit) return null;
  const local = hit.position.clone().applyMatrix4(new THREE.Matrix4().copy(worldMatrix).invert());
  return { local, world: hit.position.clone(), normal: hit.normal, index: hit.index };
}

export class PnpPanel {
  constructor({ onSolved, onToast, onFlyTo, onArmedChange, onClose } = {}) {
    this.onSolved = onSolved;
    this.onToast = onToast;
    this.onFlyTo = onFlyTo;
    this.onClose = onClose;
    // 这个回调**必须有默认实现**：它负责告诉外部「现在该去三维视口点对应点了」。
    // 一旦漏传，用户就只会在视频画面上点出一个点、却不知道该去三维里点第二个，
    // 标定直接卡死（这正是实际反馈的问题）。默认实现直接给视口加提示描边。
    this.onArmedChange =
      onArmedChange ??
      ((armed) => {
        const vp = document.getElementById('viewport');
        if (vp) vp.classList.toggle('pnp-armed', !!armed);
      });

    this.pairs = [];
    this.ctx = null;
    this.annotation = null;
    this.video = null;
    this.imageSize = { width: 1920, height: 1080 };
    this.intrinsics = null;
    this.active = false;
    this.lastSolve = null;
    this._pendingImage = null; // 已点像素、等待在三维里点对应点

    // 先建 DOM（只 createElement + innerHTML，不碰 this.dom），
    // 再收集元素引用，最后才接线 —— 顺序颠倒会导致 _wire 里读不到 this.dom。
    this.el = this._buildDom();
    this._cacheDom();
    this._wire();
    document.body.appendChild(this.el);
  }

  // ------------------------------------------------------------ DOM

  _buildDom() {
    const root = document.createElement('div');
    root.className = 'pnp-panel';
    root.hidden = true;
    root.innerHTML = `
      <div class="pnp-backdrop"></div>
      <div class="pnp-dialog" role="dialog" aria-label="PnP 相机标定">
        <header class="pnp-head">
          <strong>相机标定（PnP）</strong>
          <span class="pnp-sub" data-sub></span>
          <button class="ghost small" data-close>关闭</button>
        </header>

        <!-- 两个步骤做成子选项，同一个框架内切换：
             步骤一在面板里点画面像素，步骤二去三维视口点对应位置。 -->
        <div class="pnp-tabs" role="tablist">
          <button class="pnp-tab is-active" data-tab="1" role="tab">
            <span class="num">1</span> 点画面特征点
            <span class="badge" data-badge1>未选</span>
          </button>
          <button class="pnp-tab" data-tab="2" role="tab">
            <span class="num">2</span> 点三维对应点
            <span class="badge" data-badge2>等待</span>
          </button>
        </div>

        <div class="pnp-body">
          <!-- ============ 步骤 1：画面 ============ -->
          <section class="pnp-step" data-step="1">
            <div class="pnp-pane-title">监控 / 勘查画面<span class="pnp-hint" data-frameinfo></span></div>
            <div class="pnp-video-wrap">
              <canvas data-framecanvas></canvas>
              <div class="pnp-marks" data-marks></div>
            </div>
            <div class="pnp-framerow">
              <button class="ghost small" data-prev-frame title="上一帧">◀</button>
              <input type="range" data-seek min="0" max="1000" value="0" />
              <button class="ghost small" data-next-frame title="下一帧">▶</button>
              <span class="pnp-time" data-time>0.00s</span>
            </div>
            <p class="pnp-tip" data-steptip>
              在画面上点一个容易辨认的特征点（墙角、门框、物证边角），
              然后切到「② 点三维对应点」。
            </p>
          </section>

          <!-- ============ 步骤 2：三维 ============ -->
          <section class="pnp-step" data-step="2" hidden>
            <div class="pnp-pane-title">三维现场<span class="pnp-hint">在右侧视口里点同一位置</span></div>
            <div class="pnp-3d-note" data-3dnote>
              请先在「① 点画面特征点」里选一个像素点。
            </div>
            <div class="pnp-picked" data-pickedbox hidden>
              <span>已选像素</span>
              <strong data-pickedpx>—</strong>
            </div>
            <button class="ghost small" data-flyto>把视角转到该机位附近</button>
            <dl class="pnp-stats">
              <dt>对应点</dt><dd data-count>0</dd>
              <dt>重投影误差</dt><dd data-error>—</dd>
            </dl>
          </section>

          <!-- ============ 公共：内参 ============ -->
          <section class="pnp-step pnp-right" data-step="both">
            <div class="pnp-pane-title">相机内参<span class="pnp-hint">未知时用估算值</span></div>
            <label><span>水平视场角</span><input type="number" data-fov step="1" min="5" max="170" /></label>
            <div class="pnp-fovrow">
              <input type="range" data-fovrange min="10" max="150" step="1" />
            </div>
            <label><span>fx</span><input type="number" data-fx step="1" /></label>
            <label><span>fy</span><input type="number" data-fy step="1" /></label>
            <label><span>cx</span><input type="number" data-cx step="1" /></label>
            <label><span>cy</span><input type="number" data-cy step="1" /></label>
            <button class="ghost small" data-reset-k>按视场角重算内参</button>
            <p class="pnp-tip">
              监控摄像头若知道焦距/传感器尺寸，可在此精确填写。留空则按视场角估算。
            </p>
          </section>
        </div>

        <div class="pnp-pairlist">
          <div class="pnp-pane-title">对应点列表<span class="pnp-hint" data-assess></span></div>
          <ol data-pairs></ol>
        </div>

        <footer class="pnp-foot">
          <button class="ghost small" data-undo>撤销上一个点</button>
          <button class="ghost small" data-clear>全部清空</button>
          <label class="pnp-blur">
            <input type="checkbox" data-blurtoggle checked /> 模糊背景
          </label>
          <span class="pnp-spacer"></span>
          <button class="primary" data-solve>解算并贴回现场</button>
        </footer>
      </div>
    `;

    return root;
  }

  /** 收集面板内的元素引用到 this.dom */
  _cacheDom() {
    const root = this.el;
    this.dom = {};
    // 属性名 → key 的映射。注意：data-3dnote 读出来是 el.dataset['3dnote']，
    // 不是 note3d（数字开头不是合法的 camelCase 属性），所以这里改用显式映射表，
    // 避免再出现「元素没被缓存 → 运行到一半 Cannot set properties of undefined」。
    const MAP = {
      framecanvas: 'frameCanvas',
      marks: 'marks',
      seek: 'seek',
      time: 'time',
      'prev-frame': 'prevFrame',
      'next-frame': 'nextFrame',
      count: 'count',
      error: 'error',
      assess: 'assess',
      pairs: 'pairs',
      fov: 'fov',
      fovrange: 'fovRange',
      fx: 'fx',
      fy: 'fy',
      cx: 'cx',
      cy: 'cy',
      solve: 'solve',
      undo: 'undo',
      clear: 'clear',
      close: 'close',
      flyto: 'flyTo',
      '3dnote': 'note3d',
      sub: 'sub',
      frameinfo: 'frameInfo',
      'reset-k': 'resetK',
      steptip: 'steptip',
      badge1: 'badge1',
      badge2: 'badge2',
      pickedbox: 'pickedBox',
      pickedpx: 'pickedPx',
      blurtoggle: 'blurToggle',
    };
    // 选择器可以直接由 MAP 生成，避免以后加了新元素忘记同步列表。
    const selector = Object.keys(MAP)
      .map((attr) => `[data-${attr}]`)
      .join(',');
    for (const el of root.querySelectorAll(selector)) {
      for (const [attr, key] of Object.entries(MAP)) {
        if (el.hasAttribute(`data-${attr}`) && this.dom[key] === undefined) {
          this.dom[key] = el;
        }
      }
    }
    // 缺元素要在控制台明确报出来，而不是等到运行时才 undefined
    for (const key of Object.values(MAP)) {
      if (!this.dom[key]) console.error(`[pnp-panel] 面板元素缺失：${key}`);
    }
  }

  _wire() {
    const d = this.dom;
    d.close.addEventListener('click', () => this.close());
    d.seek.addEventListener('input', () => this.seekTo(Number(d.seek.value) / 1000));
    d.prevFrame.addEventListener('click', () => this.stepFrame(-1));
    d.nextFrame.addEventListener('click', () => this.stepFrame(1));
    d.frameCanvas.addEventListener('click', (ev) => this._onImageClick(ev));
    d.undo.addEventListener('click', () => this.undo());
    d.clear.addEventListener('click', () => this.clearPairs());
    d.solve.addEventListener('click', () => this.solve());
    d.flyTo.addEventListener('click', () => this.onFlyTo?.(this.annotation, this.pairs));
    d.resetK.addEventListener('click', () => {
      const fov = Number(d.fov.value) || 60;
      this.intrinsics = defaultIntrinsics(this.imageSize.width, this.imageSize.height, fov);
      this._writeIntrinsics();
    });
    d.fovRange.addEventListener('input', () => {
      d.fov.value = d.fovRange.value;
      d.resetK.click();
    });
    for (const key of ['fx', 'fy', 'cx', 'cy']) {
      d[key].addEventListener('change', () => {
        this.intrinsics = {
          ...this.intrinsics,
          [key]: Number(d[key].value),
        };
      });
    }
    // 步骤切换（两个子选项）
    for (const tab of this.el.querySelectorAll('[data-tab]')) {
      tab.addEventListener('click', () => this.setStep(Number(tab.dataset.tab)));
    }
    // 背景模糊开关：三维视口需要看清才能点对应点，默认开启但可关掉
    d.blurToggle.addEventListener('change', () => {
      this.el.querySelector('.pnp-backdrop').classList.toggle('no-blur', !d.blurToggle.checked);
    });
    window.addEventListener('keydown', (ev) => {
      if (!this.active) return;
      if (ev.key === 'Escape') {
        ev.stopPropagation();
        this.close();
      }
    }, true);
  }

  /**
   * 切换步骤。步骤 1 = 在面板画面上点像素；步骤 2 = 去三维视口点对应点。
   * 两个步骤放在同一个对话框的子选项里，不用来回关开面板。
   */
  setStep(step) {
    const s = step === 2 ? 2 : 1;
    this.step = s;
    for (const tab of this.el.querySelectorAll('[data-tab]')) {
      tab.classList.toggle('is-active', Number(tab.dataset.tab) === s);
    }
    for (const section of this.el.querySelectorAll('[data-step]')) {
      const own = section.dataset.step;
      if (own === 'both') continue;
      section.hidden = Number(own) !== s;
    }
    this.dom.pickedBox.hidden = !this._pendingImage;
    this._updateStepTip();
  }

  /** 把 intrinsics 数值回写到输入框 */
  _writeIntrinsics() {
    const d = this.dom;
    if (!this.intrinsics) return;
    d.fx.value = this.intrinsics.fx.toFixed(1);
    d.fy.value = this.intrinsics.fy.toFixed(1);
    d.cx.value = this.intrinsics.cx.toFixed(1);
    d.cy.value = this.intrinsics.cy.toFixed(1);
  }

  // ------------------------------------------------------------ 打开 / 关闭

  /**
   * @param {object} opts
   * @param {object} opts.annotation 相机标注
   * @param {string} opts.mediaUrl   视频或图片 URL
   * @param {object} opts.ctx        { scene, camera, canvas, picker, worldMatrix, overlay }
   */
  async open({ annotation, mediaUrl, ctx, initialPose }) {
    this.annotation = annotation;
    this.ctx = ctx;
    this.pairs = [];
    this.lastSolve = null;
    this._pendingImage = null;
    this.active = true;
    this.el.hidden = false;

    this.dom.sub.textContent = annotation?.label ? `对象：${annotation.label}` : '';

    const pose = annotation?.pose ?? initialPose;
    const fov = pose?.fovY ?? 60;
    this.dom.fov.value = String(Math.round(fov));
    this.dom.fovRange.value = String(Math.round(fov));
    this.imageSize = { width: 1920, height: 1080 };
    this.intrinsics = defaultIntrinsics(this.imageSize.width, this.imageSize.height, fov);

    await this._loadMedia(mediaUrl);
    this._renderFrame();
    this._renderPairs();
    this.setStep(1);
    return this;
  }

  close() {
    this.active = false;
    this.el.hidden = true;
    this._pendingImage = null;
    this.onArmedChange?.(false);
    if (this.video) {
      this.video.pause();
      this.video.removeAttribute('src');
      this.video.load();
      this.video = null;
    }
    this.onClose?.();
  }

  // ------------------------------------------------------------ 画面

  async _loadMedia(url) {
    this._loadError = null;
    if (!url) {
      this._loadError = '该相机尚未关联视频/图片，请先导入画面';
      this.dom.note3d.textContent = this._loadError;
      return;
    }
    const isImage = /\.(png|jpe?g|webp|bmp|gif)$/i.test(url);
    if (isImage) {
      const img = new Image();
      await new Promise((resolve) => {
        img.onload = resolve;
        img.onerror = () => {
          this._loadError = '图片加载失败：' + url;
          resolve();
        };
        img.src = url;
      });
      this._imageEl = img;
      if (img.naturalWidth) this.imageSize = { width: img.naturalWidth, height: img.naturalHeight };
      this._imageIsStill = true;
      this.dom.seek.disabled = true;
      this.dom.prevFrame.disabled = true;
      this.dom.nextFrame.disabled = true;
      this.dom.frameInfo.textContent = `静态图片 ${this.imageSize.width}×${this.imageSize.height}`;
    } else {
      const video = document.createElement('video');
      video.src = url;
      video.muted = true;
      video.playsInline = true;
      video.preload = 'auto';
      video.crossOrigin = 'anonymous';
      await new Promise((resolve) => {
        video.onloadeddata = resolve;
        video.onerror = () => {
          this._loadError = '视频加载失败：' + url;
          resolve();
        };
        setTimeout(resolve, 8000);
      });
      this.video = video;
      this._imageIsStill = false;
      if (video.videoWidth) this.imageSize = { width: video.videoWidth, height: video.videoHeight };
      this.dom.seek.disabled = false;
      this.dom.prevFrame.disabled = false;
      this.dom.nextFrame.disabled = false;
      this.dom.frameInfo.textContent = `${this.imageSize.width}×${this.imageSize.height} · ${video.duration ? video.duration.toFixed(1) + 's' : '时长未知'}`;
    }
    // 内参随真实分辨率重算
    this.intrinsics = defaultIntrinsics(this.imageSize.width, this.imageSize.height, Number(this.dom.fov.value) || 60);
    this._writeIntrinsics();
    // 视频：等首帧真正可画（readyState>=2）并 seek 到 0，避免抽出一张黑图
    if (this.video && !this._imageIsStill) {
      if (this.video.readyState < 2) {
        await new Promise((resolve) => {
          this.video.addEventListener('loadeddata', resolve, { once: true });
          setTimeout(resolve, 4000);
        });
      }
      try {
        if (this.video.currentTime > 0.001) {
          await new Promise((resolve) => {
            this.video.addEventListener('seeked', resolve, { once: true });
            this.video.currentTime = 0;
            setTimeout(resolve, 1500);
          });
        }
      } catch {
        /* 某些容器不支持 seek，忽略 */
      }
      this.dom.seek.value = '0';
      this.dom.time.textContent = '0.00s';
    }
  }

  get duration() {
    return Number.isFinite(this.video?.duration) ? this.video.duration : 0;
  }

  get currentTime() {
    return this.video?.currentTime ?? 0;
  }

  async seekTo(frac) {
    if (!this.video) return;
    const dur = this.duration;
    if (!dur) return;
    const t = Math.max(0, Math.min(dur - 0.001, frac * dur));
    await new Promise((resolve) => {
      const done = () => resolve();
      this.video.addEventListener('seeked', done, { once: true });
      this.video.currentTime = t;
      setTimeout(done, 1500);
    });
    this.dom.seek.value = String(Math.round((t / dur) * 1000));
    this.dom.time.textContent = `${t.toFixed(2)}s`;
    this._renderFrame();
    this._renderPairs();
  }

  stepFrame(dir) {
    if (!this.video) return;
    const fps = 25; // 未知帧率时按 25fps 步进
    const dur = this.duration || 0;
    const t = Math.max(0, Math.min((dur || 1e9) - 0.001, this.video.currentTime + dir / fps));
    const frac = dur ? t / dur : 0;
    this.seekTo(frac);
  }

  _renderFrame() {
    const c = this.dom.frameCanvas;
    const { width, height } = this.imageSize;
    if (!width || !height) return;
    c.width = width;
    c.height = height;
    const g = c.getContext('2d');
    g.clearRect(0, 0, width, height);
    try {
      if (this._imageIsStill && this._imageEl?.naturalWidth) {
        g.drawImage(this._imageEl, 0, 0, width, height);
      } else if (this.video && this.video.readyState >= 2) {
        g.drawImage(this.video, 0, 0, width, height);
      } else {
        g.fillStyle = '#1b2129';
        g.fillRect(0, 0, width, height);
        g.fillStyle = '#8b98a5';
        g.font = `${Math.round(height / 24)}px sans-serif`;
        g.textAlign = 'center';
        g.fillText(this._loadError ?? '正在读取画面…', width / 2, height / 2);
      }
    } catch (err) {
      this._loadError = '画面绘制失败（可能是跨域限制）：' + err.message;
    }
    // 画布按容器宽度等比缩放，坐标换算统一走 getBoundingClientRect
    const wrapW = this.dom.frameCanvas.parentElement?.clientWidth || width;
    c.style.width = '100%';
    c.style.height = 'auto';
    void wrapW;
  }

  /** 把画布上的鼠标事件换算成原始像素坐标 */
  _eventToPixel(ev) {
    const c = this.dom.frameCanvas;
    const rect = c.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const x = ((ev.clientX - rect.left) / rect.width) * this.imageSize.width;
    const y = ((ev.clientY - rect.top) / rect.height) * this.imageSize.height;
    if (x < 0 || y < 0 || x > this.imageSize.width || y > this.imageSize.height) return null;
    return [x, y];
  }

  _onImageClick(ev) {
    const px = this._eventToPixel(ev);
    if (!px) return;
    this._pendingImage = px;
    this._updateStepTip();
    this.onToast?.(
      `已记录像素点 (${px[0].toFixed(0)}, ${px[1].toFixed(0)}) —— 切到「② 点三维对应点」后，在右侧三维视口里点同一位置`
    );
    this._renderPairs();
    // 自动进入步骤 2：用户的下一步动作必然是在三维里点对应点
    this.setStep(2);
  }

  /**
   * 由 app.js 在三维视口点击时调用。
   * @returns {boolean} 是否消费了这次点击
   */
  handleSceneClick(ndc) {
    if (!this.active || !this._pendingImage) return false;
    const hit = pickLocalPoint(ndc, this.ctx);
    if (!hit) {
      this.onToast?.('这里没有可拾取的高斯点，换一处表面再点', { error: true });
      return true;
    }
    this.pairs.push({
      image: this._pendingImage,
      object: [hit.local.x, hit.local.y, hit.local.z],
      world: hit.world.toArray(),
      time: this.currentTime,
    });
    this._pendingImage = null;
    this._updateStepTip();
    this._renderPairs();
    this._refreshSolveState();
    return true;
  }

  undo() {
    if (this._pendingImage) {
      this._pendingImage = null;
    } else if (this.pairs.length) {
      this.pairs.pop();
    }
    this._updateStepTip();
    this._renderPairs();
    this._refreshSolveState();
  }

  clearPairs() {
    this.pairs = [];
    this._pendingImage = null;
    this.lastSolve = null;
    this._updateStepTip();
    this._renderPairs();
    this._refreshSolveState();
  }

  // ------------------------------------------------------------ 渲染

  _renderPairs() {
    const marks = this.dom.marks;
    const { width, height } = this.imageSize;
    marks.innerHTML = '';
    marks.style.width = '100%';
    marks.style.height = '100%';
    const add = (x, y, label, cls) => {
      const dot = document.createElement('div');
      dot.className = `pnp-mark ${cls}`;
      dot.style.left = `${(x / width) * 100}%`;
      dot.style.top = `${(y / height) * 100}%`;
      dot.textContent = label;
      marks.appendChild(dot);
      return dot;
    };
    this.pairs.forEach((p, i) => add(p.image[0], p.image[1], String(i + 1), 'done'));
    if (this._pendingImage) add(this._pendingImage[0], this._pendingImage[1], '?', 'pending');

    // 解算后把重投影点也画出来，偏差一眼可见
    if (this.lastSolve) {
      this.lastSolve.perPoint?.forEach((pt, i) => {
        if (!pt?.projected) return;
        const d = document.createElement('div');
        d.className = 'pnp-mark reproj';
        d.style.left = `${(pt.projected[0] / width) * 100}%`;
        d.style.top = `${(pt.projected[1] / height) * 100}%`;
        d.title = `第 ${i + 1} 点重投影误差 ${pt.error.toFixed(2)}px`;
        marks.appendChild(d);
      });
    }

    // 列表
    this.dom.pairs.innerHTML = this.pairs
      .map((p, i) => {
        const err = this.lastSolve?.perPoint?.[i]?.error;
        const errTxt = Number.isFinite(err) ? `${err.toFixed(2)}px` : '—';
        return `<li>
          <span class="idx">${i + 1}</span>
          <span class="px">像素 (${p.image[0].toFixed(0)}, ${p.image[1].toFixed(0)})</span>
          <span class="obj">三维 (${p.object.map((v) => v.toFixed(3)).join(', ')})</span>
          <span class="err">${errTxt}</span>
        </li>`;
      })
      .join('');

    this.dom.count.textContent = String(this.pairs.length);
    const assess = assessPairs(this.pairs);
    this.dom.assess.textContent = assess.message;
    this.dom.assess.className = `pnp-hint level-${assess.level}`;
  }

  _refreshSolveState() {
    // ⚠️ 这里**刻意不用 `disabled`**。
    //
    // 原来写的是 `this.dom.solve.disabled = this.pairs.length < MIN_PAIRS`，
    // 而**禁用按钮不派发 click 事件** —— 用户点下去什么都不会发生：
    // 没有提示、没有 toast、没有控制台报错。实测用户反馈的
    // 「选了 4 组点后点解算没反应」就是这个（实际只记到 3 组，按钮一直是禁用的）。
    //
    // 现在改成：按钮**永远可点**，用文案和样式表达"还差几组"，
    // 点了就给明确提示。宁可多说一句话，也不要让按钮静默失效。
    const need = MIN_PAIRS - this.pairs.length;
    const ready = need <= 0;
    const btn = this.dom.solve;
    this._solveReady = ready;
    btn.classList.toggle('is-not-ready', !ready);
    btn.setAttribute('aria-disabled', ready ? 'false' : 'true');
    btn.title = ready
      ? '解算相机位姿并贴回现场'
      : `还需 ${need} 组对应点（至少 ${MIN_PAIRS} 组；当前 ${this.pairs.length} 组）`;
    btn.textContent = ready ? '解算并贴回现场' : `解算并贴回现场（还需 ${need} 组）`;
  }

  _updateStepTip() {
    const tip = this.dom.steptip;
    const note = this.dom.note3d;
    const armed = !!this._pendingImage;
    this.onArmedChange?.(armed);

    // 两个子选项上的状态徽标
    if (this.dom.badge1) {
      this.dom.badge1.textContent = armed
        ? `已选 (${this._pendingImage[0].toFixed(0)}, ${this._pendingImage[1].toFixed(0)})`
        : '待选择';
      this.dom.badge1.classList.toggle('ok', armed);
    }
    if (this.dom.badge2) {
      this.dom.badge2.textContent = armed ? '点这里 →' : '等待';
      this.dom.badge2.classList.toggle('ok', armed);
    }
    if (this.dom.pickedBox && this.dom.pickedPx) {
      this.dom.pickedBox.hidden = !armed;
      this.dom.pickedPx.textContent = armed
        ? `(${this._pendingImage[0].toFixed(0)}, ${this._pendingImage[1].toFixed(0)})`
        : '—';
    }

    if (tip) {
      tip.textContent = armed
        ? '已选好像素点 —— 切到「② 点三维对应点」，再到右侧三维视口里点它的对应位置'
        : '在画面上点一个容易辨认的特征点（墙角、门框、物证边角），然后切到「② 点三维对应点」';
    }
    if (note) {
      note.innerHTML = armed
        ? `<strong class="ok">请在右侧三维视口中点击该特征点</strong>`
        : `请先在「① 点画面特征点」里选一个像素点。`;
    }
    this._refreshSolveState();
  }

  // ------------------------------------------------------------ 解算

  solve() {
    if (this.pairs.length < MIN_PAIRS) {
      // 用 sticky：这条提示是用户点下去的唯一反馈，飘 3 秒就没了等于没提示。
      const need = MIN_PAIRS - this.pairs.length;
      this.onToast?.(
        `还不能解算：还需 ${need} 组对应点（当前 ${this.pairs.length}/${MIN_PAIRS}）。` +
          `步骤是「在面板画面上点一个特征点 → 再到三维视口点它的对应位置」，一组要两个点。`,
        { error: true, sticky: true }
      );
      return null;
    }
    try {
      const { result, pose, perPoint, meanError, maxError } = solveFromPairs(
        this.pairs,
        this.intrinsics,
        this.imageSize
      );
      this.lastSolve = { result, pose, perPoint, meanError, maxError };
      this.dom.error.textContent = `${meanError.toFixed(3)} px（最大 ${maxError.toFixed(3)} px）`;
      this._renderPairs();
      this.onSolved?.(this.annotation, pose, result, { pairs: this.pairs, perPoint });
      // 成功也要说清楚「发生了什么」，否则用户会以为没反应：
      // 明说用了几组点、误差多少、位姿已经写回并贴回现场。
      const warn = maxError > 5 ? '；⚠ 有较大残差，建议检查误点' : '';
      this.onToast?.(
        `标定完成：用 ${this.pairs.length} 组对应点，重投影误差 ${meanError.toFixed(2)} px` +
          `（最大 ${maxError.toFixed(2)} px），位姿已写回该相机并贴回现场${warn}`
      );
      return { pose, result };
    } catch (err) {
      this.onToast?.(`解算失败：${err.message}`, { error: true, sticky: true });
      return null;
    }
  }

  /** 供测试使用：直接注入一组对应点 */
  setPairs(pairs) {
    this.pairs = pairs.map((p) => ({ ...p, image: [...p.image], object: [...p.object] }));
    this._renderPairs();
    this._refreshSolveState();
  }

  /** 供测试使用：注入内参与图像尺寸而不加载媒体 */
  setIntrinsics(intrinsics, imageSize) {
    this.intrinsics = { ...intrinsics };
    if (imageSize) this.imageSize = { ...imageSize };
    this._writeIntrinsics();
  }

  /** 供测试使用：跳过媒体加载直接打开面板 */
  openForTest({ imageSize = { width: 1920, height: 1080 }, fov = 60 } = {}) {
    this.active = true;
    this.el.hidden = false;
    this.imageSize = { ...imageSize };
    this.intrinsics = defaultIntrinsics(imageSize.width, imageSize.height, fov);
    this.dom.fov.value = String(fov);
    this._writeIntrinsics();
    this._renderPairs();
    this.setStep(1);
    return this;
  }
}

export { MIN_PAIRS };
