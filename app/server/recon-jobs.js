/**
 * 现场建模任务（reconstruction job）
 *
 * 把「视频/图片 → 抽帧 → COLMAP 位姿 → gsplat 训练 → .splat」整条流水线
 * 封装成一个可轮询、可取消、可续看日志的后台任务。
 *
 * 关键工程约束（本机实测）：
 *   1) 服务端跑在 Node 的 embeddable 运行时里，**没有 torch / gsplat**；
 *      而训练必须在 `pyruntime312` 环境 + MSVC(v143) + CUDA 12.8 下执行。
 *      所以这里统一通过 `tools\msvc-env.bat` 包一层 cmd 来调用 Python。
 *   2) DSH 受限模式下程序不能开命名管道；虽然本机实测 `stdio:'pipe'` 也能用，
 *      但为了稳妥（并且让 Web 端能随时拉取完整日志、刷新页面不丢进度），
 *      这里把子进程的 stdout/stderr **重定向到文件描述符**，
 *      日志落盘到 `<jobDir>/pipeline.log`，对外只暴露「读文件」这一条路径。
 *   3) 训练是长任务（几分钟到几十分钟），必须异步执行，
 *      并且只允许同时跑一个（8 GB 显存 + CPU 抽帧都不适合并发）。
 */

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

/** 任务阶段（也是前端进度条的分段） */
export const STAGES = [
  { key: 'queued', label: '排队中', weight: 0 },
  { key: 'extracting', label: '抽取视频帧', weight: 0.08 },
  { key: 'reconstructing', label: 'COLMAP 位姿重建', weight: 0.27 },
  { key: 'training', label: '高斯泼溅训练', weight: 0.55 },
  { key: 'converting', label: '导出模型文件', weight: 0.1 },
  { key: 'done', label: '完成', weight: 0 },
  // failed 也必须在这里：否则 STAGE_INDEX['failed'] 取不到 → 回退成 STAGES[0]，
  // 界面上会把「失败」显示成「排队中」（用户以为任务还在排队，其实早就失败了）。
  { key: 'failed', label: '失败', weight: 0 },
];

const STAGE_INDEX = Object.fromEntries(STAGES.map((s, i) => [s.key, i]));

function nowIso() {
  return new Date().toISOString();
}

/** 找一个可用的 Python 解释器（优先 3.12，训练依赖都在那里） */
export function findPython(root) {
  const candidates = [
    process.env.DSH_RECON_PYTHON,
    path.join(root, 'pyruntime312', 'python.exe'),
    path.join(root, 'pyruntime310', 'python.exe'),
  ].filter(Boolean);
  for (const c of candidates) {
    try {
      if (fs.existsSync(c)) return c;
    } catch {
      /* ignore */
    }
  }
  return null;
}

export class ReconstructionJobs {
  /**
   * @param {object} opts
   * @param {string} opts.root        仓库根目录
   * @param {string} opts.dataDir     data 目录（模型写到这里）
   * @param {string} opts.modelsDir   共享模型目录（<dataDir>/models）
   * @param {(msg:object)=>void} opts.warn  日志回调
   */
  constructor({ root, dataDir, modelsDir, onEvent }) {
    this.root = root;
    this.dataDir = dataDir;
    this.modelsDir = modelsDir;
    this.onEvent = onEvent ?? (() => {});
    /** @type {Map<string, any>} */
    this.jobs = new Map();
    this.running = null;
  }

  /** 列出任务（新的在前） */
  list() {
    return [...this.jobs.values()].sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1)).map((j) => this.status(j.id));
  }

  /** 取某个任务的对外状态（不含内部字段） */
  status(id) {
    const job = this.jobs.get(id);
    if (!job) return null;
    return {
      id: job.id,
      projectId: job.projectId,
      name: job.name,
      stage: job.stage,
      stageLabel: STAGES[STAGE_INDEX[job.stage] ?? 0].label,
      progress: job.progress,
      message: job.message,
      ok: job.ok,
      error: job.error ?? null,
      createdAt: job.createdAt,
      startedAt: job.startedAt,
      finishedAt: job.finishedAt,
      elapsedSec: job.startedAt ? Math.round(((job.finishedAt ? Date.parse(job.finishedAt) : Date.now()) - Date.parse(job.startedAt)) / 1000) : 0,
      input: job.input,
      params: job.params,
      result: job.result ?? null,
      logLines: job.logLines,
    };
  }

  /** 读日志尾部（供 Web 端轮询） */
  readLog(id, { tail = 200, fromLine = 0 } = {}) {
    const job = this.jobs.get(id);
    if (!job) return null;
    if (!fs.existsSync(job.logPath)) return { id, stage: job.stage, progress: job.progress, lines: [], total: 0 };
    let text = '';
    try {
      text = fs.readFileSync(job.logPath, 'utf8');
    } catch {
      text = '';
    }
    const all = text.split(/\r?\n/).filter((l) => l.length > 0);
    const slice = fromLine > 0 ? all.slice(fromLine) : all.slice(-tail);
    return { id, stage: job.stage, progress: job.progress, total: all.length, from: fromLine, lines: slice };
  }

  /** 取消任务 */
  cancel(id) {
    const job = this.jobs.get(id);
    if (!job) return { ok: false, error: '任务不存在' };
    if (job.child && !job.finishedAt) {
      job.canceled = true;
      try {
        // Windows 上需要连同子进程树一起结束（cmd → python → nvcc/cl）
        spawn('taskkill', ['/pid', String(job.child.pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true });
      } catch {
        try {
          job.child.kill();
        } catch {
          /* ignore */
        }
      }
      this._append(job, '[系统] 已请求取消任务');
      return { ok: true };
    }
    return { ok: false, error: '任务已结束' };
  }

  /**
   * 启动一个建模任务。
   *
   * @param {object} p
   * @param {string} p.projectId
   * @param {string} p.projectDir   案件目录（data/projects/<id>）
   * @param {string} p.name         模型名称（最终 data/models/<name>.splat）
   * @param {Array<{path:string,name:string,kind:string}>} p.inputs 素材（绝对路径）
   * @param {object} p.params       抽帧/重建/训练参数
   */
  start({ projectId, projectDir, name, inputs, params = {} }) {
    if (this.running && !this.jobs.get(this.running)?.finishedAt) {
      const runningJob = this.jobs.get(this.running);
      return {
        ok: false,
        error: `已有建模任务在运行（${runningJob?.name ?? this.running}）。同一时间只允许一个 —— 8GB 显存不适合并发训练。`,
      };
    }
    const python = findPython(this.root);
    if (!python) {
      return { ok: false, error: '找不到 Python 运行时（pyruntime312/pyruntime310）' };
    }
    if (!inputs || inputs.length === 0) {
      return { ok: false, error: '没有可用的素材：请先上传视频或图片' };
    }

    const id = `recon-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
    const jobDir = path.join(projectDir, 'recon', id);
    fs.mkdirSync(jobDir, { recursive: true });

    const safeName = String(name || `模型-${new Date().toISOString().slice(0, 10)}`)
      .replace(/[\\/:*?"<>|]/g, '_')
      .slice(0, 80);

    const merged = {
      fps: Number(params.fps ?? 2),
      longSide: Number(params.longSide ?? 1600),
      blurThreshold: Number(params.blurThreshold ?? 60),
      matcher: params.matcher === 'sequential' ? 'sequential' : 'exhaustive',
      cameraModel: params.cameraModel ?? 'OPENCV',
      iterations: Number(params.iterations ?? 30000),
      downscale: Number(params.downscale ?? 2),
      shDegree: Number(params.shDegree ?? 3),
      lrScaleFactor: Number(params.lrScaleFactor ?? 0.4),
      // 默认 0.10，与 pipeline/train_gsplat.py 的默认值一致。
      // 曾经这里是 0.05（比训练器更激进），A/B 实测在 12MB/24MB 两份素材上
      // 分别比 0.10 低 4.78 dB 与 2.14 dB PSNR —— 详见 docs/模型质量与空间结构.md §4.3
      pruneScale: Number(params.pruneScale ?? 0.1),
      evalViews: Number(params.evalViews ?? 0),
      useGpu: params.useGpu !== false,
    };

    const job = {
      id,
      projectId,
      name: safeName,
      jobDir,
      logPath: path.join(jobDir, 'pipeline.log'),
      scriptPath: path.join(jobDir, 'run-recon.bat'),
      stage: 'queued',
      progress: 0,
      message: '等待启动',
      ok: null,
      error: null,
      createdAt: nowIso(),
      startedAt: null,
      finishedAt: null,
      logLines: 0,
      child: null,
      canceled: false,
      input: {
        count: inputs.length,
        videos: inputs.filter((i) => i.kind === 'video').length,
        images: inputs.filter((i) => i.kind === 'image').length,
        files: inputs.map((i) => i.name),
      },
      params: merged,
      result: null,
      python,
      inputs,
    };
    this.jobs.set(id, job);
    this.running = id;

    // 把素材清单写成 JSON，交给 Python 编排脚本读（避免命令行超长）
    const manifest = path.join(jobDir, 'inputs.json');
    fs.writeFileSync(
      manifest,
      JSON.stringify(
        {
          jobId: id,
          projectId,
          name: safeName,
          jobDir,
          dataDir: this.dataDir,
          modelsDir: this.modelsDir,
          inputs,
          params: merged,
        },
        null,
        2
      ),
      'utf8'
    );

    this._writeRunnerScript(job, manifest);
    this._spawn(job);
    this.onEvent({ type: 'recon.start', job: this.status(id), projectId });
    return { ok: true, job: this.status(id) };
  }

  /**
   * 生成运行批处理。
   *
   * 为什么要包一层 .bat：
   *   gsplat 的训练必须在 MSVC v143 + CUDA 12.8 + ninja 的环境里跑，
   *   这些环境变量由 tools\msvc-env.bat 设置。
   *   直接调 python.exe 会因为找不到 cl.exe / nvcc 而失败。
   */
  _writeRunnerScript(job, manifest) {
    const envBat = path.join(this.root, 'tools', 'msvc-env.bat');
    const orch = path.join(this.root, 'pipeline', 'recon_job.py');
    const lines = [
      '@echo off',
      'chcp 65001 >nul',
      `set "DSH_ROOT=${this.root}"`,
      `cd /d "${this.root}"`,
      `if exist "${envBat}" call "${envBat}" >nul 2>&1`,
      `"${job.python}" -u "${orch}" --manifest "${manifest}"`,
      'exit /b %errorlevel%',
    ];
    fs.writeFileSync(job.scriptPath, lines.join('\r\n'), 'ascii');
  }

  _spawn(job) {
    job.startedAt = nowIso();
    job.stage = 'extracting';
    job.progress = 0.02;
    job.message = '正在启动流水线…';
    fs.writeFileSync(job.logPath, `[系统] 任务 ${job.id} 开始\n${'-'.repeat(60)}\n`, 'utf8');

    // 把 stdout/stderr 都重定向到同一个文件描述符：
    //   · 不走管道 → 不受命名管道限制
    //   · 落盘 → Web 端可随时拉取，刷新页面不丢进度
    const fd = fs.openSync(job.logPath, 'a');
    let child;
    try {
      child = spawn('cmd.exe', ['/c', job.scriptPath], {
        cwd: this.root,
        stdio: ['ignore', fd, fd],
        windowsHide: true,
      });
    } catch (err) {
      fs.closeSync(fd);
      this._finish(job, false, `无法启动流水线进程：${err.message}`);
      return;
    }
    job.child = child;
    job.fd = fd;
    this.onEvent({ type: 'recon.progress', job: this.status(job.id) });

    child.on('error', (err) => {
      this._finish(job, false, `流水线进程错误：${err.code ?? err.message}`);
    });
    child.on('exit', (code) => {
      try {
        fs.closeSync(fd);
      } catch {
        /* ignore */
      }
      job.child = null;
      if (job.canceled) {
        this._finish(job, false, '任务已取消（用户操作）');
      } else if (code === 0) {
        const produced = this._collectResult(job);
        if (produced) this._finish(job, true, null, produced);
        else this._finish(job, false, '流水线返回成功，但没有找到导出的模型文件');
      } else {
        this._finish(job, false, this._explainExit(code));
      }
    });
  }

  /** 从日志里抓阶段与进度，供轮询时更新状态 */
  refresh(id) {
    const job = this.jobs.get(id);
    if (!job || job.finishedAt) return job ? this.status(id) : null;
    if (!fs.existsSync(job.logPath)) return this.status(id);
    let text = '';
    try {
      text = fs.readFileSync(job.logPath, 'utf8');
    } catch {
      return this.status(id);
    }
    const lines = text.split(/\r?\n/).filter(Boolean);
    job.logLines = lines.length;

    // 只解析**最后**出现的阶段标记，避免早期日志覆盖当前状态
    const marks = [];
    for (const line of lines) {
      const m = /\[阶段\]\s*([a-z]+)\s*\|([\d.]+)\|(.*)$/.exec(line);
      if (m) marks.push({ stage: m[1], progress: Number(m[2]), message: m[3].trim() });
    }
    if (marks.length) {
      const last = marks[marks.length - 1];
      if (STAGE_INDEX[last.stage] !== undefined) {
        const changed = job.stage !== last.stage;
        job.stage = last.stage;
        job.progress = Math.max(0, Math.min(1, last.progress));
        job.message = last.message;
        if (changed) this.onEvent({ type: 'recon.progress', job: this.status(id) });
      }
    }
    return this.status(id);
  }

  /** 任务结束后，找找有没有产出 .splat（Python 侧一般已放好，这里作为兜底） */
  _collectResult(job) {
    const candidates = [
      path.join(job.jobDir, 'model.json'),
      path.join(this.modelsDir, `${job.name}.json`),
    ];
    for (const c of candidates) {
      try {
        if (fs.existsSync(c)) {
          const meta = JSON.parse(fs.readFileSync(c, 'utf8'));
          return { model: meta.name ?? job.name, json: c, meta };
        }
      } catch {
        /* ignore */
      }
    }
    const splat = path.join(this.modelsDir, `${job.name}.splat`);
    if (fs.existsSync(splat)) {
      const st = fs.statSync(splat);
      return { model: job.name, bytes: st.size };
    }
    return null;
  }

  _finish(job, ok, error, result) {
    job.ok = ok;
    job.error = error ?? null;
    job.finishedAt = nowIso();
    job.stage = ok ? 'done' : 'failed';
    job.progress = ok ? 1 : job.progress;
    job.message = ok ? '建模完成' : error ?? '失败';
    if (result) job.result = result;
    if (this.running === job.id) this.running = null;
    this._append(job, `\n[系统] 任务${ok ? '成功结束' : `失败：${error}`}`);
    this.onEvent({ type: ok ? 'recon.done' : 'recon.failed', job: this.status(job.id), projectId: job.projectId });
  }

  _append(job, text) {
    try {
      fs.appendFileSync(job.logPath, `${text}\n`, 'utf8');
    } catch {
      /* ignore */
    }
  }

  /** 把退出码翻译成有用的提示 */
  _explainExit(code) {
    const map = {
      2: '参数或数据错误（素材不足、目录缺失）',
      3: '缺少 torch / gsplat 依赖',
      4: '显存不足（可调大 --downscale 或减少迭代）',
      5: '训练失败（详见日志）',
      6: 'COLMAP 重建未达标（注册图片太少，建议重拍素材）',
    };
    const extra = map[code] ? `：${map[code]}` : '';
    return `流水线退出码 ${code}${extra}。完整日志见任务日志。`;
  }
}
