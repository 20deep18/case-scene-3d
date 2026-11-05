#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
现场建模流水线编排脚本（被服务端以子进程方式调用）。

它把下面四步串起来，并在每一步打印**机器可解析的阶段标记**，
服务端读日志就能知道进度：

    [阶段] extracting|0.05|正在抽取视频帧

完整流程：
    1) 素材整理：视频 → 抽帧（含模糊过滤）；图片 → 直接用
    2) COLMAP：相机位姿 + 稀疏点云
    3) gsplat：训练 3DGS
    4) 导出：PLY → .splat，并落到 data/models/<名称>.splat（自动出现在「模型文件」里）

退出码约定（服务端会翻译成中文提示）：
    0 成功
    2 参数/素材错误
    3 缺少依赖（torch/gsplat）
    4 显存不足
    5 训练失败
    6 COLMAP 注册率不达标

用法（一般由服务端调用，不手工执行）：
    python pipeline/recon_job.py --manifest <任务目录>/inputs.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm")

THRESHOLD_FILE = ROOT / "pipeline" / "recon_threshold.json"


def load_min_bytes() -> tuple[int, str]:
    """读取「建模所需最小素材体积」（单一口径：pipeline/recon_threshold.json）。

    这个值由**真实训练实测**得到（tools/threshold_experiment.py），
    服务端接口 GET /api/projects/:id/recon/materials 读的是同一个文件。
    这里读它只是为了在日志里也明确写出「体积是否够」，方便事后追责/复盘。
    """
    try:
        raw = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
        min_bytes = int(raw.get("minBytes") or 0)
        note = str(raw.get("note") or "")
        return (min_bytes, note)
    except Exception:  # noqa: BLE001
        return (0, "")


def human_bytes(num: float) -> str:
    if num >= 2**30:
        return f"{num / 2**30:.2f} GB"
    if num >= 2**20:
        return f"{num / 2**20:.1f} MB"
    if num >= 2**10:
        return f"{num / 2**10:.0f} KB"
    return f"{num:.0f} B"


def resolve_media_path(raw: str | Path) -> Path | None:
    """把素材路径解析成真实存在的文件；不存在就返回 None。

    为什么需要「解码后重试」这一层：
      媒体 URL 里的中文名是百分号编码的（`%E7%8E%B0%E5%9C%BA.mp4`）。
      服务端已经会解码，但**旧的 manifest / 第三方调用**仍可能把编码串当路径传进来，
      这时 `文件不存在` 会被下游误报成「素材不足」，排查起来非常费劲。
      这里做一次兜底：解码后如果那个文件真的在，就用它。
    """
    p = Path(raw)
    if p.exists():
        return p
    text = str(raw)
    if "%" in text:
        try:
            from urllib.parse import unquote

            decoded = Path(unquote(text))
            if decoded.exists():
                return decoded
        except Exception:  # noqa: BLE001
            pass
    return None


def stage(name: str, progress: float, message: str = "") -> None:
    """打印机器可解析的阶段标记（服务端按这个更新进度条）。"""
    print(f"[阶段] {name}|{progress:.3f}|{message}", flush=True)


def log(message: str) -> None:
    print(message, flush=True)


def run_script(script: str, args: list[str], label: str) -> int:
    """
    以子进程方式跑一个流水线脚本。

    为什么用子进程而不是直接 import 调 main()：
      · 各脚本自己管理日志前缀与退出码，子进程边界最干净；
      · 某个脚本内部崩溃（例如 CUDA 报错）不会把编排进程一起带走；
      · 输出会原样进日志，便于排查。
    """
    cmd = [sys.executable, "-u", str(PIPELINE / script), *args]
    log(f"\n{'=' * 72}\n[{label}] {' '.join(cmd)}\n{'=' * 72}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    log(f"[{label}] 退出码 {proc.returncode}，耗时 {time.time() - t0:.1f}s")
    return proc.returncode


def gather_video_frames(video: Path, out_dir: Path, params: dict) -> int:
    args = [
        "--video", str(video),
        "--out", str(out_dir),
        "--fps", str(params.get("fps", 2)),
        "--long-side", str(params.get("longSide", 1600)),
        "--blur-threshold", str(params.get("blurThreshold", 60)),
    ]
    return run_script("extract_frames.py", args, f"抽帧 {video.name}")


def count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    ap = argparse.ArgumentParser(description="现场建模流水线编排")
    ap.add_argument("--manifest", required=True, help="任务清单 JSON（服务端生成）")
    args = ap.parse_args()

    # 用 utf-8-sig 读：有些工具（含 PowerShell 的 Set-Content）会写 UTF-8 BOM，
    # 直接 json.loads 会报 "Unexpected UTF-8 BOM"。utf-8-sig 对无 BOM 文件同样适用。
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    job_dir = Path(manifest["jobDir"])
    data_dir = Path(manifest["dataDir"])
    models_dir = Path(manifest["modelsDir"])
    name = manifest["name"]
    params = manifest.get("params", {})
    inputs = manifest.get("inputs", [])

    images_dir = job_dir / "images"
    colmap_dir = job_dir / "colmap"
    train_dir = job_dir / "train"
    for d in (images_dir, colmap_dir, train_dir):
        d.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log(f"现场建模任务：{name}")
    log(f"  任务目录 : {job_dir}")
    log(f"  输出模型 : {models_dir / (name + '.splat')}")
    log(f"  素材     : {len(inputs)} 个（视频 {sum(1 for i in inputs if i['kind'] == 'video')}，"
        f"图片 {sum(1 for i in inputs if i['kind'] == 'image')}）")

    # 素材体积体检（与 Web 端同一个阈值）：
    # 前端在体积不足时已经让用户确认过一次，这里再记一笔，
    # 这样「为什么这个模型这么糊」在日志里就有据可查。
    total_bytes = 0
    for item in inputs:
        try:
            total_bytes += Path(item["path"]).stat().st_size
        except OSError:
            pass
    min_bytes, _note = load_min_bytes()
    if min_bytes > 0:
        verdict = "充足" if total_bytes >= min_bytes else "偏少"
        log(f"  素材体积 : {human_bytes(total_bytes)}（实测建模下限 {human_bytes(min_bytes)} → {verdict}）")
        if total_bytes < min_bytes:
            log(f"[警告] 素材体积只有实测下限的 {total_bytes / min_bytes:.0%}。")
            log("       体积偏少通常意味着可用视角不足：COLMAP 可能注册失败、点云过稀，")
            log("       训练出来的现场会缺面/发糊。建议补充视频或照片后重新建模。")
    else:
        log(f"  素材体积 : {human_bytes(total_bytes)}")

    log(f"  参数     : {json.dumps(params, ensure_ascii=False)}")
    log("=" * 72)

    # ---------------------------------------------------------------- 1/4
    stage("extracting", 0.03, "整理素材（视频抽帧 / 图片归集）")
    # 逐个解析成真实存在的文件（顺带兜住「URL 百分号编码没解码」的历史 manifest）
    videos: list[Path] = []
    images: list[Path] = []
    for item in inputs:
        resolved = resolve_media_path(item["path"])
        if resolved is None:
            log(f"[警告] 素材文件不存在，跳过：{item['path']}")
            continue
        (videos if item.get("kind") == "video" else images).append(resolved)

    if not videos and not images:
        log("[错误] 没有任何素材（文件全部不存在或列表为空）")
        return 2

    # 图片直接复制进 images_dir（保持平铺，COLMAP 只按文件名匹配）
    copied = 0
    for img in images:
        dest = images_dir / img.name
        if not dest.exists():
            shutil.copy2(img, dest)
            copied += 1
    if copied:
        log(f"[信息] 归集图片 {copied} 张")

    for idx, video in enumerate(videos):
        # 多个视频分别抽到子目录再合并，避免帧名冲突（frame_0001.png 会互相覆盖）
        sub = job_dir / f"frames_{idx + 1}"
        base = 0.03 + 0.05 * (idx / max(1, len(videos)))
        stage("extracting", base, f"抽取视频帧：{video.name}")
        code = gather_video_frames(video, sub, params)
        if code != 0:
            log(f"[警告] 抽帧脚本返回 {code}，继续尝试后续步骤")
        # 合并到 images_dir，冲突时加前缀重命名
        if sub.is_dir():
            n = 0
            for p in sorted(sub.iterdir()):
                if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                dest = images_dir / p.name
                if dest.exists():
                    dest = images_dir / f"v{idx + 1}_{p.name}"
                shutil.copy2(p, dest)
                n += 1
            log(f"[信息] 视频 {video.name} 贡献 {n} 帧")

    total_images = count_images(images_dir)
    log(f"\n[信息] 可用于重建的图片总数：{total_images}")
    if total_images < 8:
        log("[错误] 可用于重建的图片太少（< 8 张）。请上传更长的视频或更多图片。")
        return 2

    # ---------------------------------------------------------------- 2/4
    stage("reconstructing", 0.10, f"COLMAP 位姿重建（{total_images} 张图）")
    code = run_script(
        "colmap_recon.py",
        [
            "--images", str(images_dir),
            "--workspace", str(colmap_dir),
            "--matcher", str(params.get("matcher", "exhaustive")),
            "--camera-model", str(params.get("cameraModel", "OPENCV")),
            "--use-gpu", "1" if params.get("useGpu", True) else "0",
        ],
        "COLMAP",
    )
    sparse0 = colmap_dir / "sparse" / "0"
    if code != 0 or not (sparse0 / "images.bin").exists():
        log("[错误] COLMAP 重建未产出可用模型（注册率不足或匹配失败）。")
        log("       建议：素材要有真实平移与纹理，避免纯旋转/白墙。")
        return 6
    log(f"[信息] COLMAP 模型就绪：{sparse0}")

    # ---------------------------------------------------------------- 2.5/4
    # 房间结构骨架：从稀疏点云 + 相机位姿里拟合地面/墙/房间盒/相机轨迹。
    # 3DGS 泼溅只能让人看见「材质」，看不见「结构」——用户反馈的
    # 「没有呈现出房间的整体布局和空间线条」就是缺这一层。
    # 它是**增强**：失败不影响建模本身，所以不看退出码，只看有没有产物。
    stage("reconstructing", 0.36, "提取房间结构骨架（地面/墙/相机轨迹）")
    struct_path = models_dir / f"{name}.structure.json"
    run_script(
        "../tools/scene_structure.py",
        ["--model", str(sparse0), "--out", str(struct_path)],
        "房间结构提取",
    )
    if struct_path.exists():
        try:
            st = json.loads(struct_path.read_text(encoding="utf-8"))
            size = st.get("room", {}).get("size") or [0, 0, 0]
            ch = (st.get("trajectory") or {}).get("heightAboveFloor") or {}
            ev = st.get("sceneModeEvidence") or {}
            _mode_cn = {"room": "房间", "object": "物体（贴拍）", "uncertain": "不确定"}.get(
                st.get("sceneMode"), st.get("sceneMode")
            )
            if st.get("degraded"):
                # 点云为空/过少：结构提取会降级输出而不是崩（见 scene_structure._empty_structure）
                log(f"[警告] 结构骨架不可用：{(ev.get('rejectReasons') or ['点云不足'])[0]}")
                log("       前端不会叠加房间盒；请检查素材（纹理、平移、光照）。")
            else:
                log(
                    f"[信息] 结构骨架：场景判定 {_mode_cn}"
                    f"（方向包围度 {(ev.get('surroundCoverage') or {}).get('median')}），"
                    f"房间 {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} m，"
                    f"墙面 {len(st.get('walls') or [])} 面，"
                    f"相机离地中位 {ch.get('median')} m，"
                    f"自检 {'通过' if st.get('sane') else '**未通过（结果仅供参考）**'}"
                )
            # 判定为 object/uncertain 时前端**不画房间盒**（那会是个假户型）。
            # 这里必须把原因写进日志 —— 否则用户看到"没有户型线"会以为是坏了。
            if st.get("sceneMode") != "room":
                why = (ev.get("rejectReasons") or ["证据不足"])[0]
                log(f"[信息] 未叠加房间盒/地面网格：{why}")
                log("       相机轨迹仍会显示；如需强制按房间处理，可用 "
                    "tools/scene_structure.py --scene-mode room 重新生成结构文件。")
            # 拍摄引导：把「哪些方向没拍到」写进日志 —— 这是模型发虚最常见的根因，
            # 而且调参救不了，只能补拍。日志里留一行，事后复盘有据可查。
            cov = st.get("coverage") or {}
            if cov.get("frames"):
                log(
                    f"[信息] 方向覆盖度：{cov['frames']} 帧中 {cov.get('weakFrames')} 帧"
                    f"（{cov.get('weakRatio', 0):.1%}）几乎看不到点云；"
                    f"最少 {cov.get('minPointsInView')} 点 / 中位 {cov.get('medianPointsInView')} 点"
                )
                if cov.get("verdict") and cov["verdict"] != "覆盖良好":
                    log(f"[警告] {cov['verdict']}")
                    log("       模型在没拍到的方向只能靠插值 → 会缺面/发虚。")
                    log("       这属于素材问题（不是训练参数问题），补拍这些方向才有效。")
        except Exception as exc:  # noqa: BLE001
            log(f"[警告] 结构骨架读取失败：{exc}")
    else:
        log("[警告] 没有产出房间结构骨架（不影响建模，只是少了一层结构线框）")

    # ---------------------------------------------------------------- 3/4
    stage("training", 0.38, f"高斯泼溅训练（{params.get('iterations', 30000)} 步）")
    code = run_script(
        "train_gsplat.py",
        [
            "--data", str(colmap_dir),
            "--images", str(images_dir),
            "--out", str(train_dir),
            "--iterations", str(params.get("iterations", 30000)),
            "--downscale", str(params.get("downscale", 2)),
            "--sh-degree", str(params.get("shDegree", 3)),
            "--lr-scale-factor", str(params.get("lrScaleFactor", 0.4)),
            "--prune-scale", str(params.get("pruneScale", 0.05)),
            "--eval-views", str(params.get("evalViews", 0)),
            "--save-every", "0",
        ],
        "gsplat 训练",
    )
    ply = train_dir / "point_cloud.ply"
    if code != 0 or not ply.exists():
        log("[错误] 训练失败或没有产出 PLY。")
        if code == 4:
            log("       显存不足：请把 downscale 调大（例如 3 或 4）后重试。")
            return 4
        return 5
    log(f"[信息] 训练产物：{ply}（{ply.stat().st_size / 2**20:.1f} MB）")

    # ---------------------------------------------------------------- 4/4
    stage("converting", 0.93, "导出 .splat 并登记到模型库")
    splat_path = models_dir / f"{name}.splat"
    json_path = models_dir / f"{name}.json"
    # 尺度上限**不再硬编码 0.5**。
    #
    # 原来写死 "--max-scale 0.5"，而训练器 `_clamp_scales()` 的钳位是
    # `prune_scale × spatial_scale`（某真实任务 = 0.1 × 11.185 = 1.1185 m），
    # 导出阈值比训练器上限还紧 2.24 倍 → 训练器刻意保留的大高斯（铺墙铺地的主力）
    # 被整批删掉：实测同一份 PLY，0.5 丢掉 6358 个（51.5%）只剩 5036 个，
    # 1.1185 则丢弃 0 个、导出 11090 个。
    #
    # 现在优先用训练摘要里落盘的 maxScaleMeters（与训练器同源）；
    # 拿不到就交给 to_splat 的 `auto`（按本模型尺度分布的 P99.5 自适应），
    # 两条路都不会再一刀切。
    max_scale_arg = "auto"
    try:
        _ts_path = train_dir / "train_summary.json"
        if _ts_path.exists():
            _ts = json.loads(_ts_path.read_text(encoding="utf-8"))
            _derived = _ts.get("maxScaleMeters")
            if isinstance(_derived, (int, float)) and _derived > 0:
                max_scale_arg = f"{float(_derived):.6g}"
                log(
                    f"[信息] 尺度上限取自训练摘要：{max_scale_arg} m"
                    f"（= prune_scale {_ts.get('pruneScale')} × 场景尺度 {_ts.get('sceneExtent')}）"
                )
    except Exception as exc:  # noqa: BLE001 - 摘要缺失/损坏不应中断导出
        log(f"[警告] 读取训练摘要里的尺度上限失败（改用 auto）：{exc}")
    if max_scale_arg == "auto":
        log("[信息] 训练摘要里没有尺度信息，改用 auto（本模型尺度分布的 P99.5）")
    code = run_script(
        "to_splat.py",
        [
            "--ply", str(ply),
            "--out", str(splat_path),
            "--json", str(json_path),
            "--sh-degree", str(params.get("shDegree", 3)),
            "--max-scale", max_scale_arg,
        ],
        "PLY → .splat",
    )
    if code != 0 or not splat_path.exists():
        log("[错误] 导出 .splat 失败。")
        return 5

    # 富化元数据，便于前端展示「来源 / 参数 / 画质」
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    except Exception:  # noqa: BLE001
        meta = {}
    train_summary = {}
    try:
        ts = train_dir / "train_summary.json"
        if ts.exists():
            train_summary = json.loads(ts.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        train_summary = {}
    meta.update(
        {
            "name": name,
            "source": "web-recon",
            "sourceProject": manifest.get("projectId"),
            "jobId": manifest.get("jobId"),
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "imageCount": total_images,
            # ⚠️ Bug B：这里原来写的是
            #     meta.get("gaussianCount") or train_summary.get("gaussians")
            #   而 to_splat 的输出里**没有** gaussianCount 这个键（它写的是 count），
            #   于是永远落到 train_summary["gaussians"] —— 那是**训练产出数**。
            #   实测：导出 5036 个，元数据却写 12338（虚高 2.4 倍），
            #   界面上的「N 高斯」、体积评估、回归断言全部跟着失真。
            #   现在优先取**导出后实际写入的数量**。
            "gaussianCount": (
                meta.get("count")
                or (meta.get("filters") or {}).get("kept")
                or meta.get("gaussianCount")
                or train_summary.get("gaussians")
            ),
            "quality": train_summary.get("eval"),
            "params": params,
        }
    )
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 给任务目录也放一份，服务端可以据此确认产出
    (job_dir / "model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    size_mb = splat_path.stat().st_size / 2**20
    stage("done", 1.0, f"建模完成：{name}.splat（{size_mb:.1f} MB）")
    log("\n" + "=" * 72)
    log(f"✔ 建模完成")
    log(f"  模型文件：{splat_path}（{size_mb:.1f} MB）")
    log(f"  高斯数量：{meta.get('gaussianCount', '未知')}")
    # 导出前后的数量差本身就是重要信号：差得多说明导出过滤砍掉了大量高斯，
    # 直接说出来，别再让「界面显示的数」和「文件里的数」对不上。
    _trained = train_summary.get("gaussians")
    _exported = meta.get("gaussianCount")
    if _trained and _exported and _trained != _exported:
        _lost = 1.0 - (_exported / _trained)
        log(f"  导出过滤：训练产出 {_trained} → 导出 {_exported}（丢弃 {_lost:.0%}）")
        if _lost > 0.35:
            log("  [警告] 超过三分之一的高斯被导出过滤丢掉。大尺度高斯是铺墙铺地的主力，")
            log("         丢这么多通常意味着训练时尺度失控（见文档 Bug A），会引起模型发空发虚。")
    if train_summary.get("eval"):
        ev = train_summary["eval"]
        log(f"  留出视角：PSNR {ev.get('psnr')} dB / SSIM {ev.get('ssim')}")
    log("  该模型已登记到模型库，可在「模型文件」中选择并载入。")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[错误] 被中断", flush=True)
        raise SystemExit(130)
