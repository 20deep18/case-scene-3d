#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
「建模所需最小素材体积」实测脚本。

目标：用**真实训练**回答一个问题 ——
    **用户提供的视频/图片总体积小于多少时，现场建模会失败或产出不可用模型？**

思路
----
体积本身不是物理量，但它在本系统里是「素材够不够」的一个**可测量代理**：
同一台设备、同一编码参数下，体积基本正比于**时长**，而时长在固定抽帧帧率下
决定了**可用视角数**；视角数不足 → COLMAP 注册不上 → 稀疏点云太稀 →
高斯泼溅没有足够约束 → 模型糊掉或直接失败。

所以本脚本做一条「体积阶梯」（用 ffmpeg 把实测视频按 `-c copy -t` 截断，
编码参数不变，因此体积 ∝ 时长 ∝ 抽帧数），逐档跑：

    1) extract_frames.py   抽帧（fps=2, long-side=1600, 模糊阈值=60）
    2) colmap_recon.py     COLMAP 位姿重建（exhaustive）
    3) train_gsplat.py     高斯泼溅训练（降到 7000 步，够看出模型是否可用）

并把每档的「体积 / 可用帧数 / 注册率 / 稀疏点数 / 重投影误差 / PSNR / SSIM」
写进报告。取**稳定通过的最小档位**作为阈值，并在文档里保留安全余量。

判定标准（脚本里写死，避免事后挑数据）
--------------------------------------
  COLMAP 通过：注册率 >= 0.60 且 注册帧数 >= 12
  训练通过   ：退出码 0 且 高斯数 >= 5000 且 留出视角 PSNR >= 15 dB
  该档通过   ：COLMAP 与训练都通过

用法
----
    python tools/threshold_experiment.py --video "测试视频（较大型）.mp4" --out output/threshold
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

import config  # noqa: E402  （项目自带依赖探测）

# 体积阶梯（MB，目标值；实际体积由 ffmpeg 截断后测量）
LADDER_MB = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 100]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm")


def log(msg: str) -> None:
    print(msg, flush=True)


def count_images(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def run(cmd: list[str], log_path: Path, label: str) -> tuple[int, float]:
    """跑一个子进程，stdout/stderr 落盘（不用管道，避免受限环境下的命名管道限制）。"""
    t0 = time.time()
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{'=' * 72}\n[{label}] {' '.join(cmd)}\n{'=' * 72}\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n[{label}] 退出码 {proc.returncode}，耗时 {dt:.1f}s\n")
    return proc.returncode, dt


def ffmpeg_exe() -> str:
    exe = config.find_ffmpeg()
    if not exe:
        raise SystemExit("找不到 ffmpeg")
    return exe


def cut_clip(src: Path, dst: Path, seconds: float) -> None:
    """按秒数截断（-c copy：不重编码，体积与时长严格成比例）。"""
    exe = ffmpeg_exe()
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error",
           "-ss", "0", "-i", str(src), "-t", f"{seconds:.3f}", "-c", "copy",
           "-avoid_negative_ts", "make_zero", str(dst)]
    subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not dst.exists() or dst.stat().st_size == 0:
        # 有些容器 -c copy 截断会失败：退回重编码
        cmd2 = [exe, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(src), "-t", f"{seconds:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", str(dst)]
        subprocess.run(cmd2, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser(description="建模最小素材体积实测")
    ap.add_argument("--video", default="测试视频（较大型）.mp4", help="用于切档的源视频")
    ap.add_argument("--out", default="output/threshold", help="输出目录")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--long-side", type=int, default=1600)
    ap.add_argument("--blur", type=float, default=60.0)
    ap.add_argument("--iterations", type=int, default=7000, help="训练步数（默认 7000，够判别可用性）")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--ladder", default="", help="覆盖阶梯，逗号分隔的 MB 值")
    ap.add_argument("--only", default="", help="只跑指定档位（逗号分隔 MB）")
    args = ap.parse_args()

    src = (ROOT / args.video) if not Path(args.video).is_absolute() else Path(args.video)
    if not src.is_file():
        raise SystemExit(f"源视频不存在：{src}")
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = out_dir / "clips"

    ladder = [float(x) for x in args.ladder.split(",") if x.strip()] if args.ladder else [float(x) for x in LADDER_MB]
    if args.only:
        keep = {float(x) for x in args.only.split(",") if x.strip()}
        ladder = [x for x in ladder if x in keep]

    src_mb = src.stat().st_size / 2**20
    log(f"源视频 {src.name}：{src_mb:.1f} MB")
    log(f"阶梯（目标 MB）：{ladder}")

    report = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {"name": src.name, "bytes": src.stat().st_size, "mb": round(src_mb, 2)},
        "params": {
            "fps": args.fps, "longSide": args.long_side, "blurThreshold": args.blur,
            "iterations": args.iterations, "downscale": args.downscale, "matcher": "exhaustive",
        },
        "criteria": {
            "colmapRegisteredRatio": 0.60,
            "colmapRegisteredImages": 12,
            # 稀疏点是**最干净**的判据：实测里「可用」的档位点云都在 1 万以上，
            # 「崩掉/勉强」的档位只有 0~6116。相比之下 PSNR 会被「留出视角与训练
            # 视角几乎是近邻」这种情况抬高（10.5MB 的 9 秒视频只有 958 个点，
            # PSNR 却有 33 dB），所以把点数也写进判据。
            "sparseMinPoints": 10000,
            "trainMinGaussians": 5000,
            "trainMinPsnrDb": 15.0,
        },
        "rungs": [],
    }

    for target_mb in ladder:
        # 用「源视频的字节/秒」把目标体积换算成秒数
        bytes_per_sec = src.stat().st_size / max(1e-6, _duration_sec(src))
        seconds = max(1.2, (target_mb * 2**20) / bytes_per_sec)
        tag = f"{target_mb:g}mb".replace(".", "p")
        clip = clips / f"clip-{tag}.mp4"
        if not clip.exists():
            cut_clip(src, clip, seconds)
        clip_bytes = clip.stat().st_size if clip.exists() else 0

        run_dir = out_dir / f"run-{tag}"
        images = run_dir / "images"
        colmap_dir = run_dir / "colmap"
        train_dir = run_dir / "train"
        for d in (run_dir, images, colmap_dir, train_dir):
            d.mkdir(parents=True, exist_ok=True)
        plog = run_dir / "experiment.log"

        log(f"\n{'#' * 72}\n# 档位 {target_mb:g} MB（截断 {seconds:.1f}s，实际 {clip_bytes/2**20:.2f} MB）\n{'#' * 72}")

        entry: dict = {
            "targetMb": target_mb,
            "clipFile": clip.name,
            "clipBytes": clip_bytes,
            "clipMb": round(clip_bytes / 2**20, 3),
            "clipSeconds": round(seconds, 2),
        }

        py = sys.executable
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

        # ---- 1) 抽帧
        if not count_images(images):
            code, dt = run([py, "-u", str(PIPELINE / "extract_frames.py"),
                            "--video", str(clip), "--out", str(images),
                            "--fps", str(args.fps), "--long-side", str(args.long_side),
                            "--blur-threshold", str(args.blur)], plog, "抽帧")
        n_img = count_images(images)
        n_rej = count_images(images / "_rejected")
        entry.update({"frames": n_img, "rejected": n_rej})
        log(f"  抽帧：可用 {n_img} 张，模糊剔除 {n_rej} 张")
        if n_img < 8:
            entry.update({"ok": False, "reason": f"可用帧数 {n_img} < 8，流水线会直接判素材不足"})
            report["rungs"].append(entry)
            _save(report, out_dir)
            continue

        # ---- 2) COLMAP
        summary_path = colmap_dir / "colmap_summary.json"
        if not summary_path.exists():
            code, dt = run([py, "-u", str(PIPELINE / "colmap_recon.py"),
                            "--images", str(images), "--workspace", str(colmap_dir),
                            "--matcher", "exhaustive", "--camera-model", "OPENCV",
                            "--use-gpu", "1"], plog, "COLMAP")
        summary = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                summary = {}
        reg = int(summary.get("registeredImages") or 0)
        ratio = float(summary.get("registeredRatio") or 0.0)
        pts = int(summary.get("points3D") or 0)
        reproj = summary.get("meanReprojectionError")
        min_points = int(report["criteria"]["sparseMinPoints"])
        sparse_ok = pts >= min_points
        colmap_ok = ratio >= 0.60 and reg >= 12
        entry.update({
            "registeredImages": reg, "registeredRatio": round(ratio, 4),
            "points3D": pts, "meanReprojectionError": reproj,
            "colmapOk": colmap_ok,
            "sparseOk": sparse_ok,
            "colmapElapsedSec": summary.get("elapsedSec"),
        })
        log(f"  COLMAP：注册 {reg}/{n_img}（{ratio:.1%}），点 {pts}（≥{min_points}? {sparse_ok}），误差 {reproj}")

        if not colmap_ok:
            entry.update({"ok": False, "reason": f"COLMAP 注册率 {ratio:.1%}（{reg} 帧）不达标"})
            report["rungs"].append(entry)
            _save(report, out_dir)
            continue
        if not sparse_ok:
            # 注册上了但点云太稀 —— 这种最容易骗人：COLMAP 报成功，训练出来却是一团糊
            entry.update({
                "ok": False,
                "reason": f"稀疏点云只有 {pts} 个（< {min_points}），模型会缺面/发糊",
            })
            report["rungs"].append(entry)
            _save(report, out_dir)
            continue

        # ---- 3) 训练
        ts_path = train_dir / "train_summary.json"
        if not ts_path.exists():
            code, dt = run([py, "-u", str(PIPELINE / "train_gsplat.py"),
                            "--data", str(colmap_dir), "--images", str(images),
                            "--out", str(train_dir), "--iterations", str(args.iterations),
                            "--downscale", str(args.downscale), "--sh-degree", "1",
                            "--eval-views", "4", "--save-every", "0"], plog, "gsplat 训练")
        ts = {}
        if ts_path.exists():
            try:
                ts = json.loads(ts_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                ts = {}
        gauss = int(ts.get("gaussians") or ts.get("count") or 0)
        ev = ts.get("eval") or {}
        psnr = ev.get("psnr")
        ssim = ev.get("ssim")
        train_ok = gauss >= 5000 and (psnr is None or float(psnr) >= 15.0) and gauss > 0
        entry.update({
            "gaussians": gauss, "psnr": psnr, "ssim": ssim,
            "trainOk": bool(train_ok), "trainSteps": args.iterations,
        })
        log(f"  训练：高斯 {gauss}，PSNR {psnr}，SSIM {ssim}，通过={train_ok}")
        entry.update({"ok": bool(train_ok), "reason": None if train_ok else "训练产物不可用"})
        report["rungs"].append(entry)
        _save(report, out_dir)

    _save(report, out_dir)
    _write_markdown(report, out_dir)
    log("\n" + "=" * 72)
    log(f"报告：{out_dir / 'threshold-report.json'} / threshold-report.md")
    log("=" * 72)
    return 0


def _duration_sec(video: Path) -> float:
    """用 ffprobe 不可用时退化为「按 1.1 MB/s 估算」；这里优先用 ffmpeg 解析。"""
    exe = config.find_ffmpeg()
    if exe:
        probe = Path(exe).with_name("ffprobe.exe")
        candidates = [probe, Path(exe)]
        for cand in candidates:
            if not cand.exists():
                continue
            try:
                out = subprocess.run(
                    [str(cand), "-hide_banner", "-i", str(video)],
                    capture_output=True, text=True, errors="ignore",
                ).stderr
            except Exception:  # noqa: BLE001
                continue
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("Duration:"):
                    hms = line.split("Duration:")[1].split(",")[0].strip()
                    try:
                        h, m, s = hms.split(":")
                        return int(h) * 3600 + int(m) * 60 + float(s)
                    except Exception:  # noqa: BLE001
                        pass
    return video.stat().st_size / (1.1 * 2**20)


def _save(report: dict, out_dir: Path) -> None:
    (out_dir / "threshold-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_markdown(report: dict, out_dir: Path) -> None:
    lines = [
        "# 建模所需最小素材体积 · 实测报告",
        "",
        f"> 生成时间：{report['generatedAt']}",
        f"> 源视频：{report['source']['name']}（{report['source']['mb']} MB）",
        f"> 参数：fps={report['params']['fps']}，长边={report['params']['longSide']}，"
        f"训练 {report['params']['iterations']} 步，downscale={report['params']['downscale']}",
        "",
        "## 判定标准",
        "",
        f"- COLMAP：注册率 ≥ {report['criteria']['colmapRegisteredRatio']:.0%} 且注册帧数 ≥ {report['criteria']['colmapRegisteredImages']}",
        f"- 训练：高斯数 ≥ {report['criteria']['trainMinGaussians']} 且 PSNR ≥ {report['criteria']['trainMinPsnrDb']} dB",
        "",
        "## 阶梯结果",
        "",
        "| 目标体积 | 实际体积 | 截断时长 | 可用帧 | 模糊剔除 | 注册帧 | 注册率 | 稀疏点 | 重投影误差 | 高斯数 | PSNR | SSIM | 结论 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report["rungs"]:
        lines.append(
            "| {t:g} MB | {a:.2f} MB | {s:.1f}s | {f} | {rej} | {reg} | {ratio} | {pts} | {err} | {g} | {p} | {ss} | {ok} |".format(
                t=r["targetMb"], a=r["clipMb"], s=r["clipSeconds"], f=r.get("frames", 0),
                rej=r.get("rejected", 0), reg=r.get("registeredImages", 0),
                ratio=(f"{r['registeredRatio']:.1%}" if r.get("registeredRatio") is not None else "—"),
                pts=r.get("points3D", 0),
                err=(f"{r['meanReprojectionError']:.3f}" if isinstance(r.get("meanReprojectionError"), (int, float)) else "—"),
                g=r.get("gaussians", "—"),
                p=(f"{r['psnr']:.2f}" if isinstance(r.get("psnr"), (int, float)) else "—"),
                ss=(f"{r['ssim']:.4f}" if isinstance(r.get("ssim"), (int, float)) else "—"),
                ok=("✅ 通过" if r.get("ok") else f"❌ {r.get('reason') or '失败'}"),
            )
        )
    out_dir.joinpath("threshold-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
