#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
归因：为什么留出视角的 PSNR 差这么多（实测 12.99 ~ 25.02 dB，差 12 dB）？

对每个留出视角同时算三类指标，再和**全部训练帧的分布**比：

  1. **清晰度**：源帧的拉普拉斯方差（越小人越糊 / 运动模糊）
  2. **曝光**：平均亮度与对比度（亮度跳变会让 3DGS 用「糊」去覆盖）
  3. **几何覆盖**：该视角视锥里落了多少个稀疏三维点、附近有多少个训练相机
     —— 这一项最要紧：视锥里点很少 = 这块区域**没被拍到过**，
     模型只能靠插值，必然是软的。

用法:
    python tools/diag_eval_views.py --job <recon 任务目录>
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "tools", ROOT / "pipeline"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import colmap_model_io as cmio  # noqa: E402


def laplacian_variance(gray: np.ndarray) -> float:
    """拉普拉斯方差：经典的对焦/清晰度指标（越大越清晰）。

    ⚠️ 必须用 **0~255 的 8 位灰度**而不是 0~1：
    `extract_frames.py` 的 `--blur-threshold 60` 就是在 8 位尺度上定的，
    用 0~1 算出的是 0.002 这种量级，`{:.1f}` 打印出来会全是 0.0（我踩过）。
    """
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    out = np.zeros((h - 2, w - 2), dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            if k[dy, dx]:
                out += k[dy, dx] * gray[dy : dy + h - 2, dx : dx + w - 2]
    return float(out.var())


def read_gray(path: Path) -> np.ndarray | None:
    """读成 8 位灰度（0~255），并把长边统一到 640 以消除分辨率影响。"""
    try:
        from PIL import Image

        img = Image.open(path).convert("L")
        if img.width > 640:
            img = img.resize((640, max(1, int(img.height * 640 / img.width))))
        return np.asarray(img, dtype=np.float64)
    except Exception:  # noqa: BLE001
        return None


def parse_eval_from_log(log_path: Path) -> dict[str, dict]:
    """从 pipeline.log 里抠出留出视角的 PSNR/SSIM/L1 与留出帧名。"""
    out: dict[str, dict] = {}
    if not log_path.is_file():
        return out
    pat = re.compile(r"视角\s+(\S+?):\s+PSNR\s+([\d.]+)\s+dB\s+SSIM\s+([\d.]+)\s+L1\s+([\d.]+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if m:
            out[m.group(1)] = {"psnr": float(m.group(2)), "ssim": float(m.group(3)), "l1": float(m.group(4))}
    return out


def frustum_counts(model: dict, cam_xyz: np.ndarray, names: list[str], xyz: np.ndarray) -> dict[str, dict]:
    """每个视角：视锥内的稀疏点数 + 附近训练相机数。"""
    images = {im["name"]: im for im in model["images"]}
    cams = model["cameras"]
    out: dict[str, dict] = {}
    for name in names:
        im = images.get(name)
        if im is None:
            out[name] = {"pointsInView": None, "camerasWithin1m": None, "camerasWithin2m": None}
            continue
        cam = cams.get(im["camera_id"])
        if cam is None:
            out[name] = {"pointsInView": None, "camerasWithin1m": None, "camerasWithin2m": None}
            continue
        R = cmio.quat_to_R(np.asarray(im["qvec"], float))
        t = np.asarray(im["tvec"], float)
        # intrinsics_of 返回的是 (fx, fy, cx, cy)；宽高在相机记录里
        fx, fy, cx, cy = cmio.intrinsics_of(cam)
        cw = float(cam.get("width") or 0) or cx * 2
        ch = float(cam.get("height") or 0) or cy * 2
        # 相机系：x 右、y 下、z 前
        p_cam = (R @ xyz.T).T + t
        z = p_cam[:, 2]
        ok = z > 1e-6
        if not ok.any():
            out[name] = {"pointsInView": 0, "camerasWithin1m": 0, "camerasWithin2m": 0}
            continue
        u = fx * p_cam[ok, 0] / z[ok] + cx
        v = fy * p_cam[ok, 1] / z[ok] + cy
        inside = (u >= 0) & (u < cw) & (v >= 0) & (v < ch) & (z[ok] < 50.0)
        c = cam_xyz[names.index(name)] if name in names else None
        near1 = near2 = None
        if c is not None and len(cam_xyz) > 1:
            d = np.linalg.norm(cam_xyz - c, axis=1)
            near1 = int(((d > 1e-6) & (d < 1.0)).sum())
            near2 = int(((d > 1e-6) & (d < 2.0)).sum())
        out[name] = {
            "pointsInView": int(inside.sum()),
            "camerasWithin1m": near1,
            "camerasWithin2m": near2,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="留出视角 PSNR 差异归因")
    ap.add_argument("--job", required=True, help="recon 任务目录（含 pipeline.log / images / colmap）")
    ap.add_argument("--json", default="", help="可选：把结果写成 JSON")
    args = ap.parse_args()

    job = Path(args.job)
    if not job.is_dir():
        print(f"[错误] 目录不存在：{job}")
        return 2
    log_path = job / "pipeline.log"
    evals = parse_eval_from_log(log_path)
    if not evals:
        print("[错误] 在 pipeline.log 里找不到留出视角的 PSNR 行")
        return 2

    sparse = job / "colmap" / "sparse" / "0"
    model = cmio.read_model(sparse)
    xyz = np.asarray(model["points3D"]["xyz"], dtype=float)
    err = np.asarray(model["points3D"]["error"], dtype=float)
    xyz = xyz[(err < 2.0) & np.isfinite(xyz).all(axis=1)]

    img_dir = job / "images"
    names = sorted(im["name"] for im in model["images"])
    cams = []
    for nm in names:
        im = next(i for i in model["images"] if i["name"] == nm)
        cams.append(cmio.camera_center(np.asarray(im["qvec"], float), np.asarray(im["tvec"], float)))
    cam_xyz = np.asarray(cams, dtype=float)

    # 全部训练帧的基线分布
    base_blur, base_lum = [], []
    all_files = {p.name: p for p in img_dir.glob("*.png")}
    for nm in names:
        g = read_gray(all_files[nm]) if nm in all_files else None
        if g is None:
            continue
        base_blur.append(laplacian_variance(g))
        base_lum.append(float(g.mean()))
    base_blur_a = np.asarray(base_blur)
    base_lum_a = np.asarray(base_lum)

    print("=" * 92)
    print("留出视角 PSNR 差异归因")
    print(f"  任务：{job}")
    print(f"  训练帧 {len(names)} 张；稀疏点 {len(xyz)} 个（已按重投影误差 <2px 过滤）")
    if len(base_blur_a):
        print(f"  全帧清晰度（拉普拉斯方差，8 位尺度）：中位 {np.median(base_blur_a):.1f}，"
              f"P10 {np.percentile(base_blur_a, 10):.1f}，P90 {np.percentile(base_blur_a, 90):.1f}"
              f"（抽帧阈值是 60，低于它的帧会被判为模糊）")
        print(f"  全帧平均亮度：中位 {np.median(base_lum_a):.1f}/255")
    print("-" * 92)
    print(f"{'视角':<18}{'PSNR':>7}{'SSIM':>8}{'清晰度':>10}{'清晰度分位':>12}{'亮度':>8}{'视锥内点':>10}{'1m内相机':>10}")
    print("-" * 92)

    rows = []
    for nm, ev in sorted(evals.items(), key=lambda kv: kv[1]["psnr"]):
        g = read_gray(all_files[nm]) if nm in all_files else None
        blur = laplacian_variance(g) if g is not None else None
        lum = float(g.mean()) if g is not None else None
        pct = None
        if blur is not None and len(base_blur_a):
            pct = float((base_blur_a < blur).mean() * 100)
        fr = frustum_counts(model, cam_xyz, [nm], xyz)[nm]
        rows.append({"name": nm, **ev, "blur": blur, "blurPercentile": pct, "luminance": lum, **fr})
        print(
            f"{nm:<18}{ev['psnr']:>7.2f}{ev['ssim']:>8.4f}"
            f"{(f'{blur:.1f}' if blur is not None else '—'):>10}"
            f"{(f'{pct:.0f}%' if pct is not None else '—'):>12}"
            f"{(f'{lum:.3f}' if lum is not None else '—'):>8}"
            f"{(fr['pointsInView'] if fr['pointsInView'] is not None else '—'):>10}"
            f"{(fr['camerasWithin1m'] if fr['camerasWithin1m'] is not None else '—'):>10}"
        )
    print("-" * 92)

    # 归因：把最差与最好的视角对比
    rows.sort(key=lambda r: r["psnr"])
    worst, best = rows[0], rows[-1]
    print(f"最差 {worst['name']}（{worst['psnr']:.2f} dB） vs 最好 {best['name']}（{best['psnr']:.2f} dB）")
    reasons = []
    if worst["blur"] and best["blur"] and worst["blur"] < best["blur"] * 0.6:
        reasons.append(f"清晰度差 {best['blur'] / max(worst['blur'], 1e-6):.1f}×（运动模糊/失焦）")
    if worst["pointsInView"] is not None and best["pointsInView"]:
        ratio = worst["pointsInView"] / max(1, best["pointsInView"])
        if ratio < 0.5:
            reasons.append(f"视锥内稀疏点只有最好视角的 {ratio:.0%}（{worst['pointsInView']} vs {best['pointsInView']}）→ 这块区域没被拍到")
    if worst["camerasWithin2m"] is not None and best["camerasWithin2m"] is not None:
        if worst["camerasWithin2m"] < best["camerasWithin2m"] * 0.5:
            reasons.append(f"附近训练相机少（{worst['camerasWithin2m']} vs {best['camerasWithin2m']}）→ 视角覆盖稀疏")
    if worst["luminance"] is not None and best["luminance"] is not None:
        if abs(worst["luminance"] - best["luminance"]) > 0.12:
            reasons.append(f"亮度相差 {abs(worst['luminance'] - best['luminance']):.3f}（曝光跳变）")
    if reasons:
        for r in reasons:
            print(f"  · {r}")
    else:
        print("  · 三项指标都没拉开差距 → 更像是位姿精度或全局欠拟合（增加迭代/提高分辨率）")

    if args.json:
        Path(args.json).write_text(json.dumps({"job": str(job), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写出 {args.json}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

