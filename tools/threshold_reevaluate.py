#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把多次「最小素材体积」实测的原始报告**重新按最终判据**汇总成一张表。

为什么需要它：
  `threshold_experiment.py` 是边跑边写结论的，判据在实验过程中被修正过
  （例如后来发现「稀疏点云数」比 PSNR 更能分辨可用/不可用）。
  这个脚本只做**离线重算**，不改动原始数据，保证文档里的表与原始测量一一对应。

用法：
    python tools/threshold_reevaluate.py --reports output/threshold/threshold-report.json `
        output/threshold-retest/threshold-report.json output/threshold-user/threshold-report.json `
        --out output/threshold/summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 最终判据（与 threshold_experiment.py 保持一致）
CRITERIA = {
    "colmapRegisteredRatio": 0.60,
    "colmapRegisteredImages": 12,
    "sparseMinPoints": 10000,
    "trainMinGaussians": 5000,
    "trainMinPsnrDb": 15.0,
}


def verdict(r: dict) -> tuple[str, str]:
    """返回 (符号+结论, 原因)。"""
    frames = int(r.get("frames") or 0)
    if frames < 8:
        return "❌ 崩", f"可用帧 {frames} < 8（流水线直接判素材不足）"
    ratio = r.get("registeredRatio")
    reg = int(r.get("registeredImages") or 0)
    if ratio is None or ratio < CRITERIA["colmapRegisteredRatio"] or reg < CRITERIA["colmapRegisteredImages"]:
        return "❌ 崩", f"COLMAP 注册率只有 {(ratio or 0):.1%}（{reg} 帧）"
    pts = int(r.get("points3D") or 0)
    if pts < CRITERIA["sparseMinPoints"]:
        return "⚠️ 勉强", f"注册成功但稀疏点只有 {pts} 个（< {CRITERIA['sparseMinPoints']}），模型缺面/发糊"
    gauss = int(r.get("gaussians") or 0)
    psnr = r.get("psnr")
    if gauss < CRITERIA["trainMinGaussians"]:
        return "❌ 崩", f"训练后高斯只有 {gauss} 个"
    if psnr is None:
        return "❌ 崩", "训练未产出可用 PLY"
    if float(psnr) < CRITERIA["trainMinPsnrDb"]:
        return "❌ 崩", f"PSNR 只有 {psnr} dB（模型没学出来）"
    return "✅ 可用", ""


def fmt(v, digits=2, dash="—"):
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}"
    return dash


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = []
    for rp in args.reports:
        p = Path(rp)
        if not p.exists():
            print(f"[跳过] 找不到 {p}")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        src = data.get("source", {}).get("name", "?")
        for r in data.get("rungs", []):
            mark, why = verdict(r)
            rows.append(
                {
                    "source": src,
                    "mb": r.get("clipMb"),
                    "seconds": r.get("clipSeconds"),
                    "frames": r.get("frames"),
                    "rejected": r.get("rejected"),
                    "reg": r.get("registeredImages"),
                    "ratio": r.get("registeredRatio"),
                    "points": r.get("points3D"),
                    "reproj": r.get("meanReprojectionError"),
                    "gauss": r.get("gaussians"),
                    "psnr": r.get("psnr"),
                    "ssim": r.get("ssim"),
                    "mark": mark,
                    "why": why,
                }
            )

    rows.sort(key=lambda x: (x["source"], x["mb"] or 0))

    lines = [
        "| 源素材 | 体积 | 时长 | 帧 | 注册帧 | 注册率 | **稀疏点** | 重投影误差 | 高斯数 | PSNR | SSIM | 结论 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        pts = r["points"]
        pts_cell = f"**{pts}**" if isinstance(pts, int) and pts and pts >= CRITERIA["sparseMinPoints"] else str(pts if pts is not None else "—")
        lines.append(
            "| {src} | {mb} MB | {sec}s | {f} | {reg} | {ratio} | {pts} | {err} | {g} | {psnr} | {ssim} | {mark}{why} |".format(
                src=r["source"],
                mb=fmt(r["mb"]),
                sec=fmt(r["seconds"], 1),
                f=r["frames"] if r["frames"] is not None else "—",
                reg=r["reg"] if r["reg"] is not None else "—",
                ratio=f"{r['ratio']:.1%}" if isinstance(r["ratio"], (int, float)) else "—",
                pts=pts_cell,
                err=fmt(r["reproj"], 3),
                g=r["gauss"] if r["gauss"] else "—",
                psnr=fmt(r["psnr"]),
                ssim=fmt(r["ssim"], 4),
                mark=r["mark"],
                why=("" if not r["why"] else f"（{r['why']}）"),
            )
        )

    table = "\n".join(lines)
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n", encoding="utf-8")
        print(f"\n已写入 {out}")

    usable = [r for r in rows if r["mark"] == "✅ 可用"]
    if usable:
        print(f"\n可用档位最小体积：{min(r['mb'] for r in usable):.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
