#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量出「点云自身的特征尺度」，作为训练器尺度上限的依据。

为什么不用 spatial_scale（相机包围球半径×1.1）：
    它是**相机轨迹**的尺度，不是**表面**的尺度。某任务 spatial_scale = 11.185 m，
    上限因此 = 0.1 × 11.185 = 1.1185 m；而实测健康模型的尺度中位只有 0.05~0.21 m。
    挂在相机尺度上等于给了 8 倍的余量，优化器必然把大量高斯顶到上限 →
    又大又白又淡的雾团。

这里算出两个候选口径：
    nn_median  最近邻距离中位数（表面特征尺度，最贴近「一个高斯该多大」）
    bbox_diag  包围盒对角线（场景整体尺度）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for e in (ROOT / "tools", ROOT / "pipeline"):
    if str(e) not in sys.path:
        sys.path.insert(0, str(e))
import colmap_model_io as cmio  # noqa: E402


def load_points(model_dir: Path) -> np.ndarray:
    model = cmio.read_model(model_dir)
    xyz = np.asarray(model["points3D"]["xyz"], float)
    err = np.asarray(model["points3D"]["error"], float)
    trk = np.asarray(model["points3D"]["track_length"], float)
    keep = np.isfinite(xyz).all(axis=1) & (err < 2.0) & (trk >= 3)
    return xyz[keep]


def nn_median(P: np.ndarray, sample: int = 2000, seed: int = 7) -> float:
    """随机子采样的最近邻距离中位数（避免 N² 全量）。"""
    rng = np.random.default_rng(seed)
    n = len(P)
    m = min(sample, n)
    sub = P[rng.choice(n, size=m, replace=False)]
    # 分块算距离，控制内存
    best = np.full(m, np.inf)
    for i in range(0, m, 500):
        blk = sub[i:i + 500]
        d = np.linalg.norm(blk[:, None, :] - sub[None, :, :], axis=2)
        d[np.arange(len(blk)), np.arange(i, i + len(blk))] = np.inf
        best[i:i + len(blk)] = d.min(axis=1)
    return float(np.median(best))


CASES = [
    ("发雾（小视频 17 帧）", "data/projects/case-20261003/recon/recon-muvf7pbo-k9nz1/colmap/sparse/0"),
    ("健康 room-real（178 帧）", "data/scene-big/colmap-ex/sparse/0"),
]


def main() -> int:
    print(f"{'数据集':<26}{'点数':>8}{'nn中位':>10}{'包围盒对角':>12}{'nn×4':>9}{'nn×6':>9}{'extent×2%':>11}")
    print("-" * 92)
    for label, rel in CASES:
        d = ROOT / rel
        if not d.is_dir():
            print(f"{label:<26}  缺少 {rel}")
            continue
        P = load_points(d)
        diag = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
        nn = nn_median(P)
        print(
            f"{label:<26}{len(P):>8}{nn:>10.4f}{diag:>12.2f}"
            f"{nn*4:>9.3f}{nn*6:>9.3f}{diag*0.02:>11.3f}"
        )
    print("\n参考：健康模型实际尺度中位 0.138 m / P95 0.334 m；发雾模型中位 0.59 m")
    print("目标：上限要 ≥ 健康模型的 P95（0.334），同时明显低于 1.1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
