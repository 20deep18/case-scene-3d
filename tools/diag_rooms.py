#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断：房间分割为什么切不出格子 —— 把墙在「对齐后坐标系」里的朝向与切点打印出来。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for e in (ROOT / "tools", ROOT / "pipeline"):
    if str(e) not in sys.path:
        sys.path.insert(0, str(e))
import colmap_model_io as cmio  # noqa: E402
import scene_structure as ss  # noqa: E402


def diag(model_dir: Path) -> None:
    model = cmio.read_model(model_dir)
    xyz = np.asarray(model["points3D"]["xyz"], float)
    err = np.asarray(model["points3D"]["error"], float)
    trk = np.asarray(model["points3D"]["track_length"], float)
    keep = np.isfinite(xyz).all(axis=1) & (err < 2.0) & (trk >= 3)
    P = xyz[keep]
    frame = ss.robust_frame(P)
    scale = frame["p95"]
    eps = max(0.015 * scale, 0.01)

    imgs = sorted(model["images"], key=lambda im: im["name"])
    cams = np.asarray([cmio.camera_center(np.asarray(i["qvec"], float), np.asarray(i["tvec"], float)) for i in imgs])
    g_down, _ = ss.gravity_from_cameras(imgs)
    n_up = -g_down
    u, v = ss.orthonormal_basis(n_up)

    rel = P - frame["center"]
    hh = rel @ n_up
    h_cam = float(np.median((cams - frame["center"]) @ n_up))
    below = hh[hh < h_cam - 0.1]
    h_floor, _ = ss.density_peak(below, float(np.percentile(below, 0.5)), h_cam - 0.1, max(0.03, 0.01 * scale))
    if not (0.8 <= h_cam - h_floor <= 2.4):
        h_floor = h_cam - 1.5
    floor_pt = frame["center"] + n_up * h_floor
    hh2 = (P - floor_pt) @ n_up
    h_ceil = h_floor + 2.6
    height = min(2.6, 2.6)

    band = (hh2 > 0.2) & (hh2 < height - 0.2)
    wall_pool = P[band]
    rng = np.random.default_rng(20261005)
    walls = []
    remaining = np.ones(len(wall_pool), bool)
    for _ in range(8):
        sub = wall_pool[remaining]
        if len(sub) < 80:
            break
        w = ss.ransac_plane(sub, eps=eps, iters=1500, rng=rng, perp_to=n_up, min_perp_deg=72.0)
        if w is None or w["count"] < max(80, 0.012 * len(P)):
            break
        idx = np.nonzero(remaining)[0][w["inliers"]]
        m = np.zeros(len(wall_pool), bool)
        m[idx] = True
        w["inliers"] = m
        walls.append(w)
        remaining = remaining & ~m

    print(f"\n=== {model_dir.parent.parent.name} ===")
    print(f"检测到 {len(walls)} 面墙（法向与重力垂直，容差 72°）")
    # ⚠️ 2026-10-06：`align_frame_to_walls` 的 n_up 已从模块级全局改成**显式参数**
    # （拆分模块后隐式全局依赖会变成"导入顺序一错就算错"的坑）。
    a_axis, b_axis, theta = ss.align_frame_to_walls(u, v, walls, n_up)
    print(f"对齐旋转 = {theta:.1f}°")
    print(f"{'#':>2} {'点':>7} {'|n·a|':>8} {'|n·b|':>8} {'切点a':>9} {'切点b':>9}  归属")
    a_cuts, b_cuts = [], []
    for i, w in enumerate(walls):
        n = np.asarray(w["normal"], float)
        na, nb = abs(float(n @ a_axis)), abs(float(n @ b_axis))
        c = n * float(w.get("d", 0.0))
        ca, cb = float(c @ a_axis), float(c @ b_axis)
        if na >= nb:
            own = "A族(切a)" if na > 0.94 else f"斜墙({na:.2f})"
            if na > 0.94:
                a_cuts.append(ca)
        else:
            own = "B族(切b)" if nb > 0.94 else f"斜墙({nb:.2f})"
            if nb > 0.94:
                b_cuts.append(cb)
        print(f"{i:>2} {w['count']:>7} {na:>8.3f} {nb:>8.3f} {ca:>9.2f} {cb:>9.2f}  {own}")
    relf = P - floor_pt
    uu, vv = relf @ a_axis, relf @ b_axis
    print(f"点云在 a 方向 {np.percentile(uu,1):.2f} ~ {np.percentile(uu,99):.2f} m（跨度 {np.percentile(uu,99)-np.percentile(uu,1):.2f}）")
    print(f"点云在 b 方向 {np.percentile(vv,1):.2f} ~ {np.percentile(vv,99):.2f} m（跨度 {np.percentile(vv,99)-np.percentile(vv,1):.2f}）")
    print(f"A 族切点（排序去重后）: {sorted(round(x,2) for x in a_cuts)}")
    print(f"B 族切点（排序去重后）: {sorted(round(x,2) for x in b_cuts)}")
    print(f"→ 严格阈值(0.94)下可用切点：A {len(a_cuts)} 个、B {len(b_cuts)} 个"
          f"（切开成 {len(a_cuts)+1} × {len(b_cuts)+1} 张网）")


if __name__ == "__main__":
    for md in ["data/projects/case-20261003/recon/recon-muuw12xi-8pzis/colmap/sparse/0",
               "data/scene-big/colmap-ex/sparse/0"]:
        diag(ROOT / md)
