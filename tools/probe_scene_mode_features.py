#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""探针：给多个 COLMAP 模型算一批**候选的"房间/物体"判据特征**，用真实数据挑判据。

为什么需要这个脚本
------------------
交接文档提议用「视线汇聚度」当新判据。我先把它在**真实素材**上量了一遍
（`tools/probe_ray_convergence.py`），结果是它在这份柜子素材上**完全失效**：

    柜子现场（130 台相机）  minSpreadRel = 1.0000   ← 最小值落在 d=0，等于"完全不汇聚"
    room-real（178 台相机） minSpreadRel = 0.9652
    两者几乎不可分，而且柜子被判成了"房间"（正好判反）

原因看素材就明白了：那段柜子视频是**贴着柜子近距离巡拍**（相机就在柜门前 0.4~1 m，
左右摆动着拍隔板、纸箱、柜门内侧），既不是"绕着物体转"（视线汇聚），
也不是"在房间里走"。所以判据必须换一个能区分**近距离贴拍**与**房间走动**的量。

本脚本量的是「相机到最近重建点云的距离」这一类**局部**统计量：
近距离贴拍 → 相机紧贴表面（dNear 小）；房间走动 → 相机在房间中央，到墙有 1.5~4 m。
关键是它**局部**，不受"点云里混了背景、整体跨度被撑大"的影响
（交接文档已论证整体尺度判据不可靠）。

用法：
    pyruntime312\\python.exe tools\\probe_scene_mode_features.py <模型目录> [更多目录...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "tools", ROOT / "pipeline"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import colmap_model_io as cmio  # noqa: E402
import geom_common as gc  # noqa: E402

try:
    from scipy.spatial import cKDTree
except Exception:  # noqa: BLE001
    cKDTree = None


def load(model_dir: Path) -> dict:
    model = cmio.read_model(model_dir)
    # ⚠️ 相机位姿走**生产代码**（geom_common），不再自己算一遍：
    #    两份实现只要有一处不同（比如前向符号），探针结论就与线上相反。
    #    交接文档里那个「-Z 是观察方向」的错误就属于这类。
    C, F = gc.camera_centers_and_forwards(sorted(model["images"], key=lambda im: im["name"]))

    xyz = np.asarray(model["points3D"]["xyz"], float)
    err = np.asarray(model["points3D"]["error"], float)
    trk = np.asarray(model["points3D"]["track_length"], float)
    keep = np.isfinite(xyz).all(axis=1) & (err < 2.0) & (trk >= 3)
    return {"C": C, "F": F, "P": xyz[keep], "nAll": int(len(xyz)), "cams": model["cameras"], "images": model["images"]}


def features(d: dict) -> dict:
    C, F, P = d["C"], d["F"], d["P"]
    out: dict = {"nCams": int(len(C)), "nPoints": int(len(P))}
    if len(C) < 4 or len(P) < 50:
        out["ok"] = False
        return out
    out["ok"] = True

    centroid = C.mean(axis=0)
    dc = np.linalg.norm(C - centroid, axis=1)
    R_cam = float(np.median(dc))
    out["camRadius"] = R_cam
    out["camSpanP99"] = float(np.percentile(dc, 99))

    # 相机轨迹（按帧名顺序）
    step = np.linalg.norm(np.diff(C, axis=0), axis=1)
    out["camPathLen"] = float(step.sum())
    out["camNetDisp"] = float(np.linalg.norm(C[-1] - C[0]))
    out["camStepMedian"] = float(np.median(step)) if len(step) else 0.0
    # 相邻帧转动角（度）：近距离贴拍靠"转头"扫视，房间走动靠"位移"
    cosang = np.clip(np.sum(F[:-1] * F[1:], axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    out["camTurnMedianDeg"] = float(np.median(ang)) if len(ang) else 0.0
    # 转动 / 位移：贴拍应该远大于 1（deg per meter 归一化后再看）
    out["turnPerMeter"] = float(out["camTurnMedianDeg"] / max(out["camStepMedian"], 1e-6))

    # 点云尺度（只作参考，判据不用它 —— 交接文档已论证不可靠）
    pc = np.median(P, axis=0)
    dp = np.linalg.norm(P - pc, axis=1)
    out["cloudSpanP99"] = float(np.percentile(dp, 99))
    out["cloudSpanP50"] = float(np.median(dp))
    out["cloudOverCam"] = float(out["cloudSpanP99"] / max(R_cam, 1e-9))

    if cKDTree is None:
        return out
    tree = cKDTree(P)
    # ---- 特征 A：相机到最近重建点的距离（局部量，抗背景野点）----
    dNear, _ = tree.query(C, k=1)
    out["dNearMedian"] = float(np.median(dNear))
    out["dNearP90"] = float(np.percentile(dNear, 90))
    out["dNearOverCamR"] = float(np.median(dNear) / max(R_cam, 1e-9))

    # ---- 特征 B：每台相机"视线第一层表面"的距离（视锥内点的低分位）----
    first = []
    for i in range(len(C)):
        rel = P - C[i]
        dist = np.linalg.norm(rel, axis=1)
        z = rel @ F[i]
        ok = z > 1e-6
        if not ok.any():
            continue
        # 只取带一定前向夹角（30° 内）的点，近似"正对着的表面"
        cosf = np.clip(z[ok] / np.maximum(dist[ok], 1e-9), -1.0, 1.0)
        sel = cosf > 0.87
        if sel.sum() < 5:
            continue
        first.append(float(np.percentile(dist[ok][sel], 10)))
    if first:
        out["dSurfaceMedian"] = float(np.median(first))
        out["dSurfaceOverCamR"] = float(np.median(first) / max(R_cam, 1e-9))

    # ---- 特征 C：相机"嵌在点云里"的程度（房间：相机四周都有点；贴拍：只有前面有）----
    around = tree.query_ball_point(C, r=0.5 * R_cam, return_length=True)
    out["ptsWithinHalfCamR"] = float(np.median(around))

    # ---- 特征 D：**方向包围度** —— 重建点在相机周围占了多少方向的立体角 -------------
    #
    # 这是最符合物理直觉、又完全**尺度无关**的判据：
    #   · 房间：相机站在屋子中间，前后左右上下都是重建出来的表面 → 方向几乎铺满
    #   · 贴着一个物体拍（柜子）：重建点只在正前方一个锥里 → 大面积方向是空的
    # 它只看**方向**，不看距离，所以不受"点云整体跨度被背景撑大"的影响。
    #
    # ⚠️ 2026-10-06：**改成直接调用生产代码** `geom_common.camera_surround_coverage()`。
    #    原来探针自己实现了一份（Fibonacci 方向 + 最近 K 点）——两份实现只要有一处
    #    参数不同（K、容差角、方向数），"探针量出来的阈值"就和"线上真正用的数"不是一回事，
    #    而这种漂移**不会有任何报错**，只会让调出来的阈值悄悄失效。
    #    现在探针与 scene_structure 用的是同一个函数、同一套默认参数。
    cov_info = gc.camera_surround_coverage(C, P)
    if cov_info.get("ok"):
        # 同时保留"最近 K 点"口径的旧数值，便于与历史结论对照
        K = min(1000, len(P))
        _, idx = tree.query(C, k=K)
        out["sphereCoverageMedian"] = cov_info["median"]
        out["sphereCoverageP10"] = cov_info["p10"]
        out["sphereCoverageSource"] = "geom_common.camera_surround_coverage"
        out["sphereCoverageParams"] = {"k": cov_info["k"], "directions": cov_info["directions"],
                                       "toleranceDeg": cov_info["toleranceDeg"]}
    else:
        out["sphereCoverageMedian"] = None
        out["sphereCoverageP10"] = None
        out["sphereCoverageSource"] = f"不可用：{cov_info.get('reason')}"

    # ---- 特征 E：视锥内点的"深度跨度 / 中位深度"（低视差+贴拍会异常大）----
    return out


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    rows = []
    for a in args:
        p = Path(a)
        if not p.is_dir():
            print(f"[跳过] {p}")
            continue
        try:
            f = features(load(p))
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {p}: {type(exc).__name__}: {exc}")
            continue
        f["dir"] = str(p)
        rows.append(f)
        print("=" * 78)
        print(p)
        if not f.get("ok"):
            print("  ✗ 数据太少，无法计算")
            continue
        print(f"  相机 {f['nCams']} 台   稀疏点 {f['nPoints']} 个")
        print(f"  相机群半径 R_cam      = {f['camRadius']:.3f} m   (P99 {f['camSpanP99']:.3f})")
        print(f"  轨迹 总长/净位移       = {f['camPathLen']:.2f} / {f['camNetDisp']:.2f} m")
        print(f"  相邻帧 位移/转角       = {f['camStepMedian']:.4f} m / {f['camTurnMedianDeg']:.2f}°"
              f"   → 转动每米 {f['turnPerMeter']:.1f}°")
        print(f"  点云 稳健跨度(P99)/中位 = {f['cloudSpanP99']:.2f} / {f['cloudSpanP50']:.2f} m"
              f"   点云跨度/相机半径 = {f['cloudOverCam']:.2f}")
        if "dNearMedian" in f:
            print(f"  ★ 相机到最近点 中位     = {f['dNearMedian']:.3f} m  (P90 {f['dNearP90']:.3f})"
                  f"   /R_cam = {f['dNearOverCamR']:.4f}")
            print(f"  ★ 视线正前方表面距离 中位 = {f.get('dSurfaceMedian', float('nan')):.3f} m"
                  f"   /R_cam = {f.get('dSurfaceOverCamR', float('nan')):.4f}")
            print(f"  ★ 半径 0.5·R_cam 球内的点数 中位 = {f['ptsWithinHalfCamR']:.0f}")
            print(f"  ★★ 方向包围度 中位            = {f['sphereCoverageMedian']:.4f}"
                  f"  (P10 {f['sphereCoverageP10']:.4f})   ← 房间应接近 1，贴拍会很低")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
