#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""探针：物体包围盒**什么时候可信**？（决定前端要不要画它）

为什么要先量（docs §九 第 4 条）
---------------------------------
真实现状：柜子那份素材算出来的物体盒是 **4.42 × 16.77 × 8.60 m**，
而相机群半径只有 3.53 m、相机到盒中心中位 **15.12 m** ——
一个柜子被算成 16 米长的东西，直接画出来就是重演"假房间盒"。

难点：我手上**只有坏样本**（柜子、run-32mb 都是"一次贴面拍"的退化素材），
没有一份"可信的物体重建"当正样本，于是**没法定阈值**。
所以这里用**合成真值**造正样本 —— `structure_object.build_object_model()`
是「点云 + 相机位姿」的纯函数，可以直接喂进去，不需要跑 COLMAP：

    正样本 A：1.2×0.6×1.9 m 的两层柜，相机绕半径 1.6~2.2 m 一圈，看向柜心
    正样本 B：2.5×1.0×2.2 m 的大柜，相机绕半径 4.0 m（物体大、机位也远）
    正样本 C：同 A，但加 15% 的背景野点（模拟真实重建里的房间点）
    负样本  ：真实柜子素材（recon-muwcu0a4-889pb）

量的量（都做成**尺度无关**的比值，因为 COLMAP 的尺度是任意的）：

    · 盒最长边 / 相机群半径
    · 盒中心到相机质心的距离 / 相机群半径      ← 最关键：相机是"围着物体看"，
                                                 所以盒心必须靠近相机质心
    · 相机群半径内所包含的点占比（2×半径内占比）
    · 点云稳健跨度(P99) / 相机群半径
    · 水平层的突出度与承载点数

用法：
    pyruntime312\\python.exe tools/probe_object_trust.py
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
from structure_object import build_object_model  # noqa: E402


def cabinet_points(size, *, shelves=None, n_per_face=1200, seed=7) -> np.ndarray:
    """造一个"两层柜"的表面点：三块水平面（顶/中/底）+ 背板 + 两块侧板。

    `shelves` 默认按**高度比例**取（0 / h·0.5 / h）。为什么不能写死：
    第一版写死成 (0, 0.95, 1.9)，给一个 2.2 m 高的柜子用 1.9 m 的搁板 ——
    真值口径自己就自相矛盾了（测试里表现为"高度差 5 cm"这种看不懂的误差）。
    """
    rng = np.random.default_rng(seed)
    w, d, h = size
    if shelves is None:
        shelves = (0.0, h * 0.5, h)
    pts = []
    # 水平面（顶/中搁板/底）：两层柜的结构信息就在这里
    for z in shelves:
        n = n_per_face
        pts.append(np.stack([rng.uniform(-w / 2, w / 2, n), rng.uniform(-d / 2, d / 2, n),
                             np.full(n, z)], axis=1))
    # 背板：x ∈ [-w/2, w/2]，y = -d/2，z ∈ [0, h]
    # ⚠️ 第一版这里把轴搞混了（侧板被写成 z ∈ [-w/2, w/2]），于是"真值"自己就越过了地面，
    #    看起来像"生产代码算错了高度"、其实是**测试数据错**。
    #    教训：合成真值的生成器要能被一句话讲清"每个点落在哪个面上"。
    n = n_per_face
    pts.append(np.stack([rng.uniform(-w / 2, w / 2, n),
                         np.full(n, -d / 2),
                         rng.uniform(0, h, n)], axis=1))
    for sx in (-1, 1):
        pts.append(np.stack([np.full(n, sx * w / 2), rng.uniform(-d / 2, d / 2, n),
                             rng.uniform(0, h, n)], axis=1))
    allp = np.concatenate(pts, axis=0)
    return allp + rng.normal(0, 0.004, allp.shape)   # 一点点噪声


def orbit_cameras(center, radius, *, n=60, h_lo=-0.15, h_hi=0.25, seed=11):
    """相机绕物体一圈（带轻微高度抖动），全部看向物体中心。"""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.normal(0, 0.05, n)
    r = radius * (1 + rng.normal(0, 0.08, n))
    z = rng.uniform(h_lo, h_hi, n)
    c = np.stack([np.cos(ang) * r, np.sin(ang) * r, z], axis=1) + np.asarray(center, float)
    fwd = np.asarray(center, float)[None, :] - c
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    return c, fwd


def metrics(P: np.ndarray, C: np.ndarray, F: np.ndarray, *, tag: str, n_up: np.ndarray) -> dict:
    """把 build_object_model 跑一遍，并算出所有候选判据。"""
    u, v = gc.orthonormal_basis(n_up)
    # 物体模式锚点 = 相机质心（与 build_object_model 一致）
    anchor = C.mean(axis=0)
    obj, layers, lev = build_object_model(
        P, C, u=u, v=v, n_up=n_up, floor_pt=anchor, scale=float(np.percentile(np.linalg.norm(P - np.median(P, axis=0), axis=1), 95)),
        mode="object", layers=[], layers_ev={},
    )
    cam_r = float(np.median(np.linalg.norm(C - anchor, axis=1)))
    out = {"tag": tag, "nPoints": int(len(P)), "nCams": int(len(C)), "camRadius": round(cam_r, 3)}
    if obj is None:
        out["ok"] = False
        return out
    size = np.asarray(obj["size"], float)
    center = np.asarray(obj["center"], float)
    d_center = float(np.linalg.norm(center - anchor))
    dp = np.linalg.norm(P - anchor, axis=1)
    med = np.median(P, axis=0)
    span = float(np.percentile(np.linalg.norm(P - med, axis=1), 99))
    out.update({
        "ok": True,
        "size": [round(float(x), 2) for x in size],
        "longestEdgeOverCamR": round(float(size.max() / max(cam_r, 1e-9)), 2),
        "centerOffsetOverCamR": round(d_center / max(cam_r, 1e-9), 2),
        "fracPtsWithin2CamR": round(float((dp <= 2 * cam_r).mean()), 3),
        "fracPtsWithin1CamR": round(float((dp <= cam_r).mean()), 3),
        "cloudSpanP99OverCamR": round(span / max(cam_r, 1e-9), 2),
        "nLayers": len(layers),
        "layerProminence": [l.get("prominence") for l in layers],
        "layerPoints": [l.get("points") for l in layers],
        "layerRatio": [l.get("ratio") for l in layers],
    })
    return out


def real_case(model_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = cmio.read_model(model_dir)
    images = sorted(m["images"], key=lambda i: i["name"])
    C, F = gc.camera_centers_and_forwards(images)
    g_down, _ = gc.gravity_from_cameras(images)      # 与生产同源
    xyz = np.asarray(m["points3D"]["xyz"], float)
    err = np.asarray(m["points3D"]["error"], float)
    trk = np.asarray(m["points3D"]["track_length"], float)
    keep = np.isfinite(xyz).all(axis=1) & (err < 2.0) & (trk >= 3)
    P = xyz[keep]
    P = P[gc.drop_statistical_outliers(P)]
    return P, C, F, -g_down


def main() -> int:
    rows = []

    # ---- 正样本 A/B/C：合成真值 ----------------------------------------
    for tag, size, radius, bg in (
        ("合成A 小柜 1.2×0.6×1.9 绕1.9m", (1.2, 0.6, 1.9), 1.9, 0.0),
        ("合成B 大柜 2.5×1.0×2.2 绕4.0m", (2.5, 1.0, 2.2), 4.0, 0.0),
        ("合成C 小柜+15%背景野点", (1.2, 0.6, 1.9), 1.9, 0.15),
    ):
        P = cabinet_points(size)
        if bg > 0:
            rng = np.random.default_rng(3)
            n_bg = int(len(P) * bg / (1 - bg))
            # 背景点：撒在 8~14 m 外（模拟房间/远处结构，正是真实素材里撑大盒子的那批）
            r = rng.uniform(8, 14, n_bg)
            dirv = rng.normal(size=(n_bg, 3))
            dirv /= np.linalg.norm(dirv, axis=1, keepdims=True)
            P = np.concatenate([P, dirv * r[:, None]], axis=0)
        C, F = orbit_cameras((0, 0, size[2] / 2), radius)
        rows.append(metrics(P, C, F, tag=tag, n_up=np.array([0.0, 0.0, 1.0])))

    # ---- 负样本：真实柜子 ----------------------------------------------
    P, C, F, n_up_real = real_case(Path(r"data\projects\case-20261003\recon\recon-muwcu0a4-889pb\colmap\sparse\0"))
    rows.append(metrics(P, C, F, tag="真实 柜子（一次贴面拍）", n_up=n_up_real))

    # ---- 打印 -----------------------------------------------------------
    print(f"{'样本':<34}{'点数':>7}{'相机':>5}{'相机半径':>9}{'最长边/R':>9}{'盒心偏移/R':>11}{'≤2R点占比':>10}{'跨度/R':>8}{'层':>3}")
    print("-" * 108)
    for r in rows:
        if not r.get("ok"):
            print(f"{r['tag']:<34}{r['nPoints']:>7}{r['nCams']:>5}{r['camRadius']:>9.2f}   （算不出物体盒）")
            continue
        print(f"{r['tag']:<34}{r['nPoints']:>7}{r['nCams']:>5}{r['camRadius']:>9.2f}"
              f"{r['longestEdgeOverCamR']:>9.2f}{r['centerOffsetOverCamR']:>11.2f}"
              f"{r['fracPtsWithin2CamR']:>10.2f}{r['cloudSpanP99OverCamR']:>8.2f}{r['nLayers']:>3}")
    print("\n各样本物体盒尺寸：")
    for r in rows:
        if r.get("ok"):
            print(f"  {r['tag']:<34} size={r['size']}  层突出度={r['layerProminence']} 层点数={r['layerPoints']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
