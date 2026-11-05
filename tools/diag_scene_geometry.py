#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断图：把「相机位姿 / 视线 / 稀疏点云」的关系画出来，并做重投影核对。

为什么需要：判断"房间 vs 物体"的判据必须建立在**真实几何**上。
只算标量很容易自我说服（我就差点按交接文档的"视线汇聚度"直接改代码，
量完才发现它在真实的柜子素材上判反）。看图能一眼看出相机是在房间里
走动、还是贴着某个东西在拍，以及点云到底有没有覆盖相机正对的那个面。

四个面板：
  (a) 俯视：点云 + 相机位置 + 每台相机的视线（沿重力对齐的坐标系）
  (b) 侧视：水平距离 vs 高度，同样叠加
  (c) 相机到最近重建点的距离分布
  (d) 抽一帧把稀疏点云重投影上去，核对"点云是否落在画面内容上"

用法：
    $env:MPLCONFIGDIR=".buildtmp\mpl"
    pyruntime312\\python.exe tools\\diag_scene_geometry.py --model <稀疏模型目录> --out <png> [--frame 1]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "tools", ROOT / "pipeline"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import colmap_model_io as cmio  # noqa: E402


def gravity_from_cameras(images: list[dict]) -> np.ndarray:
    downs = []
    for im in images:
        try:
            R = cmio.quat_to_R(np.asarray(im["qvec"], float))
        except Exception:  # noqa: BLE001
            continue
        downs.append(R.T @ np.array([0.0, 1.0, 0.0]))
    if not downs:
        return np.array([0.0, 1.0, 0.0])
    m = np.asarray(downs).mean(axis=0)
    n = float(np.linalg.norm(m))
    return m / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])


def robust_clean(P: np.ndarray, k: int = 2) -> np.ndarray:
    """统计式洗点：点到第 k 近邻距离过大的丢掉（与 scene_structure 同思路的简化版）。"""
    from scipy.spatial import cKDTree

    if len(P) < 50:
        return P
    t = cKDTree(P)
    d, _ = t.query(P, k=min(k + 1, len(P)))
    dk = d[:, -1]
    med = np.median(dk)
    mad = np.median(np.abs(dk - med)) + 1e-12
    return P[dk <= med + 6.0 * 1.4826 * mad]


def load(model_dir: Path):
    model = cmio.read_model(model_dir)
    ims = sorted(model["images"], key=lambda im: im["name"])
    C, F = [], []
    for im in ims:
        q = np.asarray(im["qvec"], float)
        t = np.asarray(im["tvec"], float)
        R = cmio.quat_to_R(q)
        C.append(cmio.camera_center(q, t))
        F.append(R.T @ np.array([0.0, 0.0, 1.0]))
    C = np.asarray(C, float)
    F = np.asarray(F, float)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12

    xyz = np.asarray(model["points3D"]["xyz"], float)
    err = np.asarray(model["points3D"]["error"], float)
    trk = np.asarray(model["points3D"]["track_length"], float)
    keep = np.isfinite(xyz).all(axis=1) & (err < 2.0) & (trk >= 3)
    P_raw = xyz[keep]
    P = robust_clean(P_raw)
    return model, ims, C, F, P_raw, P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--images", default=None, help="原图目录（做重投影核对用）")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, ims, C, F, P_raw, P = load(Path(args.model))
    g_down = gravity_from_cameras(ims)
    up = -g_down

    # 水平基
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(float(tmp @ up)) > 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(up, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    origin = np.median(P, axis=0) if len(P) else C.mean(axis=0)
    Pc = P - origin
    Cc = C - origin
    pu, pv, ph = Pc @ e1, Pc @ e2, Pc @ up
    cu, cv, ch = Cc @ e1, Cc @ e2, Cc @ up
    fu, fv, fh = F @ e1, F @ e2, F @ up

    from scipy.spatial import cKDTree
    dNear, _ = cKDTree(P).query(C, k=1)
    dNearRaw, _ = cKDTree(P_raw).query(C, k=1)

    fig, ax = plt.subplots(2, 2, figsize=(15, 12))

    # (a) 俯视
    a = ax[0][0]
    a.scatter(pu, pv, s=0.4, c="#9aa4b2", alpha=0.45, linewidths=0)
    a.scatter(cu, cv, s=12, c="#d92b2b", zorder=3)
    L = max(np.percentile(np.linalg.norm(Cc, axis=1), 90), 1e-6) * 0.6
    a.quiver(cu, cv, fu, fv, color="#1f6feb", angles="xy", scale=1.0 / L, width=0.003, zorder=4)
    a.set_title(f"(a) 俯视（重力对齐）\n相机 {len(C)} 台（红），蓝色短线=视线方向")
    a.set_aspect("equal")
    a.grid(alpha=0.25)
    a.set_xlabel("水平轴 1 (m)")
    a.set_ylabel("水平轴 2 (m)")

    # (b) 侧视
    b = ax[0][1]
    b.scatter(pu, ph, s=0.4, c="#9aa4b2", alpha=0.45, linewidths=0)
    b.scatter(cu, ch, s=12, c="#d92b2b", zorder=3)
    b.quiver(cu, ch, fu, fh, color="#1f6feb", angles="xy", scale=1.0 / L, width=0.003, zorder=4)
    b.set_title("(b) 侧视（水平轴 1 vs 高度）")
    b.grid(alpha=0.25)
    b.set_xlabel("水平轴 1 (m)")
    b.set_ylabel("高度 (m)")

    # (c) 距离分布
    c = ax[1][0]
    c.hist(dNearRaw, bins=30, alpha=0.6, label=f"洗点前（中位 {np.median(dNearRaw):.2f} m）", color="#e0a020")
    c.hist(dNear, bins=30, alpha=0.6, label=f"洗点后（中位 {np.median(dNear):.2f} m）", color="#2f9e44")
    c.set_title("(c) 每台相机到最近重建点的距离")
    c.set_xlabel("距离 (m)")
    c.set_ylabel("相机数")
    c.legend()
    c.grid(alpha=0.25)

    # (d) 重投影核对
    d = ax[1][1]
    imgs_dir = Path(args.images) if args.images else (Path(args.model).parent / "images")
    shown = False
    if imgs_dir.is_dir():
        names = [im["name"] for im in ims]
        idx = max(0, min(args.frame, len(names) - 1))
        target = names[idx]
        cand = [imgs_dir / target]
        for ext in (".png", ".jpg", ".jpeg", ".JPG"):
            cand.append(imgs_dir / (Path(target).stem + ext))
        p = next((q for q in cand if q.exists()), None)
        if p is not None:
            from PIL import Image

            img = Image.open(p).convert("RGB")
            W, H = img.size
            im = {im["name"]: im for im in model["images"]}[target]
            R = cmio.quat_to_R(np.asarray(im["qvec"], float))
            t = np.asarray(im["tvec"], float)
            cam = model["cameras"][im["camera_id"]]
            fx, fy, cx, cy = cmio.intrinsics_of(cam)
            fx *= W / (float(cam.get("width") or W) or W)
            fy *= H / (float(cam.get("height") or H) or H)
            cx *= W / (float(cam.get("width") or W) or W)
            cy *= H / (float(cam.get("height") or H) or H)
            pc = (R @ P.T).T + t
            z = pc[:, 2]
            ok = z > 1e-6
            u = fx * pc[ok, 0] / z[ok] + cx
            v = fy * pc[ok, 1] / z[ok] + cy
            m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            d.imshow(img)
            d.scatter(u[m], v[m], s=6, facecolors="none", edgecolors="#00ff88", linewidths=0.7)
            zz = z[ok][m]
            d.set_title(f"(d) 重投影核对：{target}\n绿圈=稀疏点投影（视锥内 {int(m.sum())} 个，深度中位 "
                        f"{np.median(zz) if m.sum() else float('nan'):.2f} m）")
            d.axis("off")
            shown = True
    if not shown:
        d.text(0.5, 0.5, "（缺少原图，跳过重投影核对）", ha="center", va="center")
        d.axis("off")

    fig.suptitle(f"{Path(args.model)}", fontsize=11)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=95)
    print(f"已写出 {out}")
    print(f"  相机 {len(C)} 台；点云 洗点前 {len(P_raw)} → 洗点后 {len(P)}")
    print(f"  相机到最近点：洗点前中位 {np.median(dNearRaw):.3f} m，洗点后中位 {np.median(dNear):.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
