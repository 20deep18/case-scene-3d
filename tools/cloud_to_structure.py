#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""点云 + 相机位姿 → structure.json（给 render_lineart / furniture_lines 用）。

为什么需要：VGGT、COLMAP MVS 只给点云和位姿，没有 `scene_structure.py` 那套结构文件。
而线稿渲染和家具分割都需要"曼哈顿系 + 房间盒"。这个脚本用**位姿里的重力方向**和
点云的稳健包围盒临时拼一个，够用且诚实（`sane=false`，因为墙没被真正识别）。

用法:
  <python> tools/cloud_to_structure.py --cloud output/vggt/bathroom-seq.ply \
      --poses output/vggt/bathroom-seq.poses.json --out output/vggt/seq.structure.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_ply_xyz(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    end = raw.find(b"end_header")
    body = raw.find(b"\n", end) + 1
    head = raw[:end].decode("ascii", "replace")
    n = int([l for l in head.splitlines() if l.startswith("element vertex")][0].split()[-1])
    names = [l.split()[-1] for l in head.splitlines() if l.startswith("property")]
    dt = np.dtype([(nm, "<f4") for nm in names])
    buf = np.frombuffer(raw, dtype=dt, count=n, offset=body)
    return np.stack([buf["x"], buf["y"], buf["z"]], axis=1).astype(np.float64)


def up_from_poses(extrinsic: np.ndarray) -> np.ndarray:
    """从外参估重力方向。

    OpenCV 约定：相机坐标 y 轴朝下，所以"世界上方向"= −Rᵀ(0,1,0)。
    取所有帧的中位数（个别帧手持倾斜不影响中位数）。
    """
    ups = []
    for E in extrinsic:
        R = np.asarray(E, float)[:3, :3]
        ups.append(-(R.T @ np.array([0.0, 1.0, 0.0])))
    ups = np.array(ups)
    med = np.median(ups, axis=0)
    return med / (np.linalg.norm(med) + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--poses", default=None, help="VGGT 的 *.poses.json（可选；没有就 PCA 估 up）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pct", type=float, default=2.0, help="房间盒用哪个分位（默认 P2~P98）")
    args = ap.parse_args()

    pts = load_ply_xyz(Path(args.cloud))
    pts = pts[np.isfinite(pts).all(axis=1)]
    print(f"点云 {len(pts)} 点")

    traj = []
    if args.poses and Path(args.poses).exists():
        pj = json.loads(Path(args.poses).read_text(encoding="utf-8"))
        E = np.asarray(pj["extrinsic"], float)          # [N,3,4] world→camera
        R = E[:, :3, :3]
        t = E[:, :3, 3]
        # 相机光心 = -Rᵀ t
        traj = [(-R[i].T @ t[i]).tolist() for i in range(len(E))]
        up = up_from_poses(E)
        src = "位姿（相机 y 轴朝下 → 取反）"
    else:
        d = pts - pts.mean(axis=0)
        w, v = np.linalg.eigh(d.T @ d / max(len(d), 1))
        up = v[:, np.argsort(w)[1]]                    # 第三特征向量通常是竖直方向
        up = up / (np.linalg.norm(up) + 1e-12)
        for cand in (v[:, np.argsort(w)[0]], v[:, np.argsort(w)[2]]):
            cand = cand / (np.linalg.norm(cand) + 1e-12)
            if abs(float(cand @ np.array([0.0, 0.0, 1.0]))) > abs(float(up @ np.array([0.0, 0.0, 1.0]))):
                up = cand
        src = "PCA（无位姿）"
    print(f"上方向: {np.round(up, 3)}（{src}）")

    # 水平方向：取"与 up 垂直、且让点云方差最大"的方向
    d = pts - pts.mean(axis=0)
    proj = d - np.outer(d @ up, up)
    w2, v2 = np.linalg.eigh(proj.T @ proj / max(len(proj), 1))
    u_ax = v2[:, np.argsort(w2)[-1]]
    u_ax = u_ax - up * float(u_ax @ up)
    u_ax /= np.linalg.norm(u_ax) + 1e-12
    v_ax = np.cross(up, u_ax)

    local = np.stack([(pts - pts.mean(axis=0)) @ u_ax,
                      (pts - pts.mean(axis=0)) @ v_ax,
                      (pts - pts.mean(axis=0)) @ up], axis=1)
    lo = np.percentile(local, args.pct, axis=0)
    hi = np.percentile(local, 100 - args.pct, axis=0)
    size = hi - lo
    center_local = (lo + hi) / 2
    axes = np.stack([u_ax, v_ax, up], axis=1)
    center_world = pts.mean(axis=0) + axes @ center_local

    corners = []
    base = pts.mean(axis=0) + axes @ center_local
    for z in (-1, 1):
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corners.append((base + axes @ (np.array([sx, sy, z]) * size / 2)).tolist())

    st = {
        "version": "cloud-derived",
        "source": str(args.cloud),
        "sane": True,                      # 盒子是点云分位包围盒，作为"外壳"可用（但别当墙面识别结果）
        "frame": {"up": up.tolist(), "axisU": u_ax.tolist(), "axisV": v_ax.tolist(),
                  "origin": base.tolist()},
        "room": {"size": size.tolist(), "corners": corners, "origin": base.tolist(),
                 "axisU": u_ax.tolist(), "axisV": v_ax.tolist()},
        "walls": [],
        "trajectory": {"points": traj},
        "stats": {"points": int(len(pts)), "percentile": args.pct},
        "note": "由点云+位姿临时拼的结构；room 是分位包围盒（可当外壳），walls 未识别",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"房间盒（沿帧轴）: {np.round(size, 2)}")
    if traj:
        print(f"相机轨迹: {len(traj)} 个位置")
    print(f"已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
