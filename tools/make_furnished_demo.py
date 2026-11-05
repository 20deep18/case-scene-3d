#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""造一个**带家具的合成房间**（点云 + 结构文件 + 真值），用来验证"家具级线稿"算法。

为什么要造：现有 `.splat` 素材里没有可用于演示家具的现场 ——
实测（见会话记录）room-real 剥掉房间壳后只剩 184 个点（那间屋基本是空的），
柜子那份几何已退化，sample-room/test-scene 没有结构文件、PCA 姿态也是乱的。
所以要拿**已知真值**验证算法：真值盒的位置/尺寸是我自己定的，算法输出可以逐项比。

产出：
  output/furniture/demo-room.splat          （32 字节/高斯，位置+尺度+RGBA+四元数）
  output/furniture/demo-room.structure.json （房间盒 + 曼哈顿系，供 furniture_lines / 线稿用）
  output/furniture/demo-room-truth.json     （家具真值盒）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "output" / "furniture"

ROOM = (4.6, 3.6, 2.8)     # 宽(x) 深(y) 高(z)，米

# 家具真值：名字, 中心(x,y,z), 尺寸(x,y,z)
# ⚠️ 家具之间必须留出空隙（> 2 个体素 ≈ 8 cm），否则会连成一簇 ——
#    第一版把床头柜贴着床摆，结果床+柜被聚成一个盒子。
# 坐标必须**落在房间内**（房间 4.6×3.6×2.8 m → x∈[-2.3,2.3], y∈[-1.8,1.8]）：
# 第一版把床摆到 x=3.95，直接在房间外面，剥壳时全被剔掉了（只找到 1 个盒子）。
FURNITURE = [
    ("衣柜",   (-1.35, -1.45, 1.00), (1.80, 0.60, 2.00)),  # 靠后墙
    ("床",     (0.55, 0.20, 0.28), (2.00, 1.50, 0.50)),
    ("床头柜A", (1.95, 0.20, 0.28), (0.45, 0.40, 0.52)),   # 床右侧，留 0.18 m 间隙
    ("床头柜B", (-0.90, 0.20, 0.28), (0.45, 0.40, 0.52)),  # 床左侧，留 0.22 m 间隙
]


def box_surface(center, size, n, rng) -> np.ndarray:
    """在一个盒子的 6 个面上均匀撒点（含少量法向抖动）。"""
    cx, cy, cz = center
    sx, sy, sz = size
    pts = []
    per = max(8, n // 6)
    # ±x 面
    for sx_ in (-0.5, 0.5):
        pts.append(np.stack([
            np.full(per, cx + sx_ * sx),
            rng.uniform(cy - sy / 2, cy + sy / 2, per),
            rng.uniform(cz - sz / 2, cz + sz / 2, per)], axis=1))
    # ±y 面
    for sy_ in (-0.5, 0.5):
        pts.append(np.stack([
            rng.uniform(cx - sx / 2, cx + sx / 2, per),
            np.full(per, cy + sy_ * sy),
            rng.uniform(cz - sz / 2, cz + sz / 2, per)], axis=1))
    # ±z 面
    for sz_ in (-0.5, 0.5):
        pts.append(np.stack([
            rng.uniform(cx - sx / 2, cx + sx / 2, per),
            rng.uniform(cy - sy / 2, cy + sy / 2, per),
            np.full(per, cz + sz_ * sz)], axis=1))
    return np.concatenate(pts, axis=0)


def main() -> int:
    rng = np.random.default_rng(20261101)
    OUT.mkdir(parents=True, exist_ok=True)
    W, D, H = ROOM

    clouds = []
    # 地面 + 四面墙 + 天花（间距 2cm 的网格，模拟重建出来的"房间壳"）
    g = 0.02
    xs, ys = np.arange(-W / 2, W / 2, g), np.arange(-D / 2, D / 2, g)
    X, Y = np.meshgrid(xs, ys)
    clouds.append(np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1))          # 地面
    clouds.append(np.stack([X.ravel(), Y.ravel(), np.full(X.size, H)], axis=1))        # 天花
    zs = np.arange(0, H, g)
    Zx, Yy = np.meshgrid(zs, ys)
    clouds.append(np.stack([np.full(Zx.size, -W / 2), Yy.ravel(), Zx.ravel()], axis=1))
    clouds.append(np.stack([np.full(Zx.size, +W / 2), Yy.ravel(), Zx.ravel()], axis=1))
    Zy, Xx = np.meshgrid(zs, xs)
    clouds.append(np.stack([Xx.ravel(), np.full(Zy.size, -D / 2), Zy.ravel()], axis=1))
    clouds.append(np.stack([Xx.ravel(), np.full(Zy.size, +D / 2), Zy.ravel()], axis=1))

    # 家具表面：每件 9000 点 ≈ 每面 1500 点。
# ⚠️ 密度是必须的：第一版每件 1400 点（每面 233 点、间距 ~11cm），
#    远大于体素(3.5cm)，2m 的大面碎成孤立体素 → 床和衣柜聚不起来（只找到床头柜）。
    for name, center, size in FURNITURE:
        clouds.append(box_surface(center, size, 9000, rng))

    P = np.concatenate(clouds, axis=0)
    P += rng.normal(0, 0.006, P.shape)       # 6mm 噪声，接近真实重建精度
    print(f"合成点云：{len(P)} 个")

    # ---- 写成 .splat（位置 3f + 尺度 3f + RGBA 4B + 四元数 4B = 32B/点）----
    n = len(P)
    buf = np.zeros((n, 8), dtype="<f4")
    buf[:, 0:3] = P
    buf[:, 3:6] = 0.02                       # 每个点当成 2cm 的小高斯
    raw = bytearray(buf.tobytes())
    b = np.zeros((n, 32), np.uint8)
    b[:, 24:27] = 200                        # RGB 灰白
    b[:, 27] = 255                           # alpha = 1
    b[:, 28] = 255                           # 四元数 w=1
    raw = bytearray(np.concatenate([buf.view(np.uint8).reshape(n, 32)[:, :24], b[:, 24:]], axis=1).tobytes())
    (OUT / "demo-room.splat").write_bytes(bytes(raw))
    print(f"已写出 {(OUT / 'demo-room.splat')}  ({(OUT / 'demo-room.splat').stat().st_size / 1048576:.1f} MB)")

    # ---- 结构文件（房间盒 + 曼哈顿系；坐标就是米）----
    up = [0.0, 0.0, 1.0]
    u_ax = [1.0, 0.0, 0.0]
    v_ax = [0.0, 1.0, 0.0]
    corners = []
    for sy in (-1, 1):
        for sx in (-1, 1):
            pass
    # 顺序与 structure_room 一致：先底面 4 点（-u-v, +u-v, +u+v, -u+v），再顶面 4 点
    for z in (0.0, H):
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corners.append([round(sx * W / 2, 4), round(sy * D / 2, 4), round(z, 4)])
    structure = {
        "version": "demo",
        "source": "synthetic",
        "sane": True,
        "frame": {"up": up, "axisU": u_ax, "axisV": v_ax, "origin": [0.0, 0.0, 0.0]},
        "room": {
            "size": [W, D, H],
            "corners": corners,
            "origin": [0.0, 0.0, 0.0],
            "axisU": u_ax,
            "axisV": v_ax,
        },
        "walls": [],
        "trajectory": {"points": []},
        "stats": {"scaleP95": W, "pointsUsed": n},
        "note": "合成房间：只用于验证家具级线稿算法（坐标=米）",
    }
    (OUT / "demo-room.structure.json").write_text(
        json.dumps(structure, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {OUT / 'demo-room.structure.json'}")

    truth = {"room": ROOM,
             "furniture": [{"name": nm, "center": list(c), "size": list(s)} for nm, c, s in FURNITURE]}
    (OUT / "demo-room-truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    print("真值：")
    for nm, c, s in FURNITURE:
        print(f"  {nm:6s} 中心 {c}  尺寸 {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
