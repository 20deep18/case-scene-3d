#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成一个**纹理丰富**的合成场景，用于验证 COLMAP 端到端可用。

背景：tools/make_test_dataset.py 生成的是纯色墙面的房间，SIFT 只能找到 28~50 个特征点，
匹配数为 0 —— 这是测试数据缺少纹理，不是 COLMAP 的问题。
本脚本给空间中的每个表面贴上程序生成的多尺度纹理（棋盘 + 斑点 + 条纹 + 噪声），
使 SIFT 能稳定找到大量特征点。

**不依赖任何图像库**：直接写 PPM（P6）—— COLMAP 原生支持 PPM。
透视映射用 numpy 逐像素反查（纹理空间 → 屏幕空间的单应），保证纹理不拉伸变形。

用法：
    pyruntime312\\python.exe tools/make_textured_dataset.py --out data/scene-textured
"""

import argparse
import os
import sys

import numpy as np


# ---------------------------------------------------------------- 纹理生成


def make_texture(size, seed):
    """多尺度高对比度纹理（H, W, 3 uint8）"""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 60, dtype=np.float32)

    # 1) 多层棋盘 —— 提供大量稳定的角点
    for cell, gain in ((8, 55), (16, 45), (32, 35), (64, 25)):
        ys = (np.arange(size) // cell) % 2
        xs = (np.arange(size) // cell) % 2
        board = (ys[:, None] ^ xs[None, :]).astype(bool)
        img[board] += gain

    # 2) 随机斑点 —— 局部特征
    n = (size * size) // 260
    for _ in range(n):
        cy, cx = rng.integers(0, size, 2)
        r = int(rng.integers(2, max(3, size // 48)))
        y0, y1 = max(0, cy - r), min(size, cy + r + 1)
        x0, x1 = max(0, cx - r), min(size, cx + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.ogrid[y0 - cy : y1 - cy, x0 - cx : x1 - cx]
        mask = (xx * xx + yy * yy) <= r * r
        img[y0:y1, x0:x1][mask] = rng.integers(0, 256, 3)

    # 3) 斜向条纹 —— 提供长程结构
    yy, xx = np.mgrid[0:size, 0:size]
    stripes = ((xx + yy) // 12) % 2
    img[stripes == 1] += 30

    # 4) 密集噪声 —— 提高纹理熵
    img += rng.integers(-16, 17, (size, size, 3)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 几何


def look_at(eye, target, up):
    """
    返回 (A, t)，使得相机空间坐标 **x_cam = Aᵀ·(x_world - eye)**，等价于 x_cam = A·x_world + t。

    约定（与 app/web/js/pnp.js / tests/test_pnp.mjs 完全一致）：
      - 相机空间：x 右、y 下、**z 朝前**（相机前方的点满足 z_cam > 0）
      - A 的列 = 相机基 (右, 下, 前)，即「相机→世界」旋转
      - t = -Aᵀ·eye，于是相机中心 C = -Aᵀ·t = eye

    ⚠️ 这里连续踩了两个坑，都很难靠自检查出来：
      1) 世界→相机要左乘 **Aᵀ**，不是 A（旋转方向反了）；
      2) 相机基第三列必须是 **+f**，写成 -f 会让 z 变成负数，所有点都跑到相机背后。
     `A@eye + t = 0` 只验证了「相机中心在原点」，两种错误写法下都成立，
     所以自检必须验证**前方点落到 z_cam > 0**，以及 **C = -Aᵀ·t 能还原 eye**。
    """
    f = target - eye
    n = np.linalg.norm(f)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("look_at: eye 与 target 重合")
    f = f / n

    # up 与 f 平行时叉积退化为 0（万向节死锁）——换一个参考方向兜底
    upv = np.asarray(up, dtype=float)
    if np.linalg.norm(np.cross(f, upv)) < 1e-6:
        upv = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    s = np.cross(f, upv)
    s = s / np.linalg.norm(s)
    # ★ 必须是 f × s（不是 s × f）：这样 (右, 下, 前) 才构成右手系，det = +1。
    # 用 s × f 会让 y 轴朝上（det = -1），图像上下颠倒、左右镜像。
    u = np.cross(f, s)

    R = np.stack([s, u, f], axis=1)  # 列 = 相机基 (右, 下, 前)
    A = R.T  # 世界→相机
    t = -A @ eye
    return A, t


def world_to_cam(A, t, x):
    """x_cam = A·x + t，其中 A 是 world→camera 旋转"""
    return A @ np.asarray(x, dtype=float) + t


def camera_center(A, t):
    """相机中心 C = -Aᵀ·t"""
    return -A.T @ t


def quad_homography(tex_quad, dst_quad):
    """求把纹理四边形映射到目标四边形的单应（3×3）"""
    A = []
    for (sx, sy), (dx, dy) in zip(tex_quad, dst_quad):
        A.append([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy, -dx])
        A.append([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy, -dy])
    A = np.asarray(A, dtype=float)
    _, _, Vt = np.linalg.svd(A)
    return Vt[-1].reshape(3, 3)


def warp_quad(canvas, tex, dst_quad, bbox):
    """把纹理按单应贴到 canvas 的 dst_quad 区域（逐像素反查，不依赖图像库）"""
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return
    Hm = quad_homography([(0, 0), (tex.shape[1], 0), (tex.shape[1], tex.shape[0]), (0, tex.shape[0])], dst_quad)
    try:
        Hinv = np.linalg.inv(Hm)
    except np.linalg.LinAlgError:
        return

    ys, xs = np.mgrid[y0:y1, x0:x1]
    ones = np.ones_like(xs, dtype=float)
    pts = np.stack([xs.ravel().astype(float), ys.ravel().astype(float), ones.ravel()], axis=0)
    mapped = Hinv @ pts
    w = mapped[2]
    ok = np.abs(w) > 1e-9
    u = np.full(w.shape, -1.0)
    v = np.full(w.shape, -1.0)
    u[ok] = mapped[0][ok] / w[ok]
    v[ok] = mapped[1][ok] / w[ok]

    th, tw = tex.shape[:2]
    inside = (u >= 0) & (u < tw - 1) & (v >= 0) & (v < th - 1)
    if not inside.any():
        return
    ui = u[inside].astype(np.int32)
    vi = v[inside].astype(np.int32)
    colors = tex[vi, ui]

    sub = canvas[y0:y1, x0:x1].reshape(-1, 3)
    sub[inside] = colors
    canvas[y0:y1, x0:x1] = sub.reshape(y1 - y0, x1 - x0, 3)


def quad_mask(bbox, quad):
    """向量化的凸四边形内外掩码（bbox 区域内的像素，True = 落在四边形内）"""
    x0, y0, x1, y1 = bbox
    yy, xx = np.mgrid[y0:y1, x0:x1]
    px = xx + 0.5
    py = yy + 0.5
    # 以四边形中心的绕向为参考，把 4 条边的叉积统一到「内侧为同一符号」
    cxm = sum(p[0] for p in quad) / 4.0
    cym = sum(p[1] for p in quad) / 4.0
    inside = np.ones(px.shape, dtype=bool)
    for i in range(4):
        ax, ay = quad[i]
        bx, by = quad[(i + 1) % 4]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        ref = (bx - ax) * (cym - ay) - (by - ay) * (cxm - ax)
        inside &= (cross * ref) >= -1e-9
    return inside


def point_in_quad(p, quad):
    """凸四边形内外判定（叉积同号）"""
    sign = None
    for i in range(4):
        a = np.array(quad[i])
        b = np.array(quad[(i + 1) % 4])
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if sign is None:
            sign = np.sign(cross)
        elif np.sign(cross) != sign and abs(cross) > 1e-9:
            return False
    return True


# ---------------------------------------------------------------- 写图


def write_ppm(path, array):
    h, w = array.shape[:2]
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, h))
        f.write(np.ascontiguousarray(array, dtype=np.uint8).tobytes())


# ---------------------------------------------------------------- 主流程


def self_check(views, radius):
    """
    相机约定自检。这个函数的存在是有原因的：写这个脚本时我在同一个地方
    连续犯了 3 个符号错误（Aᵀ 写成 A、第三列取 -f、C 少了负号），
    而它们**都能通过「相机中心在原点」这种自检**，只有验证
    「前方点落到 z_cam > 0」+「C = -Aᵀ·t 能还原 eye」才查得出来。
    """
    problems = []
    for i in range(views):
        ang = 2 * np.pi * i / views
        elev = 0.22 * np.sin(2 * ang)
        eye = np.array(
            [
                radius * np.cos(elev) * np.cos(ang),
                radius * np.sin(elev),
                radius * np.cos(elev) * np.sin(ang),
            ]
        )
        A, t = look_at(eye, np.zeros(3), np.array([0.0, 1.0, 0.0]))
        xc = world_to_cam(A, t, np.zeros(3))
        c = camera_center(A, t)
        if abs(xc[0]) > 1e-9 or abs(xc[1]) > 1e-9:
            problems.append(f"view {i + 1}: 原点不在光轴上 ({xc[0]:.2e}, {xc[1]:.2e})")
        if abs(xc[2] - radius) > 1e-9:
            problems.append(f"view {i + 1}: 原点不在相机前方 z={xc[2]:.4f}（应为 {radius}）")
        if np.linalg.norm(c - eye) > 1e-9:
            problems.append(f"view {i + 1}: 相机中心还原失败 {np.round(c, 4)} vs {np.round(eye, 4)}")
        det = np.linalg.det(A)
        if abs(det - 1.0) > 1e-9:
            problems.append(f"view {i + 1}: det(A)={det:.6f}，不是纯旋转")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scene-textured")
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--radius", type=float, default=4.6, help="环绕半径（米）")
    ap.add_argument("--tex-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = os.path.abspath(args.out)

    # 先做相机约定自检 —— 不通过就直接退出，避免又生成一堆只有背景的废图
    problems = self_check(args.views, args.radius)
    if problems:
        print("相机约定自检未通过：")
        for p in problems[:10]:
            print("  ✘ " + p)
        sys.exit(1)
    print(f"相机约定自检通过（{args.views} 个视角）")

    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)

    W, H = args.width, args.height
    fx = fy = 0.5 * W / np.tan(np.radians(30.0))  # 水平 60°
    cx_p, cy_p = W / 2.0, H / 2.0

    # 场景：一个立方体（6 个面），相机在外侧环绕
    half = 1.2
    faces = [
        ("+x", [(-half, -half, half), (-half, half, half), (-half, half, -half), (-half, -half, -half)], (1, 0, 0)),
        ("-x", [(half, -half, -half), (half, half, -half), (half, half, half), (half, -half, half)], (-1, 0, 0)),
        ("+y", [(-half, half, half), (-half, half, -half), (half, half, -half), (half, half, half)], (0, 1, 0)),
        ("-y", [(-half, -half, -half), (-half, -half, half), (half, -half, half), (half, -half, -half)], (0, -1, 0)),
        ("+z", [(-half, -half, half), (half, -half, half), (half, half, half), (-half, half, half)], (0, 0, 1)),
        ("-z", [(half, -half, -half), (-half, -half, -half), (-half, half, -half), (half, half, -half)], (0, 0, -1)),
    ]

    # 每个面预生成一张纹理
    textures = {}
    for idx, (name, _, _) in enumerate(faces):
        textures[name] = make_texture(args.tex_size, args.seed * 1000 + idx)

    cameras = []
    for i in range(args.views):
        ang = 2 * np.pi * i / args.views
        # 相机在半径 R 的球面上环绕，**始终朝向场景中心**（与真实手持环绕拍摄一致）。
        # 注意：必须让每一帧都真正看着中心，否则物体会漂到画面外 —— 之前用过
        # 「y 随角度起伏」的轨迹，结果大多数视角里立方体都跑出画面了。
        elev = 0.22 * np.sin(2 * ang)  # 轻度的上下起伏（±0.22 rad ≈ 12.6°）
        eye = np.array(
            [
                args.radius * np.cos(elev) * np.cos(ang),
                args.radius * np.sin(elev),
                args.radius * np.cos(elev) * np.sin(ang),
            ]
        )
        R, t = look_at(eye, np.zeros(3), np.array([0.0, 1.0, 0.0]))

        rng = np.random.default_rng(9000 + i)
        canvas = rng.integers(18, 48, (H, W, 3), dtype=np.uint8)

        # 按面中心到相机的距离排序（远的先画）
        drawable = []
        for name, verts, normal in faces:
            center = np.mean(np.array(verts), axis=0)
            if np.dot(np.array(normal), eye - center) <= 0:
                continue
            pts = []
            ok = True
            for v in verts:
                xc = R @ np.array(v) + t
                if xc[2] <= 1e-6:
                    ok = False
                    break
                pts.append((fx * xc[0] / xc[2] + cx_p, fy * xc[1] / xc[2] + cy_p))
            if not ok:
                continue
            drawable.append((np.linalg.norm(eye - center), name, pts))
        drawable.sort(reverse=True)

        for _, name, pts in drawable:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bx0 = max(0, int(np.floor(min(xs))))
            by0 = max(0, int(np.floor(min(ys))))
            bx1 = min(W, int(np.ceil(max(xs))) + 1)
            by1 = min(H, int(np.ceil(max(ys))) + 1)
            # 面可能完全在画面外或只有一部分可见 —— 必须夹紧，否则 mgrid 会因负尺寸报错
            if bx1 - bx0 < 2 or by1 - by0 < 2:
                continue
            bbox = (bx0, by0, bx1, by1)
            # 只在四边形内部贴图（warp_quad 会覆盖整个 bbox，需要再裁剪）
            before = canvas[bbox[1] : bbox[3], bbox[0] : bbox[2]].copy()
            warp_quad(canvas, textures[name], pts, bbox)
            after = canvas[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            inside = quad_mask(bbox, pts)
            after[~inside] = before[~inside]
            canvas[bbox[1] : bbox[3], bbox[0] : bbox[2]] = after

        # 传感器噪声
        canvas = np.clip(
            canvas.astype(np.int16) + rng.integers(-6, 7, canvas.shape, dtype=np.int16), 0, 255
        ).astype(np.uint8)

        name = f"view_{i + 1:03d}.ppm"
        write_ppm(os.path.join(img_dir, name), canvas)
        cameras.append((name, R, t))
        if (i + 1) % 6 == 0:
            print(f"  已写出 {i + 1}/{args.views} 个视角")

    # 真值（文本格式，便于对比）
    with open(os.path.join(out, "cameras.txt"), "w", encoding="utf-8") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {fx:.6f} {fy:.6f} {cx_p:.6f} {cy_p:.6f}\n")
    with open(os.path.join(out, "images.txt"), "w", encoding="utf-8") as f:
        for idx, (name, R, t) in enumerate(cameras, start=1):
            q = rotmat_to_quat(R)
            f.write(f"{idx} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {name}\n\n")

    print(f"\n已写出 {len(cameras)} 个视角到 {img_dir}")
    print(f"内参 fx={fx:.2f} fy={fy:.2f} cx={cx_p:.1f} cy={cy_p:.1f}  {W}x{H}")
    print(f"真值: {out}\\cameras.txt 与 images.txt")


def rotmat_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    if R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)


if __name__ == "__main__":
    main()
