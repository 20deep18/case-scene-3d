"""
tools/make_test_dataset.py —— 生成一个「完整可训练」的合成 COLMAP 数据集

做三件事：
  1. 用解析方式渲染一个合成房间（地面/墙/桌子/球/盒子/门框），得到 12 张环绕图片；
  2. 把这些图片对应的相机内外参写成 COLMAP 的 cameras.bin / images.bin；
  3. 从房间表面采样一批带颜色的三维点，写成 points3D.bin（作为 3DGS 的初始化点云）。

这样即使没有 COLMAP（本机拿不到 Windows 预编译版），也能完整验证
「抽帧 → 位姿 → 训练 → .splat → 浏览器」这条链路。

用法：
    python tools/make_test_dataset.py --out data/scene-synth --views 12 --width 960 --height 540
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# --------------------------------------------------------------- 场景定义（COLMAP 系：x 右, y 下, z 前）

ROOM_W, ROOM_D, ROOM_H = 6.0, 8.0, 3.0
FLOOR_Y = ROOM_H / 2  # y 向下，地面在 +H/2
TABLE = {"c": (0.4, FLOOR_Y - 0.78, -0.6), "s": (1.8, 0.78, 1.0)}

BOXES = [
    # (中心, 尺寸, 颜色 BGR)
    ((0.4, FLOOR_Y - 0.78, -0.6), (1.8, 0.06, 1.0), (60, 100, 160)),          # 桌面
    ((-0.05, FLOOR_Y - 0.51, -0.65), (0.12, 0.12, 0.12), (90, 200, 230)),      # 球
    ((0.75, FLOOR_Y - 0.87, -0.7), (0.36, 0.18, 0.26), (200, 130, 40)),        # 盒子
]
CYLINDERS = [
    ((1.15, FLOOR_Y - 0.87, -0.32), 0.045, 0.18, (200, 200, 120)),             # 瓶子
]
BLOOD = ((-1.6, FLOOR_Y - 0.002, 1.4), 0.22, (10, 10, 90))                     # 地面血迹


def look_at(eye, target, up=np.array([0.0, 1.0, 0.0])):
    """
    返回 COLMAP/OpenCV 相机系下的旋转矩阵 R 与平移 t：x_cam = R·X_world + t。

    R 的三行分别是相机 x/y/z 轴在世界系中的表示；相机系里 +x 向右、
    +y **向下**（图像行号增大方向）、+z 向前。本房间模型的 y 轴也向下
    （地面在 +H/2），所以传入的 up 取世界 +y，用它构造出「图像下方」。
    """
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(up, f)
    if np.linalg.norm(r) < 1e-9:
        r = np.array([1.0, 0.0, 0.0])
    r = r / np.linalg.norm(r)
    u = np.cross(f, r)  # 图像向下方向 = 相机 +y
    R = np.stack([r, u, f], axis=0)
    t = -R @ eye
    return R, t


def rotation_to_qvec(R):
    """旋转矩阵 -> COLMAP 四元数 (w,x,y,z)"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


# --------------------------------------------------------------- 软件光栅（光线-几何求交）

def ray_box(orig, dirv, center, size):
    """求射线与轴对齐盒子的最近交点，返回 (t, normal)；未命中处 t = inf"""
    half = np.asarray(size, dtype=np.float64).reshape(3) / 2.0
    center = np.asarray(center, dtype=np.float64).reshape(3)
    inv = np.where(np.abs(dirv) < 1e-9, 1e9, 1.0 / dirv)
    t1 = (center - half - orig) * inv
    t2 = (center + half - orig) * inv
    tmin = np.maximum(np.minimum(t1, t2), -1e9)
    tmax = np.minimum(np.maximum(t1, t2), 1e9)
    t_near = tmin.max(axis=1)
    t_far = tmax.min(axis=1)
    hit = (t_near <= t_far) & (t_far > 1e-4)
    t = np.where(t_near > 1e-4, t_near, t_far)
    t = np.where(hit, t, np.inf)

    # 命中面的法向 = 在 t_near 对应轴上朝向射线来向
    axis = np.argmax(tmin, axis=1)
    normal = np.zeros_like(dirv)
    rows = np.arange(dirv.shape[0])
    sign = np.where(dirv[rows, axis] > 0, -1.0, 1.0)
    normal[rows, axis] = np.where(hit, sign, 0.0)
    return t, normal


def ray_sphere(orig, dirv, center, radius):
    oc = orig - center
    b = 2.0 * (dirv * oc).sum(axis=1)
    c = (oc * oc).sum(axis=1) - radius * radius
    disc = b * b - 4 * (dirv * dirv).sum(axis=1) * c
    t = np.full(dirv.shape[0], np.inf)
    ok = disc > 0
    sq = np.sqrt(np.maximum(disc, 0))
    a2 = 2.0 * (dirv * dirv).sum(axis=1)
    t0 = (-b - sq) / a2
    t1 = (-b + sq) / a2
    t = np.where(ok & (t0 > 1e-4), t0, np.where(ok & (t1 > 1e-4), t1, np.inf))
    p = orig + dirv * t[:, None]
    n = (p - center)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(nn, 1e-9)
    return t, n


def ray_cylinder(orig, dirv, center, radius, height):
    """竖直圆柱（沿 y 轴）"""
    o = orig - center
    a = dirv[:, 0] ** 2 + dirv[:, 2] ** 2
    b = 2 * (o[:, 0] * dirv[:, 0] + o[:, 2] * dirv[:, 2])
    c = o[:, 0] ** 2 + o[:, 2] ** 2 - radius * radius
    disc = b * b - 4 * a * c
    t = np.full(dirv.shape[0], np.inf)
    ok = (disc > 0) & (a > 1e-12)
    sq = np.sqrt(np.maximum(disc, 0))
    for sign in (-1, 1):
        tt = (-b + sign * sq) / (2 * np.maximum(a, 1e-12))
        py = o[:, 1] + tt * dirv[:, 1]
        good = ok & (tt > 1e-4) & (np.abs(py) <= height / 2)
        t = np.where(good & (tt < t), tt, t)
    n = np.zeros_like(dirv)
    p = orig + dirv * t[:, None]
    n[:, 0] = p[:, 0] - center[0]
    n[:, 2] = p[:, 2] - center[2]
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(nn, 1e-9)
    return t, n


def render_view(R, t, K, width, height, noise=2.0):
    """解析求交渲染一帧（返回 BGR uint8 图）"""
    fx, fy, cx, cy = K
    j, i = np.meshgrid(np.arange(width), np.arange(height))
    xd = (j.ravel() - cx) / fx
    yd = (i.ravel() - cy) / fy
    dir_cam = np.stack([xd, yd, np.ones_like(xd)], axis=1)
    dir_world = dir_cam @ R  # R^T 作用在行向量上
    dir_world = dir_world / np.linalg.norm(dir_world, axis=1, keepdims=True)
    orig = np.tile((-R.T @ t), (dir_world.shape[0], 1))

    # 背景：天花板/地板之外的暗色
    color = np.zeros((dir_world.shape[0], 3), dtype=np.float64)
    depth = np.full(dir_world.shape[0], np.inf)

    # 地面
    ty = (FLOOR_Y - orig[:, 1]) / np.where(np.abs(dir_world[:, 1]) < 1e-9, 1e-9, dir_world[:, 1])
    p = orig + dir_world * ty[:, None]
    on_floor = (ty > 1e-4) & (np.abs(p[:, 0]) < ROOM_W / 2) & (np.abs(p[:, 2]) < ROOM_D / 2)
    color[on_floor] = (70, 68, 66)
    depth[on_floor] = ty[on_floor]

    # 后墙 z = +D/2（法向 -z）
    tz = (ROOM_D / 2 - orig[:, 2]) / np.where(np.abs(dir_world[:, 2]) < 1e-9, 1e-9, dir_world[:, 2])
    p = orig + dir_world * tz[:, None]
    on_back = (tz > 1e-4) & (tz < depth) & (np.abs(p[:, 0]) < ROOM_W / 2) & (p[:, 1] > -ROOM_H / 2) & (p[:, 1] < FLOOR_Y)
    color[on_back] = (196, 194, 190)
    depth[on_back] = tz[on_back]

    # 左墙 x = -W/2（法向 +x）
    tx = (-ROOM_W / 2 - orig[:, 0]) / np.where(np.abs(dir_world[:, 0]) < 1e-9, 1e-9, dir_world[:, 0])
    p = orig + dir_world * tx[:, None]
    on_left = (tx > 1e-4) & (tx < depth) & (np.abs(p[:, 2]) < ROOM_D / 2) & (p[:, 1] > -ROOM_H / 2) & (p[:, 1] < FLOOR_Y)
    color[on_left] = (178, 176, 172)
    depth[on_left] = tx[on_left]

    # 门框（后墙上的深色矩形）
    door_w, door_h = 1.0, 2.1
    door_x = -ROOM_W / 2 + 0.9
    px = orig[:, 0] + dir_world[:, 0] * depth
    py = orig[:, 1] + dir_world[:, 1] * depth
    pz = orig[:, 2] + dir_world[:, 2] * depth
    in_door = (np.abs(pz - ROOM_D / 2) < 1e-3) & (px > door_x) & (px < door_x + door_w) & (py > FLOOR_Y - door_h) & (py < FLOOR_Y)
    color[in_door] = (92, 88, 84)

    # 盒子 / 球 / 圆柱
    for center, size, bgr in BOXES:
        tb, nb = ray_box(orig, dir_world, np.array(center, dtype=np.float64), size)
        hit = tb < depth
        if hit.any():
            shade = 0.72 + 0.28 * np.clip(-np.einsum("ij,ij->i", nb, dir_world), 0, 1)
            for ch in range(3):
                color[hit, ch] = bgr[ch] * shade[hit]
            depth[hit] = tb[hit]
    for center, radius, bgr in ((b[0], b[1][0] / 2, b[2]) for b in [(( -0.05, FLOOR_Y - 0.51, -0.65), (0.12,), (90, 200, 230))]):
        ts, ns = ray_sphere(orig, dir_world, np.array(center, dtype=np.float64), radius)
        hit = ts < depth
        if hit.any():
            shade = 0.7 + 0.3 * np.clip(-np.einsum("ij,ij->i", ns, dir_world), 0, 1)
            for ch in range(3):
                color[hit, ch] = bgr[ch] * shade[hit]
            depth[hit] = ts[hit]
    for center, radius, height_, bgr in CYLINDERS:
        tc, nc = ray_cylinder(orig, dir_world, np.array(center, dtype=np.float64), radius, height_)
        hit = tc < depth
        if hit.any():
            shade = 0.6 + 0.4 * np.clip(-np.einsum("ij,ij->i", nc, dir_world), 0, 1)
            for ch in range(3):
                color[hit, ch] = bgr[ch] * shade[hit]
            depth[hit] = tc[hit]

    # 血迹（地面上的红色斑）
    bc, br, bgr = BLOOD
    dist2 = (p[:, 0] - bc[0]) ** 2 + (p[:, 2] - bc[2]) ** 2
    on_blood = (np.abs(py - FLOOR_Y) < 1e-3) & (dist2 < br * br)
    if on_blood.any():
        for ch in range(3):
            color[on_blood, ch] = bgr[ch]

    # 简易光照衰减 + 噪声
    fade = np.clip(1.0 - depth / 40.0, 0.35, 1.0)
    color = color * fade[:, None]
    color += np.random.normal(0, noise, color.shape)
    img = np.clip(color, 0, 255).astype(np.uint8).reshape(height, width, 3)
    return img


# --------------------------------------------------------------- COLMAP 写出

def write_cameras_bin(path: Path, cam_id: int, model_id: int, width: int, height: int, params: list[float]):
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", 1))
        fh.write(struct.pack("<iiQQ", cam_id, model_id, width, height))
        fh.write(struct.pack("<" + "d" * len(params), *params))


def write_images_bin(path: Path, entries: list[dict], points: np.ndarray):
    """entries: [{id, qvec, tvec, camera_id, name, xys: (M,2), ids: (M,)}]"""
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(entries)))
        for e in entries:
            fh.write(struct.pack("<idddddddi", e["id"], *e["qvec"], *e["tvec"], e["camera_id"]))
            fh.write(e["name"].encode("utf-8") + b"\x00")
            xys = e["xys"]
            ids = e["ids"]
            fh.write(struct.pack("<Q", len(xys)))
            for (x, y), pid in zip(xys, ids):
                fh.write(struct.pack("<ddq", float(x), float(y), int(pid)))
    return


def write_points3d_bin(path: Path, points: np.ndarray, colors: np.ndarray):
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", points.shape[0]))
        for i in range(points.shape[0]):
            p = points[i]
            c = colors[i]
            fh.write(
                struct.pack(
                    "<QdddBBBd",
                    i + 1,
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    int(c[0]),
                    int(c[1]),
                    int(c[2]),
                    0.5,
                )
            )
            fh.write(struct.pack("<Q", 0))  # track_length = 0


def sample_surface_points(rng, count: int):
    """在房间表面上采样带颜色的三维点，作为 3DGS 初始点云"""
    pts, cols = [], []
    def add(p, c, n):
        for _ in range(n):
            pts.append(p + rng.normal(0, 0.01, 3))
            shade = 0.85 + rng.random() * 0.3
            cols.append([min(255, c[i] * shade) for i in range(3)])

    for _ in range(int(count * 0.4)):
        add(np.array([rng.uniform(-ROOM_W / 2, ROOM_W / 2), FLOOR_Y, rng.uniform(-ROOM_D / 2, ROOM_D / 2)]), np.array([70, 68, 66]), 1)
    for _ in range(int(count * 0.25)):
        add(np.array([rng.uniform(-ROOM_W / 2, ROOM_W / 2), rng.uniform(-ROOM_H / 2, FLOOR_Y), ROOM_D / 2]), np.array([196, 194, 190]), 1)
    for _ in range(int(count * 0.15)):
        add(np.array([-ROOM_W / 2, rng.uniform(-ROOM_H / 2, FLOOR_Y), rng.uniform(-ROOM_D / 2, ROOM_D / 2)]), np.array([178, 176, 172]), 1)
    for center, size, bgr in BOXES:
        c = np.array(center)
        s = np.array(size)
        for _ in range(int(count * 0.06)):
            face = rng.integers(0, 6)
            p = c + (rng.random(3) - 0.5) * s
            p[face // 2] = c[face // 2] + (s[face // 2] / 2) * (1 if face % 2 else -1)
            add(p, np.array(bgr), 1)
    for _ in range(int(count * 0.04)):
        u = rng.normal(0, 1, 3)
        u /= np.linalg.norm(u)
        add(np.array([-0.05, FLOOR_Y - 0.51, -0.65]) + u * 0.12, np.array([90, 200, 230]), 1)
    return np.array(pts), np.array(cols)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成可训练的合成 COLMAP 数据集")
    ap.add_argument("--out", default="data/scene-synth")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fov", type=float, default=58.0)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()

    if cv2 is None:
        print("需要 opencv-python 来写出 PNG（pip install opencv-python）")
        return 3

    out = Path(args.out).resolve()
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    fx = 0.5 * args.width / math.tan(math.radians(args.fov) / 2)
    fy = fx
    cx = args.width / 2
    cy = args.height / 2
    K = (fx, fy, cx, cy)

    entries = []
    target = np.array([0.2, 0.1, -0.2])
    radius = 3.6
    for v in range(args.views):
        ang = 2 * math.pi * v / args.views
        eye = np.array([math.sin(ang) * radius * 0.75, 0.25 + 0.35 * math.sin(ang * 2), math.cos(ang) * radius * 0.7 - 0.3])
        R, t = look_at(eye, target)
        img = render_view(R, t, K, args.width, args.height)
        name = f"view_{v + 1:03d}.png"
        cv2.imwrite(str(out / "images" / name), img)
        q = rotation_to_qvec(R)
        entries.append({"id": v + 1, "qvec": q, "tvec": t, "camera_id": 1, "name": name, "xys": np.zeros((0, 2)), "ids": np.zeros(0)})
        print(f"  渲染 {name}  相机位置 {np.round(-R.T @ t, 3)}")

    write_cameras_bin(out / "sparse" / "0" / "cameras.bin", 1, 4, args.width, args.height, [fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0])
    write_images_bin(out / "sparse" / "0" / "images.bin", entries, np.zeros((0, 3)))
    pts, cols = sample_surface_points(rng, args.points)
    write_points3d_bin(out / "sparse" / "0" / "points3D.bin", pts, cols)

    (out / "dataset.json").write_text(
        __import__("json").dumps(
            {
                "views": args.views,
                "width": args.width,
                "height": args.height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "cameraModel": "OPENCV",
                "points": int(pts.shape[0]),
                "axes": "COLMAP (x right, y down, z forward)",
                "note": "由 tools/make_test_dataset.py 生成的合成数据集，用于在无 COLMAP 环境下验证训练链路",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n已写出数据集：{out}")
    print(f"  图像：{args.views} 张 ({args.width}x{args.height})  点云：{pts.shape[0]} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
