"""
生成一个合成「案发现场」三维高斯场景（.splat），用于在没有真实拍摄数据时
验证整条链路（Web 加载 / 渲染 / 标注 / 协同）。

房间构成：地面、后墙、左墙、桌子、桌子上的球体与盒子、门框、若干物证标记点。
所有表面用大量各向异性高斯铺满，方向与表面法向一致。

用法：
    python tools/make_sample_scene.py --out data/models/sample-room.splat [--count 120000]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:  # 允许无 numpy 时降级（会慢很多）
    np = None

SH_C0 = 0.28209479177387814


def srgb_to_linear(c: float) -> float:
    c = max(0.0, min(1.0, c))
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def pack_splat(points, scales, quats, colors, opacities) -> bytes:
    """按 antimatter15 .splat 格式打包：32 字节/高斯。"""
    n = len(points)
    buf = bytearray(n * 32)
    for i in range(n):
        off = i * 32
        x, y, z = points[i]
        sx, sy, sz = scales[i]
        struct.pack_into("<6f", buf, off, x, y, z, sx, sy, sz)
        # 颜色：写入 SH DC 系数映射到 0..255，渲染端按 linear = 0.5 + SH_C0 * (v/255) 还原
        for c in range(3):
            lin = srgb_to_linear(colors[i][c])
            dc = (lin - 0.5) / SH_C0
            buf[off + 24 + c] = max(0, min(255, int(round(dc * 255))))
        buf[off + 27] = max(0, min(255, int(round(opacities[i] * 255))))
        qw, qx, qy, qz = quats[i]
        buf[off + 28] = max(0, min(255, int(round(qw * 128 + 128))))
        buf[off + 29] = max(0, min(255, int(round(qx * 128 + 128))))
        buf[off + 30] = max(0, min(255, int(round(qy * 128 + 128))))
        buf[off + 31] = max(0, min(255, int(round(qz * 128 + 128))))
    return bytes(buf)


def quat_from_normal(nx: float, ny: float, nz: float, roll: float = 0.0) -> tuple[float, float, float, float]:
    """构造把局部 z 轴旋转到给定法向的四元数（w,x,y,z）。"""
    ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    dot = nz
    if dot > 0.999999:
        q = (1.0, 0.0, 0.0, 0.0)
    elif dot < -0.999999:
        q = (0.0, 1.0, 0.0, 0.0)  # 180° 绕 x
    else:
        ax, ay, az = -ny, nx, 0.0
        la = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        ax, ay, az = ax / la, ay / la, az / la
        half = math.acos(max(-1.0, min(1.0, dot))) / 2.0
        s = math.sin(half)
        q = (math.cos(half), ax * s, ay * s, az * s)
    # 叠加绕法向的 roll
    if roll:
        hr = roll / 2.0
        rq = (math.cos(hr), nx * math.sin(hr), ny * math.sin(hr), nz * math.sin(hr))
        q = quat_mul(rq, q)
    return q


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def rand_unit(rng):
    while True:
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        z = rng.uniform(-1, 1)
        d = x * x + y * y + z * z
        if 0.0001 < d <= 1.0:
            s = math.sqrt(d)
            return x / s, y / s, z / s


def main() -> int:
    ap = argparse.ArgumentParser(description="生成合成案件现场高斯场景")
    ap.add_argument("--out", default="data/models/sample-room.splat")
    ap.add_argument("--json", default=None)
    ap.add_argument("--count", type=int, default=120000, help="近似高斯总数")
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()

    rng = __import__("random").Random(args.seed)

    # 房间尺寸（米），y 轴向下（COLMAP 约定）
    W, D, H = 6.0, 8.0, 3.0
    pts: list[tuple[float, float, float]] = []
    scl: list[tuple[float, float, float]] = []
    qts: list[tuple[float, float, float, float]] = []
    cols: list[tuple[float, float, float]] = []
    ops: list[float] = []

    def add_surface(origin, u, v, normal, color, count, jitter=0.006, scale_uv=(0.035, 0.035), opacity=0.92, noise=0.05):
        """在一块矩形平面上铺高斯。u/v 为面内基向量（含长度）。"""
        ox, oy, oz = origin
        ux, uy, uz = u
        vx, vy, vz = v
        nx, ny, nz = normal
        for _ in range(count):
            a = rng.random()
            b = rng.random()
            # 轻微抖动，让表面有质感
            jx = rng.gauss(0, jitter)
            jy = rng.gauss(0, jitter)
            jz = rng.gauss(0, jitter)
            px = ox + ux * a + vx * b + jx
            py = oy + uy * a + vy * b + jy
            pz = oz + uz * a + vz * b + jz
            pts.append((px, py, pz))
            su = scale_uv[0] * rng.uniform(0.7, 1.4)
            sv = scale_uv[1] * rng.uniform(0.7, 1.4)
            sn = max(0.0025, min(su, sv) * 0.22)
            scl.append((su, sv, sn))
            qts.append(quat_from_normal(nx, ny, nz, rng.uniform(0, math.pi * 2)))
            shade = 1.0 + rng.gauss(0, noise)
            cols.append(
                (
                    max(0.0, min(1.0, color[0] * shade)),
                    max(0.0, min(1.0, color[1] * shade)),
                    max(0.0, min(1.0, color[2] * shade)),
                )
            )
            ops.append(min(1.0, max(0.05, opacity * rng.uniform(0.85, 1.05))))

    base = args.count
    # 面积分配
    area_floor = W * D
    area_back = W * H
    area_left = D * H
    area_total = area_floor + area_back + area_left
    n_floor = int(base * 0.36 * area_floor / (area_floor * 1.0) * 0.36 / 0.36 * (area_floor / area_total) / (area_floor / area_total) * 0.36) if False else int(base * 0.42)
    n_back = int(base * 0.24)
    n_left = int(base * 0.16)
    n_table = int(base * 0.10)
    n_objects = base - n_floor - n_back - n_left - n_table

    # 地面 y=+H/2 附近：把 y 作为向下轴，地面在 y = H/2
    floor_y = H / 2
    add_surface(
        (-W / 2, floor_y, -D / 2),
        (W, 0, 0),
        (0, 0, D),
        (0, -1, 0),
        (0.32, 0.31, 0.30),
        n_floor,
        jitter=0.004,
        scale_uv=(0.042, 0.042),
    )
    # 后墙 z = +D/2，法向 -z
    add_surface(
        (-W / 2, floor_y, D / 2),
        (W, 0, 0),
        (0, -H, 0),
        (0, 0, -1),
        (0.78, 0.77, 0.74),
        n_back,
        scale_uv=(0.045, 0.045),
    )
    # 左墙 x = -W/2，法向 +x
    add_surface(
        (-W / 2, floor_y, -D / 2),
        (0, 0, D),
        (0, -H, 0),
        (1, 0, 0),
        (0.72, 0.71, 0.69),
        n_left,
        scale_uv=(0.045, 0.045),
    )

    # 桌子：桌面 + 四条腿（简化为四个长方体侧面）
    tx, tz = 0.4, -0.6
    tw, td, th = 1.8, 1.0, 0.78
    top_y = floor_y - th
    add_surface(
        (tx - tw / 2, top_y, tz - td / 2),
        (tw, 0, 0),
        (0, 0, td),
        (0, -1, 0),
        (0.55, 0.36, 0.20),
        int(n_table * 0.7),
        scale_uv=(0.022, 0.022),
    )
    for sx in (-1, 1):
        for sz in (-1, 1):
            add_surface(
                (tx + sx * (tw / 2 - 0.08), top_y, tz + sz * (td / 2 - 0.08)),
                (0.05, 0, 0),
                (0, th, 0),
                (0, 0, sz),
                (0.42, 0.27, 0.15),
                max(200, int(n_table * 0.075)),
                scale_uv=(0.02, 0.02),
            )

    # 桌上物体：一个球（台灯）、一个盒子（证据箱）、一瓶水
    n_sphere = int(n_objects * 0.4)
    n_box = int(n_objects * 0.35)
    n_bottle = n_objects - n_sphere - n_box

    cx, cy, cz, r = tx - 0.45, top_y - 0.12, tz + 0.1, 0.12
    for _ in range(n_sphere):
        ux, uy, uz = rand_unit(rng)
        pts.append((cx + ux * r, cy + uy * r, cz + uz * r))
        s = 0.016 * rng.uniform(0.8, 1.2)
        scl.append((s, s, s))
        qts.append(quat_from_normal(ux, uy, uz))
        cols.append((0.92, 0.90, 0.55))
        ops.append(0.95)

    bx, by, bz = tx + 0.35, top_y - 0.09, tz - 0.1
    bw, bh, bd = 0.36, 0.18, 0.26
    faces = [
        ((bx - bw / 2, by, bz - bd / 2), (bw, 0, 0), (0, 0, bd), (0, -1, 0)),
        ((bx - bw / 2, by - bh, bz - bd / 2), (bw, 0, 0), (0, 0, bd), (0, 1, 0)),
        ((bx - bw / 2, by, bz - bd / 2), (0, 0, bd), (0, -bh, 0), (-1, 0, 0)),
        ((bx + bw / 2, by, bz - bd / 2), (0, 0, bd), (0, -bh, 0), (1, 0, 0)),
        ((bx - bw / 2, by, bz - bd / 2), (bw, 0, 0), (0, -bh, 0), (0, 0, -1)),
        ((bx - bw / 2, by, bz + bd / 2), (bw, 0, 0), (0, -bh, 0), (0, 0, 1)),
    ]
    per_face = max(120, n_box // len(faces))
    for origin, u, v, nrm in faces:
        add_surface(origin, u, v, nrm, (0.15, 0.45, 0.75), per_face, scale_uv=(0.018, 0.018))

    for _ in range(n_bottle):
        ang = rng.uniform(0, math.pi * 2)
        hh = rng.uniform(-0.09, 0.09)
        br = 0.032
        pts.append((tx + 0.75 + math.cos(ang) * br, top_y - 0.09 + hh, tz + 0.28 + math.sin(ang) * br))
        s = 0.012 * rng.uniform(0.8, 1.2)
        scl.append((s, s, s))
        qts.append(quat_from_normal(math.cos(ang), 0, math.sin(ang)))
        cols.append((0.35, 0.75, 0.85))
        ops.append(0.9)

    # 门框（后墙上的门洞轮廓，用亮色细高斯强调）
    door_w, door_h = 1.0, 2.1
    door_x = -W / 2 + 0.9
    n_door = max(600, int(base * 0.01))
    for i in range(n_door):
        t = i / n_door
        if t < 0.33:
            px = door_x + (door_w * (t / 0.33))
            py = floor_y - door_h
        elif t < 0.66:
            px = door_x + door_w
            py = floor_y - door_h + (door_h * ((t - 0.33) / 0.33))
        else:
            px = door_x + door_w * (1 - (t - 0.66) / 0.34)
            py = floor_y
        pts.append((px + rng.gauss(0, 0.006), py + rng.gauss(0, 0.006), D / 2 - 0.01))
        s = 0.018 * rng.uniform(0.8, 1.2)
        scl.append((s, s, s * 0.3))
        qts.append(quat_from_normal(0, 0, -1))
        cols.append((0.88, 0.86, 0.82))
        ops.append(0.95)

    # 地面血迹（深红高不透明斑点，作为物证）
    n_blood = max(400, int(base * 0.006))
    sx0, sz0 = -1.6, 1.4
    for _ in range(n_blood):
        a = rng.gauss(0, 0.16)
        b = rng.gauss(0, 0.11)
        pts.append((sx0 + a, floor_y - 0.004 + rng.gauss(0, 0.0015), sz0 + b))
        s = 0.012 * rng.uniform(0.6, 1.5)
        scl.append((s, s, s * 0.25))
        qts.append(quat_from_normal(0, -1, 0))
        d = math.hypot(a / 0.16, b / 0.11)
        shade = max(0.25, 1.0 - d * 0.12)
        cols.append((0.45 * shade, 0.05 * shade, 0.05 * shade))
        ops.append(0.98)

    n = len(pts)
    print(f"生成高斯数量：{n}")
    data = pack_splat(pts, scl, qts, cols, ops)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    mn = [min(xs), min(ys), min(zs)]
    mx = [max(xs), max(ys), max(zs)]
    center = [(mn[i] + mx[i]) / 2 for i in range(3)]
    size = [mx[i] - mn[i] for i in range(3)]
    radius = 0.5 * math.sqrt(sum(s * s for s in size))

    meta = {
        "name": out.name,
        "format": "splat",
        "bytesPerGaussian": 32,
        "count": n,
        "shDegree": 0,
        "source": "tools/make_sample_scene.py 合成场景",
        "createdAt": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "units": "meters",
        "axes": {"system": "COLMAP 3DGS", "x": "right", "y": "down", "z": "forward"},
        "bounds": {"min": mn, "max": mx, "center": center, "size": size, "radius": radius},
        "scene": {"room": [W, D, H], "table": {"center": [tx, top_y, tz], "size": [tw, th, td]}},
    }
    jpath = Path(args.json) if args.json else out.with_suffix(".json")
    jpath.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 附带一组环绕相机位姿（COLMAP 系，y 向下）
    cams = []
    for i in range(8):
        ang = i / 8 * math.pi * 2
        r_cam = 3.4
        pos = [math.sin(ang) * r_cam * 0.6, 0.1 + 0.25 * math.sin(ang * 2), math.cos(ang) * r_cam * 0.5 - 0.4]
        tgt = [0.0, -0.2, -0.2]
        cams.append({"name": f"环绕机位 {i + 1}", "position": pos, "target": tgt, "fovY": 55, "aspect": 16 / 9})
    (out.with_name(out.stem + "-cameras.json")).write_text(
        json.dumps({"cameras": cams, "axes": "COLMAP"}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"已写出：{out}  ({len(data) / 1048576:.2f} MB)")
    print(f"元数据：{jpath}")
    print(f"包围盒 min={[round(v, 3) for v in mn]} max={[round(v, 3) for v in mx]}")
    print(f"半径≈{radius:.2f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
