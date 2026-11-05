#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成一个**两层柜**的合成拍摄（纹理四边形渲染 → PPM），用于端到端验证"物体"这一路。

为什么要这个
------------
物体盒的"可信判据"与渲染门控都做好了，但**真实素材里没有一份可信的物体重建**
（柜子那份算出来 16.8 m 长，trust=false），于是"可信 → 画出来"这条路径
从没在**真实重建产物**上跑过。这个工具补上：造一个尺寸已知的柜子 →
渲染一圈视角 → 交给真 COLMAP 重建 → 再看结构提取给出的盒子对不对。

柜子几何（**真值**，写进 truth.json）
------------------------------------
    宽 W=1.2 (x) · 深 D=0.6 (y) · 高 H=1.9 (z)，落地 z=0
    三块水平面：底(0) / 中间搁板(0.95) / 顶(1.9)   ← 两层柜 = 中间隔一层
    前脸开口 + 一圈门框；背面/两侧/顶底**双面**都出图
    （背面剔除会按视角各留一面，所以内部与外部都能被正确渲染）

相机：绕柜子一圈（半径 ~2.3 m，眼高 ~柜子中心，带轻微抖动），**每帧都看向柜心**。

用法:
    pyruntime312\\python.exe tools/make_object_dataset.py --out data/object-cabinet --views 36
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "tools", ROOT / "pipeline"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

# 复用已验证过的渲染零件（纹理生成 / 相机约定 / 四边形贴图 / PPM 写出）
from make_textured_dataset import (  # noqa: E402
    look_at,
    make_texture,
    quad_mask,
    rotmat_to_quat,
    warp_quad,
    write_ppm,
)


def rect_quad(axis: str, a: float, b: float, c: float, d: float, plane: float):
    """轴对齐矩形 → 4 个角点（按朝向 +法向 逆时针给出）。

    axis='x' 表示平面 x=plane，另两轴范围 a..b（y）、c..d（z）；其余同理。
    """
    if axis == "x":
        return [(plane, a, c), (plane, b, c), (plane, b, d), (plane, a, d)]
    if axis == "y":
        return [(a, plane, c), (b, plane, c), (b, plane, d), (a, plane, d)]
    return [(a, c, plane), (b, c, plane), (b, d, plane), (a, d, plane)]


def add_double(panels, verts, normal, name):
    """同一个矩形出**两面**（法向相反的复制）。

    为什么必须双面：柜子前脸是开口的，从前面看进去要能看到背板**内侧**；
    而背板的朝外面在正面视角里会被背面剔除掉。两面都提供之后，
    背面剔除会**自动**为每个视角挑出正确的那一面 —— 不用自己去判断"里/外"。
    """
    panels.append((name + "·out", verts, normal))
    panels.append((name + "·in", verts[::-1], tuple(-n for n in normal)))


def build_cabinet(W: float, D: float, H: float, frame: float = 0.08):
    """两层柜：外壳 + 中间搁板 + 前脸门框。返回 (panels, 真值信息)。"""
    x0, x1 = -W / 2, W / 2
    y0, y1 = -D / 2, D / 2          # y0=背面，y1=前脸
    z0, z1 = 0.0, H
    zm = H / 2
    panels: list[tuple[str, list, tuple]] = []

    add_double(panels, rect_quad("y", x0, x1, z0, z1, y0), (0, -1, 0), "back")
    add_double(panels, rect_quad("x", y0, y1, z0, z1, x0), (-1, 0, 0), "left")
    add_double(panels, rect_quad("x", y0, y1, z0, z1, x1), (1, 0, 0), "right")
    add_double(panels, rect_quad("z", x0, x1, y0, y1, z1), (0, 0, 1), "top")
    add_double(panels, rect_quad("z", x0, x1, y0, y1, z0), (0, 0, -1), "bottom")
    # 中间搁板 = 两层柜的"两层"分界（也是第 2 块水平面）
    add_double(panels, rect_quad("z", x0, x1, y0, y1, zm), (0, 0, 1), "shelf")
    # 前脸门框：左右两根立梃 + 上下两根横档（给正面视角提供纹理与视差）
    add_double(panels, rect_quad("y", x0, x0 + frame, z0, z1, y1), (0, 1, 0), "stileL")
    add_double(panels, rect_quad("y", x1 - frame, x1, z0, z1, y1), (0, 1, 0), "stileR")
    add_double(panels, rect_quad("y", x0 + frame, x1 - frame, z1 - frame, z1, y1), (0, 1, 0), "railTop")
    add_double(panels, rect_quad("y", x0 + frame, x1 - frame, z0, z0 + frame, y1), (0, 1, 0), "railBot")

    truth = {
        "size": [W, D, H],
        "center": [0.0, 0.0, H / 2],
        "horizontalPlanesZ": [z0, zm, z1],   # 底 / 搁板 / 顶（层检测应当找出 3 个）
        "note": "两层柜：外壳双面出图 + 中间搁板 + 前脸门框；三块水平面是层的真值",
    }
    return panels, truth


def frames_ok(r, ang, target, corners, fx, fy, W, H, hcam, *, margin=0.88) -> bool:
    """给定环绕半径 r，柜子的 8 个角点是否都落在画面内（留 margin 比例的边距）？"""
    eye = np.array([r * np.cos(ang), r * np.sin(ang), hcam])
    R, t = look_at(eye, target, np.array([0.0, 0.0, 1.0]))
    for c in corners:
        xc = R @ c + t
        if xc[2] <= 1e-6:
            return False
        u = fx * xc[0] / xc[2] + W / 2.0
        v = fy * xc[1] / xc[2] + H / 2.0
        if not ((1 - margin) * W / 2 <= u <= (1 + margin) * W / 2):
            return False
        if not ((1 - margin) * H / 2 <= v <= (1 + margin) * H / 2):
            return False
    return True


def fit_radius(ang, target, corners, fx, fy, W, H, hcam, *, margin=0.88, r_lo=1.2, r_hi=12.0):
    """求**刚好装得下整个柜子**的环绕半径（每个视角自己算）。

    为什么不让调用者手填半径：第一版用固定 2.3 m，结果 1.9 m 高的柜子
    **近处的角点出画**（投影 y 到 -135..855，画面只有 720 高）——
    36 个视角里 29 个被自检拦掉。物体越大、越靠近相机，近角点越容易出画，
    所以这里按"所有角点都在画面内且留边距"反解半径。
    """
    for r in np.linspace(r_lo, r_hi, 160):
        if frames_ok(r, ang, target, corners, fx, fy, W, H, hcam, margin=margin):
            return float(r)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/object-cabinet")
    ap.add_argument("--views", type=int, default=36)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=70.0, help="水平视场角（度）")
    ap.add_argument("--radius", type=float, default=2.3, help="环绕半径（米，仅作参考；实际每个视角自动求）")
    ap.add_argument(
        "--elevation", type=float, default=0.0,
        help="眼高相对柜子中心的**起伏幅度**（米）。0 = 一直平视（真实「站着拍柜子」的样子，"
             "搁板是边缘对着镜头 → 层检测必然找不出层）；0.55 左右就能俯仰看到搁板面",
    )
    ap.add_argument("--elevation-bias", type=float, default=0.0,
                    help="眼高整体抬高量（米）。抬高 + 俯视更容易拍到搁板面")
    ap.add_argument("--aim-sweep", type=float, default=0.0,
                    help="瞄点相对柜心的上下移动幅度（米）：探头看柜内用 0.45 左右")
    ap.add_argument("--cabinet", default="1.2,0.6,1.9", help="柜子 宽,深,高（米）")
    ap.add_argument("--tex-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    W_m, D_m, H_m = [float(v) for v in args.cabinet.split(",")]
    out = Path(args.out).resolve()
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    W, H = args.width, args.height
    fx = fy = 0.5 * W / np.tan(np.radians(args.hfov / 2))
    cx_p, cy_p = W / 2.0, H / 2.0
    panels, truth = build_cabinet(W_m, D_m, H_m)
    textures = {name: make_texture(args.tex_size, args.seed * 1000 + i) for i, (name, _, _) in enumerate(panels)}
    print(f"柜子 {W_m}×{D_m}×{H_m} m，面片 {len(panels)} 个（双面），内参 fx={fx:.1f} {W}×{H}")

    target = np.array(truth["center"], dtype=float)
    corners = np.array(
        [[sx * W_m / 2, sy * D_m / 2, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (0, H_m)],
        dtype=float,
    )

    rng0 = np.random.default_rng(args.seed)
    cameras = []
    problems = []
    for i in range(args.views):
        ang = 2 * np.pi * i / args.views + rng0.normal(0, 0.03)
        # 眼高：`--elevation` 控制起伏幅度。
        # ⚠️ 为什么必须有这个开关：端到端测试发现，**一直平视**拍柜子时，
        #    三块水平面（底/搁板/顶）全是**边缘对着镜头**（掠射角），
        #    重建出来的点根本不集中在那些高度上 —— 竖直直方图几乎没有峰，
        #    层检测报 0 层。这不是算法的错，是**拍摄方式的物理限制**。
        #    所以生成器要能同时造出"平视（失败）"与"带俯仰（成功）"两种素材。
        elev = args.elevation * np.sin(2 * ang) + args.elevation_bias
        z = target[2] + elev + rng0.normal(0, 0.10)   # 眼高起伏 + 轻微手持抖动
        # 瞄点随俯仰上下移动：高机位看上层格、低机位看下层格
        # （真实"探头往柜子里看"的行为；一直盯柜心会让搁板永远是掠射）
        aim = target + np.array([0.0, 0.0, args.aim_sweep * np.sin(2 * ang)])
        # 自动求"刚好装下柜子"的距离（见 fit_radius 的说明：固定半径会让近角点出画）
        # 瞄点偏离柜心时（探头看柜内），柜子本来就会偏向画面一侧 ——
        # 这时放宽边距（否则"八角都在画面内"的要求会把大多数视角判废）
        # ⚠️ 必须把**同一个瞄点**交给拟合与自检：第一版这里传的是柜心 target，
        #    而渲染用的是 aim，两者不一致 → 拟合出的半径与实际画面对不上，
        #    36 个视角只剩 5 个通过（自检报"角点出画"）。
        mg = 0.98 if args.aim_sweep else 0.88
        r = fit_radius(ang, aim, corners, fx, fy, W, H, z, margin=mg, r_hi=24.0)
        if r is None:
            problems.append(f"view_{i + 1:03d}: 找不到能装下柜子的机位（{args.cabinet} 太近/太大）")
            continue
        # ⚠️ 抖动必须在**拟合边距之内**。第一版按 0.96 的边距拟合完再乘 ±4%，
        #    结果视角又被挤出画框（自检当场报出来）。
        #    这里再加一道保险：抖动后若又出画，就**退回拟合出来的那个半径**，
        #    宁可少一点抖动，也不要交出一张缺角的图。
        r_fit = r
        r = r_fit * (1 + rng0.normal(0, 0.04))
        if not frames_ok(r, ang, aim, corners, fx, fy, W, H, z, margin=mg):
            r = r_fit          # 抖动过头就退回拟合值（宁可少抖动，也不交出缺角的图）
        eye = np.array([r * np.cos(ang), r * np.sin(ang), z])
        R, t = look_at(eye, aim, np.array([0.0, 0.0, 1.0]))   # 与既有合成数据同一套相机约定

        # 自检：柜子的 8 个角点必须**都在画面内**（否则重建会因为缺角而形状不对）
        pts2 = []
        for c in corners:
            xc = R @ c + t
            if xc[2] <= 1e-6:
                pts2 = None
                break
            pts2.append((fx * xc[0] / xc[2] + cx_p, fy * xc[1] / xc[2] + cy_p))
        if pts2 is None:
            problems.append(f"view_{i + 1:03d}: 有的角点在相机后面")
            continue
        xs = [p[0] for p in pts2]
        ys = [p[1] for p in pts2]
        if min(xs) < 4 or min(ys) < 4 or max(xs) > W - 4 or max(ys) > H - 4:
            problems.append(
                f"view_{i + 1:03d}: 柜子角点出画（x {min(xs):.0f}..{max(xs):.0f} / "
                f"y {min(ys):.0f}..{max(ys):.0f}，画面 {W}×{H}）"
            )
            continue

        rng = np.random.default_rng(9000 + i)
        canvas = rng.integers(18, 48, (H, W, 3), dtype=np.uint8)
        drawable = []
        for name, verts, normal in panels:
            center = np.mean(np.array(verts), dtype=float, axis=0)
            if np.dot(np.array(normal, dtype=float), eye - center) <= 0:
                continue          # 背面剔除：这里会自动为每个视角挑出"里/外"正确的那一面
            proj, ok = [], True
            for v in verts:
                xc = R @ np.array(v, dtype=float) + t
                if xc[2] <= 1e-6:
                    ok = False
                    break
                proj.append((fx * xc[0] / xc[2] + cx_p, fy * xc[1] / xc[2] + cy_p))
            if not ok:
                continue
            drawable.append((float(np.linalg.norm(eye - center)), name, proj))
        drawable.sort(reverse=True)     # 远的先画（画家算法）

        for _, name, proj in drawable:
            xs = [p[0] for p in proj]
            ys = [p[1] for p in proj]
            bx0 = max(0, int(np.floor(min(xs))))
            by0 = max(0, int(np.floor(min(ys))))
            bx1 = min(W, int(np.ceil(max(xs))) + 1)
            by1 = min(H, int(np.ceil(max(ys))) + 1)
            if bx1 - bx0 < 2 or by1 - by0 < 2:
                continue
            bbox = (bx0, by0, bx1, by1)
            before = canvas[bbox[1] : bbox[3], bbox[0] : bbox[2]].copy()
            warp_quad(canvas, textures[name], proj, bbox)
            after = canvas[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            inside = quad_mask(bbox, proj)
            after[~inside] = before[~inside]
            canvas[bbox[1] : bbox[3], bbox[0] : bbox[2]] = after

        canvas = np.clip(
            canvas.astype(np.int16) + rng.integers(-6, 7, canvas.shape, dtype=np.int16), 0, 255
        ).astype(np.uint8)
        fname = f"view_{i + 1:03d}.ppm"
        write_ppm(str(img_dir / fname), canvas)
        cameras.append({"name": fname, "R": R.tolist(), "t": t.tolist(), "eye": eye.tolist()})
        if (i + 1) % 12 == 0:
            print(f"  已写出 {i + 1}/{args.views}")

    if problems:
        print("自检未通过（柜子出画）—— 已跳过这些视角：")
        for p in problems[:8]:
            print("  ✘ " + p)
    if len(cameras) < 8:
        print(f"只生成了 {len(cameras)} 个有效视角，不足以重建")
        return 1

    truth.update({
        "intrinsics": {"model": "PINHOLE", "width": W, "height": H, "fx": fx, "fy": fy, "cx": cx_p, "cy": cy_p},
        "radiusMeters": args.radius,
        "views": len(cameras),
        "cameras": cameras,
    })
    (out / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
    # 文本格式的真值位姿（便于人工核对；COLMAP 只读 images/ 目录，不会碰它）
    with open(out / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {fx:.6f} {fy:.6f} {cx_p:.6f} {cy_p:.6f}\n")
    with open(out / "images.txt", "w", encoding="utf-8") as f:
        for idx, cam in enumerate(cameras, start=1):
            R = np.array(cam["R"], dtype=float)
            t = np.array(cam["t"], dtype=float)
            q = rotmat_to_quat(R)
            f.write(f"{idx} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                    f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {cam['name']}\n\n")

    print(f"\n已写出 {len(cameras)} 个视角到 {img_dir}")
    print(f"真值: {out}\\truth.json（柜子 {W_m}×{D_m}×{H_m} m，水平面 z={truth['horizontalPlanesZ']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
