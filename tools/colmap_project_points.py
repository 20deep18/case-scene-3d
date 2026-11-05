#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 COLMAP 自己解出的位姿，把稀疏点云重投影回原图，输出带标记的对比图。

这是验证「COLMAP 重建结果自洽」最直接的办法：
位姿与三维点若是正确的，把三维点按位姿投影回去应当正好落在原图的特征点上。

用法：
    pyruntime312\\python.exe tools/colmap_project_points.py ^
        --model output/colmap-pipeline/sparse/0 ^
        --images output/colmap-pipeline/images ^
        --out output/verify/colmap-reproj.png

不依赖 Pillow：输入支持 PPM；其他格式优先用 imageio，读不到就跳过。
"""

import argparse
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_model_io import intrinsics_of, quat_to_R, read_model  # noqa: E402


def read_ppm(path):
    with open(path, "rb") as f:
        if f.readline().strip() != b"P6":
            return None
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        f.readline()
        data = f.read(w * h * 3)
        if len(data) != w * h * 3:
            return None
        return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)


def read_image_any(path):
    """按 PPM → imageio 的顺序尝试读取，统一转成 JPEG 时自动处理通道数"""
    if os.path.exists(path) and path.lower().endswith((".ppm", ".pgm")):
        img = read_ppm(path)
        if img is not None:
            return img
    try:
        import imageio.v2 as imageio

        arr = np.asarray(imageio.imread(path))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr[:, :, :3]
    except Exception:  # noqa: BLE001
        return None


def write_png(path, arr):
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="output/verify/colmap-reproj.png")
    ap.add_argument("--tiles", type=int, default=4, help="拼几张图")
    ap.add_argument("--downscale", type=int, default=2, help="输出缩略图倍数")
    args = ap.parse_args()

    model = read_model(args.model)
    cams = model["cameras"]
    images = model["images"]
    xyz = model["points3D"]["xyz"]
    ids = model["points3D"]["ids"]
    id2row = {int(p): i for i, p in enumerate(ids)}
    print(f"相机 {len(cams)}，注册图像 {len(images)}，三维点 {len(xyz)}")

    # 精确对应关系需要 point3D_id，重新读一遍带 id 的版本
    from colmap_model_io import read_images as _read_images

    images_with_ids = _read_images(os.path.join(args.model, "images.bin"), with_points=True)

    tiles = []
    all_err = []
    for rec in images_with_ids[: args.tiles]:
        img = read_image_any(os.path.join(args.images, rec["name"]))
        if img is None:
            print(f"  跳过 {rec['name']}（读不到图片）")
            continue
        cam = cams[rec["camera_id"]]
        fx, fy, cx, cy = intrinsics_of(cam)
        R = quat_to_R(rec["qvec"])
        t = np.asarray(rec["tvec"], dtype=float)

        # 只用该图真正观测到的三维点算精确重投影误差
        obs = rec["points2D"]
        rows, obs_xy = [], []
        for ox, oy, pid in obs:
            pid = int(pid)
            if pid == 0xFFFFFFFFFFFFFFFF or pid not in id2row:
                continue
            rows.append(id2row[pid])
            obs_xy.append((ox, oy))
        errs = np.asarray([np.nan])
        if rows:
            P = xyz[np.asarray(rows)]
            Xc = (R @ P.T).T + t
            front = Xc[:, 2] > 1e-6
            uv = np.full((len(P), 2), np.nan)
            uv[front, 0] = fx * Xc[front, 0] / Xc[front, 2] + cx
            uv[front, 1] = fy * Xc[front, 1] / Xc[front, 2] + cy
            ob = np.asarray(obs_xy, dtype=float)
            errs = np.hypot(uv[:, 0] - ob[:, 0], uv[:, 1] - ob[:, 1])
            errs = errs[np.isfinite(errs)]

        H, W = img.shape[:2]
        sx = W / cam["width"]
        sy = H / cam["height"]
        canvas = img.copy()

        # 画全部三维点的投影（看整体覆盖）
        Xall = (R @ xyz.T).T + t
        front = Xall[:, 2] > 1e-6
        uall = np.full(len(xyz), np.nan)
        vall = np.full(len(xyz), np.nan)
        uall[front] = fx * Xall[front, 0] / Xall[front, 2] + cx
        vall[front] = fy * Xall[front, 1] / Xall[front, 2] + cy
        inside = 0
        for uu, vv in zip(uall, vall):
            if not np.isfinite(uu):
                continue
            x = int(round(uu * sx))
            y = int(round(vv * sy))
            if 1 <= x < W - 1 and 1 <= y < H - 1:
                inside += 1
                canvas[y, x] = (255, 40, 40)

        med = float(np.median(errs)) if len(errs) and np.isfinite(errs).any() else float("nan")
        all_err.append(med)
        print(
            f"  {rec['name']}: 有效观测 {len(rows)}，投影落在画面内 {inside}，"
            f"忽略畸变的重投影偏差 中位 {med:.4f}px"
        )

        step = max(1, args.downscale)
        tiles.append(canvas[: H - H % step, : W - W % step][::step, ::step])

    if tiles:
        h = min(t.shape[0] for t in tiles)
        w = min(t.shape[1] for t in tiles)
        sheet = np.concatenate([t[:h, :w] for t in tiles], axis=1)
        write_png(args.out, sheet)
        print(f"\n已写出 {args.out}（{sheet.shape[1]}x{sheet.shape[0]}）")
        valid = [e for e in all_err if np.isfinite(e)]
        if valid:
            print(
                f"各图「忽略畸变的重投影偏差」中位数平均 {np.mean(valid):.4f}px\n"
                "  说明：本脚本为了不依赖任何图像库，只做了针孔投影，**没有应用畸变系数**；\n"
                "  因此这个数值会明显大于 COLMAP 自报的误差，只适合用来确认「点位大致对得上」。\n"
                "  权威误差请用官方工具查看：\n"
                "      colmap model_analyzer --path <模型目录>\n"
                "  其中的 Mean reprojection error 才是带完整相机模型的结果。"
            )


if __name__ == "__main__":
    main()
