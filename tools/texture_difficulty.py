#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
判断训练集纹理的「可重建难度」。

思路：如果一张**纯色图**（只取训练集平均色）的 PSNR 已经很高，
说明画面本身接近常量，谁也重建不差；
如果纯色图的 PSNR 很低（例如 8 dB），说明画面接近随机噪声 ——
这种内容任何 3DGS 都重建不好，模型的高 PSNR 上限本来就被素材锁死了。

用法：python tools/texture_difficulty.py --images <图片目录>
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))


def read_image(path):
    """复用本项目的图像读取（PIL → imageio → OpenCV → 内置 PNG/PPM 解码）。"""
    try:
        import extract_frames  # noqa: PLC0415

        return extract_frames.load_image_pixels(path)
    except Exception:  # noqa: BLE001
        pass
    try:
        from PIL import Image  # noqa: PLC0415

        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:  # noqa: BLE001
        pass
    try:
        import imageio.v2 as imageio  # noqa: PLC0415

        a = imageio.imread(path)
        return np.asarray(a)[..., :3]
    except Exception:  # noqa: BLE001
        return None


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((255.0**2) / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--max", type=int, default=8)
    args = ap.parse_args()

    files = sorted(
        f
        for f in glob.glob(os.path.join(args.images, "*"))
        if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".ppm", ".pgm", ".bmp", ".tif", ".tiff", ".webp")
    )[: args.max]
    if not files:
        print("没找到图片")
        return 1

    imgs = []
    for f in files:
        a = read_image(f)
        if a is None:
            print(f"  读不了 {f}")
            continue
        imgs.append(a.astype(np.float32))
    if not imgs:
        print("一张都读不了")
        return 1

    mean_img = np.mean(np.stack(imgs), axis=0)  # 所有图的均值，作为「常量图」基线

    print(f"图片数 {len(imgs)}   尺寸 {imgs[0].shape[1]}x{imgs[0].shape[0]}")
    print()
    print("=== 常量图基线 PSNR（越高说明画面越平坦、越好重建）===")
    vals = []
    for f, a in zip(files, imgs):
        # 用全局均值当预测，衡量这张图自身偏离常量的程度
        p = psnr(a, mean_img)
        vals.append(p)
        # 另算「逐图均值」——即只用一个颜色去预测这张图
        flat = np.full_like(a, a.mean())
        p2 = psnr(a, flat)
        # 高频能量占比：相邻像素差分的能量 / 总能量（接近噪声的图会很高）
        d = np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()
        print(
            f"  {os.path.basename(f):18s} 常量图PSNR {p2:5.2f} dB   "
            f"相邻像素平均差 {d:6.2f}（越大越高频/越像噪声）"
        )

    avg_flat = float(np.mean(vals))
    print()
    print(f"平均常量图 PSNR: {avg_flat:.2f} dB")
    print()
    print("=== 结论 ===")
    if avg_flat < 12:
        print("  ✘ 画面接近随机噪声：任何 3DGS 都难以重建出高 PSNR，")
        print("     此时 20 dB 左右基本就是这套素材的上限，问题在素材不在算法。")
    elif avg_flat < 18:
        print("  ⚠ 画面纹理偏强（细节密集），重建难度较大，PSNR 天花板不高。")
    else:
        print("  ✔ 画面较平坦，3DGS 应当能给出较高的 PSNR。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
