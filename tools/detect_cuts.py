#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检测视频帧序列里的「硬切」（镜头跳变）。

为什么重要：sequential_matcher 依赖相邻帧连续。如果视频里有硬切
（比如手机录制中断、反复按暂停/继续），相邻两帧完全没有重叠，
顺序匹配就会在切点断掉，导致后面整段无法注册。

方法：比较相邻帧的「降采样灰度直方图」与「平均绝对差」。
        连续运动  → 差异小且平滑
        硬切      → 差异出现明显尖峰

用法: python tools/detect_cuts.py --images <帧目录> [--threshold 0.35]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))


def load_gray(path, size=64):
    try:
        import imageio.v2 as imageio  # noqa: PLC0415

        a = np.asarray(imageio.imread(path))
    except Exception:  # noqa: BLE001
        try:
            from PIL import Image  # noqa: PLC0415

            a = np.asarray(Image.open(path).convert("RGB"))
        except Exception:  # noqa: BLE001
            return None
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    h, w = a.shape[:2]
    ys = np.linspace(0, h - 1, min(size, h)).astype(int)
    xs = np.linspace(0, w - 1, min(size, w)).astype(int)
    return a[np.ix_(ys, xs)].astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--threshold", type=float, default=0.30, help="判定为硬切的平均绝对差阈值")
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, "*.png")))
    if len(files) < 3:
        files = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    if len(files) < 3:
        print("帧数太少")
        return 1

    frames = [load_gray(f, args.size) for f in files]
    valid = [i for i, f in enumerate(frames) if f is not None]
    print(f"帧数 {len(files)}（可读 {len(valid)}）")
    print()

    diffs = []
    for i in range(1, len(frames)):
        if frames[i] is None or frames[i - 1] is None:
            diffs.append(None)
            continue
        d = float(np.abs(frames[i] - frames[i - 1]).mean())
        diffs.append(d)

    nums = [d for d in diffs if d is not None]
    med = float(np.median(nums))
    p90 = float(np.percentile(nums, 90))
    print("=== 相邻帧差异 ===")
    print(f"  中位 {med:.4f}   P90 {p90:.4f}   最大 {max(nums):.4f}")
    print()

    cuts = [(i, d) for i, d in enumerate(diffs, start=1) if d is not None and d > args.threshold]
    print(f"=== 疑似硬切（差异 > {args.threshold}）===")
    if not cuts:
        print("  没有检测到硬切，序列连续 ✔")
    else:
        for i, d in cuts:
            print(f"  第 {i} 帧 与 第 {i-1} 帧 之间：差异 {d:.4f}  ({os.path.basename(files[i-1])} -> {os.path.basename(files[i])})")
        print()
        print(f"  共 {len(cuts)} 处硬切。建议：")
        print(f"    · 用 --start/--end 只取连续段，或把视频切成多段分别重建")
        print(f"    · 顺序匹配对硬切很敏感（这正是注册率上不去的原因之一）")
        print(f"    · 改穷举匹配 --matcher exhaustive 能跨过硬切，但慢一些")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
