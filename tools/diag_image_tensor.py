#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断训练用的图像张量是否正常：亮度、通道顺序、是否为目标图。

「渲染一片黑但 loss 很低」这类怪象，常见原因是：
  · 图像读取路径把颜色通道搞错（BGR / 灰度误判）
  · 归一化倍数错误
  · 评估时拿错张量（比如把 alpha 当 RGB）

用法: python tools/diag_image_tensor.py --data <场景目录> [--images <图片目录>]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402
import extract_frames  # noqa: E402
import train_gsplat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--downscale", type=int, default=3)
    ap.add_argument("--count", type=int, default=6)
    args = ap.parse_args()

    scene = train_gsplat.parse_scene(Path(args.data), args.images) if hasattr(train_gsplat, "parse_scene") else None
    if scene is None:
        # 回退：直接用 resolve_scene_paths
        imgs_dir, model_dir = train_gsplat.resolve_scene_paths(Path(args.data), args.images)
        print(f"图片目录: {imgs_dir}")
        print(f"模型目录: {model_dir}")
        files = sorted(
            p for p in imgs_dir.iterdir() if p.is_file() and p.suffix.lower() in train_gsplat.IMAGE_SUFFIXES
        )[: args.count]
    else:
        files = [f.path for f in scene.frames[: args.count]]

    print(f"\n=== 前 {len(files)} 张图的张量统计 ===")
    for p in files:
        try:
            t = train_gsplat.image_to_tensor(p, args.downscale, None, None)
        except Exception as exc:  # noqa: BLE001
            print(f"  {p.name}: 读取失败 {type(exc).__name__}: {exc}")
            continue
        arr = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
        print(
            f"  {p.name:22s} shape {tuple(arr.shape)}  "
            f"min {arr.min():.3f}  mean {arr.mean():.3f}  max {arr.max():.3f}  "
            f"通道均值 {np.round(arr.reshape(3, -1).mean(1), 3)}"
        )

    # 直接用原始解码器读第一张，作为对照
    if files:
        p = files[0]
        print(f"\n=== 对照：用 extract_frames 的解码器直接读 {p.name} ===")
        raw = None
        for name in ("load_image_pixels", "read_image_pixels", "decode_image"):
            fn = getattr(extract_frames, name, None)
            if fn:
                try:
                    raw = fn(p)
                    print(f"  用 {name}() 读取成功")
                except Exception as exc:  # noqa: BLE001
                    print(f"  {name}() 失败: {exc}")
                if raw is not None:
                    break
        if raw is not None:
            a = np.asarray(raw)
            print(f"  shape {a.shape} dtype {a.dtype} min {a.min()} mean {a.mean():.2f} max {a.max()}")
            if a.ndim == 3:
                print(f"  通道均值 {np.round(a.reshape(-1, a.shape[2]).mean(0), 2)}")
        else:
            print("  没有可用的解码函数名，跳过")


if __name__ == "__main__":
    main()
