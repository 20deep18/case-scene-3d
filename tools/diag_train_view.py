#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
关键诊断：训练出的模型能否复现**训练视角**？

如果训练视角也渲染成黑的，说明模型/光栅化有问题；
如果训练视角很好、只有留出视角差，说明是「过拟合 + 视角外推」问题
（纯旋转拍摄时常见：模型只记住了训练过的朝向）。

用法: python tools/diag_train_view.py --data <场景> --images <图片目录> --iterations 300
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import train_gsplat  # noqa: E402


def build_trainer(data, images, iterations, downscale, out, lr_factor):
    args = train_gsplat.build_parser().parse_args(
        [
            "--data", str(data),
            "--out", str(out),
            "--iterations", str(iterations),
            "--downscale", str(downscale),
            "--eval-views", "0",
            "--save-every", "0",
            "--log-every", str(iterations),
            "--lr-scale-factor", str(lr_factor),
        ]
    )
    if images:
        args.images = str(images)
    scene = train_gsplat.load_scene(Path(data), str(images) if images else None)
    trainer = train_gsplat.Trainer(scene, Path(out), args)
    return trainer, args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--downscale", type=int, default=3)
    ap.add_argument("--out", default="output/diag-tv")
    ap.add_argument("--lr-scale-factor", type=float, default=0.5)
    args = ap.parse_args()

    trainer, targs = build_trainer(
        args.data, args.images, args.iterations, args.downscale, args.out, args.lr_scale_factor
    )
    print(f"帧数 {len(trainer.frames)}   高斯 {trainer.params['means'].shape[0]}")
    trainer.train()

    # 在**训练用到的**几个视角上评估
    print()
    print("=== 训练视角上的渲染质量（应当很好）===")
    # 同时对比「按 evaluate() 的取帧方式」拿到的视角，确认两者是否一致
    holdout_step = max(1, len(trainer.frames) // 6)
    eval_like = list(range(0, len(trainer.frames), holdout_step))[:6]
    print(f"  总帧数 {len(trainer.frames)}   evaluate() 会取这些下标: {eval_like}")
    indices = [0, 1, 2, 3] + eval_like
    seen = set()
    for idx in indices:
        if idx in seen or idx >= len(trainer.frames):
            continue
        seen.add(idx)
        frame = trainer.frames[idx]
        target = trainer.images[frame.image_id]
        h, w = int(target.shape[1]), int(target.shape[2])
        fx, fy, cx, cy = frame.intrinsics()
        s = args.downscale
        K = torch.tensor(
            [[fx / s, 0, cx / s], [0, fy / s, cy / s], [0, 0, 1]], dtype=torch.float32, device=trainer.device
        )
        R = torch.tensor(frame.image.rotation_matrix(), dtype=torch.float32, device=trainer.device)
        t = torch.tensor(frame.image.tvec, dtype=torch.float32, device=trainer.device)
        vm = torch.eye(4, dtype=torch.float32, device=trainer.device)
        vm[:3, :3] = R
        vm[:3, 3] = t
        with torch.no_grad():
            rendered, alpha = trainer.rasterize(trainer.max_sh_degree, vm, K, w, h)
            img = rendered.permute(2, 0, 1).clamp(0, 1)
            mse = torch.mean((img - target) ** 2).clamp_min(1e-10)
            psnr = float(-10 * torch.log10(mse))
            print(
                f"  {frame.name:20s} 渲染均值 {img.mean():.3f}  真值均值 {target.mean():.3f}  "
                f"alpha 均值 {alpha.mean():.3f}  PSNR {psnr:5.2f} dB"
            )
        if idx == 0:
            rgb = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            ref = (target.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pair = np.concatenate([rgb, ref], axis=0)
            p = Path(args.out) / "train_view0.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                from PIL import Image  # noqa: PLC0415

                Image.fromarray(pair).save(p)
            except Exception:  # noqa: BLE001
                import imageio.v2 as imageio  # noqa: PLC0415

                imageio.imwrite(p, pair)
            print(f"  对比图已写出：{p}")


if __name__ == "__main__":
    main()
