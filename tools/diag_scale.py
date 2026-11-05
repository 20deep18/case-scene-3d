#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断 COLMAP 稀疏点云的尺度特征，用于判断高斯初始尺度是否合理。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import torch  # noqa: E402

import colmap_recon  # noqa: E402

model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "output/colmap-verify/sparse/0")
model = colmap_recon.read_model(model_dir)
pts = torch.tensor([p.xyz for p in model.points3D.values()], dtype=torch.float32)
total = pts.shape[0]
print(f"点数: {total}")

span = (pts.max(0).values - pts.min(0).values).tolist()
print(f"包围盒范围(米): [{span[0]:.2f}, {span[1]:.2f}, {span[2]:.2f}]")
print(f"场景尺度(半对角线): {(pts.max(0).values - pts.min(0).values).norm().item()/2:.3f}")

# 复刻 _estimate_point_scale
g = torch.Generator().manual_seed(0)
idx = torch.randperm(total, generator=g)[: min(total, 8192)]
sample = pts[idx]
d = torch.cdist(pts, sample)
nn = d.topk(3, dim=1, largest=False).values
sc = nn.mean(dim=1)
print()
print("=== 3DGS 式初始尺度（到最近 3 点平均距离）===")
print(f"  min    {sc.min():.5f}")
print(f"  中位   {sc.median():.5f}")
print(f"  均值   {sc.mean():.5f}")
print(f"  P95    {sc.percentile(sc, 95):.5f}")
print(f"  max    {sc.max():.5f}")
print()
print("=== 对比：原先用的固定初始尺度 ===")
print("  旧代码写成 log(spatial_scale * 0.01) —— 实际见 _init_params 分支")
print()
# 高斯在该尺度下覆盖的像素大小估算：scale / (深度 * tan(fov/2)) * (H/2)
fx = 823.67 / 2  # downscale 2
depth = 6.0
for label, s in (("P25", sc.percentile(sc, 25)), ("中位", sc.median()), ("P95", sc.percentile(sc, 95))):
    px = s / depth * fx
    print(f"  {label} 尺度 {s:.4f} m 在 6m 远处约 {px:.2f} 像素（太小则重建不出细节）")
