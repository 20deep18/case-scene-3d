#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断训练出的 3DGS PLY：尺度/不透明度/位置分布是否正常。"""
import struct
import sys
from pathlib import Path

import numpy as np


def read_ply(path):
    with open(path, "rb") as f:
        header = b""
        while b"end_header" not in header:
            header += f.readline()
        lines = header.decode("ascii", "replace").splitlines()
        n = 0
        props = []
        in_vertex = False
        for ln in lines:
            if ln.startswith("element vertex"):
                n = int(ln.split()[-1])
                in_vertex = True
            elif ln.startswith("element"):
                in_vertex = False
            elif ln.startswith("property") and in_vertex:
                props.append(ln.split()[-1])
        data = np.frombuffer(f.read(n * len(props) * 4), dtype=np.float32).reshape(n, len(props))
    return props, data


def main(path, scales_are_log=False):
    props, data = read_ply(path)
    idx = {name: i for i, name in enumerate(props)}
    print(f"文件      : {path}")
    print(f"高斯数量  : {len(data):,}")
    print(f"属性      : {len(props)} 个")
    print()

    xyz = data[:, [idx["x"], idx["y"], idx["z"]]]
    # 标准 3DGS PLY 的 scale_* 是**线性**尺度（已激活），不要再取 exp。
    # 只有早期 train_gsplat.py 的产物是 log 值，那种情况加 --scales-are-log。
    scales = data[:, [idx[f"scale_{i}"] for i in range(3)]]
    if scales_are_log:
        scales = np.exp(scales)
    opac = 1.0 / (1.0 + np.exp(-data[:, idx["opacity"]]))
    max_scale = scales.max(axis=1)

    print("=== 位置分布 ===")
    print(f"  min    {np.round(xyz.min(axis=0), 3)}")
    print(f"  max    {np.round(xyz.max(axis=0), 3)}")
    print(f"  包围盒对角线 {(np.linalg.norm(xyz.max(axis=0)-xyz.min(axis=0))):.2f} m")
    print(f"  中位点 {np.round(np.median(xyz, axis=0), 3)}")
    print()

    print("=== 尺度分布（世界单位，米）===")
    for q in (1, 5, 25, 50, 75, 90, 95, 99):
        print(f"  P{q:<3} max_scale = {np.percentile(max_scale, q):9.4f}")
    print(f"  >0.1m  : {(max_scale > 0.1).sum():>8,}  ({(max_scale>0.1).mean()*100:5.1f}%)")
    print(f"  >0.5m  : {(max_scale > 0.5).sum():>8,}  ({(max_scale>0.5).mean()*100:5.1f}%)")
    print(f"  >1.0m  : {(max_scale > 1.0).sum():>8,}  ({(max_scale>1.0).mean()*100:5.1f}%)")
    print(f"  >5.0m  : {(max_scale > 5.0).sum():>8,}  ({(max_scale>5.0).mean()*100:5.1f}%)")
    print()

    print("=== 不透明度分布 ===")
    for q in (5, 25, 50, 75, 95):
        print(f"  P{q:<3} = {np.percentile(opac, q):.4f}")
    print(f"  >0.5 : {(opac > 0.5).sum():,} ({(opac>0.5).mean()*100:.1f}%)")
    print()

    print("=== 判断 ===")
    frac_big = (max_scale > 1.0).mean()
    if frac_big > 0.3:
        print(f"  ✘ 尺度失控：{frac_big*100:.0f}% 的高斯大于 1 米，模型被「大气团」撑坏")
        print("     → 训练缺少尺度正则 / 尺度 clamp，需要修")
    elif frac_big > 0.05:
        print(f"  ⚠ 部分高斯偏大（{frac_big*100:.0f}% > 1m），建议加尺度约束")
    else:
        print("  ✔ 尺度分布正常")


if __name__ == "__main__":
    main(Path(sys.argv[1]), scales_are_log="--scales-are-log" in sys.argv)
