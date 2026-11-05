#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一致性检查：训练出的高斯位置，是否与 COLMAP 的相机/稀疏点云处在同一个空间范围。

这是判断「训练是否把模型撑坏」最直接的证据 ——
如果高斯的包围盒比相机+稀疏点云的包围盒大出好几倍，说明尺度/位置失控。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colmap_model_io as io  # noqa: E402


def read_ply_positions(path):
    d = Path(path).read_bytes()
    i = d.find(b"end_header")
    j = i + len(b"end_header")
    while d[j : j + 1] in (b"\n", b"\r"):
        j += 1
    hdr = d[:i].decode("ascii", "replace")
    names = [ln.split()[-1] for ln in hdr.splitlines() if ln.startswith("property")]
    n = int([ln for ln in hdr.splitlines() if ln.startswith("element vertex")][0].split()[-1])
    dt = np.dtype([(x, "<f4") for x in names])
    a = np.frombuffer(d[j : j + n * dt.itemsize], dtype=dt)
    pos = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    scl = np.stack([a["scale_0"], a["scale_1"], a["scale_2"]], 1).astype(np.float64)
    opa = 1.0 / (1.0 + np.exp(-a["opacity"].astype(np.float64)))
    return pos, scl, opa


def main(sparse_dir, ply_path):
    sp = Path(sparse_dir)
    imgs = io.read_images(sp / "images.bin")
    cams = np.array([io.camera_center(i["qvec"], i["tvec"]) for i in imgs])
    pts = io.read_points3d(sp / "points3D.bin")
    # read_points3d 返回的是「按列打包」的字典（ids/xyz/rgb/error/track_length）
    xyz = np.asarray(pts["xyz"], dtype=np.float64)

    print("=== COLMAP 空间范围 ===")
    print(f"  相机光心  min {np.round(cams.min(0),2)}  max {np.round(cams.max(0),2)}")
    print(f"  稀疏点云  min {np.round(xyz.min(0),2)}  max {np.round(xyz.max(0),2)}")
    ref_lo = np.minimum(cams.min(0), xyz.min(0))
    ref_hi = np.maximum(cams.max(0), xyz.max(0))
    print(f"  合并范围  边长 {np.round(ref_hi-ref_lo,2)}")

    pos, scl, opa = read_ply_positions(ply_path)
    print()
    print("=== 训练出的高斯 ===")
    print(f"  数量      {len(pos):,}")
    print(f"  位置      min {np.round(pos.min(0),2)}  max {np.round(pos.max(0),2)}")
    print(f"  位置边长  {np.round(pos.max(0)-pos.min(0),2)}")
    print(f"  线性尺度  中位 {np.median(scl):.5f} m   中位最大轴 {np.median(scl.max(1)):.5f} m")
    print(f"  不透明度  中位 {np.median(opa):.4f}   >0.5 占 {(opa>0.5).mean()*100:.1f}%")

    inside = ((pos >= ref_lo) & (pos <= ref_hi)).all(1)
    print()
    print("=== 一致性判断 ===")
    span_g = (pos.max(0) - pos.min(0)).max()
    span_r = (ref_hi - ref_lo).max()
    print(f"  高斯跨度 / 参考跨度 = {span_g/span_r:.2f} 倍")
    print(f"  落在参考包围盒内的高斯: {inside.mean()*100:.1f}%")
    if span_g / span_r > 2.0:
        print("  ✘ 高斯范围远超场景 → 位置/尺度失控，模型被撑坏")
    elif inside.mean() < 0.8:
        print("  ⚠ 相当多高斯跑到场景外 → 可能有漂浮噪点")
    else:
        print("  ✔ 高斯范围与 COLMAP 场景一致，训练正常")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/colmap-verify/sparse/0", sys.argv[2])
