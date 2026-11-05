#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""直接写 COLMAP 的**二进制**模型（cameras.bin / images.bin / points3D.bin）。

为什么要自己写：`colmap model_converter` 在本机崩溃（0xC0000409），
而 pycolmap 读"带占位 track 的文本模型"会 IndexError。二进制格式其实很简单，
自己写反而最稳，而且能把 VGGT 的稠密点云直接写成 3DGS 的初始点。

格式（little-endian）：
  cameras.bin : uint64 数量；每台：int32 id, int32 model_id, uint64 w, uint64 h, double 参数[]
                PINHOLE 的 model_id = 1，参数 = fx fy cx cy
  images.bin  : uint64 数量；每张：int32 id, double qvec[4], double tvec[3], int32 camera_id,
                name(以 \\0 结尾), uint64 点数；每个点：double x, double y, int64 point3D_id
  points3D.bin: uint64 数量；每个点：int64 id, double xyz[3], uint8 rgb[3], double error,
                uint64 轨迹长度；每条轨迹：int32 image_id, int32 point2D_idx
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

PINHOLE_MODEL_ID = 1


def write_cameras(path: Path, *, w: int, h: int, fx: float, fy: float, cx: float, cy: float) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, PINHOLE_MODEL_ID, w, h))
        f.write(struct.pack("<4d", fx, fy, cx, cy))


def write_images(path: Path, *, names: list[str], qvecs: np.ndarray, tvecs: np.ndarray,
                 n_points_per_image: list[int]) -> None:
    """每张图的 2D 点用 (0,0) 占位，只为让 track 自洽（训练只用 3D 位置）。"""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for i, name in enumerate(names):
            f.write(struct.pack("<i", i + 1))
            f.write(struct.pack("<4d", *qvecs[i].tolist()))
            f.write(struct.pack("<3d", *tvecs[i].tolist()))
            f.write(struct.pack("<i", 1))          # camera_id 共享一台
            f.write(name.encode("utf-8") + b"\x00")
            n = n_points_per_image[i]
            f.write(struct.pack("<Q", n))
            for k in range(n):
                f.write(struct.pack("<2d", 0.0, 0.0))
                f.write(struct.pack("<q", -1))


def write_points3d(path: Path, xyz: np.ndarray, rgb: np.ndarray, *, track_image_id: int = 1) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i in range(len(xyz)):
            f.write(struct.pack("<q", i + 1))
            f.write(struct.pack("<3d", *xyz[i].tolist()))
            f.write(struct.pack("<3B", *[int(c) for c in rgb[i]]))
            f.write(struct.pack("<d", 0.0))
            f.write(struct.pack("<Q", 1))           # 一条轨迹
            f.write(struct.pack("<ii", track_image_id, i))
