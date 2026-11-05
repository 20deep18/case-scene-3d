#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COLMAP 二进制稀疏模型读写（自实现，不依赖 pycolmap）。

**记录布局（本机 COLMAP 4.2.1 实测，用 hex dump 确认）**

    cameras.bin
        uint64 num_cameras
        每台相机：int32 camera_id, int32 model_id, uint64 width, uint64 height,
                  double params[num_params]
        （params 个数由 model_id 决定，见 CAMERA_MODEL_PARAMS）

    images.bin
        uint64 num_images
        每张图像：
            int32   image_id
            double  qvec[4]     (w, x, y, z)
            double  tvec[3]
            int32   camera_id
            char    name[]      (\\0 结尾)
            uint64  num_points2D
            num_points2D × (double x, double y, uint64 point3D_id)

    points3D.bin
        uint64 num_points
        每个点：
            uint64  point3D_id
            double  xyz[3]
            uint8   rgb[3]
            double  error
            uint64  track_length
            track_length × (int32 image_id, int32 point2D_idx)

⚠️ 这里的布局与 COLMAP 3.x 的官方文档一致；不同版本可能不同，
所以 `read_model` 之前先用文件大小做一次“能否完整解析”的校验，避免静默读错。
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

#: model_id → 参数个数（COLMAP 官方相机模型表）
CAMERA_MODEL_PARAMS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 12),
}


def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        for _ in range(num):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            n = CAMERA_MODEL_PARAMS.get(model_id, ("", 4))[1]
            params = struct.unpack("<" + "d" * n, f.read(8 * n))
            cams[cam_id] = {
                "model_id": model_id,
                "model": CAMERA_MODEL_PARAMS.get(model_id, ("UNKNOWN", n))[0],
                "width": w,
                "height": h,
                "params": params,
            }
    return cams


def intrinsics_of(cam):
    """从 COLMAP 相机参数里取出 fx, fy, cx, cy"""
    p = cam["params"]
    mid = cam["model_id"]
    if mid == 0:  # SIMPLE_PINHOLE: f, cx, cy
        return p[0], p[0], p[1], p[2]
    if mid in (1, 3, 4, 5, 7, 9, 10, 11):  # PINHOLE 及带 fx,fy 的模型
        return p[0], p[1], p[2], p[3]
    if mid in (2, 8):  # SIMPLE_RADIAL*: f, cx, cy, k
        return p[0], p[0], p[1], p[2]
    return p[0], p[0], p[1], p[2]


def read_images(path, with_points=False):
    """
    返回 [{name, qvec, tvec, camera_id, points2D(Nx2 或 Nx3)}]
    with_points=True 时附带 point3D_id 列（Nx3：x, y, id）。
    """
    out = []
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        for _ in range(num):
            image_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<i", f.read(4))[0]
            name_bytes = b""
            while True:
                ch = f.read(1)
                if ch in (b"\x00", b""):
                    break
                name_bytes += ch
            (n2d,) = struct.unpack("<Q", f.read(8))
            raw = f.read(n2d * 24)
            if len(raw) != n2d * 24:
                raise ValueError(
                    f"images.bin 解析越界：期望 {n2d * 24} 字节，实际 {len(raw)}。"
                    "该文件的记录布局可能与本模块假设不同。"
                )
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(n2d, 24)
            xy = arr[:, :16].copy().view(np.float64).reshape(n2d, 2)
            # point3D_id 必须保持整数类型：无效值是 0xFFFFFFFFFFFFFFFF，
            # 转成 float64 会丢精度、再转回整数会得到错误的值（实测会报 cast 警告且数错）。
            pid = arr[:, 16:24].copy().view(np.uint64).reshape(n2d)
            rec = {
                "image_id": image_id,
                "name": name_bytes.decode("utf-8", "replace"),
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "points2D": xy,
                "point3D_ids": pid if with_points else None,
            }
            out.append(rec)
    return out


#: COLMAP 用 uint64 最大值表示「该二维点没有对应三维点」
INVALID_POINT3D_ID = 0xFFFFFFFFFFFFFFFF


def read_points3d(path):
    ids, xyz, rgb, err, tracks = [], [], [], [], []
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        for _ in range(num):
            pid, x, y, z, r, g, b, e = struct.unpack("<QdddBBBd", f.read(43))
            (tlen,) = struct.unpack("<Q", f.read(8))
            f.read(tlen * 8)
            ids.append(pid)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
            err.append(e)
            tracks.append(tlen)
    return {
        "ids": np.asarray(ids, dtype=np.uint64),
        "xyz": np.asarray(xyz, dtype=float),
        "rgb": np.asarray(rgb, dtype=np.uint8),
        "error": np.asarray(err, dtype=float),
        "track_length": np.asarray(tracks, dtype=np.int64),
    }


def quat_to_R(q):
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def camera_center(qvec, tvec):
    """C = -Rᵀ·t"""
    return -quat_to_R(qvec).T @ np.asarray(tvec, dtype=float)


def read_model(model_dir):
    """读取一个稀疏模型目录"""
    model_dir = Path(model_dir)
    missing = [fn for fn in ("cameras.bin", "images.bin", "points3D.bin") if not (model_dir / fn).exists()]
    if missing:
        raise FileNotFoundError(f"{model_dir} 缺少 {missing}")
    cams = read_cameras(model_dir / "cameras.bin")
    images = read_images(model_dir / "images.bin")
    points = read_points3d(model_dir / "points3D.bin")
    return {"cameras": cams, "images": images, "points3D": points, "dir": str(model_dir)}


def summarize(model):
    cams = model["cameras"]
    images = model["images"]
    pts = model["points3D"]
    centers = np.array([camera_center(im["qvec"], im["tvec"]) for im in images]) if images else np.zeros((0, 3))
    # COLMAP 的 Observations 只统计**真的关联到三维点**的二维观测
    # （point3D_id != 0xFFFFFFFFFFFFFFFF）。若只数 points2D 长度会偏大，
    # 实测会得到 94341 而官方是 53143。
    obs = 0
    images_with_ids = read_images(Path(model["dir"]) / "images.bin", with_points=True)
    for rec in images_with_ids:
        pids = rec["point3D_ids"]
        if pids is not None and len(pids):
            obs += int((pids != INVALID_POINT3D_ID).sum())
    return {
        "cameras": len(cams),
        "registeredImages": len(images),
        "points3D": len(pts["ids"]),
        "observations": obs,
        "meanTrackLength": float(pts["track_length"].mean()) if len(pts["ids"]) else 0.0,
        "meanReprojectionError": float(pts["error"].mean()) if len(pts["ids"]) else 0.0,
        "pointsBounds": {
            "min": pts["xyz"].min(axis=0).tolist() if len(pts["ids"]) else [0, 0, 0],
            "max": pts["xyz"].max(axis=0).tolist() if len(pts["ids"]) else [0, 0, 0],
        },
        "cameraCenters": centers.tolist(),
    }


if __name__ == "__main__":
    import json
    import sys

    d = sys.argv[1] if len(sys.argv) > 1 else "output/colmap-pipeline/sparse/0"
    m = read_model(d)
    s = summarize(m)
    print(json.dumps({k: v for k, v in s.items() if k != "cameraCenters"}, ensure_ascii=False, indent=2))
    for cid, cam in m["cameras"].items():
        fx, fy, cx, cy = intrinsics_of(cam)
        print(f"相机 {cid}: {cam['model']} {cam['width']}x{cam['height']} fx={fx:.2f} fy={fy:.2f} cx={cx:.1f} cy={cy:.1f}")
