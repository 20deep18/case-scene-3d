#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取 COLMAP 二进制稀疏模型，输出统计信息，并与合成数据集的真值对比。

用途：验证 COLMAP 端到端可用的证据。
注意：COLMAP 4.2.1 的 `model_converter`（二进制→文本）在 Windows 上会崩溃
（0xC0000409 栈缓冲溢出），所以这里自己解析二进制格式。

用法：
    pyruntime312\\python.exe tools/colmap_report.py --model output/colmap-textured/sparse/1 ^
        --truth data/scene-textured/images.txt
"""

import argparse
import os
import struct
import sys

import numpy as np


def read_images_bin(path):
    """
    解析 images.bin：返回 [(name, q(w,x,y,z), t(3))]

    注意 COLMAP 4.x 的记录布局与 3.x 不同：
      4.x: image_id(int32) + qvec(4×double) + tvec(3×double)  = 4 + 32 + 24 = **60 字节**
      3.x: 还多一个 camera_id(int32) = 64 字节
    这里两种都试，避免版本差异导致解析失败。
    """
    out = []
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        for _ in range(num):
            record_start = f.tell()
            chunk = f.read(64)
            if len(chunk) < 60:
                break
            # 先按 4.x 的 60 字节解析
            iid, qw, qx, qy, qz, tx, ty, tz = struct.unpack("<iddddddd", chunk[:60])
            cam_id = None
            # 判断后面是不是紧随字符串（4.x）还是还有一个 int32（3.x）
            nxt = chunk[60:64]
            if nxt[:4] == b"\x00\x00\x00\x00":
                f.seek(record_start + 64)  # 3.x：camera_id 在 60..64
            else:
                f.seek(record_start + 60)  # 4.x
            name_bytes = b""
            while True:
                ch = f.read(1)
                if ch in (b"\x00", b""):
                    break
                name_bytes += ch
            (num_pts,) = struct.unpack("<Q", f.read(8))
            f.read(num_pts * 24)
            out.append((name_bytes.decode("utf-8", "replace"), (qw, qx, qy, qz), (tx, ty, tz), cam_id))
    return out


def read_points3d_bin(path):
    """解析 points3D.bin：返回 N×3 的坐标数组"""
    pts = []
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        for _ in range(num):
            # id(Q) xyz(3d) rgb(3B) error(d) track_length(Q) then track data
            head = struct.unpack("<QdddBBBd", f.read(8 + 24 + 3 + 8))
            pid, x, y, z = head[0], head[1], head[2], head[3]
            pts.append((x, y, z))
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.read(track_len * 8)  # 每条 track: image_id(uint32) + point2D_idx(uint32)
    return np.asarray(pts, dtype=float)


def read_cameras_bin(path):
    """解析 cameras.bin：返回 {camera_id: (model_id, w, h, params)}"""
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        cams = {}
        for _ in range(num):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            # 参数个数取决于 model_id，这里按 COLMAP 的模型表
            nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 5, 11: 8}.get(model_id, 4)
            params = struct.unpack("<" + "d" * nparams, f.read(8 * nparams))
            cams[cam_id] = (model_id, w, h, params)
        return cams


def centers_from(images):
    """由 (q, t) 求相机中心 C = -Rᵀ·t"""
    out = {}
    for name, (qw, qx, qy, qz), t in [(n, q, t) for n, q, t, _ in images]:
        R = np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
            ]
        )
        out[name] = -R.T @ np.asarray(t)
    return out


def read_truth(path):
    """读取我们生成的 COLMAP 文本真值"""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 10:
                continue
            q = [float(v) for v in p[1:5]]
            t = [float(v) for v in p[5:8]]
            name = p[9]
            qw, qx, qy, qz = q
            R = np.array(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                    [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                    [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
                ]
            )
            out[name] = -R.T @ np.asarray(t)
    return out


def similarity_align(src, dst):
    """Umeyama 求把 src 对齐到 dst 的 (s, R, t)，返回对齐后的残差"""
    a = np.asarray(src, dtype=float)
    b = np.asarray(dst, dtype=float)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    A = a - ca
    B = b - cb
    H = A.T @ B / len(a)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    var_a = (A**2).sum() / len(a)
    s = (S * np.array([1, 1, d])).sum() / var_a if var_a > 0 else 1.0
    t = cb - s * R @ ca
    aligned = (s * (R @ A.T)).T + cb
    err = np.linalg.norm(aligned - b, axis=1)
    return s, R, t, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="稀疏模型目录（含 cameras.bin 等）")
    ap.add_argument("--truth", help="合成数据集的 images.txt（可选，用于精度对比）")
    args = ap.parse_args()

    model = os.path.abspath(args.model)
    for fn in ("cameras.bin", "images.bin", "points3D.bin"):
        if not os.path.exists(os.path.join(model, fn)):
            print(f"缺少 {fn}，目录：{model}")
            sys.exit(1)

    cams = read_cameras_bin(os.path.join(model, "cameras.bin"))
    images = read_images_bin(os.path.join(model, "images.bin"))
    pts = read_points3d_bin(os.path.join(model, "points3D.bin"))

    print(f"模型目录: {model}")
    print(f"  相机数      : {len(cams)}")
    print(f"  注册图像数  : {len(images)}")
    print(f"  三维点数    : {len(pts)}")
    if len(images):
        obs = 0
        with open(os.path.join(model, "images.bin"), "rb") as f:
            f.read(8)
            for _ in range(len(images)):
                f.read(64)
                while f.read(1) not in (b"\x00", b""):
                    pass
                (n,) = struct.unpack("<Q", f.read(8))
                obs += n
                f.read(n * 24)
        print(f"  2D 观测数   : {obs}")
    for cid, (mid, w, h, params) in cams.items():
        print(f"  相机 {cid}: model_id={mid} {w}x{h} params={[round(float(v), 3) for v in params]}")
    if len(pts):
        print(f"  点云范围    : min={np.round(pts.min(axis=0), 3).tolist()} max={np.round(pts.max(axis=0), 3).tolist()}")
        print(f"  点云对角线  : {np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)):.3f}")

    if args.truth and os.path.exists(args.truth):
        truth = read_truth(args.truth)
        centers = centers_from(images)
        names = [n for n in centers if n in truth]
        if names:
            src = [centers[n] for n in names]
            dst = [truth[n] for n in names]
            s, R, t, err = similarity_align(src, dst)
            print("\n与真值对比（Umeyama 相似变换对齐：单目重建尺度不可观测）")
            print(f"  对比图像数  : {len(names)}")
            print(f"  估计尺度    : {s:.4f}")
            print(f"  中心误差    : 中位 {np.median(err):.4f} m，均值 {err.mean():.4f} m，最大 {err.max():.4f} m")
            # 真值场景尺度约 2.4m
            rel = np.median(err) / 2.4 * 100
            print(f"  相对误差    : {rel:.2f}%（相对 2.4m 的场景尺寸）")
            if s and abs(s - 1) > 1e-6:
                print(f"  对齐后尺度  : 缩放 {s:.4f}（原始重建尺度与米制差 {abs(1 - s) * 100:.1f}%）")
        else:
            print("\n真值里没有匹配的图像名，跳过对比")


if __name__ == "__main__":
    main()
