#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VGGT 位姿 → COLMAP 文本模型（供 train_gsplat.py 直接训练）。

为什么走这条路：VGGT 的点图在弱纹理场景偏噪（实测：连续 12 帧的云聚出的"盒子"
都是 0.17×0.15×0.05 的薄片、每块仅 ~1,200 点）。但它的**位姿**是可靠的 ——
把位姿交给 3DGS，让多视角光度监督去长出"贴表面"的稠密高斯，正好补上
本机 COLMAP（CPU 版、无 PatchMatch）拿不到的那部分。

关键：为了让内参自洽，训练用的图必须是 **VGGT 预处理后的同款方形图**
（`load_and_preprocess_images_square` 会补边成正方形）。所以这里：
  1) 用 VGGT 的预处理把原帧转成方形图并落盘（训练用图）
  2) 跑一次 VGGT 拿 extrinsic/intrinsic（或读已存的 poses.json）
  3) 写 cameras.txt / images.txt / points3D.txt（COLMAP 文本格式）

用法:
  <python> tools/vggt_poses_to_colmap.py --frames output/bathroom/vggt-dense \
      --weights models/vggt/model.pt --out output/vggt/colmap-ws --frames-limit 12
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "models" / "vggt" / "repo" / "vggt-main"
sys.path.insert(0, str(REPO))

from vggt.utils.load_fn import load_and_preprocess_images_square  # noqa: E402


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 四元数 (w, x, y, z)（COLMAP 的顺序）。"""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], float)
    return q / (np.linalg.norm(q) + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="原始帧目录")
    ap.add_argument("--out", required=True, help="输出工作区（会写 images/ 与 sparse/0/）")
    ap.add_argument("--weights", default=str(ROOT / "models" / "vggt" / "model.pt"))
    ap.add_argument("--poses", default=None, help="已有的 poses.json（有就不重跑网络）")
    ap.add_argument("--frames-limit", type=int, default=12)
    ap.add_argument("--init-ply", default=None,
                    help="稠密点云 PLY：写成 points3D.txt 作为 3DGS 的初始点"
                    "（VGGT 的位姿模型没有 3D 点，不喂初始点就长不出高斯）")
    ap.add_argument("--init-points", type=int, default=20000)
    ap.add_argument("--load-res", type=int, default=1024)
    ap.add_argument("--resolution", type=int, default=518)
    args = ap.parse_args()

    out = Path(args.out)
    images_dir = out / "images"
    sparse_dir = out / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(p for p in Path(args.frames).iterdir() if p.suffix.lower() in exts)
    if args.frames_limit and len(files) > args.frames_limit:
        idx = np.linspace(0, len(files) - 1, args.frames_limit).round().astype(int)
        files = [files[i] for i in dict.fromkeys(idx)]
    print(f"帧数 {len(files)}")

    # 1) 用 VGGT 的预处理生成"方形图"，训练与推理用同一份几何
    images, _ = load_and_preprocess_images_square([str(p) for p in files], args.load_res)
    arr = (images.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    for i, p in enumerate(files, 1):
        Image.fromarray(arr[i - 1]).save(images_dir / f"{i:04d}.png")
    h, w = arr.shape[1], arr.shape[2]
    print(f"已写方形训练图 {len(files)} 张（{w}×{h}）")

    # 2) 位姿：优先用已存的 poses.json，否则跑一次 VGGT
    if args.poses and Path(args.poses).exists():
        pj = json.loads(Path(args.poses).read_text(encoding="utf-8"))
        E = np.asarray(pj["extrinsic"], float)
        K = np.asarray(pj["intrinsic"], float)
        print(f"读取已有位姿 {len(E)} 帧")
    else:
        import torch
        import torch.nn.functional as F
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = VGGT()
        model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
        model.eval().to(device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        imgs = F.interpolate(images.to(device), size=(args.resolution, args.resolution),
                             mode="bilinear", align_corners=False)[None]
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            agg, ps_idx = model.aggregator(imgs)
            pose_enc = model.camera_head(agg)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])
        E = extrinsic.squeeze(0).cpu().numpy()
        K = intrinsic.squeeze(0).cpu().numpy()
        print(f"VGGT 推理得位姿 {len(E)} 帧")

    n = min(len(E), len(files))
    E, K = E[:n], K[:n]

    # 3) 内参从 518（推理分辨率）缩放到训练图分辨率（load_res）
    scale = args.load_res / args.resolution
    K = K.copy()
    K[:, :2, :] *= scale
    # ⚠️ VGGT 的逐帧内参很不稳（实测 fx 200~441、fy 285~829），而方形图里 fx 必须等于 fy。
    #    这里改成"全帧共享 + 各向同性"：f 取所有帧的中位数，光心固定在图像中心。
    fx = fy = float(np.median(np.concatenate([K[:, 0, 0], K[:, 1, 1]])))
    cx, cy = w / 2.0, h / 2.0
    print(f"内参（训练分辨率 {w}×{h}）: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    (sparse_dir / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n", encoding="utf-8")

    lines = ["# Image list with two lines of data per image:\n"]
    for i in range(n):
        R = E[i, :3, :3]
        t = E[i, :3, 3]
        q = rotmat_to_qvec(R)
        lines.append(f"{i + 1} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                     f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 {i + 1:04d}.png\n\n")
    (sparse_dir / "images.txt").write_text("".join(lines), encoding="utf-8")
    # 初始 3D 点：从稠密点云抽样写成 COLMAP 的 points3D.txt
    if args.init_ply and Path(args.init_ply).exists():
        raw = Path(args.init_ply).read_bytes()
        end = raw.find(b"end_header")
        body = raw.find(b"\n", end) + 1
        head = raw[:end].decode("ascii", "replace")
        m = int([l for l in head.splitlines() if l.startswith("element vertex")][0].split()[-1])
        # 属性类型要照原样映射（颜色常是 uchar，一律当 f4 会让 itemsize 对不上）
        props = [(l.split()[1], l.split()[-1]) for l in head.splitlines() if l.startswith("property")]
        dt = np.dtype([(nm, "<f4" if tname in ("float", "float32", "double") else "u1")
                       for tname, nm in props])
        buf = np.frombuffer(raw, dtype=dt, count=m, offset=body)
        P = np.stack([buf["x"], buf["y"], buf["z"]], axis=1).astype(np.float64)
        # ⚠️ **必须带颜色**：之前这里写死 200 200 200（白灰），导致 3DGS 从白色开始优化，
        #    12k 步收敛不到真实颜色 —— 网页里看到的模型"全是白的"就是这个原因（实测）。
        if "red" in buf.dtype.names:
            C = np.stack([buf["red"], buf["green"], buf["blue"]], axis=1).astype(np.uint8)
        else:
            C = np.full((len(P), 3), 200, np.uint8)
        ok = np.isfinite(P).all(axis=1)
        P, C = P[ok], C[ok]
        if len(P) > args.init_points:
            sel = np.random.default_rng(0).choice(len(P), args.init_points, replace=False)
            P, C = P[sel], C[sel]
        # 每条点给一个占位 track（训练只用到位置/颜色）
        lines = []
        for k, (p, c) in enumerate(zip(P, C), 1):
            lines.append(f"{k} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                         f"{int(c[0])} {int(c[1])} {int(c[2])} 0.0 1 {k % n + 1} {k}\n")
        (sparse_dir / "points3D.txt").write_text("".join(lines), encoding="utf-8")
        print(f"初始点 {len(P)} 个（来自 {Path(args.init_ply).name}，"
              f"颜色{'取自点云' if 'red' in buf.dtype.names else '缺失→灰' }）")
    else:
        (sparse_dir / "points3D.txt").write_text("", encoding="utf-8")

    # 相机中心（给家具算法当轨迹用）
    centers = [(-E[i, :3, :3].T @ E[i, :3, 3]).tolist() for i in range(n)]
    r = float(np.median(np.linalg.norm(np.asarray(centers) - np.mean(centers, axis=0), axis=1)))
    print(f"相机群半径（中位）= {r:.3f}；相机高度建议再按实测校正")
    (out / "camera_centers.json").write_text(
        json.dumps({"centers": centers, "medianRadius": r}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"已写出 COLMAP 文本模型：{sparse_dir}")
    print("\n下一步训练：")
    print(f'  python pipeline/train_gsplat.py --data "{out}" --out "{out}/train" --images "{images_dir}" '
          f"--iterations 12000 --downscale 1 --sh-degree 0 --densify-grad-threshold 1e-4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
