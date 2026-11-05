#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VGGT 一跑：帧 → 位姿 + 稠密点云（PLY）→ 直接喂给家具分割流水线。

为什么用它：COLMAP 在弱纹理（白瓷砖）上特征太少，稠密重建（PatchMatch）又是 CUDA-only；
VGGT 是**一次前向**由网络回归出位姿 + 深度 + 点图，对弱纹理更稳，产物是 PLY，
正好能被 `tools/furniture_lines.py` 吃下（已支持 PLY）。

⚠️ 显存：VGGT 的全局注意力随**帧数平方**增长。8 GB 上先从小帧数试（8~16），
   不够再降 `--resolution` 或减 `--frames`。

用法:
  # 干跑（不加载权重，只验证取帧/预处理/写 PLY 的链路）
  <python> tools/vggt_cloud.py --frames-dir output/bathroom/vggt-in --out output/vggt/dry.ply --dry-run
  # 真跑
  <python> tools/vggt_cloud.py --frames-dir output/bathroom/vggt-in --out output/vggt/bathroom.ply --frames 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "models" / "vggt" / "repo" / "vggt-main"
sys.path.insert(0, str(REPO))          # 项目 python 是精简版，PYTHONPATH 不生效，必须显式加


def list_frames(frames_dir: Path, limit: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise SystemExit(f"{frames_dir} 里没有图片")
    if limit and len(files) > limit:
        idx = np.linspace(0, len(files) - 1, limit).round().astype(int)
        files = [files[i] for i in dict.fromkeys(idx)]
    return files


def write_ply(path: Path, pts: np.ndarray, rgb: np.ndarray | None = None) -> None:
    """写二进制 PLY（x,y,z + 可选 RGB）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(pts)
    has_rgb = rgb is not None and len(rgb) == n
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
            "property float x", "property float y", "property float z"]
    if has_rgb:
        head += ["property uchar red", "property uchar green", "property uchar blue"]
    head.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        if has_rgb:
            rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
            rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
            rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        else:
            rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        f.write(rec.tobytes())
    print(f"已写出 {path}（{n} 点，{path.stat().st_size / 1048576:.1f} MB）")


def main() -> int:
    ap = argparse.ArgumentParser(description="用 VGGT 从视频帧重建稠密点云")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True, help="输出的 .ply")
    ap.add_argument("--weights", default=str(ROOT / "models" / "vggt" / "model.pt"))
    ap.add_argument("--frames", type=int, default=12, help="使用多少帧（显存随帧数平方增长）")
    ap.add_argument("--resolution", type=int, default=518, help="VGGT 推理分辨率（官方 518）")
    ap.add_argument("--load-res", type=int, default=1024, help="图片加载分辨率（官方 1024）")
    ap.add_argument("--conf", type=float, default=5.0, help="深度置信度阈值（官方 5.0）")
    ap.add_argument("--conf-pct", type=float, default=70.0,
                    help="阈值过严导致 0 点时，按该分位数兜底（默认 P70）")
    ap.add_argument("--max-points", type=int, default=200000)
    ap.add_argument("--save-depth", action="store_true",
                    help="把深度图存成 .depth.npz（供 2D 检测框反投影成 3D 盒）")
    ap.add_argument("--dry-run", action="store_true", help="不加载模型，只跑取帧/预处理/写盘，验证链路")
    args = ap.parse_args()

    frames = list_frames(Path(args.frames_dir), args.frames)
    print(f"使用 {len(frames)} 帧：{frames[0].name} … {frames[-1].name}")

    import torch
    import torch.nn.functional as F

    from vggt.utils.load_fn import load_and_preprocess_images_square

    print(f"预处理（加载分辨率 {args.load_res}）…")
    images, original_coords = load_and_preprocess_images_square([str(p) for p in frames], args.load_res)
    print(f"  图像张量: {tuple(images.shape)}  dtype={images.dtype}")

    if args.dry_run:
        # 干跑：用图像均值生成一个占位点云，验证"写 PLY → furniture_lines"的链路
        rng = np.random.default_rng(0)
        fake = rng.normal(0, 1.0, (2000, 3)).astype(np.float32)
        write_ply(Path(args.out), fake, (rng.random((len(fake), 3)) * 255).astype(np.uint8))
        print("干跑完成（点云是随机占位，仅用于验证链路）")
        return 0

    weights = Path(args.weights)
    if not weights.exists():
        print(f"✗ 权重还没下好：{weights}")
        return 2
    print(f"加载权重 {weights}（{weights.stat().st_size / 1073741824:.2f} GB）…")

    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT()
    state = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(state if isinstance(state, dict) and "model" not in state else state["model"])
    model.eval().to(device)
    print(f"  模型已加载到 {device}，dtype={dtype}")

    images = images.to(device)
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            imgs = F.interpolate(images, size=(args.resolution, args.resolution),
                                 mode="bilinear", align_corners=False)[None]
            agg, ps_idx = model.aggregator(imgs)
            pose_enc = model.camera_head(agg)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])
            depth_map, depth_conf = model.depth_head(agg, imgs, ps_idx)
            points = unproject_depth_map_to_point_map(
                depth_map.squeeze(0).cpu().numpy(),
                extrinsic.squeeze(0).cpu().numpy(),
                intrinsic.squeeze(0).cpu().numpy())
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(f"✗ 显存不足（帧数 {len(frames)}、分辨率 {args.resolution}）。"
                  f"把 --frames 降到 {max(4, len(frames)//2)} 或 --resolution 降到 336 再试。")
            return 3
        raise

    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    print(f"  点图: {points.shape}  置信度: {depth_conf.shape}")

    pct = np.percentile(depth_conf, [50, 70, 90, 99])
    print("  置信度分布: P50=%.2f P70=%.2f P90=%.2f P99=%.2f max=%.2f"
          % (pct[0], pct[1], pct[2], pct[3], float(depth_conf.max())))
    mask = depth_conf >= args.conf
    if int(mask.sum()) == 0:
        # 官方默认阈值是 5.0，但它是针对官方示例数据的；换数据后置信度分布会变。
        # 一个点都没有时按分位数兜底（默认取 P70），并把实际阈值打出来 —— 不要静默写出空点云。
        thr = float(np.percentile(depth_conf, args.conf_pct))
        print(f"  ⚠ 阈值 {args.conf} 太严（0 点），按 P{args.conf_pct:.0f} 兜底为 {thr:.2f}")
        mask = depth_conf >= thr
    print(f"  保留点数: {int(mask.sum())} / {mask.size}")
    pts = points[mask]
    # ⚠️ **必须带颜色**：VGGT 的 point_map 不含颜色，之前写 PLY 时只写 xyz，
    #    于是 3DGS 的初始化点全是灰色的 —— 网页里模型"一片白"就是这个原因（实测）。
    #    颜色直接从输入图像按同一像素坐标取（点图与图像是同分辨率、同坐标系）。
    rgb = None
    if "imgs" in dir():
        try:
            imgs_np = imgs.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()      # [N,H,W,3]
            rgb = np.clip(imgs_np[mask] * 255.0, 0, 255).astype(np.uint8)
            print(f"  颜色: 取自输入图像（{rgb.shape[0]} 点）")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ 取颜色失败（{type(exc).__name__}），退回灰色：{exc}")
            rgb = None
    if len(pts) > args.max_points:
        sel = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
        pts = pts[sel]
        if rgb is not None:
            rgb = rgb[sel]
    write_ply(Path(args.out), pts.astype(np.float32), rgb)

    poses = Path(args.out).with_suffix(".poses.json")
    poses.write_text(json.dumps({
        "frames": [p.name for p in frames],
        "extrinsic": extrinsic.squeeze(0).cpu().numpy().tolist(),
        "intrinsic": intrinsic.squeeze(0).cpu().numpy().tolist(),
        "resolution": args.resolution,
        "note": "OpenCV 约定（world→camera 的 3x4）；尺度为任意单位",
    }, ensure_ascii=False), encoding="utf-8")
    print(f"已写出 {poses}")
    if args.save_depth:
        depth_npz = Path(args.out).with_suffix(".depth.npz")
        np.savez_compressed(
            depth_npz,
            depth=depth_map.squeeze(0).cpu().numpy().astype("float16"),
            conf=depth_conf.astype("float16"),
        )
        print(f"已写出 {depth_npz}（深度图 {depth_map.shape[1:]}，与 poses.json 的帧顺序一致）")

    print("\n下一步：")
    print(f"  python tools/compare_clouds.py --cloud \"VGGT={args.out}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
