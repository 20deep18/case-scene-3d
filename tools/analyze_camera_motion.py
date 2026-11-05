#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析 COLMAP 恢复出的相机运动，判断 SfM 的可解性（well-posedness）。

为什么要看这些量：
  SfM 的病态程度由「基线 / 景深」比（baseline-to-depth ratio）决定，而**不是**
  单纯由基线长度决定。原地转身哪怕转了 360°，基线也是 0，景深完全无解；
  沿走廊走 10 米但两侧都是 50 米外的墙，基线/深度只有 0.2，同样很难。

  另一个关键量是「旋转主导度」：如果两帧之间的旋转远大于平移带来的视差，
  深度就几乎无法与相机旋转区分开 —— 这是手持拍摄最常见的退化情形。

输出：
  · 轨迹总长度、相邻帧位移、每个相机到场景中心的距离
  · 相邻帧旋转角、以及「平移视差等效像素」与「旋转量」的对比
  · 基线/景深 比的分布 → 给出可解性结论

用法: python tools/analyze_camera_motion.py --model <sparse/0 目录>
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colmap_model_io as io  # noqa: E402


def rot_angle_deg(R):
    """两个旋转之间的夹角（度）。"""
    c = (np.trace(R) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="COLMAP sparse/0 目录")
    ap.add_argument("--focal", type=float, default=None, help="像素焦距（不填则从相机读出）")
    ap.add_argument("--width", type=int, default=None)
    args = ap.parse_args()

    model = Path(args.model)
    cams = io.read_cameras(model / "cameras.bin")
    imgs = io.read_images(model / "images.bin")
    pts = io.read_points3d(model / "points3D.bin")

    # ---------------------------------------------------------------
    # 关键 1：必须按**文件名里的时间序号**排序。
    # COLMAP 的 image_id 顺序是内部顺序，**不等于拍摄顺序**。
    # 如果直接用 read_images 的顺序当作时间序列，算出来的"相邻帧位移/转角"
    # 完全是无意义的（实测会得到 1.68 m/帧、16.7°/帧 这种荒谬数字，
    # 而真实相邻帧只差 0.5 秒）。
    # ---------------------------------------------------------------
    def time_key(img):
        import re

        nums = re.findall(r"\d+", Path(img["name"]).name)
        return (int(nums[-1]) if nums else 0, img["name"])

    imgs = sorted(imgs, key=time_key)

    # ---------------------------------------------------------------
    # 关键 2：过滤离群稀疏点，否则包围盒/场景中心会被极少数野点主导。
    # 用中位数 + 分位数做稳健区间，而不是直接用 min/max。
    # ---------------------------------------------------------------
    xyz_all = np.asarray(pts["xyz"], dtype=np.float64)
    med = np.median(xyz_all, axis=0)
    dist = np.linalg.norm(xyz_all - med, axis=0 if xyz_all.ndim == 1 else 1)
    keep = dist < np.percentile(dist, 95)
    xyz = xyz_all[keep]
    outliers = int((~keep).sum())

    # 相机内参
    cam = next(iter(cams.values()))
    fx, fy, cx, cy = cam["params"][0], cam["params"][1], cam["params"][2], cam["params"][3]
    W, H = int(cam["width"]), int(cam["height"])
    fx = args.focal or fx

    centers = np.array([io.camera_center(i["qvec"], i["tvec"]) for i in imgs])
    Rs = np.array([io.quat_to_R(i["qvec"]) for i in imgs])

    # 场景中心用**稳健**的点云中心，而不是均值（均值受野点影响极大）
    scene_center = np.median(xyz, axis=0)
    depths = np.linalg.norm(centers - scene_center, axis=1)

    print("=" * 74)
    print("相机运动与 SfM 可解性分析")
    print("=" * 74)
    print(f"图像 {len(imgs)} 张（已按文件名时间序号排序）   稀疏点 {len(xyz)}（剔除 {outliers} 个离群点）")
    print(f"相机 {W}×{H}  fx={fx:.1f}")
    print(f"稳健场景尺度（P95 点云包围盒对角线） {np.linalg.norm(xyz.max(0)-xyz.min(0)):.2f} m")
    print()
    print(f"时间序前 3 张: {[Path(i['name']).name for i in imgs[:3]]}")
    print(f"时间序后 3 张: {[Path(i['name']).name for i in imgs[-3:]]}")
    print()

    # --- 轨迹 ---
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    total_path = steps.sum()
    net_disp = np.linalg.norm(centers[-1] - centers[0])
    print("=== 平移 ===")
    print(f"  轨迹总长（累加路径）   {total_path:8.3f} m")
    print(f"  首尾净位移             {net_disp:8.3f} m")
    print(f"  包围盒尺寸             {np.round(centers.max(0)-centers.min(0), 3)} m")
    print(f"  相邻帧位移  中位 {np.median(steps):.4f} m   P90 {np.percentile(steps,90):.4f} m   max {steps.max():.4f} m")
    print(f"  相机到场景中心距离  中位 {np.median(depths):.3f} m   范围 {depths.min():.3f}~{depths.max():.3f} m")
    print()

    # --- 旋转 ---
    angs = []
    for i in range(1, len(Rs)):
        angs.append(rot_angle_deg(Rs[i - 1].T @ Rs[i]))
    angs = np.array(angs)
    print("=== 旋转 ===")
    print(f"  相邻帧旋转角  中位 {np.median(angs):6.3f}°   P90 {np.percentile(angs,90):6.3f}°   max {angs.max():6.3f}°")
    # 整段视频的累计转角
    total_rot = angs.sum()
    print(f"  累计转角（各帧相加）   {total_rot:8.2f}°")
    print()

    # --- 关键指标：基线/景深 ---
    # 对每一对「间隔 k 帧」的相机，算基线并除以平均景深
    print("=== 基线 / 景深 比（SfM 可解性的核心指标）===")
    print("  经验：>0.1 好；0.03~0.1 勉强；<0.03 病态（深度基本靠猜）")
    print()
    print(f"  {'间隔帧数':>8s} {'基线中位(m)':>12s} {'景深中位(m)':>12s} {'基线/景深':>10s} {'旋转中位(°)':>12s}")
    for k in (1, 2, 5, 10, 20, 40, 80):
        if k >= len(centers):
            continue
        b = np.linalg.norm(centers[k:] - centers[:-k], axis=1)
        d = (depths[k:] + depths[:-k]) / 2.0
        ratio = b / np.maximum(d, 1e-9)
        rk = np.array([rot_angle_deg(Rs[i].T @ Rs[i + k]) for i in range(len(Rs) - k)])
        print(
            f"  {k:>8d} {np.median(b):>12.4f} {np.median(d):>12.4f} "
            f"{np.median(ratio):>10.4f} {np.median(rk):>12.3f}"
        )
    print()

    # --- 旋转主导度：把旋转折算成像素位移，与平移视差比较 ---
    print("=== 旋转主导度（相邻帧）===")
    # 纯旋转造成的等效像素位移 ≈ f * tan(角度)
    rot_px = fx * np.tan(np.radians(angs))
    # 平移造成的视差像素 ≈ f * baseline / depth
    trans_px = fx * steps / depths[1:]
    ratio = trans_px / np.maximum(rot_px, 1e-9)
    print(f"  旋转等效像素位移  中位 {np.median(rot_px):8.1f} px")
    print(f"  平移等效像素视差  中位 {np.median(trans_px):8.1f} px")
    print(f"  平移/旋转 比       中位 {np.median(ratio):8.4f}")
    print()
    if np.median(ratio) < 0.1:
        print("  ✘ 旋转远大于平移视差：相邻帧的视差几乎全部来自旋转，")
        print("     深度信息极弱 → 典型「手持视频」退化情形")
    elif np.median(ratio) < 0.3:
        print("  ⚠ 旋转明显主导：深度可解但噪声大，重建会偏糊")
    else:
        print("  ✔ 平移视差与旋转相当或更大：几何条件良好")
    print()

    # --- 结论 ---
    med_ratio_k5 = None
    k = min(10, len(centers) - 1)
    if k >= 1:
        b = np.linalg.norm(centers[k:] - centers[:-k], axis=1)
        d = (depths[k:] + depths[:-k]) / 2.0
        med_ratio_k5 = float(np.median(b / np.maximum(d, 1e-9)))
    print("=== 结论 ===")
    print(f"  轨迹总长 {total_path:.2f} m，净位移 {net_disp:.2f} m，累计转角 {total_rot:.1f}°")
    if med_ratio_k5 is not None:
        print(f"  相隔 {k} 帧的基线/景深 = {med_ratio_k5:.4f}")
        if med_ratio_k5 >= 0.1:
            print("  → 基线是**够用**的。画质问题不该归因于「没平移」。")
        elif med_ratio_k5 >= 0.03:
            print("  → 基线偏小但可用；需要靠更多帧与更好的匹配来补偿。")
        else:
            print("  → 基线确实不足，深度病态。")


if __name__ == "__main__":
    main()
