#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""背景分离（mask）在训练侧与 COLMAP 侧的接线检查。

为什么单独测这一层
------------------
`tools/object_mask.py` 生成的 mask 是对的，不代表**它被用上了**。
这一层最容易出的错是"看起来做了、其实没做"：

    · mask 与图片下采样后尺寸不一致 → 广播报错，或者更糟：静默广播到错的位置
    · 只遮了 L1、忘了遮 SSIM → 背景照样有梯度（暗门）
    · mask 目录写成相对/绝对路径不一致 → 一张都找不到，但日志里没有明显报错
    · COLMAP 的 mask 选项名对版本敏感 → 传了不被识别的选项，COLMAP 直接退出

本测试把这几条钉住（不需要 GPU、不需要跑 COLMAP）。

用法: pyruntime312\\python.exe tests/test_mask_wiring.py
退出码 0 通过 / 1 失败
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "tools", ROOT / "pipeline"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import colmap_recon  # noqa: E402
import train_gsplat as tg  # noqa: E402

passed = 0
failed = 0


def check(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [OK]   {msg}")
    else:
        failed += 1
        print(f"  [FAIL] {msg}")


def main():
    global passed, failed
    print("=" * 72)
    print("背景分离接线（训练侧 / COLMAP 侧）")
    print("=" * 72)
    tmp = Path(tempfile.mkdtemp(prefix="maskwire-"))
    cpu = torch.device("cpu")

    # ---- 1) mask 与图片下采样后必须逐像素对齐 ------------------------------
    print("\n[1] mask 与图片下采样后尺寸一致（错位会让遮罩盖在错误的区域）")
    h, w = 40, 60
    img_arr = (np.random.default_rng(3).random((h, w, 3)) * 255).astype(np.uint8)
    m_arr = np.zeros((h, w), np.uint8)
    m_arr[:, : w // 2] = 255
    ip, mp = tmp / "frame_0001.png", tmp / "mask_0001.png"
    Image.fromarray(img_arr).save(ip)
    Image.fromarray(m_arr).save(mp)
    for ds in (1, 2, 4):
        t = tg.image_to_tensor(ip, ds, torch, cpu)
        m = tg.mask_to_tensor(mp, ds, torch, cpu)
        check(t.shape[1:] == m.shape[1:],
              f"downscale={ds}：图片 {tuple(t.shape[1:])} == mask {tuple(m.shape[1:])}")
        # 左半保留、右半忽略的语义必须还在
        # ⚠️ mask 是 (1, H, W)，切宽度要切 **第 3 维** —— 我第一版写成 m[:, :w//2]，
        #    切的是高度维，于是左右都算成了全图均值 0.5，看着像"代码错了"。
        half = m.shape[2] // 2
        left = float(m[:, :, :half].mean())
        right = float(m[:, :, half:].mean())
        check(left > 0.9 and right < 0.1,
              f"downscale={ds}：左半保留 {left:.2f} / 右半忽略 {right:.2f}")

    # ---- 2) mask 必须是二值的 0/1 -------------------------------------------
    print("\n[2] mask 张量必须是二值 0/1（灰度值混进来会让 loss 权重失控）")
    m = tg.mask_to_tensor(mp, 2, torch, cpu)
    uniq = sorted(set(m.flatten().tolist()))
    check(set(uniq) <= {0.0, 1.0}, f"取值只有 {uniq}")

    # ---- 3) SSIM 也必须被遮住（不能只有 L1 遮）-----------------------------
    print("\n[3] SSIM 的遮罩语义（只遮 L1 会留下「背景照样有梯度」的暗门）")

    class _Probe(tg.Trainer):  # 只借它的 _ssim / _gaussian_window，不跑 __init__
        def __init__(self):  # noqa: D107
            self.torch = torch

    probe = _Probe()
    # ⚠️ 必须固定随机种子：这节的输入是随机的，而"边界泄漏"的**幅度**也随内容变化
    #    （实测 12 个种子：0.6% ~ 7.9%）。不固定种子这条断言就是**天生 flaky** 的 ——
    #    我第一版正是如此：手跑过一次 20/0，跑回归时变成 19/1。
    torch.manual_seed(11)
    a = torch.rand(1, 3, 16, 16)
    b = torch.rand(1, 3, 16, 16)
    full = tg.Trainer._ssim(probe, a, b)
    zero = tg.Trainer._ssim(probe, a, b, mask=torch.zeros(1, 1, 16, 16))
    ones = tg.Trainer._ssim(probe, a, b, mask=torch.ones(1, 1, 16, 16))
    half_mask = torch.zeros(1, 1, 16, 16)
    half_mask[:, :, :, :8] = 1.0
    half = tg.Trainer._ssim(probe, a, b, mask=half_mask)
    check(bool(torch.isfinite(zero)), f"全 0 遮罩不产生 nan/inf（得到 {float(zero)}）")
    check(abs(float(ones) - float(full)) < 1e-6,
          f"全 1 遮罩 == 不遮（{float(ones):.6f} vs {float(full):.6f}）")
    check(float(half) != float(full),
          f"半图遮罩真的改变了结果（全图 {float(full):.4f} / 半图 {float(half):.4f}）")
    # 也要接受一个**真实的**性质：11×11 窗口会**跨过遮罩边界**，
    #    所以被遮区域的像素仍会通过边界那几列轻微影响结果。
    #    实测（12 个随机种子）：相对差异 0.6% ~ 7.9%，中位 3.6%。
    #    这里把它**测出来并设上界**（而不是假装是 0），超过 12% 才算回归。
    a2 = a.clone()
    a2[:, :, :, 8:] = torch.rand(1, 3, 16, 8)
    half2 = tg.Trainer._ssim(probe, a2, b, mask=half_mask)
    rel = abs(float(half2) - float(half)) / max(abs(float(half)), 1e-9)
    check(0.0 < rel < 0.12,
          f"被遮区域的影响有界且只在边界（相对差异 {rel:.2%}；实测区间 0.6%~7.9%）")

    # ---- 4) 训练器读 mask 目录的行为 ---------------------------------------
    print("\n[4] --mask-dir 的读取：只认同名、前景过少的帧要跳过并记录")
    src = (ROOT / "pipeline" / "train_gsplat.py").read_text(encoding="utf-8")
    check("mask_to_tensor(candidate" in src, "训练器按「与帧同名 .png」查找 mask")
    check("self.masks[frame.image_id] = m" in src, "mask 以 image_id 挂在训练器上")
    check("ratio < 0.05" in src and "skipped_by_mask" in src,
          "前景 <5% 的帧被跳过并记录（避免 loss 变成 0/0 的假训练）")
    check("mask=mask.unsqueeze(0)" in src, "训练时把 mask 传给了 SSIM")
    check("(torch.abs(image - target) * m).sum()" in src, "L1 按 mask 归一化（不是简单乘）")

    # ---- 5) COLMAP 侧：选项与缺 mask 的处理 --------------------------------
    print("\n[5] COLMAP 命令行：mask 选项、缺 mask 时必须明确失败")
    ws = tmp / "ws"
    ws.mkdir()
    mask_dir = tmp / "masks"
    mask_dir.mkdir()
    Image.fromarray(m_arr).save(mask_dir / "frame_0001.png")
    cmds = colmap_recon.colmap_commands(
        "colmap", tmp, ws, camera_model="OPENCV", matcher="exhaustive",
        use_gpu=False, mask_dir=mask_dir,
    )
    fe = cmds[0]
    check("--ImageReader.mask_path" in fe, f"feature_extractor 带上了 mask 路径")
    check(str(mask_dir) in fe, "路径就是传入的 mask 目录")
    cmds2 = colmap_recon.colmap_commands(
        "colmap", tmp, ws, camera_model="OPENCV", matcher="exhaustive", use_gpu=False,
    )
    check("--ImageReader.mask_path" not in cmds2[0], "不传 mask 时命令行里没有这个选项（默认行为不变）")
    # 选项探测：假装这个版本不支持 mask → 必须**忽略而不是硬传**
    orig = colmap_recon._supported_options
    try:
        colmap_recon._supported_options = lambda *a, **k: {"--ImageReader.camera_model"}
        cmds3 = colmap_recon.colmap_commands(
            "colmap", tmp, ws, camera_model="OPENCV", matcher="exhaustive",
            use_gpu=False, mask_dir=mask_dir,
        )
        check("--ImageReader.mask_path" not in cmds3[0],
              "版本不支持时不硬传该选项（硬传会让 COLMAP 报 unrecognised option 直接退出）")
    finally:
        colmap_recon._supported_options = orig

    print("\n" + "=" * 72)
    print(f"结果：{passed} 通过 / {failed} 失败")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
