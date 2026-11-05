#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回归：模糊帧**不能**留在图目录里（否则 COLMAP 递归扫描会把它们又读进重建）。

背景（2026-10-07 实测的真 bug）
-------------------------------
`extract_frames.py` 的模糊过滤原来把剔除的帧移到 `<out>/_rejected/`。
但 COLMAP 的 `--image_path` 会**递归扫描子目录** —— 那批模糊帧照样进了重建。
实测用户房间视频（fps 3）：保留 227 + 剔除 45，数据库里却是 **272 张**，
其中 45 张路径为 `_rejected/frame_00XX.png`；而重建断链的帧区间（72~85、118~125）
正好就是这些模糊帧。文档里早记过同一现象（"COLMAP 把 8 张模糊帧也注册了，占 3%"），
当时量到 3%，这次 16.5%。

现在：剔除帧放到**同级目录** `<out>-rejected/`，并且 `colmap_recon.py` 会
在发现图目录里还有含图片的子目录时**明确告警**（防历史数据/人工再踩）。

用法: pyruntime312\\python.exe tests/test_frame_reject_dir.py
退出码 0 通过 / 1 失败
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "pipeline", ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import config  # noqa: E402
import extract_frames  # noqa: E402
import colmap_recon  # noqa: E402

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


def make_frames(dirpath: Path, n_sharp: int = 3, n_blur: int = 2) -> None:
    """造几张"清晰"与"模糊"的图。

    用 PIL：清晰图 = 高频棋盘格；模糊图 = 把它高斯模糊到几乎没细节。
    模糊判据是拉普拉斯方差，所以这样造出来的分数差得很开（实测 sharp≈数千 / blur≈个位数）。
    """
    from PIL import Image, ImageFilter

    dirpath.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    # 高频纹理（棋盘 + 噪声）：拉普拉斯方差高
    base = np.zeros((240, 320, 3), np.uint8)
    yy, xx = np.mgrid[0:240, 0:320]
    base[((yy // 8 + xx // 8) % 2) == 0] = 230
    base[((yy // 8 + xx // 8) % 2) == 1] = 25
    base = np.clip(base.astype(np.int16) + rng.integers(-30, 30, base.shape), 0, 255).astype(np.uint8)
    sharp = Image.fromarray(base)
    blur = sharp.filter(ImageFilter.GaussianBlur(9))
    for i in range(n_sharp):
        sharp.save(dirpath / f"frame_{i + 1:04d}.png")
    for i in range(n_blur):
        blur.save(dirpath / f"frame_{n_sharp + i + 1:04d}.png")


def count_images(dirpath: Path) -> int:
    if not dirpath.is_dir():
        return 0
    return sum(1 for p in dirpath.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def main() -> int:
    global passed, failed
    print("=" * 74)
    print("模糊帧不能留在图目录里（COLMAP 会递归读进去）")
    print("=" * 74)

    tmp = ROOT / ".buildtmp" / "frame-reject-test"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    src = tmp / "src"
    make_frames(src, n_sharp=3, n_blur=2)

    print("\n[1] 跑模糊过滤（阈值 100，故意让模糊图被剔除）")
    result = extract_frames.blur_filter(
        src, threshold=100.0, has_analysis=False, numpy=np
    )
    kept, rejected = result["kept"], result["rejected"]
    print(f"       保留 {len(kept)} 张，剔除 {len(rejected)} 张")
    check(len(rejected) == 2, f"两张模糊图被判为模糊并剔除（实际 {len(rejected)}）")
    check(len(kept) == 3, f"三张清晰图被保留（实际 {len(kept)}）")

    print("\n[2] 剔除的帧必须**不在**图目录里（这是 bug 的核心）")
    inside = count_images(src)
    check(inside == len(kept), f"图目录里只剩保留的 {len(kept)} 张（实际 {inside}）")
    subdirs = [p.name for p in src.iterdir() if p.is_dir()]
    check(subdirs == [], f"图目录里没有任何子目录（实际 {subdirs}）")

    print("\n[3] 剔除的帧要放在**同级目录**，且能被看到")
    rej_dir = tmp / f"{src.name}-rejected"
    check(rej_dir.is_dir(), f"剔除目录存在：{rej_dir.name}")
    check(count_images(rej_dir) == len(rejected),
          f"剔除目录里有 {len(rejected)} 张（实际 {count_images(rej_dir)}）")
    check(all((rej_dir / name).is_file() for name in rejected),
          "frames.json 里记录的每个剔除帧都能在剔除目录里找到")

    print("\n[4] colmap_recon 的安全网：图目录里若有含图片的子目录必须告警")
    # 手工造一个"历史数据"式的图目录：子目录里塞图片
    legacy = tmp / "legacy-images"
    make_frames(legacy, n_sharp=2, n_blur=0)
    (legacy / "_rejected").mkdir(parents=True, exist_ok=True)
    make_frames(legacy / "_rejected", n_sharp=2, n_blur=0)
    stray = colmap_recon.find_image_subdirs(legacy)
    check(len(stray) == 1 and stray[0][1] == 2,
          f"查出 1 个含图片的子目录、里面 2 张（实际 {[(p.name, n) for p, n in stray]}）")
    clean = colmap_recon.find_image_subdirs(src)
    check(clean == [], f"干净的图目录查不出子目录（实际 {clean}）")
    # 而且 list_images 只看本层，不会被带偏
    check(len(colmap_recon.list_images(legacy)) == 2,
          f"list_images 只看本层：{len(colmap_recon.list_images(legacy))} 张（应为 2）")

    print("\n[5] 重跑时不会把上一轮的剔除帧混进来")
    # 先放一个"上一轮遗留"的垃圾文件
    (rej_dir / "frame_9999.png").write_bytes(b"stale")
    result2 = extract_frames.blur_filter(
        src, threshold=100.0, has_analysis=False, numpy=np
    )
    leftover = (rej_dir / "frame_9999.png").exists()
    check(not leftover, "重跑会清掉上一轮遗留在剔除目录里的文件")
    check(count_images(rej_dir) == len(result2["rejected"]),
          f"剔除目录内容与本轮结果一致（{count_images(rej_dir)} 张）")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + "=" * 74)
    print(f"结果：{passed} 通过 / {failed} 失败")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
