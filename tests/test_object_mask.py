#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回归：背景分离（mask）必须**真的把物体留下、把背景去掉**，而且要能跨帧传播。

为什么用合成素材而不是真实帧
--------------------------
真实柜子素材上"哪些像素属于柜子"本身就需要人眼判断，没法当判据。
所以这里造一段**已知真值**的小序列：一个亮色方块在一个有纹理的背景上平移。
于是可以精确断言：

    · 关键帧：框住方块后，mask 覆盖方块 ≥90%，且背景被排除 ≥90%
    · 传播帧：**不给框**，只靠上一帧结果继续推进，仍要 ≥80% 命中
    · 方块消失/全屏时不能"跑飞"（占比异常要报可疑帧，而不是默默产出废 mask）
    · mask 文件名与图片同名（COLMAP 的约定，错一个字整条链路静默失效）

为什么这条测试重要：mask 是"看不见的输入"——它错了，COLMAP 与训练都不会报错，
只会**安静地**用一个更差的数据集跑完，最后表现为"模型还是不行"。
用户对这个的容忍度极低（原话：「不要用颜色/深度阈值这种糊弄做法」）。

用法: pyruntime312\\python.exe tests/test_object_mask.py
退出码 0 通过 / 1 失败
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import object_mask  # noqa: E402

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


def make_sequence(dirpath: Path, *, n: int = 8, size=(160, 240)) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """造 n 帧：一个有纹理背景 + 一个亮色方块（真值 mask 已知）。"""
    cv2 = object_mask._cv2()
    rng = np.random.default_rng(4)
    h, w = size
    # 固定纹理背景（每帧加一点噪声，模拟视频）
    base = np.zeros((h, w, 3), np.uint8)
    for y in range(0, h, 16):
        for x in range(0, w, 16):
            base[y:y + 16, x:x + 16] = (60 + (x * 7 + y * 3) % 120, 70 + (x * 3) % 100, 90 + (y * 5) % 90)
    images, truths = [], []
    for i in range(n):
        frame = base.copy()
        frame = np.clip(frame.astype(np.int16) + rng.integers(-6, 7, frame.shape), 0, 255).astype(np.uint8)
        # 亮色方块向右平移；约 0.6 m 见方 → 60x60 像素
        cx = 50 + i * 6
        cy = 60
        s = 30
        frame[cy - s:cy + s, cx - s:cx + s] = (245, 240, 235)
        truth = np.zeros((h, w), np.uint8)
        truth[cy - s:cy + s, cx - s:cx + s] = 255
        images.append(frame)
        truths.append(truth)
        cv2.imwrite(str(dirpath / f"frame_{i + 1:04d}.png"), frame)
    return images, truths


def overlap(mask: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """返回 (物体被留下的比例, 背景被误留的比例)。"""
    m = mask > 127
    t = truth > 127
    kept = float((m & t).sum()) / max(1, int(t.sum()))
    bg_kept = float((m & ~t).sum()) / max(1, int((~t).sum()))
    return kept, bg_kept


def main():
    global passed, failed
    print("=" * 72)
    print("背景分离（GrabCut + 时序传播）回归 —— 合成素材、真值已知")
    print("=" * 72)
    tmp = Path(tempfile.mkdtemp(prefix="objmask-"))
    img_dir = tmp / "images"
    out_dir = tmp / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    try:
        images, truths = make_sequence(img_dir)
        h, w = images[0].shape[:2]
        # 关键帧选中间（同时检验"前后两个方向都要传播"）
        key_index = 4
        # 框由**真值**推出来（外扩 8 px，模拟用户"把物体框进去"这个动作）。
        # ⚠️ 第一版我手算了一个框，结果没把移动后的方块完全框住，
        #    GrabCut 按定义把框外当背景，命中率掉到 0 —— 那是**我的框错了**，
        #    不是算法坏了。测试里就不该手算这种数字。
        ys, xs = np.nonzero(truths[key_index] > 127)
        pad = 8
        x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
        y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
        box_px = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        box = (box_px[0] / w, box_px[1] / h, box_px[2] / w, box_px[3] / h)
        print(f"       关键帧 {key_index + 1} 的框（像素）：{box_px}")

        print("\n[1] 关键帧：框选分割")
        key_mask = object_mask.segment_by_rect(images[key_index], box_px, iters=5)
        key_mask = object_mask.clean_mask(key_mask)
        kept, bg_kept = overlap(key_mask, truths[key_index])
        check(kept >= 0.90, f"物体被留下 {kept:.1%}（应 ≥90%）")
        check(bg_kept <= 0.10, f"背景被误留 {bg_kept:.1%}（应 ≤10%）")

        print("\n[2] 传播：不给框，只用上一帧结果推进")
        prev = key_mask
        hits = []
        for i in range(key_index + 1, len(images)):
            m = object_mask.segment_by_prev(images[i], prev, iters=5, dilate_px=8)
            m = object_mask.clean_mask(m)
            k, b = overlap(m, truths[i])
            hits.append((k, b))
            prev = m
        for i, (k, b) in enumerate(hits):
            print(f"       第 {key_index + 2 + i} 帧：留下 {k:.1%} / 误留背景 {b:.1%}")
        check(len(hits) > 0 and min(k for k, _ in hits) >= 0.80,
              f"往后传播每帧都 ≥80% 命中（最差 {min(k for k, _ in hits):.1%}）")
        check(max(b for _, b in hits) <= 0.15,
              f"往后传播误留背景 ≤15%（最差 {max(b for _, b in hits):.1%}）")

        print("\n[3] 反向传播（关键帧在中间时，前半段不能没有 mask）")
        prev = key_mask
        back = []
        for i in range(key_index - 1, -1, -1):
            m = object_mask.segment_by_prev(images[i], prev, iters=5, dilate_px=8)
            m = object_mask.clean_mask(m)
            k, b = overlap(m, truths[i])
            back.append((k, b))
            prev = m
        check(len(back) > 0 and min(k for k, _ in back) >= 0.80,
              f"往前传播每帧都 ≥80% 命中（最差 {min(k for k, _ in back):.1%}）")

        print("\n[4] 端到端（grabcut 路径）：文件名约定与取证报告")
        report = object_mask.build_masks(img_dir, out_dir, key_index=key_index, box=box,
                                        iters=5, method="grabcut")
        pngs = sorted(p.name for p in out_dir.glob("*.png"))
        expect = sorted(p.stem + ".png" for p in object_mask.list_images(img_dir))
        check(pngs == expect, f"mask 与图片同名（{len(pngs)} 个，COLMAP 就靠这个匹配）")
        check((out_dir / "masks.json").is_file(), "写出了 masks.json（含每帧前景占比，可事后复核）")
        check(report["ok"] is True, f"没有可疑帧（suspects={report['suspectFrames']}）")
        check(0.0 < report["foregroundRatio"]["median"] < 1.0,
              f"前景占比中位 {report['foregroundRatio']['median']}（在 0~1 之间，没跑飞）")
        # 真值核验：每一帧的 mask 都要落在方块上
        worst = 1.0
        for i, p in enumerate(object_mask.list_images(img_dir)):
            m = object_mask._cv2().imread(str(out_dir / (p.stem + ".png")), object_mask._cv2().IMREAD_GRAYSCALE)
            k, _ = overlap(m, truths[i])
            worst = min(worst, k)
        check(worst >= 0.80, f"端到端每一帧都 ≥80% 命中（最差 {worst:.1%}）")

        print("\n[7] box 路径：语义必须就是「框内保留、框外忽略」，且不能假装在跟踪")
        box_dir = tmp / "masks-box"
        rep_box = object_mask.build_masks(img_dir, box_dir, key_index=key_index, box=box, method="box")
        check(rep_box["method"] == "box", "报告里写明了 method=box（事后能分清用的是哪条路）")
        # mask 必须与框逐像素一致
        p0 = object_mask.list_images(img_dir)[0]
        m0 = object_mask._cv2().imread(str(box_dir / (p0.stem + ".png")), object_mask._cv2().IMREAD_GRAYSCALE)
        hh, ww = m0.shape[:2]
        px, py, pw, ph = object_mask.box_to_pixels(box, ww, hh)
        want = np.zeros((hh, ww), np.uint8)
        want[py:py + ph, px:px + pw] = 255
        check(int((m0 != want).sum()) == 0, "box 模式的 mask 与框逐像素一致（没有偷偷改边界）")
        # ⚠️ 固定框**不跟踪**会移动的主体：这里是**已知局限**，必须体现为可观测的数字，
        #    而不是"看起来成功了"。第 8 帧方块已经移出框外，命中率必然下降。
        last = object_mask.list_images(img_dir)[-1]
        m_last = object_mask._cv2().imread(str(box_dir / (last.stem + ".png")), object_mask._cv2().IMREAD_GRAYSCALE)
        k_last, _ = overlap(m_last, truths[-1])
        print(f"       固定框在第 8 帧（方块已移出框）的命中率：{k_last:.1%}")
        check(k_last < 1.0, "固定框不跟踪移动主体 —— 命中率会下降（这说明它**没有**假装在跟踪）")
        check(rep_box["foregroundRatio"]["min"] == rep_box["foregroundRatio"]["max"],
              f"box 模式前景占比恒定 {rep_box['foregroundRatio']['median']}（每帧同一块区域）")

        print("\n[8] 框开到整幅画面必须被警告（等于没做背景分离）")
        rep_wide = object_mask.build_masks(img_dir, tmp / "masks-wide", key_index=0,
                                          box=(0.0, 0.0, 1.0, 1.0), method="box")
        check(any("等于没有做背景分离" in w for w in rep_wide.get("warnings", [])),
              f"给出了警告：{rep_wide.get('warnings')}")

        print("\n[5] 跑飞保护：占比异常必须报可疑帧，而不是默默产出废 mask")
        empty_dir = tmp / "empty-images"
        empty_dir.mkdir()
        cv2 = object_mask._cv2()
        for i in range(4):
            cv2.imwrite(str(empty_dir / f"frame_{i + 1:04d}.png"), np.full((80, 120, 3), 30, np.uint8))
        rep2 = object_mask.build_masks(empty_dir, tmp / "masks-empty",
                                       key_index=0, box=(0.4, 0.4, 0.1, 0.1), iters=2)
        print(f"       纯色画面的结果：前景占比中位 {rep2['foregroundRatio']['median']}，"
              f"可疑帧 {len(rep2['suspectFrames'])}")
        check(rep2["ok"] is False or rep2["foregroundRatio"]["median"] < 0.5,
              "纯色画面不会被当成'整幅都是物体'（要么报可疑，要么占比很小）")

        print("\n[6] 归一化框与像素框必须等价")
        b1 = object_mask.box_to_pixels((0.25, 0.25, 0.5, 0.5), 400, 200)
        check(b1 == (100, 50, 200, 100), f"0.25,0.25,0.5,0.5 @400x200 → {b1}")
        b2 = object_mask.box_to_pixels((0.99, 0.99, 0.5, 0.5), 100, 100)
        check(b2[0] + b2[2] <= 100 and b2[1] + b2[3] <= 100,
              f"越界框被夹回画面内：{b2}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 72)
    print(f"结果：{passed} 通过 / {failed} 失败")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
