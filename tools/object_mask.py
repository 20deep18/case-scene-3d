#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""物体背景分离：关键帧画框 → GrabCut 分割 → 逐帧传播 → 输出 COLMAP/训练用的 mask。

为什么需要它（交接文档 §四，用户点名要"可靠做法"）
---------------------------------------------------
用户的抱怨是「柜子之外还有很多高斯」「拍摄点附近凑成一团」。根因是：
柜子那段素材里，**背景（房间、地面、旁边的家具）也在被重建**，
COLMAP 会在背景上找到大量特征点，训练时这些高斯又被"拉"到镜头附近。

用户明确要求用**可靠做法**，不要"颜色/深度阈值"那种糊弄做法。所以这里走
**交互式分割 + 时序传播**这条工业界标准路线：

  1. 关键帧上画一个框（框住要保留的东西）；
  2. `cv2.grabCut(..., GC_INIT_WITH_RECT)` 得到该帧的二值 mask；
  3. **相邻帧用上一帧的结果当初始化**（`GC_INIT_WITH_MASK`）逐帧推进 ——
     视频相邻帧高度相似，这个传播很稳，不需要每帧都画框；
  4. 输出 PNG mask（**黑 = 忽略**，与 COLMAP 的 `--ImageReader.mask_path` 约定一致），
     文件名与图片同名。

为什么是"物体算法专有"：房间拍摄时**没有"背景"** —— 墙、地、天花板都是主体。
给房间加 mask 只会把房间的一部分挖掉。所以这个步骤只在 `sceneMode == "object"` 时走。

用法
----
    pyruntime312\\python.exe tools/object_mask.py ^
        --images <帧目录> --out <mask 目录> ^
        --key-frame frame_0060.png --box 0.08,0.02,0.84,0.96
    # --box 用**归一化** x,y,w,h（0~1，相对画面宽高），换分辨率也不用改；
    # 也可以 --box-pixels x,y,w,h 直接给像素。

产物
----
    <out>/*.png           与图片同名（扩展名换成 .png）的二值 mask：255=保留，0=忽略
    <out>/masks.json      取证用：参数、每帧前景占比、可疑帧清单
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "tools", ROOT / "pipeline"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# 前景占比的健康区间。超出就说明分割跑飞了（整幅保留 / 整幅丢掉），
# 必须报出来而不是默默产出一堆废 mask。
FG_RATIO_MIN = 0.02
FG_RATIO_MAX = 0.97


def list_images(images_dir: Path) -> list[Path]:
    """按文件名排序返回图片（与 COLMAP 的时序、抽帧的 frame_0001… 一致）。"""
    return sorted(
        (p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name,
    )


def _load_bgr(path: Path) -> np.ndarray:
    cv2 = _cv2()
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读不到图片：{path}")
    return img


def _cv2():
    try:
        import cv2  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "背景分离需要 OpenCV（cv2）。请确认 pydeps312 里有 opencv-python："
            "项目自带的 pyruntime312 已经带了（实测 cv2 5.0.0）。"
        ) from exc
    return cv2


def box_to_pixels(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """归一化 (x, y, w, h) → 像素矩形 (x, y, w, h)，并夹到画面内。"""
    x, y, w, h = box
    px = int(round(x * width))
    py = int(round(y * height))
    pw = int(round(w * width))
    ph = int(round(h * height))
    px = max(0, min(px, width - 2))
    py = max(0, min(py, height - 2))
    pw = max(2, min(pw, width - px))
    ph = max(2, min(ph, height - py))
    return px, py, pw, ph


def clean_mask(mask: np.ndarray, *, close_px: int = 5, min_area_ratio: float = 0.002) -> np.ndarray:
    """形态学闭运算 + 去掉零星小块。

    为什么要清：GrabCut 在低纹理区域会留下麻点，这些麻点在 COLMAP 里意味着
    "东一个西一个的特征点被丢掉"，反而破坏匹配的连续性。
    ⚠️ 但**不做"只留最大连通块"** —— 柜子的搁板/边框本来就是多个连通块，
    只留最大块会把柜子的结构切掉一半（这是"看起来更干净、实际更糟"的典型）。
    """
    cv2 = _cv2()
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if min_area_ratio > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
        area_min = max(1, int(mask.size * min_area_ratio))
        keep = np.zeros_like(mask)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= area_min:
                keep[labels == label] = 255
        mask = keep
    return mask


def _gc_masks(prev: np.ndarray, *, dilate_px: int) -> np.ndarray:
    """（保留给外部调用/排查用）用上一帧的 mask 构造 GrabCut 初始化。"""
    cv2 = _cv2()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    fg = (prev > 127).astype(np.uint8)
    sure_fg = cv2.erode(fg, k, iterations=1)
    sure_bg = (cv2.dilate(fg, k, iterations=1) == 0).astype(np.uint8)
    init = np.full(prev.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    init[fg > 0] = cv2.GC_PR_FGD
    init[sure_fg > 0] = cv2.GC_FGD
    init[sure_bg > 0] = cv2.GC_BGD
    return init


def segment_by_rect(img: np.ndarray, rect_px: tuple[int, int, int, int], *, iters: int) -> np.ndarray:
    """关键帧：给定矩形，用 GrabCut 分出前景。返回 0/255。"""
    cv2 = _cv2()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect_px, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def segment_by_prev(img: np.ndarray, prev: np.ndarray, *, iters: int, dilate_px: int) -> np.ndarray:
    """后续帧：以上一帧的结果为初始化，继续 GrabCut（时序传播）。

    ⚠️ 这里有两条**必须存在**的守卫，它们都是踩出来的：
      1. 上一帧没有前景（例如关键帧框选失败）→ 直接返回空，
         否则 `cv2.grabCut` 会抛内部断言
         `!bgdSamples.empty() && !fgdSamples.empty() in function 'initGMMs'`，
         整条链路当场崩掉；
      2. 容差带比前景本身还宽（物体很小 / --dilate-px 给大了）→ sure-FG 被腐蚀空，
         同样触发上面的断言。此时退回"只有 probable 区域"的初始化。
    两种情况都只做"尽量不崩 + 不产出垃圾"，出问题由调用方按前景占比记入可疑帧。
    """
    cv2 = _cv2()
    if not (prev > 127).any():
        return np.zeros_like(prev)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    fg = (prev > 127).astype(np.uint8)
    sure_fg = cv2.erode(fg, k, iterations=1)
    if not sure_fg.any():
        small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_fg = cv2.erode(fg, small, iterations=1)
    sure_bg = (cv2.dilate(fg, k, iterations=1) == 0).astype(np.uint8)
    if not sure_bg.any():
        sure_bg = (1 - fg).astype(np.uint8)
    if not sure_fg.any() or not sure_bg.any():
        return prev.copy()
    init = np.full(prev.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    init[fg > 0] = cv2.GC_PR_FGD
    init[sure_fg > 0] = cv2.GC_FGD
    init[sure_bg > 0] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, init, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    return np.where((init == cv2.GC_FGD) | (init == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def interpolate_boxes(
    count: int,
    base_box: tuple[float, float, float, float],
    anchors: dict[int, tuple[float, float, float, float]] | None = None,
) -> list[tuple[float, float, float, float]]:
    """逐帧框 = 在所有「锚点」之间做**分段线性插值**。

    为什么需要它（2026-10-06 实测）：柜子那段的**前 2/3 帧里柜子铺满画面**，
    但**最后十几帧**画面底部露出了地面/房间（约 8% 高度）。一个固定框要么
    把地面留着（背景分离白做），要么把框上移、连柜子底部一起切掉。
    而"自动跟踪"又是另一类不可靠（跟丢了会静默切掉主体）。

    所以给一个**可预测**的办法：用户只在需要的那几帧上钉一个框，
    中间由本函数线性插值。没有任何隐式跟踪，每帧用的框都写进报告里可复核。
    """
    anchors = dict(anchors or {})
    anchors[0] = anchors.get(0, base_box)
    anchors[count - 1] = anchors.get(count - 1, base_box)
    marks = sorted(anchors)
    out: list[tuple[float, float, float, float]] = []
    for i in range(count):
        lo = max((m for m in marks if m <= i), default=marks[0])
        hi = min((m for m in marks if m >= i), default=marks[-1])
        if lo == hi:
            out.append(tuple(anchors[lo]))  # type: ignore[arg-type]
            continue
        t = (i - lo) / (hi - lo)
        a, b = anchors[lo], anchors[hi]
        out.append(tuple(a[k] + (b[k] - a[k]) * t for k in range(4)))  # type: ignore[arg-type]
    return out


def build_masks(
    images_dir: Path,
    out_dir: Path,
    *,
    key_index: int = 0,
    box: tuple[float, float, float, float] = (0.05, 0.05, 0.9, 0.9),
    box_anchors: dict[int, tuple[float, float, float, float]] | None = None,
    iters: int = 5,
    dilate_px: int = 8,
    close_px: int = 5,
    min_area_ratio: float = 0.002,
    both_directions: bool = True,
    limit: int | None = None,
    method: str = "box",
) -> dict:
    """跑完整条 mask 生成链路，返回取证用的报告。

    `method` 两种，**默认是更可预测的那个**：

    | 方法 | 做什么 | 什么时候对 | 什么时候错 |
    | --- | --- | --- | --- |
    | `box`（默认） | 框内全保留、框外全忽略；同一个归一化框用在所有帧 | 主体在画面里的位置基本不变（贴拍一个大件、机位固定） | 相机会大幅平移/变焦时主体跑出框 |
    | `grabcut` | 关键帧框选 + GrabCut 精修，再逐帧传播 | 主体**外观独特、与背景对比强**（人、彩色物体） | 主体是**低对比的大面积**（木质柜体）时，GrabCut 会去抓框内更"显眼"的**内容物**，反而把主体本身判成背景 |

    ⚠️ **为什么默认不是 grabcut**（2026-10-06 实测，见 docs 里那张叠加图）：
    在真实柜子帧上，用 8%,2%,84%,96% 的框跑 GrabCut，得到的前景只占 38% ——
    它保留了纸箱、白袋子、标签这些"显眼的东西"，
    却把**木质柜体（门板、背板、搁板）判成了背景**。
    而用户要看的恰恰是柜体的结构（「完全不能分辨柜子的空间结构」）。
    也就是说在这个素材上，GrabCut 去掉的正是我们要留的。
    `grabcut` 这条路仍然保留（它对"主体对比强"的素材是对的，
    并有 tests/test_object_mask.py 的合成真值用例守着）。
    """
    cv2 = _cv2()
    images = list_images(images_dir)
    if not images:
        raise RuntimeError(f"目录里没有图片：{images_dir}")
    if limit:
        images = images[:limit]
    key_index = max(0, min(int(key_index), len(images) - 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    if method not in ("box", "grabcut"):
        raise ValueError(f"未知的 mask 方法：{method}（可选 box / grabcut）")

    t0 = time.time()
    report: dict = {
        "imagesDir": str(images_dir),
        "outDir": str(out_dir),
        "count": len(images),
        "method": method,
        "keyFrame": images[key_index].name,
        "boxNormalized": list(box),
        "params": {"iters": iters, "dilatePx": dilate_px, "closePx": close_px,
                   "minAreaRatio": min_area_ratio, "bothDirections": both_directions},
        "perFrame": [],
    }
    masks: dict[int, np.ndarray] = {}

    def write_mask(index: int, mask: np.ndarray) -> None:
        # COLMAP 约定：mask 与图片**同名**（扩展名换成 .png），黑 = 忽略该区域
        target = out_dir / (images[index].stem + ".png")
        cv2.imwrite(str(target), mask)

    def record(index: int, mask: np.ndarray, how: str) -> None:
        ratio = float((mask > 127).mean())
        report["perFrame"].append({
            "name": images[index].name,
            "foregroundRatio": round(ratio, 4),
            "method": how,
        })

    # ---- box 模式：逐帧框（可含关键帧锚点插值），按各自分辨率换算成像素矩形 ----
    if method == "box":
        per_frame_boxes = interpolate_boxes(len(images), box, box_anchors)
        report["boxAnchors"] = {
            images[i].name: [round(v, 4) for v in b] for i, b in sorted((box_anchors or {}).items())
        }
        for index, path in enumerate(images):
            img = _load_bgr(path)
            h, w = img.shape[:2]
            px, py, pw, ph = box_to_pixels(per_frame_boxes[index], w, h)
            mask = np.zeros((h, w), np.uint8)
            mask[py:py + ph, px:px + pw] = 255
            # box 模式的价值就在于"边界就是用户画的那条线"，不做任何隐式形态学修改。
            masks[index] = mask
            write_mask(index, mask)
            record(index, mask, "box")
            report["perFrame"][-1]["box"] = [round(v, 4) for v in per_frame_boxes[index]]
        report["boxPixels"] = list(box_to_pixels(per_frame_boxes[0], *_load_bgr(images[0]).shape[:2][::-1]))
    else:
        # ---- grabcut 模式：关键帧框选 + 双向时序传播 ----
        key_img = _load_bgr(images[key_index])
        h, w = key_img.shape[:2]
        rect = box_to_pixels(box, w, h)
        key_mask = clean_mask(segment_by_rect(key_img, rect, iters=iters),
                              close_px=close_px, min_area_ratio=min_area_ratio)
        masks[key_index] = key_mask
        write_mask(key_index, key_mask)
        record(key_index, key_mask, "rect")
        report["boxPixels"] = list(rect)

        def propagate(order: list[int], start: int) -> None:
            prev = masks[start]
            for index in order:
                try:
                    img = _load_bgr(images[index])
                    raw = segment_by_prev(img, prev, iters=iters, dilate_px=dilate_px)
                    mask = clean_mask(raw, close_px=close_px, min_area_ratio=min_area_ratio)
                    ratio = float((mask > 127).mean())
                    how = "propagated"
                    if ratio < FG_RATIO_MIN or ratio > FG_RATIO_MAX:
                        # 传播跑飞了：退回上一帧的结果并标记，避免把整幅图丢掉/保留。
                        mask = prev.copy()
                        how = "fallback-prev"
                    masks[index] = mask
                    write_mask(index, mask)
                    record(index, mask, how)
                    prev = mask
                except Exception as exc:  # noqa: BLE001
                    masks[index] = prev
                    write_mask(index, prev)
                    record(index, prev, f"error:{type(exc).__name__}")
                    report.setdefault("errors", []).append(f"{images[index].name}: {exc}")

        propagate(list(range(key_index + 1, len(images))), key_index)
        if both_directions:
            propagate(list(range(key_index - 1, -1, -1)), key_index)

    ratios = np.asarray([f["foregroundRatio"] for f in report["perFrame"]], dtype=float)
    suspects = [f["name"] for f in report["perFrame"]
                if f["foregroundRatio"] < FG_RATIO_MIN or f["foregroundRatio"] > FG_RATIO_MAX
                or f["method"].startswith("fallback") or f["method"].startswith("error")]
    report["foregroundRatio"] = {
        "median": round(float(np.median(ratios)), 4),
        "min": round(float(ratios.min()), 4),
        "max": round(float(ratios.max()), 4),
    }
    report["suspectFrames"] = suspects
    # box 模式下"前景占比 ≈ 1"意味着一刀没切（等于没做背景分离）——必须说出来
    if method == "box" and report["foregroundRatio"]["median"] > 0.95:
        report.setdefault("warnings", []).append(
            "框几乎覆盖整幅画面（前景占比 >95%），等于没有做背景分离；请把框收紧到主体上"
        )
    report["elapsedSec"] = round(time.time() - t0, 2)
    report["ok"] = len(suspects) == 0
    report["note"] = (
        "mask 约定：黑=忽略该区域（COLMAP --ImageReader.mask_path；训练里 mask 外不计入损失）"
    )
    (out_dir / "masks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def parse_box(text: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in text.replace("，", ",").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--box 需要 4 个数：x,y,w,h")
    return parts[0], parts[1], parts[2], parts[3]


def main() -> int:
    ap = argparse.ArgumentParser(description="物体背景分离：关键帧画框 → GrabCut → 逐帧传播 → mask")
    ap.add_argument("--images", required=True, help="帧目录（extract_frames.py 的输出）")
    ap.add_argument("--out", required=True, help="mask 输出目录（喂给 COLMAP --mask-dir / 训练 --mask-dir）")
    ap.add_argument("--key-frame", default=None, help="关键帧文件名（默认第一张）")
    ap.add_argument("--box", type=parse_box, default=(0.05, 0.05, 0.9, 0.9),
                    help="归一化框 x,y,w,h（0~1）")
    ap.add_argument(
        "--box-at", action="append", default=None, metavar="帧名=x,y,w,h",
        help="在指定帧上钉一个框（可重复）。锚点之间**线性插值**。"
             "用途：只在前段/后段露出背景时，把框在那几帧收紧（实测柜子素材最后十几帧"
             "底部露出地面，一个固定框切不掉它）",
    )
    ap.add_argument("--box-pixels", type=parse_box, default=None, help="像素框 x,y,w,h（给了就用它）")
    ap.add_argument("--iters", type=int, default=5, help="GrabCut 迭代次数（默认 5）")
    ap.add_argument(
        "--method", choices=["box", "grabcut"], default="box",
        help="box（默认）：框内保留、框外忽略，可预测；"
             "grabcut：关键帧框选精修 + 逐帧传播（主体与背景对比强时才更好）",
    )
    ap.add_argument("--dilate-px", type=int, default=8, help="传播时的容差带宽（像素）")
    ap.add_argument("--close-px", type=int, default=5, help="形态学闭运算半径（像素）")
    ap.add_argument("--min-area-ratio", type=float, default=0.002, help="丢弃小于该面积占比的碎块")
    ap.add_argument("--forward-only", action="store_true", help="只从关键帧往后传播")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 张（冒烟测试用）")
    args = ap.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        print(f"[错误] 帧目录不存在：{images_dir}")
        return 2
    images = list_images(images_dir)
    if not images:
        print(f"[错误] 目录里没有图片：{images_dir}")
        return 2
    # ⚠️ 先把 --limit 应用到图片列表，再解析 --key-frame。
    #    第一版是"先按全量列表解析关键帧、再把列表截断"，
    #    结果 `--limit 40 --key-frame frame_0060.png` 会**悄悄**改用 frame_0040 ——
    #    用户要的帧和实际用的帧不一致，而输出里只有一行 keyFrame 能看出来。
    if args.limit:
        images = images[: args.limit]
    key_index = 0
    if args.key_frame:
        names = [p.name for p in images]
        if args.key_frame not in names:
            print(f"[错误] --key-frame 不在待处理列表里：{args.key_frame}"
                  + (f"（--limit {args.limit} 只取前 {len(images)} 张）" if args.limit else ""))
            return 2
        key_index = names.index(args.key_frame)

    box = args.box
    if args.box_pixels is not None:
        first = _load_bgr(images[min(key_index, len(images) - 1)])
        h, w = first.shape[:2]
        px, py, pw, ph = args.box_pixels
        box = (px / w, py / h, pw / w, ph / h)

    # --box-at 帧名=框 → 帧下标锚点
    box_anchors: dict[int, tuple[float, float, float, float]] = {}
    if args.box_at:
        names = [p.name for p in images]
        for spec in args.box_at:
            if "=" not in spec:
                print(f"[错误] --box-at 格式应为 帧名=x,y,w,h，收到：{spec}")
                return 2
            frame_name, raw = spec.split("=", 1)
            frame_name = frame_name.strip()
            if frame_name not in names:
                print(f"[错误] --box-at 的帧不在待处理列表里：{frame_name}")
                return 2
            try:
                box_anchors[names.index(frame_name)] = parse_box(raw)
            except Exception:  # noqa: BLE001
                print(f"[错误] --box-at 的框解析失败：{spec}")
                return 2

    try:
        report = build_masks(
            images_dir, Path(args.out),
            key_index=key_index, box=box, box_anchors=box_anchors,
            iters=args.iters, dilate_px=args.dilate_px,
            close_px=args.close_px, min_area_ratio=args.min_area_ratio,
            both_directions=not args.forward_only,
            limit=len(images),   # 已在上面截断，这里传总数保持一致
            method=args.method,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 生成 mask 失败：{type(exc).__name__}: {exc}")
        return 3

    print("=" * 68)
    print("背景分离（mask）生成完成")
    print(f"  方法           : {report['method']}")
    print(f"  帧数           : {report['count']}（关键帧 {report['keyFrame']}）")
    print(f"  框（像素）      : {report.get('boxPixels')}")
    print(f"  前景占比        : 中位 {report['foregroundRatio']['median']:.3f}"
          f"（最小 {report['foregroundRatio']['min']:.3f} ~ 最大 {report['foregroundRatio']['max']:.3f}）")
    if report["suspectFrames"]:
        print(f"  ⚠ 可疑帧 {len(report['suspectFrames'])} 张（传播跑飞或占比异常）："
              f"{report['suspectFrames'][:5]}")
    else:
        print("  可疑帧          : 无")
    print(f"  输出            : {args.out}（{report['count']} 个 png + masks.json）")
    print(f"  耗时            : {report['elapsedSec']}s")
    for _w in report.get("warnings", []):
        print(f"  ⚠ {_w}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
