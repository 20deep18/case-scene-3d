#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""2D 检测 + 深度反投影 → 3D 家具盒（专家 P1 路线）。

为什么换这条路（实测依据）：
  几何分割（连通性 + 包围盒）在宿舍这间房里**拿不出上下铺**：
  上下铺本质是"大片平面（床板）+ 少量立柱"，而流水线为了去掉墙/地/天棚会**剥平面**，
  于是把床板当背景剥掉；不剥平面时，墙-地-床又连成一团，一聚类就是一整块。
  专家意见：1~2 人周内应走"开放词表检测 + 深度反投影"，语义检测能直接框住整张床。

实现：
  1. 用 Grounding DINO（开放词表）在每帧上检出候选框与类别
  2. 框内像素取 VGGT 的**稠密深度**，用 VGGT 的内参/外参反投影成世界坐标点
  3. 取稳健包围盒（P5~P95，抗边缘膨胀），得到该帧下的 3D 盒
  4. 跨帧按"中心距离 + 尺寸相似"合并同一件家具
  5. 输出与 `furniture.json` **同结构**（frame + objects），这样现有
     评测台 / 网页图层 / 线稿渲染都能直接吃

坐标约定：VGGT 的 518×518 是**方图补边**后的图（与 vggt_cloud.py 一致），
所以检测输入也用同一套 `load_and_preprocess_images_square`，像素坐标才对得上。

用法:
  <python> tools/detect_furniture.py --frames output/dorm/vggt-24 \
      --depth output/dorm/dorm.depth.npz --poses output/dorm/dorm.poses.json \
      --model-dir models/grounding-dino-tiny --out output/dorm/detected.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "models" / "vggt" / "repo" / "vggt-main"
sys.path.insert(0, str(REPO))

DEFAULT_PROMPTS = "bed. bunk bed. door. wardrobe. desk. chair. cabinet. toilet. sink. sofa."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="帧目录（与 poses.json 的帧顺序一致）")
    ap.add_argument("--depth", required=True, help="VGGT 深度 npz（vggt_cloud.py --save-depth 产出）")
    ap.add_argument("--poses", required=True, help="VGGT 的 poses.json")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "grounding-dino-tiny"))
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--threshold", type=float, default=0.32, help="检测置信度阈值")
    ap.add_argument("--max-area", type=float, default=0.45,
                    help="框最大占图比例（超过就丢：实测阈值过低时模型会给整张图一个框）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-points", type=int, default=300, help="一个框至少要有多少有效深度像素")
    ap.add_argument("--merge-ratio", type=float, default=0.05, help="跨帧合并半径 = 场景对角线 × 该比例（尺度无关）")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForZeroShotObjectDetection, AutoTokenizer

    from vggt.utils.load_fn import load_and_preprocess_images_square

    # ---- 数据 ------------------------------------------------------
    poses = json.loads(Path(args.poses).read_text(encoding="utf-8"))
    frame_names = poses["frames"]
    E = np.asarray(poses["extrinsic"], float)      # world→camera
    K = np.asarray(poses["intrinsic"], float)      # 518 分辨率下的内参
    d = np.load(args.depth)
    # ⚠️ VGGT 的 depth 保存下来是 [N,H,W,1]（多一个通道维），要 squeeze 掉，
    #    否则后面的 depth[yy, xx] 索引会变成 4 维数组（实测踩到）。
    depth_all = np.squeeze(d["depth"].astype(np.float32))
    conf_all = np.squeeze(d["conf"].astype(np.float32))
    if depth_all.ndim == 2:                      # 单帧
        depth_all, conf_all = depth_all[None], conf_all[None]
    n = min(len(frame_names), len(E), len(depth_all))
    H, W = depth_all.shape[1], depth_all.shape[2]
    print(f"帧 {n}  深度 {depth_all.shape}  内参 f={K[0,0,0]:.1f}")

    # 与 VGGT 完全一致的预处理（方图补边），保证像素坐标对得上
    paths = [str(Path(args.frames) / name) for name in frame_names[:n]]
    images, _ = load_and_preprocess_images_square(paths, W)
    print(f"预处理图: {tuple(images.shape)}")

    # ---- 模型 ------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载检测模型 {args.model_dir} → {device}")
    proc = AutoImageProcessor.from_pretrained(args.model_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_dir).to(device).eval()
    # 提示词拆成若干短语，**逐个短语推理**：这样每个框的类别天然确定，
    # 不依赖 post_process_grounded_object_detection 里的 token→短语映射
    # （transformers 4.49 里这个后处理函数根本不存在，实测确认）。
    phrases = [p.strip(" .") for p in args.prompts.replace(";", ".").split(".") if p.strip(" .")]
    print(f"短语 {len(phrases)} 个：{phrases}")

    def unproject(px, py, depth, Ei):
        """像素 + 深度 → 世界坐标（VGGT 的 extrinsic 是 world→camera）。

        ⚠️ 必须传**当前帧**的外参 Ei：第一版直接用了整份数组 E（形状 [N,3,4]），
        取 E[:3,3] 会越界（实测 IndexError）。
        """
        fx, fy = K[0, 0, 0], K[1, 1, 1]
        cx, cy = K[0, 0, 2], K[1, 0, 2]
        x = (px - cx) / fx * depth
        y = (py - cy) / fy * depth
        cam = np.stack([x, y, depth], axis=-1)
        R = Ei[:3, :3]      # world→camera 的旋转
        tvec = Ei[:3, 3]
        return (cam - tvec[None, :]) @ R      # R 正交：逆 = 转置

    boxes: list[dict] = []
    for i in range(n):
        img = (images[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(img)
        img_in = proc(images=[pil], return_tensors="pt")
        pix = img_in["pixel_values"].to(device)
        dets = []
        for ph in phrases:
            ti = tok([ph + "."], return_tensors="pt", padding=True)
            ti = {k: v.to(device) for k, v in ti.items()}
            with torch.no_grad():
                out = model(pixel_values=pix, **ti)
            probs = out.logits.sigmoid()[0]              # [Q, T]
            scores = probs.max(dim=-1).values            # 每个 query 的最佳短语得分
            boxes_norm = out.pred_boxes[0]               # [Q, 4] cx,cy,w,h（归一化）
            keep = scores > args.threshold
            for b, s in zip(boxes_norm[keep].tolist(), scores[keep].tolist()):
                cx, cy, bw, bh = b
                dets.append(([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], s, ph))
        depth = depth_all[i]
        conf = conf_all[i]
        vi = np.clip(depth > 0, 0, 1) & (conf > 0)
        for box_n, score, label in dets:
            # 归一化 → 像素（输入就是 518×518 的方图，与深度图同尺寸）
            x0, y0, x1, y1 = [int(round(v * s)) for v, s in
                              zip(box_n, (W, H, W, H))]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W - 1, x1), min(H - 1, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            # 框太大 = 模型在"框整张图"，这种框反投影出来就是整个房间（实测踩到）。
            area_ratio = ((x1 - x0 + 1) * (y1 - y0 + 1)) / float(W * H)
            if area_ratio > args.max_area:
                continue
            m = np.zeros_like(vi)
            m[y0:y1 + 1, x0:x1 + 1] = True
            m &= vi
            if int(m.sum()) < args.min_points:
                continue
            yy, xx = np.nonzero(m)
            zz = depth[yy, xx]
            world = unproject(xx.astype(np.float32), yy.astype(np.float32), zz, E[i])
            # 稳健包围盒：P5~P95，抑制边缘"膨胀"（专家提到深度反投影在边缘会外扩）
            lo = np.percentile(world, 5, axis=0)
            hi = np.percentile(world, 95, axis=0)
            size = hi - lo
            if (size <= 0).any() or size.max() > 8.0:
                continue
            # 抽样保留世界坐标：后面按"同一物体累积所有帧的点"再取一次盒，
            # 这样部分视角能拼出全貌（按尺寸合并会把同一物体的不同视角当成不同物体）
            step = max(1, len(world) // 2000)
            boxes.append({
                "frame": i, "label": label, "score": round(float(score), 4),
                "center": ((lo + hi) / 2).tolist(), "size": size.tolist(),
                "points": int(m.sum()),
                "world": world[::step].astype(np.float32),
            })
    print(f"原始检测（含跨帧重复）：{len(boxes)} 个")

    # ---- 跨帧合并 --------------------------------------------------
    merged: list[dict] = []
    # ---- 按"中心聚类 + 点并集"合并，而不是按尺寸合并 --------------------
    # 为什么：同一件家具从不同角度看，可见尺寸差异很大（宿舍的上下铺实测
    # 在不同帧量出 0.9~1.7 单位），按尺寸合并会把它们判成不同物体；
    # 而把各帧反投影出的**世界点**并起来，部分视角就能拼出完整轮廓。
    # 场景对角线（用所有反投影点估计）→ 合并半径**按比例**给。
    # ⚠️ 为什么要按比例：绝对距离在不同尺度下完全不可比 —— 实测 merge_dist=0.45
    #    时半径 0.9 单位 ≈ 整间房，于是 171 个检测全被并成 1 个盒子。
    all_pts = np.concatenate([b["world"] for b in boxes], axis=0) if boxes else np.zeros((0, 3))
    diag = float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))) if len(all_pts) else 1.0
    merge_r = max(args.merge_ratio * diag, 1e-3)
    print(f"场景对角线 {diag:.2f} → 合并半径 {merge_r:.3f}（比例 {args.merge_ratio}）")

    clusters: list[dict] = []
    for b in sorted(boxes, key=lambda x: -x["score"]):
        c = np.asarray(b["center"], float)
        hit = None
        for cl in clusters:
            if np.linalg.norm(c - cl["center"]) <= merge_r:
                hit = cl
                break
        if hit is None:
            clusters.append({"center": c.copy(), "labels": {}, "pts": [b["world"]],
                             "frames": {b["frame"]}, "score": b["score"]})
        else:
            k = len(hit["pts"])
            hit["center"] = (hit["center"] * k + c) / (k + 1)
            hit["pts"].append(b["world"])
            hit["frames"].add(b["frame"])
            hit["labels"][b["label"]] = hit["labels"].get(b["label"], 0) + 1
            hit["score"] = max(hit["score"], b["score"])

    for cl in clusters:
        P_w = np.concatenate(cl["pts"], axis=0)
        if len(P_w) < args.min_points:
            continue
        lo = np.percentile(P_w, 5, axis=0)
        hi = np.percentile(P_w, 95, axis=0)
        size = hi - lo
        if (size <= 0).any() or size.max() > 10.0:
            continue
        label = max(cl["labels"].items(), key=lambda kv: kv[1])[0] if cl["labels"] else "object"
        merged.append({
            "center": ((lo + hi) / 2).tolist(), "size": size.tolist(),
            "points": int(len(P_w)), "label": label,
            "score": round(float(cl["score"]), 4),
            "votes": len(cl["frames"]),
            "sizeSpread": 0.0,
        })

    # 保留条件：① 至少 3 帧看到（偶发误检只出现 1~2 帧）；② 各帧尺寸一致（离散 ≤25%）
    keep = [m for m in merged
            if m["votes"] >= 3 and m.get("sizeSpread", 0.0) <= 0.25]
    if not keep:
        keep = sorted(merged, key=lambda m: -m["votes"])[:3]   # 兜底：至少给出最稳的几个
    print(f"票数≥3 且尺寸一致：{len(keep)} 件（原 {len(merged)} 件）")
    merged = keep
    merged.sort(key=lambda m: -np.prod(m["size"]))
    print(f"合并后：{len(merged)} 件")

    # ---- 输出（与 furniture.json 同结构）---------------------------
    up = [0.0, 0.0, 1.0]
    out = {
        "source": "detect_furniture.py (Grounding DINO + VGGT depth)",
        "frame": {"origin": [0.0, 0.0, 0.0], "axisU": [1.0, 0.0, 0.0], "axisV": [0.0, 1.0, 0.0], "up": up},
        "objects": [
            {"center": m["center"], "size": m["size"], "points": m["points"],
             "label": m["label"], "score": m["score"], "votes": m["votes"]}
            for m in merged
        ],
        "note": "2D 检测 + 深度反投影；尺寸受深度尺度影响，需用一次实测标定",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {args.out}")
    for m in merged[:8]:
        print(f"  {m['label']:<12} 尺寸 {np.round(m['size'], 2)}  票 {m['votes']}  "
              f"置信 {m['score']:.2f}  点 {m['points']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
