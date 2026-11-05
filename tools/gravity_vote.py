"""补丁：重力方向改为「点云平面投票 + 相机朝向」联合判定。

背景（实测）：
  · 白瓷砖卫生间是竖屏 900×1600，且拍洁具时镜头朝下俯拍 →
    用「每帧图像下方取中位」估出的重力偏离真实竖直 27°~45°。
  · up 一歪，房间盒就变成「15.5 × 12.8 × 26.4」这种竖塔（高 > 长宽），
    单墙补全、地面、层高、家具剥壳全部连带失真。

做法：
  1. 候选方向 = 相机估出的上方向 + 点云 PCA 的三个主轴（各自正负）
  2. 给每个候选打分：把点云投影到该轴上做细直方图，
     真实房间的地面/天花会形成**尖锐的高峰** → 用「最密柱占比」当分数
  3. 再加一条硬约束：相机必须落在点云的竖向范围内（房间的地面与天花之间）
  4. 取分数最高者；把候选、分数、最终选择写进证据，便于事后复核

为什么不直接用 PCA 第一主轴：房间里家具会主导方差方向，
而地面/天花是「薄而大」的平面 —— 直方图峰值对这种结构更敏感。
"""

from __future__ import annotations

import numpy as np


def _bin_score(v: np.ndarray, binw: float) -> dict:
    """评估某个方向"像不像竖直轴"。

    只找"最密柱"是不够的 —— 实测那样选出来的常常是墙面方向（墙也是薄而大的平面）。
    房间的竖直轴有两个特征：
      1. 最密的柱必须落在**轴的两端**（地面在最下、天花在最上），而不是中间
      2. 面积极大（薄而大）→ 最密柱占比高
    用这两条的乘积当分数。
    """
    out = {"peak": 0.0, "peakPos": None, "endBonus": 0.0}
    if len(v) < 50 or binw <= 0:
        return out
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < binw * 4:
        return out
    nb = int(np.ceil((hi - lo) / binw))
    idx = np.clip(((v - lo) / binw).astype(np.int64), 0, nb - 1)
    counts = np.bincount(idx, minlength=nb)
    # 取相邻两根柱的和：地面点常有 1~2 个柱宽的厚度
    pair = counts[:-1] + counts[1:] if nb > 1 else counts
    k = int(np.argmax(pair))
    peak = float(pair[k]) / float(len(v))
    pos = (k + 0.5) / nb            # 0 = 轴负端，1 = 轴正端
    # 地面/天花应当贴近两端；越靠近中间越不像"竖直轴"
    edge = max(0.0, 1.0 - min(pos, 1.0 - pos) / 0.22)   # 在 ±22% 以内线性衰减到 0
    out.update({"peak": peak, "peakPos": round(float(pos), 4), "endBonus": round(float(edge), 3)})
    return out


def refine_up_with_planes(P: np.ndarray, cam_xyz: np.ndarray, up_cam: np.ndarray,
                          scale: float) -> tuple[np.ndarray, dict]:
    """返回 (best_up, evidence)。"""
    P = np.asarray(P, float)
    if len(P) < 200:
        return up_cam, {"source": "camera-orientation", "reason": "点太少，不投票"}

    center = P.mean(axis=0)
    binw = max(0.01 * float(scale), 1e-6)
    rng_ratio = float(np.percentile(np.linalg.norm(P - center, axis=1), 95))

    # 相机高度的**集中度**：正确竖直轴上，人站着拍 → 高度集中；轴歪 → 摊开
    def cam_spread(d: np.ndarray) -> float:
        if len(cam_xyz) == 0:
            return 1.0
        hc = (cam_xyz - center) @ d
        return float(np.std(hc) / max(rng_ratio, 1e-9))

    # 相机在点云竖向范围里的占比：相机必须落在"地面与天花之间"
    def cam_inside(d: np.ndarray) -> float:
        if len(cam_xyz) == 0:
            return 0.0
        hp = (P - center) @ d
        hc = (cam_xyz - center) @ d
        lo, hi = np.percentile(hp, 1.0), np.percentile(hp, 99.0)
        return float(((hc >= lo) & (hc <= hi)).mean())

    # 候选：相机上方向、点云 PCA 三轴（±），以及它们与"竖直近似"的粗筛
    cands: list[tuple[str, np.ndarray]] = [("camera", np.asarray(up_cam, float))]
    d = P - center
    # 用协方差特征向量作候选（3 个轴 × 2 个方向）
    try:
        w, V = np.linalg.eigh(d.T @ d / max(len(d), 1))
        for k in range(3):
            ax = V[:, k]
            if np.linalg.norm(ax) > 1e-9:
                cands.append((f"pca{k}", ax / np.linalg.norm(ax)))
                cands.append((f"pca{k}-", -ax / np.linalg.norm(ax)))
    except np.linalg.LinAlgError:
        pass

    scored = []
    for name, ax in cands:
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        v = d @ ax
        bs = _bin_score(v, binw)
        s_cam = cam_inside(ax)
        spread = cam_spread(ax)
        score = (bs["peak"] * (0.15 + 0.85 * bs["endBonus"])
                 * (1.0 / (1.0 + 4.0 * spread)) * (0.4 + 0.6 * s_cam))
        scored.append({"name": name, "score": round(float(score), 4),
                       "planePeak": round(float(bs["peak"]), 4),
                       "peakPos": bs["peakPos"], "endBonus": bs["endBonus"],
                       "camInside": round(float(s_cam), 3),
                       "camSpread": round(float(spread), 4),
                       "axis": [round(float(x), 6) for x in ax]})

    scored.sort(key=lambda s: -s["score"])
    best = scored[0]
    # 相机方向明显更差时才替换（避免把本来正确的方向换掉）
    cam_entry = next(s for s in scored if s["name"] == "camera")
    if best["name"] != "camera" and best["score"] > cam_entry["score"] * 1.15:
        chosen = np.asarray(best["axis"], float)
        src = f"plane-vote:{best['name']}"
    else:
        chosen = cands[0][1]
        chosen = chosen / (np.linalg.norm(chosen) + 1e-12)
        src = "camera-orientation"

    return chosen, {
        "source": src,
        "binWidth": round(float(binw), 6),
        "p95Radius": round(float(rng_ratio), 4),
        "cameraScore": cam_entry["score"],
        "chosenScore": best["score"],
        "candidates": scored[:6],
        "note": "候选方向按「点云在该轴上的平面峰值 × 相机是否落在竖向范围内」打分；"
                "相机方向明显更差时才替换（阈值 1.15×）",
    }
