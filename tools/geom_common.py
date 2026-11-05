#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""几何与统计的通用原语（从 tools/scene_structure.py 拆出来的）。

为什么单独成文件（2026-10-06，交接文档 §一 的步骤 ②）
------------------------------------------------------
`scene_structure.py` 原来是一个 1600 多行的单文件，里面混着三类东西：
  1. **纯几何/统计原语**（本文件）—— 平面拟合、重力估计、密度峰、方向包围度、离群剔除；
  2. **房间算法**（structure_room.py）—— 地面/天花板/竖墙/房间盒/单间/平面轮廓；
  3. **物体算法**（structure_object.py）—— 以相机轨迹为锚的包围盒 + 水平层。
用户批准的方向就是"场景与物体各写一套算法"，混在一个文件里根本没法各改各的。

本文件里的函数**不许**依赖任何"房间/物体"语义：它们只吃点云、位姿、直方图，
返回测量值。这样它们可以被探针（tools/probe_*.py）、自检（tools/selfcheck_*.py）
和两套算法同时复用。
"""

from __future__ import annotations

import os

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "tools", ROOT / "pipeline"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import colmap_model_io as cmio  # noqa: E402


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
#
# ⚠️ 这里**只做测量，不做判断**：本文件里没有任何"房间/物体"的分类阈值。
#    分类阈值（SURROUND_ROOM_MIN / SURROUND_OBJECT_MAX）属于**策略**，
#    放在调度层 tools/scene_structure.py 里 —— 这样"量"与"判"分开，
#    换判据时不必动几何原语，几何原语也能被两套算法与探针无条件复用。


def robust_frame(points: np.ndarray) -> dict:
    """稳健中心与尺度（中位数 + P95 距离），避免被野点带跑。"""
    center = np.median(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    return {
        "center": center,
        "p95": float(np.percentile(dist, 95)),
        "p99": float(np.percentile(dist, 99)),
        "max": float(dist.max()),
    }


def fit_plane_lsq(Q: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """对一组点做最小二乘平面拟合，返回 (单位法向, d, 质心)，满足 n·x = d。"""
    c = Q.mean(axis=0)
    # SVD 的最后一个右奇异向量就是法向
    _u, _s, vt = np.linalg.svd(Q - c, full_matrices=False)
    n = vt[2]
    n = n / (np.linalg.norm(n) + 1e-12)
    return n, float(n @ c), c


def ransac_plane(
    points: np.ndarray,
    *,
    eps: float,
    iters: int = 1500,
    rng: np.random.Generator,
    score_subset: int = 20000,
    normal_hint: np.ndarray | None = None,
    perp_to: np.ndarray | None = None,
    min_perp_deg: float = 70.0,
) -> dict | None:
    """RANSAC 拟合平面。

    `perp_to` 给定时，只接受法向与它**近似垂直**的平面（用来找墙面）；
    `normal_hint` 给定时，只接受法向与它**近似平行**的平面（用来找天花板）。
    """
    n = len(points)
    if n < 12:
        return None
    sample_pool = points if n <= score_subset else points[rng.choice(n, score_subset, replace=False)]
    best = None
    cos_perp = math.cos(math.radians(min_perp_deg))
    for _ in range(iters):
        idx = rng.choice(len(sample_pool), 3, replace=False)
        p0, p1, p2 = sample_pool[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = float(np.linalg.norm(nrm))
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        if perp_to is not None and abs(float(nrm @ perp_to)) > cos_perp:
            continue  # 与参考法向不够垂直 → 不是墙
        if normal_hint is not None and abs(float(nrm @ normal_hint)) < 0.94:
            continue  # 与参考法向差太远 → 不是天花/地面
        d = float(nrm @ p0)
        dist = np.abs(points @ nrm - d)
        cnt = int((dist < eps).sum())
        if best is None or cnt > best[0]:
            best = (cnt, nrm, d)

    if best is None or best[0] < 12:
        return None

    # 用全部内点做最小二乘精修，再重算内点
    _cnt, nrm, d = best
    for _ in range(3):
        dist = np.abs(points @ nrm - d)
        inl = dist < eps
        if int(inl.sum()) < 12:
            break
        nrm2, d2, _c = fit_plane_lsq(points[inl])
        if float(nrm2 @ nrm) < 0:  # 保持法向朝向一致
            nrm2, d2 = -nrm2, -d2
        nrm, d = nrm2, d2
    dist = np.abs(points @ nrm - d)
    inl = dist < eps
    if int(inl.sum()) < 12:
        return None
    return {
        "normal": nrm,
        "d": float(d),
        "inliers": inl,
        "count": int(inl.sum()),
        "rms": float(np.sqrt((dist[inl] ** 2).mean())),
        "extent": float(np.linalg.norm(points[inl].max(axis=0) - points[inl].min(axis=0))),
    }


def orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """给定单位法向，构造平面内的正交基 (u, v)。"""
    a = np.array([1.0, 0.0, 0.0])
    if abs(float(a @ n)) > 0.9:
        a = np.array([0.0, 0.0, 1.0])
    u = a - n * float(a @ n)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    return u, v / (np.linalg.norm(v) + 1e-12)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def gravity_from_cameras(images: list[dict]) -> tuple[np.ndarray, dict]:
    """从**相机自身的朝向**估计重力方向 —— 比「找最大的平面」稳得多。

    原理：手持视频里人基本是站着的，所以相机画面里的「图像下方」在绝大多数帧上
    都指向重力方向。对全部帧的「图像下方」取中位数方向，就能拿到世界系里的重力。

    ⚠️ 为什么不能拿「点数最多的平面」当地面（我第一版就是这么写的，踩了坑）：
    实测这个案件里点数最多的那个平面**穿过了相机停留的位置** ——
    因为它其实是一扇门/一面墙（视频开头就在拍木门），
    于是后面的「房间尺寸 10.45 × 13.06 × 3.92 m」只是碰巧看着合理，
    而 167 个相机里有 75 个落在了「地面」下方，一眼就露馅。
    """
    downs = []
    for im in images:
        try:
            R = cmio.quat_to_R(np.asarray(im["qvec"], dtype=float))
        except Exception:  # noqa: BLE001
            continue
        # COLMAP 相机系：x 右、y 下、z 前。世界系里的「图像下方」= Rᵀ·(0,1,0)
        downs.append(R.T @ np.array([0.0, 1.0, 0.0]))
    if not downs:
        return np.array([0.0, 1.0, 0.0]), {"source": "fallback", "frames": 0, "spreadDeg": None}
    D = np.asarray(downs)
    mean = D.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-6:
        return np.array([0.0, 1.0, 0.0]), {"source": "fallback", "frames": len(D), "spreadDeg": None}
    g_down = mean / norm
    # 一致性：每帧的「下方」与估计值的夹角，用来判断这个视频是不是一直在乱转
    cos = np.clip(D @ g_down, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos))
    return g_down, {
        "source": "camera-orientation",
        "frames": int(len(D)),
        "spreadMedianDeg": round(float(np.median(angles)), 2),
        "spreadP90Deg": round(float(np.percentile(angles, 90)), 2),
        "consistent": bool(np.median(angles) < 30.0),
    }


def density_peak(values: np.ndarray, lo: float, hi: float, bin_width: float) -> tuple[float, float]:
    """在一段高度里找**最密的那一层**（返回 (峰值位置, 峰值占比)）。

    为什么不用分位数找地面：稀疏点云里往往夹杂着**地板以下的野点**
    （实测这个案件里，高度的 1% 分位比相机低了 6.6 m，
    而人手持拍摄时相机离地只有 ~1.5 m）。
    分位数会被这些野点拖走，而「最密的那一层」才是真正的表面 —— 地面是一大片平面，
    在高度直方图上必然是一个尖峰。
    """
    if hi <= lo or len(values) < 20:
        return (lo + hi) / 2.0, 0.0
    nb = max(6, min(400, int((hi - lo) / max(bin_width, 1e-6))))
    hist, edges = np.histogram(values, bins=nb, range=(lo, hi))
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    smooth = np.convolve(hist.astype(float), kernel, mode="same")
    i = int(np.argmax(smooth))
    return float((edges[i] + edges[i + 1]) / 2.0), float(hist[i] / max(1, len(values)))


def _local_maxima(y: np.ndarray) -> list[int]:
    """所有**局部极大**的下标（平台取中点）。"""
    n = len(y)
    out: list[int] = []
    i = 1
    while i < n - 1:
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            j = i
            while j + 1 < n and y[j + 1] == y[i]:
                j += 1
            out.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    return out


def _prominence(y: np.ndarray, idx: int, *, lo_idx: int = 0, hi_idx: int | None = None) -> float:
    """峰的**突出度**（prominence）：峰高减去两侧最低鞍部里较高的那个。

    `lo_idx`/`hi_idx`：**数据范围**（闭区间）。鞍部搜索只在这个范围内走。

    ⚠️ 为什么要这个范围：为了让"落在点云极值上的峰"（柜子的顶面/底面）能被发现，
    直方图两侧各外扩了 3 个**空** bin。如果不把外扩的空 bin 排除在鞍部搜索之外，
    边界处的**斜坡肩部**会被算成「峰高 − 0 = 满突出度」——
    于是**纯均匀噪声也会冒出一个假层**（实测：这条被 tests/test_layer_detection.py
    的"均匀噪声 → 0 个峰"当场抓住）。限定在数据范围内找鞍部之后：
      · 真峰在边界上：内侧鞍部是真实的低点 → 突出度大 → 保留；
      · 均匀噪声的边界 bin：内侧邻居跟它差不多高 → 突出度小 → 丢弃。
    """
    hi_idx = len(y) - 1 if hi_idx is None else hi_idx
    peak = float(y[idx])
    left = peak
    for i in range(idx - 1, lo_idx - 1, -1):
        if y[i] > peak:
            break
        left = min(left, float(y[i]))
    right = peak
    for i in range(idx + 1, hi_idx + 1):
        if y[i] > peak:
            break
        right = min(right, float(y[i]))
    # 一侧完全没有数据（idx 就在数据边界上）时，只用另一侧的鞍部
    if idx <= lo_idx:
        return peak - right
    if idx >= hi_idx:
        return peak - left
    return peak - max(left, right)


def detect_horizontal_layers(
    values: np.ndarray,
    *,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
    target_bins: int = 200,
    min_points: int = 120,
    prominence_ratio: float = 0.05,
    k_sigma: float = 6.0,
    min_sep_bins: int = 4,
) -> tuple[list[dict], dict]:
    """找出**真实存在的水平层**（顶面/搁板/底面），返回 (层列表, 判据证据)。

    ⚠️ 为什么把原来的「找最大 bin → 挖掉 ±3 bin → 再找」整段换掉
    ------------------------------------------------------------------
    原做法有个**机械性伪影**：挖掉 ±3 bin 之后，剩余分布的极大值往往正好落在
    挖掘窗口的**边界外一格**，于是再生出一个峰，且间距恒等于窗口宽度。
    实测证据（`柜子现场-致密`，2026-10-06）：输出 5 个峰，
    相邻间距精确等于 **0.583 / 0.583 / 0.874 / 1.748 m**，
    而当时 bin 宽 =(P99−P1)/60≈0.146 m → **4 bin = 0.583 m**。
    两个间距与 4×bin 完全吻合 —— 那不是搁板，是自己挖出来的回声。

    新做法只依赖分布本身的形状，不依赖"上一轮挖掉了什么"：
      1. 直方图平滑后取**全部局部极大**（`_local_maxima`）；
      2. 用**突出度**（`_prominence`，峰高 − 两侧较高鞍部）筛掉涟漪；
      3. 要求该层实际承载的点数够多；
      4. 相邻峰太近（< min_sep_bins 个 bin）时只留突出度大的那个。
    因为不再有"挖掉"这一步，**间距不会被人为锁定**，谐波无从产生。

    ⚠️ 突出度阈值为什么是"两条线取大"
    ----------------------------------
    只按"最大值的一个比例"当阈值是不够的：**纯均匀噪声**的直方图里，
    最大值只比平均值高几个 σ，于是 5%×最大值这条线落得比噪声涟漪还低，
    结果是 6000 个均匀点被判出 **20 个"层"**（我第一版就是这样，被合成用例当场抓住）。
    所以再加一条**噪声底**判据：用平滑直方图的中位数绝对偏差估噪声 σ
    （MAD 对"少数几个真层把上尾抬高"稳健），要求突出度同时超过 k_sigma·σ。
    合成用例的验收：单个层 → 恰好 1 个峰；三个层 → 恰好 3 个峰且高度准；
    纯均匀 → **0 个峰**（见 tests/test_layer_detection.py）。
    `k_sigma` 取 6：实测 6~10 的结论完全一致（是平台不是拟合出来的点），
    而 4~5 会让纯均匀噪声漏出 1~3 个假层。

    输出每个层的 `prominence` 与 `binWidth`，便于事后复核（不要只给一个结论）。
    """
    ev: dict = {"binWidth": None, "rangeMeters": None, "rawPeakCount": 0,
                "prominenceRatio": prominence_ratio, "kSigma": k_sigma,
                "minSepBins": min_sep_bins}
    if len(values) < min_points:
        ev["reason"] = f"点太少（{len(values)} < {min_points}）"
        return [], ev
    lo = float(np.percentile(values, lo_pct))
    hi = float(np.percentile(values, hi_pct))
    if not (hi > lo):
        ev["reason"] = "高度范围退化"
        return [], ev
    nb = int(max(24, min(400, target_bins)))
    bin_width_raw = (hi - lo) / nb
    # ⚠️ 2026-10-06：把直方图范围**向外扩 3 个 bin**。
    #    为什么：物体（柜子）的**顶面/底面正好落在点云的极值上**，而落在第 0 个
    #    （或最后一个）bin 里的峰**永远不会被"局部极大"发现** —— `_local_maxima`
    #    的循环从 i=1 开始，边界 bin 天然出局。
    #    合成真值实测（tools/probe_object_trust.py）：一个顶/中/底三块水平面的
    #    1.2×0.6×1.9 m 柜子，只报出 **1 层**；外扩之后 3 层都在。
    #    外扩不改变 bin 宽（两侧各加 3 个 bin、总数加 6），所以峰的**位置**口径不变；
    #    但突出度必须**只在数据范围内**算鞍部（见 _prominence 的说明），
    #    否则外扩的空 bin 会把边界上的斜坡肩部算成满突出度 → 纯噪声也会冒出假层。
    pad = 3
    lo -= pad * bin_width_raw
    hi += pad * bin_width_raw
    hist, edges = np.histogram(values, bins=nb + 2 * pad, range=(lo, hi))
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    smooth = np.convolve(hist.astype(float), kernel, mode="same")
    binw = bin_width_raw          # 真正的 bin 宽（外扩不改变它）
    ev["binWidth"] = round(binw, 5)
    ev["rangeMeters"] = round(hi - lo, 4)
    ev["rangePaddedBins"] = pad  # 两侧各外扩的 bin 数（让边界上的峰能被发现）
    data_lo, data_hi = pad, pad + nb - 1

    peaks = _local_maxima(smooth)
    ev["rawPeakCount"] = len(peaks)
    if not peaks:
        ev["reason"] = "分布里没有局部极大"
        return [], ev
    top = max(float(smooth[i]) for i in peaks)
    # 噪声底：MAD 估计（对少数真层抬高上尾稳健）
    med = float(np.median(smooth))
    mad = float(np.median(np.abs(smooth - med)))
    sigma = 1.4826 * mad
    thr_sigma = k_sigma * sigma
    thr = max(prominence_ratio * top, thr_sigma)
    ev["prominenceTop"] = round(prominence_ratio * top, 3)
    ev["noiseSigma"] = round(sigma, 3)
    ev["prominenceThreshold"] = round(thr, 3)

    cand = []
    for i in peaks:
        prom = _prominence(smooth, i, lo_idx=data_lo, hi_idx=data_hi)
        n_local = int(hist[max(0, i - 2): i + 3].sum())
        if prom < thr or n_local < min_points:
            continue
        cand.append({"bin": i, "prominence": prom, "points": n_local,
                     "height": float((edges[i] + edges[i + 1]) / 2.0)})

    # 相邻太近：保留突出度大的（这是唯一的"抑制"，不再依赖挖掘窗口）
    cand.sort(key=lambda c: c["bin"])
    kept: list[dict] = []
    for c in cand:
        if kept and (c["bin"] - kept[-1]["bin"]) < min_sep_bins:
            if c["prominence"] > kept[-1]["prominence"]:
                kept[-1] = c
            continue
        kept.append(c)

    total = max(1, int(hist.sum()))
    layers = []
    for c in kept:
        layers.append({
            "height": round(c["height"], 4),
            "ratio": round(c["points"] / total, 4),
            "points": int(c["points"]),
            "prominence": round(c["prominence"] / max(top, 1e-9), 4),
        })
    layers.sort(key=lambda x: x["height"])
    for i, ly in enumerate(layers):
        ly["index"] = i
    ev["peakCount"] = len(layers)
    ev["rejectedByProminence"] = int(
        sum(1 for i in peaks if _prominence(smooth, i, lo_idx=data_lo, hi_idx=data_hi) < thr)
    )
    ev["note"] = "局部极大 + 突出度筛选（不做挖掘，故不会产生等间距谐波峰）"
    return layers, ev


def camera_ray_convergence(cam_xyz: np.ndarray, cam_fwd: np.ndarray, *,
                           n_scan: int = 240, d_max_factor: float = 3.0) -> dict:
    """「视线汇聚度」：所有视线在空间里是否汇到一团（绕物体拍的特征）。

    ⚠️ **实测它在本项目的真实素材上不能当判据**（2026-10-06，详见
    tools/probe_ray_convergence.py）：柜子那份素材（真值是"贴着柜子近距离巡拍"）
    得到 minSpreadRel = 1.0000，而 room-real 是 0.9652 —— 两者不可分，且柜子是**判反**的。
    原因：那段柜子视频既不是"绕着物体转"（视线汇聚），也不是"在房间里走"，
    而是近距离摆动扫拍，视线既不发散也不汇聚。
    所以这里只把它当**证据**记录（交接文档也要求"不要只写一个布尔结论"），
    真正的判定用 camera_surround_coverage()。

    视线方向取 +Z：COLMAP 相机系 x 右 / y 下 / **z 前**，世界系前向 = Rᵀ·(0,0,1)。
    （交接文档写的"-Z 是观察方向"是 Three.js 的约定，用在这里会整体反号；
     实测校验见 tools/probe_ray_convergence.py 的 forwardCheck，+Z 余弦 +0.888。）
    """
    if len(cam_xyz) < 4 or len(cam_fwd) != len(cam_xyz):
        return {"ok": False, "reason": "相机太少或方向缺失"}
    centroid = cam_xyz.mean(axis=0)
    r_cam = float(np.median(np.linalg.norm(cam_xyz - centroid, axis=1)))
    if r_cam < 1e-9:
        return {"ok": False, "reason": "相机位置几乎重合（纯旋转拍摄）"}

    def spread_at(d: float) -> float:
        q = cam_xyz + cam_fwd * d
        return float(np.mean(np.linalg.norm(q - np.median(q, axis=0), axis=1))) / np.sqrt(3.0)

    s0 = spread_at(0.0)
    if s0 < 1e-9:
        return {"ok": False, "reason": "相机位置退化"}
    d_max = d_max_factor * r_cam
    ds = np.linspace(0.0, d_max, n_scan)
    rel = np.array([spread_at(float(d)) for d in ds]) / s0
    i_star = int(np.argmin(rel))
    d_star = float(ds[i_star])
    f_mean = cam_fwd.mean(axis=0)
    f_mean = f_mean / (np.linalg.norm(f_mean) + 1e-12)
    proj = cam_xyz @ f_mean
    total_var = float(np.var(cam_xyz, axis=0).sum())
    along_frac = float(np.var(proj) / total_var) if total_var > 1e-12 else 1.0
    q_star = np.median(cam_xyz + cam_fwd * d_star, axis=0)
    to_q = q_star[None, :] - cam_xyz
    to_q = to_q / (np.linalg.norm(to_q, axis=1, keepdims=True) + 1e-12)
    cos_q = np.sum(cam_fwd * to_q, axis=1)
    return {
        "ok": True,
        "minSpreadRel": round(float(rel[i_star]), 4),
        "dStarMeters": round(d_star, 4),
        "dStarOverCamRadius": round(d_star / r_cam, 4),
        "alongFrac": round(along_frac, 4),
        "medianCosRayToQStar": round(float(np.median(cos_q)), 4),
        "camRadiusMeters": round(r_cam, 4),
        "argminAtSearchBoundary": bool(i_star >= n_scan - 2),
        # 这条是**结论级**的提醒，防止别人看到这个字段就当真
        "verdictHint": "只作证据：实测在本项目素材上与真值不可分（柜子 1.0000 vs 房间 0.9652），勿单独用作判据",
    }


def camera_surround_coverage(cam_xyz: np.ndarray, P: np.ndarray, *,
                            k: int = 1000, n_dir: int = 256, tol_deg: float = 25.0,
                            max_pairs: int = 60_000_000) -> dict:
    """「方向包围度」：重建点云在相机周围占了多少方向的立体角。

    为什么用它顶替"视线汇聚度"（2026-10-06 实测，tools/probe_scene_mode_features.py）：

    | 素材 | 真相 | 方向包围度 |
    | --- | --- | --- |
    | room-real（178 相机） | 房间 | **0.523** |
    | run-100mb / run-64mb | 房间 | 0.523 / 0.406 |
    | 柜子现场（130 相机） | 物体（贴拍） | **0.234** |

    物理直觉：房间拍摄时相机站在屋子中间，前后左右上下都是被重建出来的表面，
    方向几乎铺满；贴着一个柜子拍时，重建点只在正前方一个锥里，大片方向是空的。
    它只看**方向**不看距离，所以不受"点云里混了背景、整体跨度被撑大"的影响。

    ⚠️ 但它**同时反映重建质量**：同一段房间视频用 sequential 匹配（只注册 61/175）
    时也只有 0.227，与柜子几乎一样。所以它是**佐证**而不是唯一判据 ——
    判定逻辑见 extract_structure 里的三态分类。
    """
    if len(cam_xyz) < 4 or len(P) < 50:
        return {"ok": False, "reason": f"相机 {len(cam_xyz)} / 点 {len(P)} 太少"}
    q = P
    if len(cam_xyz) * len(P) > max_pairs:
        take = max(50, int(max_pairs // max(1, len(cam_xyz))))
        q = P[np.linspace(0, len(P) - 1, min(take, len(P))).astype(int)]
    kk = int(min(k, len(q)))
    if kk < 8:
        return {"ok": False, "reason": "可用点太少"}
    nd = int(n_dir)
    kk_dir = np.arange(nd) + 0.5
    phi = np.arccos(1 - 2 * kk_dir / nd)
    theta = np.pi * (1 + 5 ** 0.5) * kk_dir
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    cos_tol = math.cos(math.radians(tol_deg))
    cov = np.empty(len(cam_xyz), dtype=float)
    for i, c in enumerate(cam_xyz):
        d = np.linalg.norm(q - c, axis=1)
        sel = np.argpartition(d, kk - 1)[:kk]
        v = q[sel] - c
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        cov[i] = float(((v @ dirs.T) > cos_tol).any(axis=0).mean())
    return {
        "ok": True,
        "median": round(float(np.median(cov)), 4),
        "p10": round(float(np.percentile(cov, 10)), 4),
        "p90": round(float(np.percentile(cov, 90)), 4),
        "k": kk,
        "directions": nd,
        "toleranceDeg": tol_deg,
        "pointsUsedForDirs": int(len(q)),
        "note": "1.0 = 相机四周每个方向都有重建点（房间）；接近 0 = 只有一面有点（贴拍物体）",
    }


def frustum_coverage(model: dict, xyz: np.ndarray, names: list[str]) -> dict:
    """逐帧统计「视锥里落了多少个稀疏点」——用来发现**从未被拍到的方向**。

    为什么值得单独做：实测某个案件的留出视角 PSNR 从 12.99 到 25.02 dB 差 12 dB，
    归因下来最差那一帧的视锥里只有 **106** 个稀疏点，而最好的一帧有 **10,563** 个
    （见 tools/diag_eval_views.py）。也就是说：**那个方向根本没人拍到过**，
    模型只能靠插值，必然塌。这属于「素材问题」，调参救不了 —— 只能提前告诉民警。
    """
    images = {im["name"]: im for im in model["images"]}
    cams = model["cameras"]
    counts: list[tuple[str, int]] = []
    for name in names:
        im = images.get(name)
        if im is None:
            continue
        cam = cams.get(im["camera_id"])
        if cam is None:
            continue
        try:
            R = cmio.quat_to_R(np.asarray(im["qvec"], float))
            t = np.asarray(im["tvec"], float)
            fx, fy, cx, cy = cmio.intrinsics_of(cam)
        except Exception:  # noqa: BLE001
            continue
        cw = float(cam.get("width") or 0) or cx * 2
        chh = float(cam.get("height") or 0) or cy * 2
        p_cam = (R @ xyz.T).T + t
        z = p_cam[:, 2]
        ok = z > 1e-6
        if not ok.any():
            counts.append((name, 0))
            continue
        u = fx * p_cam[ok, 0] / z[ok] + cx
        v = fy * p_cam[ok, 1] / z[ok] + cy
        inside = (u >= 0) & (u < cw) & (v >= 0) & (v < chh) & (z[ok] < 50.0)
        counts.append((name, int(inside.sum())))

    if not counts:
        return {"frames": 0}
    vals = np.asarray([c for _n, c in counts], dtype=float)
    med = float(np.median(vals))
    threshold = max(8.0, med * 0.05)
    weak = [n for n, c in counts if c < threshold]
    weak_ratio = len(weak) / len(counts)
    # 最差那一帧相对中位的比例：**只看数量占比会低估影响** ——
    # 实测 167 帧里只有 4% 是「弱帧」，但其中最差的一帧视锥里只有 106 个点
    # （中位 4570），它一个人就能把 4 视角的平均 PSNR 拉低约 3 dB。
    worst_ratio = float(vals.min() / med) if med > 0 else 1.0
    if weak_ratio >= 0.25 or worst_ratio <= 0.02:
        verdict = "有方向几乎没被拍到 —— 这些方向会大面积塌陷，建议补拍"
    elif weak_ratio >= 0.05 or worst_ratio <= 0.15:
        verdict = "少数方向拍到得很少，模型在这些方向会明显发虚"
    else:
        verdict = "覆盖良好"
    return {
        "frames": len(counts),
        "medianPointsInView": int(med),
        "p10PointsInView": int(np.percentile(vals, 10)),
        "minPointsInView": int(vals.min()),
        "worstToMedianRatio": round(worst_ratio, 4),
        "weakFrameThreshold": int(threshold),
        "weakFrames": len(weak),
        "weakRatio": round(weak_ratio, 4),
        "weakExamples": weak[:8],
        "verdict": verdict,
    }


def align_frame_to_walls(
    u: np.ndarray, v: np.ndarray, walls: list[dict], n_up: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """把地面坐标系的轴**对齐到墙面**。

    为什么必须做：`orthonormal_basis()` 只是随便取了一个与重力垂直的方向
    （把 x 轴投影下去），墙跟这个方向通常差几十度。后果有两个：
      1. 房间盒是**歪着量**的 —— 一个 8×6 的房间能量成 11×18（实测 room-real 就是）；
      2. 没法按墙把空间切成单间（本来只要在两个正交方向上切几刀）。
    做法：把每面墙的法向投影到地面平面得到一个方位角，用**倍角平均**处理
    「法向 ±n 等价」的 π 歧义，取主方位作为新的一根轴。

    ⚠️ 2026-10-06 拆分时把 `n_up` 从**模块级全局变量**改成**参数**：
    原来靠一个 `n_up_global` 全局变量传递地面法向，函数签名上看不出来它依赖什么，
    拆成独立模块后这种隐式全局依赖会立刻变成"导入顺序一错就算错"的坑。
    """
    angs = []
    wts = []
    for w in walls:
        n = np.asarray(w["normal"], dtype=float)
        n2 = n - u * float(n @ u) - v * float(n @ v)  # 投到地面平面
        ln = float(np.linalg.norm(n2))
        if ln < 1e-6:
            continue
        n2 = n2 / ln
        angs.append(math.atan2(float(n2 @ v), float(n2 @ u)))
        wts.append(float(w.get("count") or 1))
    if len(angs) < 2:
        return u, v, 0.0
    a = np.asarray(angs)
    wt = np.asarray(wts)
    # 倍角平均：把 π 周期变成 2π 周期，避免 +170° 与 -10° 被平均成 80°
    c = float((wt * np.cos(2 * a)).sum())
    s = float((wt * np.sin(2 * a)).sum())
    theta = 0.5 * math.atan2(s, c)
    a_axis = u * math.cos(theta) + v * math.sin(theta)
    a_axis = a_axis / (np.linalg.norm(a_axis) + 1e-12)
    b_axis = np.cross(n_up, a_axis)
    b_axis = b_axis / (np.linalg.norm(b_axis) + 1e-12)
    return a_axis, b_axis, math.degrees(theta)


def drop_statistical_outliers(
    P: np.ndarray, *, k: int = 12, std: float = 2.5, sample: int = 20000, seed: int = 5,
    protect: np.ndarray | None = None,
) -> np.ndarray:
    """统计式离群点剔除（PCL 的 StatisticalOutlierRemoval 做法）。返回保留掩码。

    为什么必须在**拟合之前**做：
      实测某份重建的点云**全跨度 229 m，而稳健(1/99)跨度只有 25 m —— 差 9 倍**。
      这些极端野点是 COLMAP 三角化出来的病态点（基线过短/视角过掠），
      数量不多但足以：
        · 把包围盒、取景半径整个撑大；
        · 让 RANSAC 去拟合「野点构成的假平面」，于是出现「墙宽 11~16 m 而房间只有 6.8 m」
          这种自相矛盾的输出，以及「相机跑到地面以下」。
      先洗点再做平面/地面/房间盒，才谈得上尺度与形状一致。

    判据：每个点到 k 个最近邻的平均距离 d_i；若 d_i > 全局中位 + std×1.4826×MAD
    就判为离群（用中位+MAD 而不是均值+标准差，避免野点自己把阈值抬高）。

    ⚠️ `protect`（可选布尔掩码）：**强制保留**这些点。
    为什么需要它：本判据是**相对密度**判据，而"贴着相机的表面"在三维里天生更稀疏
    （只被少数几个视角看到、面积小），于是会被误判成离群。实测（2026-10-06）：
        room-real：删掉 4417 点（9.2%），其中 **1828 个**是"离相机 ≤ 全体中位(4.25 m)"
                   的点，它们到最近保留点只有 **0.362 m**（贴着真实表面，不是野点）
        柜子      ：删掉 1475 点（10.2%），其中 707 个同类，到最近保留点 0.495 m
    这些近点是**视差最大、三角化最稳**的一批，删掉它们等于用更差的输入去拟合。
    调用方用"离最近相机足够近 且 在相机前方"来判定 protect（见 scene_structure）。
    """
    n = len(P)
    if n < k + 5:
        return np.ones(n, dtype=bool)
    rng = np.random.default_rng(seed)
    # 采样算阈值（20k 点以内全量算，超过就采样），再把判据推广到全体
    if n > sample:
        idx = rng.choice(n, size=sample, replace=False)
    else:
        idx = np.arange(n)
    sub = P[idx]
    m = len(sub)
    kk = min(k, m - 1)
    dmean = np.empty(m)
    block = max(1, min(1500, m))
    for i in range(0, m, block):
        b = sub[i : i + block]
        d = np.linalg.norm(b[:, None, :] - sub[None, :, :], axis=2)
        d[np.arange(len(b)), np.arange(i, i + len(b))] = np.inf
        part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        dmean[i : i + len(b)] = d[np.arange(len(b))[:, None], part].mean(axis=1)
    med = float(np.median(dmean))
    mad = float(np.median(np.abs(dmean - med)))
    if mad <= 1e-12:
        thr = float(np.percentile(dmean, 100.0 - std * 2.0))
    else:
        thr = med + std * 1.4826 * mad

    # 把阈值应用到**全体**点（用同样的 kNN 平均距离）
    keep = np.ones(n, dtype=bool)
    step = 4000
    for i in range(0, n, step):
        b = P[i : i + step]
        d = np.linalg.norm(b[:, None, :] - sub[None, :, :], axis=2)
        # 取对 sub 的 k 个最短距离（sub 是采样子集，足够近似真实 kNN）
        part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        dm = d[np.arange(len(b))[:, None], part].mean(axis=1)
        keep[i : i + len(b)] = dm <= thr

    if protect is not None:
        protect = np.asarray(protect, dtype=bool)
        if protect.shape == keep.shape:
            rescued = int((protect & ~keep).sum())
            if rescued:
                print(f"  离群点剔除：按「贴近相机」保护性保留 {rescued} 个点（它们只是天生稀疏，不是野点）")
            keep = keep | protect
    return keep


def camera_centers_and_forwards(images: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """从 COLMAP 位姿算出 (相机中心 N×3, 世界系观察方向 N×3)。

    为什么单独一个函数：房间算法与物体算法**都要**这两个量，
    而"按下标一一对应"这件事必须由同一个函数保证 —— 两处各写一遍迟早错位。

    ⚠️ 符号：COLMAP 相机系是 x 右 / y 下 / **z 前**，所以世界系前向 = Rᵀ·(0,0,1)。
    （交接文档写"-Z 是观察方向"是 Three.js 的约定；实测该代码库用 +Z，
     见 tools/probe_ray_convergence.py 的 forwardCheck：+Z 余弦 +0.888。）
    """
    centers: list[np.ndarray] = []
    forwards: list[np.ndarray] = []
    for im in images:
        try:
            q = np.asarray(im["qvec"], dtype=float)
            t = np.asarray(im["tvec"], dtype=float)
            r_mat = cmio.quat_to_R(q)
        except Exception:  # noqa: BLE001
            continue
        centers.append(cmio.camera_center(q, t))
        forwards.append(r_mat.T @ np.array([0.0, 0.0, 1.0]))
    if not centers:
        return np.zeros((0, 3)), np.zeros((0, 3))
    cam_xyz = np.asarray(centers, dtype=float)
    cam_fwd = np.asarray(forwards, dtype=float)
    cam_fwd = cam_fwd / (np.linalg.norm(cam_fwd, axis=1, keepdims=True) + 1e-12)
    return cam_xyz, cam_fwd


def reference_frame(P: np.ndarray, cam_xyz: np.ndarray, images: list[dict]) -> dict:
    """两套算法共用的**参考系**：尺度、重力「上」、地面坐标系轴、高度分布。

    这个函数是房间算法与物体算法的**公共上游**。把它单独拎出来，
    是为了让"上方向"只有一个来源 —— 交接文档记载过一次严重回归：
    曾经按点云质心翻转 `up`，导致房间盒整个倒扣，而且**自检看不出来**
    （「相机离地 0.8~2.4 m」只校验盒内自洽，倒扣之后照样成立）。
    所以这里算出的 `up_vs_gravity` 是**显式不变量**，会被写进结构文件供自检。
    """
    frame = robust_frame(P)
    scale = float(frame["p95"])
    eps = max(0.015 * scale, 0.01)
    g_down, grav_info = gravity_from_cameras(images)
    n_up = -g_down
    # ---- 重力的第二判据：点云平面投票（2026-11-02 新增）--------------------
    # 起因（实测）：竖屏 + 俯拍洁具 → "每帧图像下方取中位"估出的重力偏离真实竖直
    #   27°~45°，房间盒因此变成"15.5 × 12.8 × 26.4"的竖塔（高 > 长宽）。
    # 做法：候选方向里比"哪个方向上点云有薄而大的平面（地面/天花）、
    #   且相机落在该方向的竖向范围内"，只有候选明显更优（>1.15×）才替换相机方向。
    # 这样 room-real 这种本来正确的情况不会被乱改。
    try:
        from gravity_vote import refine_up_with_planes
    except Exception:  # noqa: BLE001
        refine_up_with_planes = None
    grav_vote = None
    # ⚠️ 默认关闭：实测它会**破坏**本来正确的模型（room-real 5 面墙 → 0 面墙）。
    #    需要时用环境变量 DSH_GRAVITY_VOTE=1 打开做对照。
    _vote_on = os.environ.get("DSH_GRAVITY_VOTE", "").strip() in ("1", "true", "yes", "on")
    if _vote_on and refine_up_with_planes is not None and len(P) >= 200 and len(cam_xyz) > 0:
        n_up, grav_vote = refine_up_with_planes(P, cam_xyz, n_up, scale)
        n_up = n_up / (np.linalg.norm(n_up) + 1e-12)
        g_down = -n_up
        grav_info = {**grav_info, "vote": grav_vote,
                     "source": f"{grav_info.get('source', 'camera')}+{grav_vote['source']}"}
    up_vs_gravity = float(n_up @ (-g_down))  # 恒为 1；作为显式不变量防回归
    u, v = orthonormal_basis(n_up)
    origin0 = np.asarray(frame["center"], dtype=float)
    hh_all = (P - origin0) @ n_up
    cam_h = (cam_xyz - origin0) @ n_up if len(cam_xyz) else np.array([])
    h_cam = float(np.median(cam_h)) if len(cam_h) else float(np.median(hh_all))
    return {
        "frame": frame,
        "scale": scale,
        "eps": eps,
        "g_down": g_down,
        "grav_info": grav_info,
        "n_up": n_up,
        "up_vs_gravity": up_vs_gravity,
        "u": u,
        "v": v,
        "origin0": origin0,
        "hh_all": hh_all,
        "cam_h": cam_h,
        "h_cam": h_cam,
        "binw": max(0.03, 0.01 * scale),
    }


def near_camera_protect_mask(P: np.ndarray, cam_xyz: np.ndarray, cam_fwd: np.ndarray) -> np.ndarray:
    """算出"应当免于被离群判据删掉"的点：**贴近相机、且落在相机前方**的那些。

    为什么是这两条：
      · 「贴近相机」= 到最近相机中心的距离 ≤ 全体中位。这些点在三维里天生稀疏
        （只被少数视角看到），会被相对密度判据误杀，但它们是**视差最大、
        三角化最准**的一批。
      · 「在相机前方」= 沿最近相机的视线方向深度 > 0。野点最常见的形态是
        **被甩到相机背后**（基线退化时三角化会翻到后面去）—— 这类点不该被保护。

    返回长度 = len(P) 的布尔掩码；点太少时返回全 False（不干预）。
    """
    n = len(P)
    if n < 16 or len(cam_xyz) < 2 or len(cam_fwd) != len(cam_xyz):
        return np.zeros(n, dtype=bool)
    # 用分块暴力最近邻（点数 1e4~1e5、相机 ~1e2，内存可控；避免引入 scipy 依赖）
    best_d = np.full(n, np.inf)
    depth = np.zeros(n)
    step = 4000
    for i in range(0, n, step):
        b = P[i : i + step]
        d = np.linalg.norm(b[:, None, :] - cam_xyz[None, :, :], axis=2)  # (m, C)
        idx = np.argmin(d, axis=1)
        best_d[i : i + len(b)] = d[np.arange(len(b)), idx]
        depth[i : i + len(b)] = np.einsum(
            "ij,ij->i", b - cam_xyz[idx], cam_fwd[idx]
        )
    near = best_d <= float(np.median(best_d))
    return near & (depth > 0)



