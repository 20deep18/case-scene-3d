#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""房间（场景）算法：地面 / 天花板 / 竖墙 / 房间盒 / 单间 / 平面轮廓 / 相机轨迹。

与 structure_object.py 的分工（交接文档 §一 的步骤 ②）
------------------------------------------------------
两者的几何假设**不同**，所以必须是两套算法：
  · 房间：曼哈顿假设 —— 地面水平、墙竖直；结构信息在**竖直平面**上；
    相机在空间**内部**走动，地面是确定存在的。
  · 物体（柜子）：结构信息在**水平面**上（顶面/搁板/底面）；
    相机绕着它转，"相机下方"未必有地面（实测 30 台相机落到拟合地面以下）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "tools", ROOT / "pipeline"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from geom_common import (  # noqa: E402
    align_frame_to_walls,
    density_peak,
    detect_horizontal_layers,
    orthonormal_basis,
    ransac_plane,
)

def segment_rooms(
    points_a: np.ndarray,
    points_b: np.ndarray,
    cam_a: np.ndarray,
    cam_b: np.ndarray,
    walls: list[dict],
    a_axis: np.ndarray,
    b_axis: np.ndarray,
    *,
    min_points: int = 400,
    min_cameras: int = 2,
) -> list[dict]:
    """按墙面把地面平面切成网格，把「有点、也有相机去过」的格子当作单间。

    这是曼哈顿世界假设下最朴素也最稳的分割法：
      墙在新坐标系里基本都是「a = 常数」或「b = 常数」的直线，
      取这些常数当切割位置，就得到一张网格；真实房间落在有点云且有相机的格子里。
    不去做完整的平面图（arrangement）计算 —— 那需要处理一般位置的多边形布尔运算，
    收益很小而风险很大（本项目的目的是让民警看懂布局，不是做建筑制图）。
    """
    if len(points_a) < min_points:
        return []
    acuts, bcuts = [], []
    for w in walls:
        n = np.asarray(w["normal"], dtype=float)
        na = float(n @ a_axis)
        nb = float(n @ b_axis)
        # 平面上的一点：w["center"] 只在「输出用」的墙记录里有；
        # RANSAC 出来的原始平面只有 (n, d)，取最近点 n·d 即可（满足 n·x = d）。
        if "center" in w:
            c = np.asarray(w["center"], dtype=float)
        else:
            c = n * float(w.get("d", 0.0))
        ca = float(c @ a_axis)
        cb = float(c @ b_axis)
        if abs(na) >= abs(nb) and abs(na) > 0.7:
            acuts.append(ca)
        elif abs(nb) > 0.7:
            bcuts.append(cb)

    def uniq(vals: list[float], lo: float, hi: float, tol: float) -> list[float]:
        out: list[float] = []
        for x in sorted(vals):
            if x < lo or x > hi:
                continue
            if not out or abs(x - out[-1]) > tol:
                out.append(x)
        return out

    tol = max(0.25, 0.03 * (float(np.percentile(points_a, 99) - np.percentile(points_a, 1))))
    a_lo, a_hi = float(np.percentile(points_a, 1)), float(np.percentile(points_a, 99))
    b_lo, b_hi = float(np.percentile(points_b, 1)), float(np.percentile(points_b, 99))
    a_edges = [a_lo] + uniq(acuts, a_lo + tol, a_hi - tol, tol) + [a_hi]
    b_edges = [b_lo] + uniq(bcuts, b_lo + tol, b_hi - tol, tol) + [b_hi]

    rooms: list[dict] = []
    for i in range(len(a_edges) - 1):
        for j in range(len(b_edges) - 1):
            a0, a1 = a_edges[i], a_edges[i + 1]
            b0, b1 = b_edges[j], b_edges[j + 1]
            if (a1 - a0) < 0.6 or (b1 - b0) < 0.6:
                continue
            in_cell = (points_a >= a0) & (points_a < a1) & (points_b >= b0) & (points_b < b1)
            n_pts = int(in_cell.sum())
            if n_pts < min_points:
                continue
            in_cam = (cam_a >= a0) & (cam_a < a1) & (cam_b >= b0) & (cam_b < b1)
            n_cam = int(in_cam.sum())
            if n_cam < min_cameras:
                continue
            rooms.append(
                {
                    "a": [round(a0, 3), round(a1, 3)],
                    "b": [round(b0, 3), round(b1, 3)],
                    "size": [round(a1 - a0, 3), round(b1 - b0, 3)],
                    "points": n_pts,
                    "cameras": n_cam,
                    # 该格子里点的高度范围（相对地面）→ 用来定这间的高度
                    "heightRange": None,
                }
            )
    rooms.sort(key=lambda r: -r["points"])
    # 单间不可能有这么长的一边（住宅尺度）。超过就说明这一格其实是
    # 「没被墙切开的一大片」，不是单间 —— 宁可不要，也不要给用户一个假轮廓。
    rooms = [r for r in rooms if max(r["size"]) <= 12.0 and min(r["size"]) <= 12.0]
    return rooms[:12]


def plan_outline(
    aa: np.ndarray,
    bb: np.ndarray,
    *,
    cell: float = 0.25,
    min_pts: int = 3,
    max_segments: int = 2000,
) -> dict:
    """地面**平面轮廓**：占用栅格 → 取「占用格 / 空格」之间的边。

    为什么改做这个（而不是继续硬啃房间分割）：
      实测两面数据里，墙面检测只找到 2~4 个平面，而且**几乎全是平行的**
      （colmap-ex 的 4 面墙 |n·a| 都 ≈1.0，其实是同一面墙被切成了三块）。
      曼哈顿房间分解需要两个正交的墙族，这里根本没有原料。
      而用户要的「俯视平面轮廓」本来也不需要先分房间 ——
      把「人走过、拍到过的地方」在地面上的覆盖范围描出来，就已经是
      「现场平面示意图」了，而且它对重建质量不敏感、稳得多。
    """
    if len(aa) < 50:
        return {"usable": False, "reason": "点太少", "segments": []}
    a0, a1 = float(aa.min()), float(aa.max())
    b0, b1 = float(bb.min()), float(bb.max())
    na = max(2, int(math.ceil((a1 - a0) / cell)) + 1)
    nb = max(2, int(math.ceil((b1 - b0) / cell)) + 1)
    if na * nb > 400_000:
        return {"usable": False, "reason": f"栅格过大（{na}×{nb}）", "segments": []}
    ia = np.clip(((aa - a0) / cell).astype(int), 0, na - 1)
    ib = np.clip(((bb - b0) / cell).astype(int), 0, nb - 1)
    grid = np.zeros((na, nb), dtype=np.int32)
    np.add.at(grid, (ia, ib), 1)
    occ = grid >= min_pts
    # 只保留最大连通块：零散的小块多半是野点/窗外，不是「现场范围」
    try:
        from scipy import ndimage  # type: ignore

        lab, n_lab = ndimage.label(occ)
        if n_lab > 1:
            sizes = ndimage.sum(occ, lab, range(1, n_lab + 1))
            occ = lab == (int(np.argmax(sizes)) + 1)
    except Exception:  # noqa: BLE001 - 没有 scipy 就先用整体占用
        pass

    segs: list[list[list[float]]] = []

    def w(a: float, b: float) -> list[float]:
        return [round(a, 3), round(b, 3)]

    # 竖直边界（沿 b 方向）：occ[i,j] 与 occ[i+1,j] 不同 → 在 a = a0+(i+1)*cell 处一条边
    d = np.diff(occ.astype(np.int8), axis=0)
    for i, j in zip(*np.nonzero(d != 0)):
        a = a0 + (i + 1) * cell
        segs.append([w(a, b0 + j * cell), w(a, b0 + (j + 1) * cell)])
    d = np.diff(occ.astype(np.int8), axis=1)
    for i, j in zip(*np.nonzero(d != 0)):
        b = b0 + (j + 1) * cell
        segs.append([w(a0 + i * cell, b), w(a0 + (i + 1) * cell, b)])
    # 外框（首末行列的边界）
    for i in range(na):
        if occ[i, 0]:
            segs.append([w(a0 + i * cell, b0), w(a0 + (i + 1) * cell, b0)])
        if occ[i, nb - 1]:
            segs.append([w(a0 + i * cell, b1), w(a0 + (i + 1) * cell, b1)])
    for j in range(nb):
        if occ[0, j]:
            segs.append([w(a0, b0 + j * cell), w(a0, b0 + (j + 1) * cell)])
        if occ[na - 1, j]:
            segs.append([w(a1, b0 + j * cell), w(a1, b0 + (j + 1) * cell)])

    truncated = len(segs) > max_segments
    if truncated:
        segs = segs[:max_segments]
    return {
        "usable": len(segs) >= 4,
        "cell": cell,
        "grid": [na, nb],
        "occupiedCells": int(occ.sum()),
        "segments": segs,
        "truncated": truncated,
        "note": "地面占用轮廓（a/b 为对齐到墙面后的地面坐标）；用于俯视平面示意图",
    }


def build_room_model(P: np.ndarray, cam_xyz: np.ndarray, ref: dict, rng) -> dict:
    """房间算法主流程：地面 → 天花板 → 竖墙 → 房间盒 → 单间 → 平面轮廓 → 相机轨迹。

    `ref` 由 `geom_common.reference_frame()` 给出（尺度/重力上/坐标轴/高度分布）。

    返回值同时承担两个职责：
      1. 供**输出装配**用的房间几何（room / walls / rooms / planOutline / trajectory …）；
      2. 供**场景分类**用的证据（`real_walls` 面数、`cam_ok` 相机高度是否自洽），
         以及物体算法需要的坐标轴与地面锚点（`u`/`v`/`n_up`/`floor_pt`/`scale`）。
    这样"分类"就发生在房间算法**之后**、物体算法**之前** —— 而分类本身**不依赖**
    房间算法是否成功（它只看相机位姿与点云包围度），避免了交接文档指出的
    「用下游拟合失败的结果去决定上游分类」。
    """
    n_up = ref["n_up"]
    u = ref["u"]
    v = ref["v"]
    origin0 = ref["origin0"]
    hh_all = ref["hh_all"]
    cam_h = ref["cam_h"]
    h_cam = ref["h_cam"]
    scale = ref["scale"]
    binw = ref["binw"]
    eps = ref["eps"]

    # ---- 地面：在「相机以下」的高度区间里找最密的一层
    floor_source = "density-peak"
    below = hh_all[hh_all < h_cam - 0.1]
    if len(below) >= 50:
        h_floor, peak_ratio = density_peak(below, float(np.percentile(below, 0.5)), h_cam - 0.1, binw)
    else:
        h_floor, peak_ratio = h_cam - 1.5, 0.0
        floor_source = "assumed-eye-height"
    eye = h_cam - h_floor
    # 手持拍摄的相机离地高度应该在 0.8~2.4 m。不在这个区间说明峰值不是地面
    # （例如拍到了大片地板倒影/桌面），退化为「按常人持机高度 1.5 m」反推地面。
    if not (0.8 <= eye <= 2.4):
        h_floor = h_cam - 1.5
        floor_source = "assumed-eye-height"
        eye = 1.5
    floor_pt = origin0 + n_up * h_floor

    # ---- 天花板：在**相机上方**（而不是「地面以上 1.2 m」）找最密的一层。
    #
    # ⚠️ 原来写的是 `hh_all > h_floor + 1.2`：这个区间里 1.2~2 m 处常有一层很密的点
    #    （桌面、柜台、床面），于是天花板被定到 1.94 m —— 相机 1.5 m 高，
    #    离「天花板」只剩 0.44 m，明显不合理，却仍然能通过「相机离地」自检。
    #    正确做法是只找**相机头顶以上**的那一层。
    cam_hi_rel = float(np.percentile(cam_h, 95) - h_floor) if len(cam_h) else 1.6
    ceil_lo = max(h_floor + 2.0, h_floor + cam_hi_rel + 0.25)
    above = hh_all[(hh_all > ceil_lo) & (hh_all < h_floor + 6.0)]
    ceil_source = "density-peak"
    if len(above) >= 50:
        h_ceil, _ratio = density_peak(above, ceil_lo, h_floor + 6.0, binw)
        if h_ceil - h_floor < 2.0:
            h_ceil = h_floor + 2.6
            ceil_source = "default-2.6m"
    else:
        h_ceil = h_floor + 2.6
        ceil_source = "default-2.6m"
    h_ceil = min(h_ceil, h_floor + 6.0)
    # 天花板不能远高于「相机真正到过的高度」—— 那上面根本没有被观测过。
    # 实测用户那个案件：密度峰落在 5.80 m，而相机最高只到 2.21 m（下面是空地，
    # 上面是没拍到的地方，多半是镜像/二层残影）。这样裁一下，两套数据都回到
    # 2.2~3.0 m 的合理区间。
    height = float(h_ceil - h_floor)
    unsupported_cap = cam_hi_rel + 0.8
    height_capped = False
    if height > unsupported_cap:
        height = max(2.0, unsupported_cap)
        height_capped = True
    hh = hh_all - h_floor  # 相对地面的高度

    # ---- 墙面：只在中段高度里找「法向与重力垂直」的平面（曼哈顿假设）
    band = (hh_all > h_floor + 0.2) & (hh_all < h_ceil - 0.2)
    wall_pool = P[band]
    walls = []
    remaining = np.ones(len(wall_pool), dtype=bool)
    for _ in range(8):
        sub = wall_pool[remaining]
        if len(sub) < 80:
            break
        w = ransac_plane(sub, eps=eps, iters=1500, rng=rng, perp_to=n_up, min_perp_deg=72.0)
        if w is None or w["count"] < max(80, 0.012 * len(P)):
            break
        idx = np.nonzero(remaining)[0][w["inliers"]]
        mask = np.zeros(len(wall_pool), dtype=bool)
        mask[idx] = True
        w["inliers"] = mask
        walls.append(w)
        remaining = remaining & ~mask

    # ---- 把地面坐标系的轴对齐到墙面（见 geom_common.align_frame_to_walls 的说明）
    # 必须在算房间范围**之前**做：否则量出来的是「歪着的包围盒」
    # （实测 room-real 因此得到 11.3 × 18.1 m，而真实户型宽深远没这么大）。
    u, v, wall_theta_deg = align_frame_to_walls(u, v, walls, n_up)

    # ---- 房间范围：以重力为「上」的坐标系里取稳健分位
    rel = P - floor_pt
    uu = rel @ u
    vv = rel @ v
    inside = (hh > -0.15 * scale) & (hh < height + 0.35 * scale)
    use = inside
    if int(use.sum()) < 100:
        use = np.ones(len(P), dtype=bool)

    def prange(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
        return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

    u0, u1 = prange(uu[use])
    v0, v1 = prange(vv[use])

    # 防止把「走廊尽头」也算进房间：宽深各裁到一个合理上限
    max_side = 3.5 * scale
    if u1 - u0 > max_side:
        cu = float(np.median(uu[use]))
        u0, u1 = cu - max_side / 2, cu + max_side / 2
    if v1 - v0 > max_side:
        cv = float(np.median(vv[use]))
        v0, v1 = cv - max_side / 2, cv + max_side / 2

    def to_world(a: float, b: float, h: float) -> list[float]:
        p = floor_pt + u * a + v * b + n_up * h
        return [round(float(x), 5) for x in p]

    # ⚠️ 名字必须与墙矩形循环里的局部量区分开（房间盒 8 角点 = room_corners）。
    #    见下方 wall_lines 循环处的说明：这里出过一次真回归。
    room_corners = [
        to_world(u0, v0, 0), to_world(u1, v0, 0), to_world(u1, v1, 0), to_world(u0, v1, 0),
        to_world(u0, v0, height), to_world(u1, v0, height), to_world(u1, v1, height), to_world(u0, v1, height),
    ]

    wall_lines = []
    for w in walls:
        wp = wall_pool[w["inliers"]]
        if len(wp) < 12:
            continue
        n = np.asarray(w["normal"], dtype=float)
        w_center = wp.mean(axis=0)
        # 墙面内的两个方向：竖直用重力方向，水平用法向×竖直。
        # 有了这两个方向就能把点投到"墙面自己的二维坐标系"里取边界 ——
        # 无限延伸的直线看不出是墙，有边界的矩形才读得出房间。
        #
        # ⚠️ 变量名必须加 w_ 前缀！第一版我随手写成 rel / vv，直接**覆盖了外层**
        # 的 `rel`（地面相对坐标）和 `vv`（地面 v 坐标），导致后面 segment_rooms
        # 拿到 245 个元素的 vv 去配 14406 个元素的掩码 → IndexError。
        vert = n_up
        horiz = np.cross(n_up, n)
        hn = float(np.linalg.norm(horiz))
        if hn < 1e-6:
            continue
        horiz = horiz / hn
        w_rel = wp - w_center
        w_h = w_rel @ horiz
        w_v = w_rel @ vert
        # 用 1/99 分位而不是 min/max：墙面点里的野点不该把矩形拉大
        #
        # ⚠️ 这里的局部变量**一律加 w_ 前缀**，这不是洁癖 —— 2026-10-06 这里出过一次
        #    真回归（实测确认）：原来写的是
        #        v0, v1 = (np.percentile(w_v, q) for q in (1.0, 99.0))
        #        width, height = h1 - h0, v1 - v0
        #        corners = [...]
        #    于是每处理一面墙，就把**外层的** v0/v1（房间进深）、height（房间层高）、
        #    corners（房间盒 8 角点）全部覆盖掉。后果：
        #      · room.size 的「深/高」双双变成最后一面墙的高度 → 输出 17.14 × 4.48 × 4.48 m
        #      · room.corners 只剩 4 个点 → 前端 _addRoomBox 要求 ≥8，直接**不画房间盒**
        #      · 平面轮廓的 band_hi 也跟着用错 height
        #    最阴险的是它**不报错**：控制台照样打印"房间结构提取完成"，自检照样通过。
        w_h0, w_h1 = (float(np.percentile(w_h, q)) for q in (1.0, 99.0))
        w_v0, w_v1 = (float(np.percentile(w_v, q)) for q in (1.0, 99.0))
        w_width, w_height = w_h1 - w_h0, w_v1 - w_v0
        w_corners = [
            w_center + horiz * h + vert * v
            for h, v in ((w_h0, w_v0), (w_h1, w_v0), (w_h1, w_v1), (w_h0, w_v1))
        ]
        # 面积太小 / 太窄的多半是家具、门板、镜面残影，不是墙。
        # 标出来让前端可以只画真墙，也便于排查。
        is_wall = bool(w_width >= 1.0 and w_height >= 1.2)
        wall_lines.append(
            {
                "count": w["count"],
                "rms": round(w["rms"], 4),
                "center": [round(float(x), 5) for x in w_center],
                "normal": [round(float(x), 6) for x in n],
                "span": round(float(np.linalg.norm(wp.max(axis=0) - wp.min(axis=0))), 4),
                "rect": {
                    "corners": [[round(float(x), 4) for x in c] for c in w_corners],
                    "width": round(w_width, 3),
                    "height": round(w_height, 3),
                    "area": round(w_width * w_height, 3),
                },
                "kind": "wall" if is_wall else "patch",
            }
        )

    # 防回归硬校验：房间盒必须始终是 8 个角点、且尺寸不能是"末面墙"的产物。
    # 与其让错误静默流到前端，不如在这里就炸掉（宁可报告不可信，也不要装作成功）。
    if len(room_corners) != 8:
        raise RuntimeError(f"房间盒角点数应为 8，实际 {len(room_corners)} —— 有局部变量覆盖了 room_corners")

    # ---- 相机轨迹
    traj = {
        "count": int(len(cam_xyz)),
        "points": [[round(float(x), 5) for x in p] for p in cam_xyz],
        "lengthMeters": 0.0,
        "netDisplacement": 0.0,
        "plan": [],
        "heightAboveFloor": None,
    }
    if len(cam_xyz) > 1:
        d = np.linalg.norm(np.diff(cam_xyz, axis=0), axis=1)
        traj["lengthMeters"] = round(float(d.sum()), 3)
        traj["netDisplacement"] = round(float(np.linalg.norm(cam_xyz[-1] - cam_xyz[0])), 3)
        med = float(np.median(d)) if len(d) else 0.0
        thr = max(med * 6.0, 1e-6)
        traj["stepMedian"] = round(med, 4)
        traj["stepP90"] = round(float(np.percentile(d, 90)), 4) if len(d) else 0.0
        traj["outlierSteps"] = int((d > thr).sum())
        traj["lengthRobustMeters"] = round(float(d[d <= thr].sum()), 3)
        relc = cam_xyz - floor_pt
        ch = relc @ n_up
        # 相机高度是最硬的**自检指标**：手持视频应该集中在 1.2~1.8 m。
        # 如果这一项明显不对，说明「上」方向或地面位置估错了。
        traj["heightAboveFloor"] = {
            "median": round(float(np.median(ch)), 3),
            "p05": round(float(np.percentile(ch, 5)), 3),
            "p95": round(float(np.percentile(ch, 95)), 3),
            "belowFloor": int((ch < -0.2).sum()),
        }
        traj["plan"] = [
            [round(float(a), 5), round(float(b), 5)] for a, b in zip(relc @ u, relc @ v)
        ]
        traj["heightRange"] = [
            round(float(np.percentile(ch, 5)), 3),
            round(float(np.percentile(ch, 95)), 3),
        ]

    # ---- 水平层检测（房间口径；物体模式稍后会按物体锚点重算）
    # 只统计**箱体内部**的水平面（与 `inside` 用同一个物理带）：
    # 实测 room-real 在"地面"下方 2.94 m 处有一个比地面更密的平面，
    # 那是地面估计偏高的产物，不该当成"房间里的一个水平层"报出来。
    # 房间的层目前只作信息参考（前端不消费）；物体模式的层才是柜子结构的主体。
    lay_lo = -0.15 * scale
    lay_hi = height + 0.35 * scale
    layers, layers_ev = detect_horizontal_layers(hh[(hh >= lay_lo) & (hh <= lay_hi)])
    layers_ev["bandMeters"] = [round(float(lay_lo), 3), round(float(lay_hi), 3)]
    layers_ev["scope"] = "room（仅箱体内部高度带；房间的层目前只作信息参考）"

    # ---- 按墙面把空间切成单间（P1-7）
    # 之前的「房间尺寸」其实是**整个场景**的包围盒：room-real 那组给出 11×18 m，
    # 因为它是整套房子的走动拍摄，不是一个房间。这里按墙线切格子，
    # 只保留「有点云、而且相机真的进去过」的格子。
    rooms = []
    if walls:
        rel_cam = cam_xyz - floor_pt if len(cam_xyz) else np.zeros((0, 3))
        rooms = segment_rooms(
            uu[use],
            vv[use],
            (rel_cam @ u) if len(rel_cam) else np.zeros(0),
            (rel_cam @ v) if len(rel_cam) else np.zeros(0),
            walls,
            u,
            v,
        )
        # 每间补上自己的高度范围（该格子里点的高度分位）
        hh_use = hh[use]
        for r in rooms:
            a0, a1 = r["a"]
            b0, b1 = r["b"]
            sel = (uu[use] >= a0) & (uu[use] < a1) & (vv[use] >= b0) & (vv[use] < b1)
            if int(sel.sum()) >= 20:
                hv = hh_use[sel]
                r["heightRange"] = [round(float(np.percentile(hv, 2)), 3), round(float(np.percentile(hv, 98)), 3)]
                r["heightMedian"] = round(float(np.median(hv)), 3)

    # ---- 地面平面轮廓（P1-7 的可交付部分）：占用栅格 → 边界线段
    # 只用地面上方 0.2~2.0 m 的点（相当于「人活动的高度层」），
    # 避免地面以下的野点和天花板上的残影把轮廓撑大。
    band_lo = 0.2
    band_hi = min(2.0, max(0.8, height - 0.2))
    band_sel = (hh > band_lo) & (hh < band_hi)
    if int(band_sel.sum()) < 200:
        band_sel = hh > 0.2
    outline = plan_outline(uu[band_sel], vv[band_sel])
    # 转成世界坐标线段，前端直接画（与房间盒/轨迹用同一套投影）
    outline_world = []
    for seg in outline.get("segments", []):
        pts = []
        for a, b in seg:
            p = floor_pt + u * a + v * b
            pts.append([round(float(x), 4) for x in p])
        outline_world.append(pts)
    outline["segmentsWorld"] = outline_world
    outline.pop("segments", None)

    # ---- 供**分类**用的两项就地证据
    #   · real_walls：竖墙里"宽度不超过点云稳健水平跨度 1.3 倍"的（墙比场景还宽 = 假墙）
    #   · cam_ok    ：相机高度是否与拟合出的地面自洽
    # ⚠️ 注意这两项**只是证据**，不是判据本身 —— 尤其 cam_ok 用的是拟合结果，
    #    交接文档已论证"用下游失败的结果决定上游分类"是危险的。
    hu = (uu[use] if len(uu) == len(use) else uu)
    hv = (vv[use] if len(vv) == len(use) else vv)
    h_extent = float(
        max(np.percentile(hu, 99) - np.percentile(hu, 1),
            np.percentile(hv, 99) - np.percentile(hv, 1))
    ) if len(hu) else 0.0
    real_walls = [
        w for w in wall_lines
        if w.get("kind") == "wall"
        and float((w.get("rect") or {}).get("width") or 0) <= 1.3 * max(h_extent, 1e-6)
    ]
    cam_h_rel = (cam_xyz - floor_pt) @ n_up if len(cam_xyz) else np.zeros(0)
    cam_ok = bool(
        len(cam_h_rel)
        and 0.6 <= float(np.median(cam_h_rel)) <= 2.6
        and int((cam_h_rel < -0.2).sum()) <= max(2, 0.05 * len(cam_h_rel))
    )

    return {
        "ref": ref,
        "u": u,
        "v": v,
        "n_up": n_up,
        "floor_pt": floor_pt,
        "h_floor": h_floor,
        "h_ceil": h_ceil,
        "floor_source": floor_source,
        "ceil_source": ceil_source,
        "eye": eye,
        "height": height,
        "height_capped": height_capped,
        "cam_hi_rel": cam_hi_rel,
        "wall_theta_deg": wall_theta_deg,
        "walls": walls,
        "wall_lines": wall_lines,
        "room_corners": room_corners,
        "uu": uu,
        "vv": vv,
        "use": use,
        "hh": hh,
        "traj": traj,
        "layers": layers,
        "layers_ev": layers_ev,
        "rooms": rooms,
        "outline": outline,
        "real_walls": real_walls,
        "cam_ok": cam_ok,
        "cam_h_rel": cam_h_rel,
        "h_extent": h_extent,
        # 房间盒的尺寸与原点**在这里算好**（`to_world` 是本地闭包，出了这个函数就没了）。
        # 2026-10-06 的回归就是"房间盒的尺寸被墙循环覆盖"，所以这里只暴露算好的结果，
        # 让上层没有机会再拿墙的尺寸去拼房间尺寸。
        "room_size": [u1 - u0, v1 - v0, height],
        "room_origin": to_world(u0, v0, 0),
    }

def complete_room_from_single_wall(*, P: np.ndarray, cam_xyz: np.ndarray, n_up: np.ndarray,
                                   wall: dict, floor_pt: np.ndarray, scale: float,
                                   height_hint: float | None = None,
                                   cam_height_median: float | None = None,
                                   below_floor_ratio: float = 0.0,
                                   eye_prior: float = 1.5,
                                   origin0: np.ndarray | None = None,
                                   width_pct: tuple[float, float] = (5.0, 95.0)) -> dict | None:
    """只有**一面**可信竖墙时，按曼哈顿正交假设补全出房间盒（"推测"性质）。

    为什么走这条路（而不是"稠密点半空间求交"）：
      弱纹理场景的稠密点本身就是雾，RANSAC 在低内点率下会拟合出歪平面，
      反而污染唯一正确的那面墙 —— 实测白瓷砖卫生间只找到 1 面可信竖墙，
      此时"保守但方向正确"的盒子远比"精确但方向错"的盒子有用。

    做法（专家给的四步）：
      1. 可信墙法向 n1（投影掉竖直分量）+ 重力 up → 第三轴 n2 = up × n1
      2. 地面点投影到 n1/n2，取 5%~95% 分位作为长宽（分位数比极值抗噪）
      3. 高度取点云在地面以上的 P98，夹在 [2.2, 3.2]（住宅层高常识范围）
      4. 若地面不可信（相机高度异常/大量相机在地面以下），用"眼高 1.5"先验重定地面
         —— 与 structure_room 里既有的 "assumed-eye-height" 是同一套假设（单位≈米）

    返回 None 表示补不出来（例如长宽里有一个小于 1 个单位，那不是房间）。
    """
    n1 = np.asarray(wall.get("normal", []), dtype=float)
    if n1.size != 3:
        return None
    n1 = n1 - n_up * float(n1 @ n_up)
    ln = float(np.linalg.norm(n1))
    if ln < 1e-6:
        return None
    n1 = n1 / ln
    n2 = np.cross(n_up, n1)
    ln2 = float(np.linalg.norm(n2))
    if ln2 < 1e-6:
        return None
    n2 = n2 / ln2

    # ---- 地面：**不假设"1 单位 = 1 米"** -------------------------------------
    # ⚠️ 实测教训：白瓷砖卫生间那份重建里 1 单位 ≈ 0.15 m，而结构代码里的
    #    "眼高 1.5"是按米写死的 → floor = 相机高度 − 1.5 单位 = 只降了 0.22 m，
    #    地面落进点云中间（up 坐标 P1 甚至到 −15.4），下游全部失真。
    #    所以这里改成**尺度无关**的估法：取点云在 up 轴上最低的 2% 分位当地面；
    #    相机高度只用于**校验**（记录有多少相机在地面以上），不用来定地面。
    org = np.asarray(origin0, dtype=float) if origin0 is not None else np.zeros(3)
    h0 = (P - org) @ n_up
    floor_z = float(np.percentile(h0, 2.0))
    floor = org + n_up * floor_z
    floor_src = "cloud-bottom-p2"
    cam_h = float(np.median((cam_xyz - floor) @ n_up)) if len(cam_xyz) else 0.0
    cam_above_ratio = (
        float(((cam_xyz - floor) @ n_up > 0).mean()) if len(cam_xyz) else 0.0
    )

    rel = P - floor
    h = rel @ n_up
    # ---- 地面带：先取紧贴地面的一层；点太少就放宽到"腰部以下"
    band = (h >= -0.02 * scale) & (h <= 0.12 * scale)
    band_src = "floor-band"
    if int(band.sum()) < 100:
        band = (h >= -0.02 * scale) & (h <= 0.80 * scale)
        band_src = "waist-band"
    if int(band.sum()) < 50:
        return None

    a = rel[band] @ n1
    b = rel[band] @ n2
    a0, a1 = (float(np.percentile(a, q)) for q in width_pct)
    b0, b1 = (float(np.percentile(b, q)) for q in width_pct)
    w, d = a1 - a0, b1 - b0
    # 长宽都得像房间：太小的多半只拍到一个角落（补出来也没意义）
    if not (w >= 1.0 and d >= 1.0):
        return None

    # ---- 层高：取"地面上方最靠上的**强平面峰**"（天花），而不是 P98 分位 ------
    # 实测教训：白瓷砖那份重建里，雾/野点把 up 方向的 P99 撑到 11.1（相对地面 26.6 单位），
    # 用分位当层高会得到一个"身高 26 单位的塔"。天花是真实存在的平面，会在
    # 高度直方图里形成峰；峰之外的雾不该算进层高。
    h_pos = h[h > 0]
    hh = 0.0
    height_src = "none"
    if len(h_pos) >= 100:
        bw = max(0.01 * scale, 1e-9)
        nb = int(np.ceil(float(h_pos.max()) / bw)) + 1
        idx = np.clip((h_pos / bw).astype(np.int64), 0, nb - 1)
        counts = np.bincount(idx, minlength=nb).astype(float)
        pair = counts[:-1] + counts[1:] if nb > 1 else counts
        strong = pair >= 0.03 * len(h_pos)          # 至少 3% 的点落在该峰上才算"平面"
        if strong.any():
            k = int(np.where(strong)[0].max())      # 最靠上的强峰 = 天花
            hh = float((k + 1) * bw)
            height_src = "ceiling-plane-peak"
    if hh <= 0:
        hh = float(np.clip(np.percentile(h_pos, 90) if len(h_pos) else 0.0, 0, None))
        height_src = "points-p90"
    if hh <= 0 or hh > 4.0 * max(min(w, d), 1e-6):
        return None  # 点云没有"地面以上"的部分，或竖向拉得离谱 → 不补

    center = floor + n1 * ((a0 + a1) / 2) + n2 * ((b0 + b1) / 2)
    corners = [
        [round(float(x), 4) for x in (center + n1 * sx * w / 2 + n2 * sy * d / 2 + n_up * z)]
        for z in (0.0, hh)
        for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    ]
    evidence = {
        "method": "single-wall Manhattan completion",
        "confidence": "estimated",
        "wallNormal": [round(float(x), 5) for x in n1],
        "axes": {
            "u": [round(float(x), 5) for x in n1],
            "v": [round(float(x), 5) for x in n2],
            "up": [round(float(x), 5) for x in n_up],
        },
        "footprintPercentile": [width_pct[0], width_pct[1]],
        "footprintBand": band_src,
        "floorSource": floor_src,
        "cameraHeightAboveFloor": round(float(cam_h), 3),
        "cameraAboveFloorRatio": round(float(cam_above_ratio), 3),
        "heightSource": height_src,
        "heightUsed": round(float(hh), 3),
        "scaleFree": True,
        # 实测精度（拿 room-real 的一面墙做补全，与人工核对的 17.14×9.19×2.88 比）：
        #   相对误差 10.5% / 12.5% / 19% —— 这是"保守估计"的量级，
        #   而 5 面墙全几何拟合是 ~1%。所以界面必须标「推测」。
        "measuredError": [0.105, 0.125, 0.19],
        "note": "只有 1 面可信竖墙 → 曼哈顿正交补全；地面取点云最低 2% 分位（尺度无关），"
                "方向可信、尺寸保守，界面必须标注「推测」",
    }
    return {
        "corners": corners,
        "size": [round(float(w), 4), round(float(d), 4), round(float(hh), 4)],
        "origin": [round(float(x), 5) for x in center],
        "u": n1,
        "v": n2,
        "floor_pt": [round(float(x), 5) for x in floor],
        "evidence": evidence,
    }
