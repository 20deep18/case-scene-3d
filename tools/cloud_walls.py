#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从**稠密点云**拟合墙面，求出房间盒（替代"分位数 + PCA 轴"的粗估）。

为什么要换：`cloud_to_structure.py` 现在用"点云 5%~95% 分位 + PCA 轴"拼盒，
- 对离群点敏感（雾点会把盒子撑大）
- PCA 轴不一定与墙平行，盒子会歪
实测宿舍（VGGT 点云）出 1.27×0.58×1.21，明显偏小，导致线稿里检测盒"撑满"。

做法（与 `scene_structure.py` 对稀疏点的做法同源，但改用稠密点）：
1. **估重力方向**：候选 = 点云 PCA 三轴；打分 = "该轴上有几个薄而大的水平峰"（地面+天花至少 2 个）
   —— 房间一定同时有地/顶或至少一个大水平面，这是比"相机朝向"更硬的证据
2. **拟合竖直墙**：RANSAC 平面，要求法向 ⟂ up（竖直面），内点足够、面积够大
3. **曼哈顿系**：取最大那面墙的水平法向当 u，v = up × u
4. **房间盒**：每个方向（±u、±v）取"最外侧的墙"的偏移；没有墙的方向退回分位估计
5. 高度：地面取"最低的强水平峰"，天花取"最高的强水平峰"；缺则用分位兜底

用法：
  <python> tools/cloud_walls.py --cloud output/dorm/auto/scene.ply --out output/dorm/auto/structure-walls.json
  <python> tools/cloud_walls.py --cloud output/furniture/demo-room-noscale.ply --out /tmp/st.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_ply(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    end = raw.find(b"end_header")
    body = raw.find(b"\n", end) + 1
    head = raw[:end].decode("ascii", "replace")
    n = int([l for l in head.splitlines() if l.startswith("element vertex")][0].split()[-1])
    names = [l.split()[-1] for l in head.splitlines() if l.startswith("property")]
    dt = np.dtype([(nm, "<f4") for nm in names])
    buf = np.frombuffer(raw, dtype=dt, count=n, offset=body)
    return np.stack([buf["x"], buf["y"], buf["z"]], axis=1).astype(np.float64)


def plane_peaks(v: np.ndarray, binw: float) -> list[dict]:
    """沿某个轴统计"薄而大的平面峰"（地面/天花/墙都会形成峰）。"""
    if len(v) < 100:
        return []
    lo, hi = float(v.min()), float(v.max())
    nb = int(np.ceil((hi - lo) / max(binw, 1e-9)))
    if nb < 2:
        return []
    idx = np.clip(((v - lo) / binw).astype(np.int64), 0, nb - 1)
    counts = np.bincount(idx, minlength=nb).astype(float)
    pair = counts[:-1] + counts[1:] if nb > 1 else counts
    total = float(len(v))
    peaks = []
    for k in range(len(pair)):
        frac = pair[k] / total
        if frac < 0.05:                       # 至少 5% 的点落在该峰上才算"平面"
            continue
        if 0 < k < len(pair) - 1 and pair[k] < pair[k - 1] and pair[k] < pair[k + 1]:
            continue                          # 只要局部极大
        peaks.append({"pos": (k + 1) * binw + lo, "frac": frac})
    peaks.sort(key=lambda p: -p["frac"])
    return peaks


def estimate_up(P: np.ndarray, binw: float) -> tuple[np.ndarray, dict]:
    """候选方向里挑"有 ≥2 个强水平峰"的那个当竖直轴。"""
    d = P - P.mean(axis=0)
    try:
        w, V = np.linalg.eigh(d.T @ d / max(len(d), 1))
    except np.linalg.LinAlgError:
        return np.array([0.0, 0.0, 1.0]), {"source": "fallback"}
    cands = []
    for k in range(3):
        ax = V[:, k]
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        for sign in (1.0, -1.0):
            a = ax * sign
            peaks = plane_peaks(d @ a, binw)
            strong = [p for p in peaks if p["frac"] >= 0.08]
            cands.append({
                "axis": a, "label": f"pca{k}{'' if sign > 0 else '-'}",
                "nStrong": len(strong),
                "topFrac": float(strong[0]["frac"]) if strong else 0.0,
                "peaks": [round(p["frac"], 3) for p in strong[:4]],
            })
    # 优先：强峰数 ≥2 且 topFrac 最大
    cands.sort(key=lambda c: (c["nStrong"] >= 2, c["nStrong"], c["topFrac"]), reverse=True)
    best = cands[0]
    return best["axis"], {"source": "plane-peak-vote", "chosen": best["label"],
                          "nStrong": best["nStrong"], "topFrac": round(best["topFrac"], 3),
                          "candidates": [{k: c[k] for k in ("label", "nStrong", "topFrac", "peaks")} for c in cands[:4]]}


def ransac_vertical_planes(P: np.ndarray, up: np.ndarray, *, band: float, trials: int = 400,
                           rng=None) -> list[dict]:
    """找出"竖直"（法向 ⟂ up）的大平面。返回按内点数排序的墙列表。"""
    rng = rng or np.random.default_rng(7)
    remain = np.ones(len(P), bool)
    walls: list[dict] = []
    for _ in range(10):
        idx = np.nonzero(remain)[0]
        if len(idx) < 200:
            break
        sub = P[idx]
        best = None
        for _t in range(trials):
            pick = rng.choice(len(sub), 3, replace=False)
            a, b, c = sub[pick]
            nrm = np.cross(b - a, c - a)
            ln = np.linalg.norm(nrm)
            if ln < 1e-9:
                continue
            nrm = nrm / ln
            if abs(float(nrm @ up)) > 0.25:      # 必须接近竖直（法向 ⟂ up）
                continue
            dd = -float(nrm @ a)
            inl = np.abs(sub @ nrm + dd) < band
            k = int(inl.sum())
            if best is None or k > best[0]:
                best = (k, nrm, dd, inl)
        if best is None or best[0] < max(200, int(0.03 * len(P))):
            break
        k, nrm, dd, inl = best
        # 让法向朝外（指向点云质心之外）：取墙两侧点的质心，法向朝"空"的一侧
        sel = idx[inl]
        walls.append({"n": nrm, "d": dd, "inliers": int(k),
                      "centroid": P[sel].mean(axis=0).tolist()})
        remain[sel] = False
    walls.sort(key=lambda w: -w["inliers"])
    return walls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--band-ratio", type=float, default=0.008, help="平面带宽 = 点云 P95 半径 × 该比例")
    ap.add_argument("--pct", type=float, default=5.0, help="没有墙的方向用该分位兜底")
    ap.add_argument("--up-from", default=None,
                    help="从已有的 structure.json 里取重力方向（稀疏路线用相机朝向算过，更可靠）。\n"
                         "实测：稠密点云用平面投票估重力会偏 25°，房间高度从 2.4 变成 0.98")
    ap.add_argument("--poses", default=None,
                    help="VGGT/COLMAP 位姿 json：有就用**相机朝向**定重力（比平面投票可靠）。\n"
                         "实测宿舍点云的 PCA 轴偏了 38°，平面投票反而给出 10 个假峰 → 0 面墙")
    args = ap.parse_args()

    P = load_ply(Path(args.cloud))
    P = P[np.isfinite(P).all(axis=1)]
    if len(P) < 200:
        print("点太少")
        return 1
    c = P.mean(axis=0)
    dist = np.linalg.norm(P - c, axis=1)
    radius = float(np.percentile(dist, 95))
    band = max(radius * args.band_ratio, 1e-4)
    print(f"点 {len(P)}  P95 半径 {radius:.3f}  平面带宽 {band:.4f}")

    d = P - c
    binw = max(band * 2.0, 1e-5)

    # 重力：优先用**相机朝向**（相机 -Y 轴在世界系的方向，OpenCV 约定 Y 向下）。
    # 为什么优先：点云 PCA 轴不一定竖直（宿舍实测偏 38°），而"平面投票"会被
    # 斜轴制造的一堆假峰骗过去（它给了 10 个强峰）。相机是"人站着拍的"，更可靠。
    up, up_ev = None, None
    # 优先级：--up-from（复用稀疏路线的相机重力）> --poses > 平面投票
    if args.up_from:
        src = Path(args.up_from)
        if src.exists():
            sj = json.loads(src.read_text(encoding="utf-8"))
            u0 = (sj.get("frame") or {}).get("up")
            if u0 and len(u0) == 3:
                uu = np.asarray(u0, float)
                n = np.linalg.norm(uu)
                if n > 1e-6:
                    up = uu / n
                    up_ev = {"source": "structure.json:frame.up", "from": str(src)}
    if up is None and args.poses:
        pose_path = Path(args.poses)
        if pose_path.exists():
            pj = json.loads(pose_path.read_text(encoding="utf-8"))
            E = np.asarray(pj.get("extrinsic", []), float)
            if E.ndim == 3 and len(E):
                ups = []
                for Ei in E:
                    R = Ei[:3, :3]                       # world→camera
                    ups.append(R.T @ np.array([0.0, -1.0, 0.0]))   # 相机 -Y 在世界系
                u_mean = np.mean(ups, axis=0)
                if np.linalg.norm(u_mean) > 1e-6:
                    up = u_mean / np.linalg.norm(u_mean)
                    # 朝向统一：让 up 与点云"竖直跨度最大"的那侧一致（保证 z 向上）
                    if (d @ up).mean() > 0 and abs(float(up[2])) < 0.9:
                        pass
                    up_ev = {"source": "camera-up", "frames": int(len(E)),
                             "spread": round(float(np.linalg.norm(np.std(ups, axis=0))), 4)}
    if up is None:
        up, up_ev = estimate_up(P, binw)
    print(f"重力（{up_ev.get('source')}）：{np.round(up, 3)}")

    walls = ransac_vertical_planes(P, up, band=band, rng=np.random.default_rng(11))
    print(f"竖直墙面：{len(walls)} 面（内点 {[w['inliers'] for w in walls[:5]]}）")

    # 曼哈顿系：**按角度聚类墙法向**，取支持最多的那个方向当 u。
    # ⚠️ 不能直接拿第一面墙（内点最多那面）当 u：实测宿舍 8 面墙里有 7 面被归到
    #    同一侧，说明 u 轴选歪了（真实房间的两组墙面应当大致垂直、各自成组）。
    def manhattan_u(ws: list[dict]) -> np.ndarray | None:
        if not ws:
            return None
        n0 = ws[0]["n"] - up * float(ws[0]["n"] @ up)
        if np.linalg.norm(n0) < 1e-6:
            return None
        n0 = n0 / np.linalg.norm(n0)
        e1 = n0
        e2 = np.cross(up, e1)
        angs, ws2 = [], []
        for w in ws:
            n = w["n"] - up * float(w["n"] @ up)
            ln = np.linalg.norm(n)
            if ln < 1e-6:
                continue
            n = n / ln
            angs.append(np.arctan2(float(n @ e2), float(n @ e1)))
            ws2.append(w)
        if not angs:
            return None
        angs = np.array(angs)
        # 门向量：4θ 的圆周平均（θ 与 θ+90° 等价）
        s = float(np.mean(np.sin(4 * angs)))
        c = float(np.mean(np.cos(4 * angs)))
        if abs(s) < 1e-9 and abs(c) < 1e-9:
            return e1
        theta = float(np.arctan2(s, c)) / 4.0
        # 两个候选方向（相差 90°），取"落在 15° 内且内点合计更多"的那个
        best_u, best_w = None, -1.0
        for cand in (theta, theta + np.pi / 2):
            uu = np.cos(cand) * e1 + np.sin(cand) * e2
            wsum = 0.0
            for a, w in zip(angs, ws2):
                d_ang = abs(((a - cand + np.pi / 2) % np.pi) - np.pi / 2)
                if d_ang < np.deg2rad(15):
                    wsum += float(w["inliers"])
            if wsum > best_w:
                best_u, best_w = uu, wsum
        print(f"  曼哈顿聚类：{len(angs)} 面墙 → u 支持内点 {int(best_w)}")
        return best_u / (np.linalg.norm(best_u) + 1e-12)

    u_m = manhattan_u(walls)
    if u_m is not None:
        u = u_m
    elif walls:
        u = walls[0]["n"] - up * float(walls[0]["n"] @ up)
        u /= np.linalg.norm(u) + 1e-12
    else:
        # 没有墙时用 PCA 第一主方向
        w2, v2 = np.linalg.eigh(d.T @ d / len(d))
        u = v2[:, int(np.argmax(w2))]
        u = u - up * float(u @ up)
        u /= np.linalg.norm(u) + 1e-12
    v = np.cross(up, u)

    # 投影到 (u, v, up)
    a = d @ u
    b = d @ v
    h = d @ up

    # 墙面给出的边界。
    # ⚠️ 不能按法向符号分"+u/-u"：RANSAC 的法向符号是**任意的**（同一面墙两次可能
    #    给出相反方向），实测这样会让 lo/hi 倒挂 → 宽度为负 → 只能全部退回分位。
    #    正确做法：把墙换算成它在轴上的坐标，再按"在点云中位数哪一侧"归到 lo 或 hi。
    lo_u_c, hi_u_c, lo_v_c, hi_v_c = [], [], [], []
    a_med = float(np.median(a))
    b_med = float(np.median(b))
    a_lim = max(abs(float(np.percentile(a, 1))), abs(float(np.percentile(a, 99)))) * 1.5 + 1e-6
    b_lim = max(abs(float(np.percentile(b, 1))), abs(float(np.percentile(b, 99)))) * 1.5 + 1e-6
    for w in walls:
        n = w["n"] - up * float(w["n"] @ up)
        ln = np.linalg.norm(n)
        if ln < 1e-6:
            continue
        n = n / ln
        cu, cv = float(n @ u), float(n @ v)
        if abs(cu) > abs(cv):
            coord = -w["d"] / cu
            if abs(coord) > a_lim:
                continue                        # 明显是野点撑出来的"墙"
            (lo_u_c if coord < a_med else hi_u_c).append(coord)
        else:
            coord = -w["d"] / cv
            if abs(coord) > b_lim:
                continue
            (lo_v_c if coord < b_med else hi_v_c).append(coord)

    # 兜底：没有墙的方向用分位
    a_lo, a_hi = (float(np.percentile(a, q)) for q in (args.pct, 100 - args.pct))
    b_lo, b_hi = (float(np.percentile(b, q)) for q in (args.pct, 100 - args.pct))
    # 有墙用"最外侧那面墙"，但**不能越过分位边界太多**（防止一面误检墙把盒子撑大）
    slack = max(radius * 0.05, 1e-3)
    lo_u = min(lo_u_c) if lo_u_c else a_lo
    hi_u = max(hi_u_c) if hi_u_c else a_hi
    lo_v = min(lo_v_c) if lo_v_c else b_lo
    hi_v = max(hi_v_c) if hi_v_c else b_hi
    lo_u = max(lo_u, a_lo - slack)
    hi_u = min(hi_u, a_hi + slack)
    lo_v = max(lo_v, b_lo - slack)
    hi_v = min(hi_v, b_hi + slack)
    if hi_u - lo_u < 0.3 or hi_v - lo_v < 0.3:      # 墙给出的范围不合理 → 退回分位
        print("  墙面范围不合理，退回分位估计")
        lo_u, hi_u, lo_v, hi_v = a_lo, a_hi, b_lo, b_hi
    print(f"  墙面边界：u [{lo_u:.2f}, {hi_u:.2f}]（{len(lo_u_c)}/{len(hi_u_c)} 面）  "
          f"v [{lo_v:.2f}, {hi_v:.2f}]（{len(lo_v_c)}/{len(hi_v_c)} 面）")

    # 高度：地面 = 最低的强水平峰；天花 = 最高的强水平峰
    hpeaks = plane_peaks(h, binw)
    strong = [p for p in hpeaks if p["frac"] >= 0.05]
    h_lo = float(h.min())
    h_hi = float(h.max())
    if strong:
        h_lo = float(min(p["pos"] for p in strong))
        h_hi = float(max(p["pos"] for p in strong))
        height_src = "plane-peaks"
    else:
        h_lo, h_hi = (float(np.percentile(h, q)) for q in (2, 98))
        height_src = "percentile"
    height = max(h_hi - h_lo, 1e-3)

    # 可靠性：墙面覆盖了几个方向（0~4）。只覆盖 1~2 个方向说明是**部分视角**，
    # 此时盒子必然偏小 —— 如实标注，不硬猜（实测宿舍：7 面墙全在一侧）。
    sides = sum(bool(x) for x in (lo_u_c, hi_u_c, lo_v_c, hi_v_c))
    if sides < 3:
        wall_note = f"墙面只覆盖 {sides}/4 个方向（部分视角），盒不可信"
        print(f"  ⚠ {wall_note}")
    else:
        wall_note = f"墙面覆盖 {sides}/4 个方向"

    size = [round(float(hi_u - lo_u), 4), round(float(hi_v - lo_v), 4), round(float(height), 4)]
    center_local = np.array([(lo_u + hi_u) / 2, (lo_v + hi_v) / 2, (h_lo + h_hi) / 2])
    axes = np.stack([u, v, up], axis=1)
    center = c + axes @ center_local
    corners = []
    for z in (-1, 1):
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corners.append([round(float(x), 4) for x in
                            (center + axes @ (np.array([sx, sy, z]) * np.array(size) / 2))])

    st = {
        "version": "cloud-walls",
        "source": str(args.cloud),
        "sane": sides >= 2,
        "estimated": sides < 3,
        "frame": {"up": [round(float(x), 6) for x in up],
                  "axisU": [round(float(x), 6) for x in u],
                  "axisV": [round(float(x), 6) for x in v],
                  "origin": [round(float(x), 5) for x in center]},
        "room": {"size": size, "corners": corners, "origin": [round(float(x), 5) for x in center],
                 "axisU": [round(float(x), 6) for x in u], "axisV": [round(float(x), 6) for x in v]},
        "walls": [{"normal": [round(float(x), 5) for x in w["n"]],
                   "inliers": w["inliers"], "centroid": [round(float(x), 4) for x in w["centroid"]]}
                  for w in walls[:6]],
        "trajectory": {"points": []},
        "stats": {"points": int(len(P)), "p95Radius": round(radius, 4), "band": round(band, 5)},
        "evidence": {"up": up_ev, "wallsFound": len(walls), "heightSource": height_src,
                     "sidesCovered": sides, "wallNote": wall_note},
        "note": "墙面拟合（RANSAC 竖直平面 + 半空间求交）；无墙方向用分位兜底",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"房间盒：{size}  （墙 {len(walls)} 面，高度来源 {height_src}）")
    print(f"已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
