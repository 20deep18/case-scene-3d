#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""清理 .splat 里的无用高斯（独立后处理，**只写新文件，不改原模型**）。

要解决的现象（实测）：
  宿舍模型 399,649 个高斯，画面上外圈一片半透明白雾，看着"糊"。
  但量过之后发现 **不是几个巨型高斯**（scale 最大只有 0.051、中位 0.0069），
  而是**大量小尺度、低不透明度的高斯叠出来的** —— 所以不能靠"删大高斯"解决。

四类过滤器（各自独立开关，逐项报告删了多少）：

  1. --opacity-pct  按**模型自身**的透明度分位删（默认 0 = 不删）。
                    ⚠️ 不能写绝对值：宿舍 opacity 中位 0.247，room-real 只有 0.020，
                    差一个数量级，写死 0.05 会把 room-real 删掉 67%。
  2. --scale-pct    按尺度分位删（默认 100 = 不删）。用于削掉异常大的"糊片"。
  3. --sor-k/std    体素密度滤波：按包围盒对角线定体素边长，统计 27 邻域内的高斯数，
                    过于稀疏的点视为漂浮噪点删除。
  4. --frustum-min  视锥覆盖：把每个高斯投到每台相机，统计"落入画面内且在前方"的相机数，
                    少于 N 台的删除（**相机没看过的东西本来就不该画**）。
  5. --box-clip     房间盒裁剪（可选，需要 structure.json）：只保留盒内 + 边距。
                    ⚠️ 房间盒不准时**不要开**（宿舍就属于不准）。

用法：
  <python> tools/prune_splat.py --splat data/models/宿舍现场-彩色-精简.splat \
      --out /tmp/dorm-pruned.splat --opacity-pct 5 --sor-k 3 --sor-std 2.0 \
      --poses output/dorm/dorm-c.poses.json --frustum-min 1 --report /tmp/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DT = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])


def load_splat(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) % DT.itemsize:
        raise SystemExit(f"文件长度不是 {DT.itemsize} 的整数倍，可能不是 .splat：{path}")
    return np.frombuffer(raw, dtype=DT, count=len(raw) // DT.itemsize).copy()


def voxel_sparse_mask(pos: np.ndarray, *, k: int, std: float, report: dict) -> np.ndarray:
    """体素密度滤波：与 SOR 同义（k 近邻平均距离超阈值判为离群），但用网格加速。

    返回 keep 掩码（True = 保留）。
    """
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    if diag <= 0:
        return np.ones(len(pos), bool)
    # 体素边长：让平均每个体素约 8 个点
    vol = float(np.prod(np.maximum(hi - lo, 1e-9)))
    cell = max((vol / max(len(pos), 1) * 8.0) ** (1.0 / 3.0), diag * 1e-4)
    idx = np.floor((pos - lo) / cell).astype(np.int64)
    dims = idx.max(axis=0) + 1
    # 27 邻域计数（用平移哈希实现，避免拉大数组）
    keys = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    count_map = dict(zip(uniq.tolist(), counts.tolist()))
    neigh = np.zeros(len(pos), np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                kk = ((idx[:, 0] + dx) * dims[1] + (idx[:, 1] + dy)) * dims[2] + (idx[:, 2] + dz)
                neigh += np.array([count_map.get(v, 0) for v in kk.tolist()], dtype=np.int64)
    keep = neigh >= max(k, 1)
    report["sor"] = {"cell": round(cell, 5), "k": k, "std": std,
                     "medianNeighbors": int(np.median(neigh)),
                     "removed": int((~keep).sum())}
    return keep


def frustum_mask(pos: np.ndarray, poses_path: Path, min_cams: int, report: dict) -> np.ndarray:
    """统计每个高斯被几台相机"看到"，少于 min_cams 的删除。"""
    pj = json.loads(poses_path.read_text(encoding="utf-8"))
    E = np.asarray(pj["extrinsic"], float)          # world→camera，[N,3,4]
    K = np.asarray(pj["intrinsic"], float)          # [N,3,3] 或 [3,3]
    names = pj.get("frames") or []
    n = min(len(E), len(K) if K.ndim == 3 else len(E))
    if n == 0:
        report["frustum"] = {"skipped": "poses 为空"}
        return np.ones(len(pos), bool)
    # 图像尺寸：从深度 npz 拿不到时用内参推算（cx*2, cy*2）
    cx = float(K[0][0, 2]) if K.ndim == 3 else float(K[0, 2])
    cy = float(K[0][1, 2]) if K.ndim == 3 else float(K[1, 2])
    W, H = cx * 2.0, cy * 2.0
    P = pos[:, :3]
    seen = np.zeros(len(pos), np.int32)
    for i in range(n):
        Ei = E[i]
        # X_cam = R · X_world + t（R 是 3×3，t 是 3 维）
        cam = P @ Ei[:3, :3].T + Ei[:3, 3][None, :]
        z = cam[:, 2]
        ok_z = z > 1e-6
        if not ok_z.any():
            continue
        Ki = K[i] if K.ndim == 3 else K
        u = (cam[:, 0] / np.where(ok_z, z, 1.0)) * Ki[0, 0] + Ki[0, 2]
        v = (cam[:, 1] / np.where(ok_z, z, 1.0)) * Ki[1, 1] + Ki[1, 2]
        # 略微放宽 5%，抵住边缘
        inside = ok_z & (u > -0.05 * W) & (u < 1.05 * W) & (v > -0.05 * H) & (v < 1.05 * H)
        seen += inside.astype(np.int32)
    keep = seen >= max(min_cams, 1)
    report["frustum"] = {"cameras": int(n), "minCams": min_cams,
                         "medianSeen": int(np.median(seen)),
                         "removed": int((~keep).sum()),
                         "seenHistogram": {int(k): int((seen == k).sum()) for k in range(0, 4)}}
    return keep


def box_mask(pos: np.ndarray, struct_path: Path, margin: float, report: dict) -> np.ndarray:
    st = json.loads(struct_path.read_text(encoding="utf-8"))
    room = st.get("room") or {}
    size = room.get("size")
    origin = room.get("origin")
    if not size or not origin or len(size) != 3:
        report["boxClip"] = {"skipped": "structure.json 没有 room.size/origin"}
        return np.ones(len(pos), bool)
    if st.get("estimated") or st.get("sane") is False:
        report["boxClip"] = {"skipped": "房间盒被标为推测/不可信，**不裁**（防止把真表面裁掉）"}
        return np.ones(len(pos), bool)
    au = np.asarray((st.get("frame") or {}).get("axisU") or room.get("axisU") or [1, 0, 0], float)
    av = np.asarray((st.get("frame") or {}).get("axisV") or room.get("axisV") or [0, 1, 0], float)
    aw = np.asarray((st.get("frame") or {}).get("up") or [0, 0, 1], float)
    axes = np.stack([au, av, aw], axis=1)
    c = np.asarray(origin, float)
    rel = (pos - c[None, :]) @ axes            # 转到房间盒坐标
    half = np.asarray(size, float) / 2.0 * (1.0 + margin)
    keep = (np.abs(rel) <= half[None, :]).all(axis=1)
    report["boxClip"] = {"margin": margin, "size": [round(float(x), 3) for x in size],
                         "removed": int((~keep).sum())}
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opacity-pct", type=float, default=0.0, help="删除透明度低于该分位的高斯（0=不删）")
    ap.add_argument("--scale-pct", type=float, default=100.0, help="删除尺度大于该分位的高斯（100=不删）")
    ap.add_argument("--sor-k", type=int, default=0, help="体素密度下限（0=不删）：27 邻域内至少多少个高斯")
    ap.add_argument("--white-large", action="store_true",
                    help="只删「近白且大」的高斯（灰白大球），**保住有色大面片**。\n"
                         "实测：按尺度 P90 一刀切会删掉 5,869 个「有色大高斯」（不透明度中位 0.69），\n"
                         "它们面积大、占屏幕颜色多 —— 这就是「抽稀后画面变白」的真正原因")
    ap.add_argument("--white-lum", type=float, default=200.0, help="近白阈值（RGB 均值）")
    ap.add_argument("--white-sat", type=float, default=25.0, help="低饱和阈值（max-min）")
    ap.add_argument("--sor-std", type=float, default=2.0, help="保留参数（当前实现未用，留作记录）")
    ap.add_argument("--poses", default=None, help="VGGT poses.json；给了才做视锥覆盖")
    ap.add_argument("--frustum-min", type=int, default=0, help="至少被几台相机看到（0=不删）")
    ap.add_argument("--structure", default=None, help="structure.json；给了才做房间盒裁剪")
    ap.add_argument("--box-margin", type=float, default=0.10, help="房间盒外扩比例")
    ap.add_argument("--report", default=None, help="过滤统计写到这个 json")
    args = ap.parse_args()

    arr = load_splat(Path(args.splat))
    n0 = len(arr)
    pos = np.stack([arr["pos"][:, 0], arr["pos"][:, 1], arr["pos"][:, 2]], axis=1)
    scale = np.stack([arr["scale"][:, 0], arr["scale"][:, 1], arr["scale"][:, 2]], axis=1)
    alpha = arr["rgba"][:, 3].astype(np.float32) / 255.0
    smax = np.nanmax(scale, axis=1)
    report: dict = {"input": str(args.splat), "gaussiansIn": int(n0), "filters": []}

    keep = np.ones(n0, bool)

    if args.opacity_pct > 0:
        thr = float(np.percentile(alpha[keep], args.opacity_pct))
        m = alpha >= thr
        report["filters"].append({"name": "opacity-pct", "pct": args.opacity_pct,
                                  "threshold": round(thr, 5), "removed": int((keep & ~m).sum())})
        keep &= m

    if args.scale_pct < 100.0:
        thr = float(np.percentile(smax[keep], args.scale_pct))
        m = smax <= thr
        report["filters"].append({"name": "scale-pct", "pct": args.scale_pct,
                                  "threshold": round(thr, 5), "removed": int((keep & ~m).sum())})
        keep &= m

    if args.white_large:
        lum = np.array([arr["rgba"][k, :3].astype(np.float32).mean() for k in range(len(arr))])
        sat = np.array([float(arr["rgba"][k, :3].max()) - float(arr["rgba"][k, :3].min())
                        for k in range(len(arr))])
        big = smax > np.percentile(smax[keep], 90)
        m = ~(big & (lum > args.white_lum) & (sat < args.white_sat))
        report["filters"].append({"name": "white-large", "lum": args.white_lum, "sat": args.white_sat,
                                  "removed": int((keep & ~m).sum())})
        keep &= m

    if args.sor_k > 0:
        m = voxel_sparse_mask(pos[keep], k=args.sor_k, std=args.sor_std, report=report)
        idx = np.nonzero(keep)[0]
        keep[idx[~m]] = False
        report["filters"].append({"name": "density", "removed": report["sor"]["removed"]})

    if args.frustum_min > 0 and args.poses:
        m = frustum_mask(pos[keep], Path(args.poses), args.frustum_min, report=report)
        idx = np.nonzero(keep)[0]
        keep[idx[~m]] = False
        report["filters"].append({"name": "frustum", "removed": report["frustum"]["removed"]})

    if args.structure:
        m = box_mask(pos[keep], Path(args.structure), args.box_margin, report=report)
        idx = np.nonzero(keep)[0]
        keep[idx[~m]] = False
        report["filters"].append({"name": "box-clip", "removed": report.get("boxClip", {}).get("removed", 0)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(arr[keep].tobytes())

    report["gaussiansOut"] = int(keep.sum())
    report["keptRatio"] = round(float(keep.sum()) / max(n0, 1), 4)
    report["bytesIn"] = Path(args.splat).stat().st_size
    report["bytesOut"] = out.stat().st_size
    print(f"输入 {n0:,} → 输出 {int(keep.sum()):,}（保留 {keep.mean():.1%}）")
    for f in report["filters"]:
        print(f"  {f['name']:<12} 删除 {f['removed']:,}")
    print(f"文件 {report['bytesIn']/1048576:.2f} MB → {report['bytesOut']/1048576:.2f} MB")
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"统计已写出 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
