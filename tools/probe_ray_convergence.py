#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""探针：用**只依赖相机位姿**的量，判断一段素材是"绕物体拍"还是"在房间里走"。

为什么单独写探针而不是直接改 scene_structure.py：用户要求「先量再改」。
交接文档提议的判据是「视线汇聚度」（ray convergence），但它有一个我以前踩过的
退化陷阱必须先量出来：

    相机沿自己的视线方向**直线前进**时，q_i(d) = c_i + f_i·d 会几乎重合
    （极端情形完全重合），spread(d) ≈ 0 —— 于是**走廊/进门推进**这种最典型的
    「房间」拍法会被误判成「物体」。

所以这个脚本除了一般的汇聚度，还量三件事用于识别退化：

    1. minSpreadRel = min_d spread(d) / spread(0)     —— 汇聚得有多紧
    2. dStar / camRadius                              —— 汇聚点离相机群多远
    3. alongFrac = std(c_i·f̄)² / total               —— 相机位移有多少是**沿着视线**的
       （绕物体拍 → 相机是绕着转的，沿视线分量小；直线推进 → 沿视线分量接近 1）

用法：
    pyruntime312\\python.exe tools\\probe_ray_convergence.py <模型目录> [更多目录...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "tools", ROOT / "pipeline"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import colmap_model_io as cmio  # noqa: E402
import geom_common as gc  # noqa: E402


def camera_rays(model: dict) -> tuple[np.ndarray, np.ndarray]:
    """相机中心与前向 —— **直接调用生产代码** `geom_common.camera_centers_and_forwards()`。

    ⚠️ 2026-10-06：原来这里自己实现了 ~25 行（含符号约定）。这类"探针自己算一遍"
    是最危险的重复：探针量出来的数一旦与线上实现不一致（哪怕只是差一个符号），
    用它推出的结论就是错的，而且**不会有任何报错**。
    交接文档里那个「-Z 是观察方向」的错误正属于这一类。
    """
    return gc.camera_centers_and_forwards(sorted(model["images"], key=lambda im: im["name"]))


def verify_forward(C: np.ndarray, F: np.ndarray, cloud_center: np.ndarray) -> dict:
    """经验校验前向符号：相机是不是朝着点云看。

    返回 +Z 与 -Z 两种取向下「从相机指向点云质心」的平均余弦。
    取正值大的那个才是真正的"观察方向"。这一步是为了**不靠记忆定符号**。
    """
    to_cloud = cloud_center[None, :] - C
    to_cloud = to_cloud / (np.linalg.norm(to_cloud, axis=1, keepdims=True) + 1e-12)
    return {
        "cosForwardPlusZ": float(np.mean(np.sum(F * to_cloud, axis=1))),
        "cosForwardMinusZ": float(np.mean(np.sum(-F * to_cloud, axis=1))),
    }


def convergence(C: np.ndarray, F: np.ndarray, *, n_scan: int = 240, d_max_factor: float = 3.0) -> dict:
    """视线汇聚度：找 q_i(d) = c_i + f_i·d 最紧的那一点。"""
    if len(C) < 4:
        return {"ok": False, "reason": "相机太少"}

    centroid = C.mean(axis=0)
    R_cam = float(np.median(np.linalg.norm(C - centroid, axis=1)))
    if R_cam < 1e-9:
        return {"ok": False, "reason": "相机位置退化（几乎没有位移）"}

    # spread(0) = 相机自身的位置离散度 —— 用它归一化，**绝不用点云尺度**
    # （交接文档已论证：点云里混了背景，尺度不可靠）。
    def spread_at(d: float) -> tuple[float, float]:
        Q = C + F * d
        med = np.median(Q, axis=0)
        dist = np.linalg.norm(Q - med, axis=1)
        # 除以 sqrt(3) 让 d=0 时的归一化基准 ≈ 各向同性时的 std 量级
        return float(np.mean(dist)) / np.sqrt(3.0), float(np.median(dist))

    s0, _ = spread_at(0.0)
    if s0 < 1e-9:
        return {"ok": False, "reason": "相机位置几乎重合（没有位移）"}

    d_max = d_max_factor * R_cam
    ds = np.linspace(0.0, d_max, n_scan)
    raw = np.array([spread_at(float(d))[0] for d in ds])
    rel = raw / s0
    i_star = int(np.argmin(rel))
    d_star = float(ds[i_star])

    # 相机位移在「平均视线」方向的占比：识别"直线推进"退化
    f_mean = F.mean(axis=0)
    f_mean = f_mean / (np.linalg.norm(f_mean) + 1e-12)
    proj = C @ f_mean
    along_var = float(np.var(proj))
    total_var = float(np.var(C, axis=0).sum())
    along_frac = along_var / total_var if total_var > 1e-12 else 1.0

    # 横跨（相机群在与视线垂直方向上的离散度）—— 绕拍的相机是"环"，这个量大
    perp = C - np.outer(proj, f_mean)
    perp_r = float(np.median(np.linalg.norm(perp - perp.mean(axis=0), axis=1)))

    # 一致性：最佳汇聚点处，每台相机的视线是否真指向 q*
    q_star = np.median(C + F * d_star, axis=0)
    to_q = q_star[None, :] - C
    to_q = to_q / (np.linalg.norm(to_q, axis=1, keepdims=True) + 1e-12)
    cos_to_q = np.sum(F * to_q, axis=1)

    return {
        "ok": True,
        "minSpreadRel": float(rel[i_star]),
        "dStar": d_star,
        "dStarOverCamRadius": d_star / R_cam,
        "camRadius": R_cam,
        "camSpanP99": float(np.percentile(np.linalg.norm(C - centroid, axis=1), 99)),
        "alongFrac": along_frac,
        "perpRadius": perp_r,
        "alongOverPerp": float(np.sqrt(along_var) / (perp_r + 1e-12)),
        "medianCosRayToQStar": float(np.median(cos_to_q)),
        "p10CosRayToQStar": float(np.percentile(cos_to_q, 10)),
        "nCams": int(len(C)),
        # 曲线形状：便于看出"最小值是不是落在搜索边界上"
        "relAt0": 1.0,
        "relAtHalfCamRadius": float(np.interp(0.5 * R_cam, ds, rel)),
        "relAtCamRadius": float(np.interp(R_cam, ds, rel)),
        "relAt2CamRadius": float(np.interp(2.0 * R_cam, ds, rel)),
        "relAtMax": float(rel[-1]),
        "argminAtBoundary": bool(i_star >= n_scan - 2),
    }


def probe(model_dir: Path) -> dict:
    model = cmio.read_model(model_dir)
    C, F = camera_rays(model)
    xyz = np.asarray(model["points3D"]["xyz"], dtype=float)
    err = np.asarray(model["points3D"]["error"], dtype=float)
    trk = np.asarray(model["points3D"]["track_length"], dtype=float)
    keep = np.isfinite(xyz).all(axis=1) & (err < 2.0) & (trk >= 3)
    pts = xyz[keep]
    cloud_center = np.median(pts, axis=0) if len(pts) else np.zeros(3)
    out = {
        "dir": str(model_dir),
        "nImages": len(model["images"]),
        "nPoints": int(len(xyz)),
        "nPointsUsed": int(keep.sum()),
        "forwardCheck": verify_forward(C, F, cloud_center) if len(C) else None,
    }
    out.update(convergence(C, F))
    return out


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    rows = []
    for a in args:
        d = Path(a)
        if not d.is_dir():
            print(f"[跳过] 不是目录：{d}")
            continue
        try:
            r = probe(d)
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {d}: {type(exc).__name__}: {exc}")
            continue
        rows.append(r)
        print("=" * 78)
        print(f"{d}")
        print(f"  相机 {r['nImages']} 台，稀疏点 {r['nPoints']}（用 {r['nPointsUsed']}）")
        if r.get("forwardCheck"):
            fc = r["forwardCheck"]
            print(f"  前向符号校验：+Z 余弦 {fc['cosForwardPlusZ']:+.3f} / -Z 余弦 {fc['cosForwardMinusZ']:+.3f}"
                  f"   → 观察方向 = {'+Z (Rᵀ·(0,0,1))' if fc['cosForwardPlusZ'] > fc['cosForwardMinusZ'] else '-Z'}")
        if not r.get("ok"):
            print(f"  ✗ 无法判定：{r.get('reason')}")
            continue
        print(f"  相机群半径 R_cam   = {r['camRadius']:.3f} m（P99 跨度 {r['camSpanP99']:.3f} m）")
        print(f"  minSpreadRel       = {r['minSpreadRel']:.4f}   ← 越小越像「绕物体拍」")
        print(f"  d*  = {r['dStar']:.3f} m  = {r['dStarOverCamRadius']:.3f} × R_cam")
        print(f"  alongFrac          = {r['alongFrac']:.4f}（1 = 相机沿视线直线推进）")
        print(f"  along/perp         = {r['alongOverPerp']:.3f}（绕拍应 << 1）")
        print(f"  视线指向 q* 的余弦 中位 {r['medianCosRayToQStar']:.3f} / P10 {r['p10CosRayToQStar']:.3f}")
        print(f"  rel(d) 曲线：0→{r['relAt0']:.3f}  0.5R→{r['relAtHalfCamRadius']:.3f}"
              f"  1R→{r['relAtCamRadius']:.3f}  2R→{r['relAt2CamRadius']:.3f}  3R→{r['relAtMax']:.3f}"
              f"{'  [最小值落在搜索边界!]' if r['argminAtBoundary'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
