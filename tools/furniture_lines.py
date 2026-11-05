#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""家具级线稿：从训练好的 `.splat` 里把**家具**找出来，输出可画的线框盒子。

为什么不用 COLMAP 稀疏点云
--------------------------
实测（本仓库 docs/场景与物体-判据实测与修复.md §十八）：稀疏点云连**水平面**都几乎没点，
更不用说家具了 —— 稀疏点云只有特征点，家具表面大多是弱纹理的平板。
而 `.splat` 里每个高斯都坐在**表面**上（训练就是为了贴合画面），
所以**家具的几何在 splat 里，不在稀疏点云里**。这条路不需要重新跑 MVS。

做法（不做深度学习的"家具识别"，只做几何聚类）
----------------------------------------------
1. 读 `.splat`（32 字节/高斯：位置 3×f32、尺度 3×f32、RGBA 4×u8、四元数 4×u8）
2. **过滤**掉不像表面的高斯：尺度太小的（噪点）、太大且很淡的（雾/天空/白墙泛光）
3. 用结构文件里的**曼哈顿坐标系**（frame.up / axisU / axisV）把点转到"房间系"，
   在房间系里做**体素连通聚类**（远处噪声不会因为投影连成一片）
4. 每个簇算**有向包围盒**（房间系里的轴对齐盒），按"尺寸像家具 + 点数够 + 不太扁"筛掉假目标
5. 输出 JSON：每个家具盒的中心/尺寸/点数/体素数 —— 交给 `tools/render_lineart.py` 画线稿

为什么用体素聚类而不是 RANSAC 平面：
  家具是**多个面围起来的体**。先聚类再取包围盒，比"先拟合平面再拼盒子"稳得多，
  而且对"柜子有多层搁板"这种情况，盒子本身就把外轮廓说清楚了；
  搁板层级可以后面在这个盒子里单独找（见 --layers）。

用法:
    <python> tools/furniture_lines.py --splat data/models/柜子现场-致密.splat --out output/furniture/guizi.json
    <python> tools/furniture_lines.py --splat data/models/room-real.splat --structure data/models/room-real.structure.json --out ...
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

GAUSS_BYTES = 32


# ---------------------------------------------------------------- 读 splat
def load_splat(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (位置 N×3, 最大尺度 N, 不透明度 N)。"""
    raw = path.read_bytes()
    n = len(raw) // GAUSS_BYTES
    f = np.frombuffer(raw[: n * GAUSS_BYTES], dtype="<f4").reshape(n, 8)
    pos = f[:, 0:3].astype(np.float64)
    scale = f[:, 3:6].astype(np.float64)
    b = np.frombuffer(raw[: n * GAUSS_BYTES], dtype=np.uint8).reshape(n, GAUSS_BYTES)
    alpha = b[:, 27].astype(np.float64) / 255.0          # RGBA 的第 4 个字节
    return pos, scale.max(axis=1), alpha


# ---------------------------------------------------------------- 读 PLY（稠密点云/学习式重建的产物）
def load_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读 PLY 点云，返回 (位置, 尺度, 不透明度)。

    为什么要支持 PLY：COLMAP 稠密重建（stereo_fusion）与 VGGT/DUSt3R 这类学习式重建
    的产物都是 PLY；而 `.splat` 只有 3DGS 会用。两者都要能进同一条家具分割流水线。

    - 普通点云（只有 x,y,z / 法向 / RGB）：尺度给 0、不透明度给 1（都过得了过滤）
    - 3DGS 的 PLY（含 scale_0..2 / opacity）：按真实值喂进去，过滤才有意义
      （3DGS 存的是 log 尺度与 logit 不透明度，要 exp / sigmoid 还原）
    """
    raw = path.read_bytes()
    if not raw.startswith(b"ply"):
        raise ValueError(f"{path} 不是 PLY")
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError("PLY 头不完整（没有 end_header）")
    head = raw[:end].decode("ascii", errors="replace")
    body_start = raw.find(b"\n", end) + 1

    fmt = None
    n_vert = 0
    props: list[tuple[str, str]] = []          # (名字, 类型)
    in_vertex = False
    for line in head.splitlines():
        s = line.strip()
        if s.startswith("format"):
            fmt = s.split()[1]
        elif s.startswith("element"):
            parts = s.split()
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                n_vert = int(parts[2])
        elif s.startswith("property") and in_vertex:
            parts = s.split()
            if parts[1] != "list":
                props.append((parts[2], parts[1]))
    if n_vert <= 0 or not props:
        raise ValueError("PLY 里没有 vertex 元素")

    np_types = {
        "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
        "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
        "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
    }
    for _, tp in props:
        if tp not in np_types:
            raise ValueError(f"不支持的 PLY 属性类型：{tp}")
    dtype = np.dtype([(nm, np_types[tp]) for nm, tp in props])

    if fmt == "ascii":
        data = np.loadtxt(path, dtype=object, skiprows=len(head.splitlines()) + 1,
                          max_rows=n_vert) if False else None
        # 用 genfromtxt 更稳（列数已知）
        arr = np.genfromtxt(path, skip_header=len(head.splitlines()),
                            max_rows=n_vert, dtype=None, encoding=None)
        cols = [arr[nm] if hasattr(arr, "dtype") and arr.dtype.names else arr[:, i]
                for i, (nm, _) in enumerate(props)]
        data = {nm: np.asarray(cols[i], float) for i, (nm, _) in enumerate(props)}
    elif fmt in ("binary_little_endian", "binary_big_endian"):
        if fmt == "binary_big_endian":
            dtype = dtype.newbyteorder(">")
        buf = np.frombuffer(raw, dtype=dtype, count=n_vert, offset=body_start)
        data = {nm: np.asarray(buf[nm], float) for nm, _ in props}
    else:
        raise ValueError(f"未知 PLY 格式：{fmt}")

    pos = np.stack([data["x"], data["y"], data["z"]], axis=1)
    if "scale_0" in data:
        s = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1)
        smax = np.exp(s).max(axis=1)          # 3DGS 存 log 尺度
    else:
        smax = np.zeros(len(pos))
    if "opacity" in data:
        alpha = 1.0 / (1.0 + np.exp(-data["opacity"]))   # logit → 概率
    else:
        alpha = np.ones(len(pos))
    return pos, smax, alpha


def load_cloud(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按扩展名分派：.splat（3DGS）或 .ply（稠密点云/学习式重建）。"""
    ext = path.suffix.lower()
    if ext == ".splat":
        return load_splat(path)
    if ext == ".ply":
        return load_ply(path)
    raise ValueError(f"不支持的格式：{ext}（只支持 .splat / .ply）")


# ---------------------------------------------------------------- 参考系
def manhattan_frame(pos: np.ndarray, structure: dict | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (origin, u, v, up)。

    优先用结构文件里已经算好的曼哈顿系（那是**按重力对齐**的，可靠）；
    没有就退回 PCA（对家具这种"一个立着的长方体"通常也能用）。
    """
    if structure:
        fr = structure.get("frame") or {}
        up = fr.get("up")
        u = fr.get("axisU")
        if up and u:
            up = np.asarray(up, float)
            u = np.asarray(u, float)
            up /= np.linalg.norm(up) + 1e-12
            u = u - up * float(u @ up)
            u /= np.linalg.norm(u) + 1e-12
            v = np.cross(up, u)
            origin = np.asarray(fr.get("origin") or np.median(pos, axis=0), float)
            return origin, u, v, up
    center = np.median(pos, axis=0)
    d = pos - center
    cov = d.T @ d / max(len(d), 1)
    w, vv = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    a1 = vv[:, order[0]]
    a2 = vv[:, order[1]]
    up0 = np.array([0.0, 0.0, 1.0])
    # 取与 a1 垂直的那个特征向量里"更接近竖直"的作为 up 的候选
    a3 = np.cross(a1, a2)
    up = a3 if abs(a3 @ up0) > abs(a1 @ up0) else a1
    u = a1 if not np.allclose(up, a1) else a2
    up = up / (np.linalg.norm(up) + 1e-12)
    u = u - up * float(u @ up)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(up, u)
    return center, u, v, up


# ---------------------------------------------------------------- 剥掉房间外壳
def strip_shell(local: np.ndarray, box_lo: np.ndarray, box_hi: np.ndarray, margin: float) -> np.ndarray:
    """剔掉**贴着房间六面**的点（地板/天花/四面墙），只留下屋里的"东西"。

    为什么必须做：一次体素连通聚类会把**地板**当成一条连着所有家具的走廊，
    于是整间屋聚成一个大簇（实测 sample-room / test-scene 就是这样：0 个家具盒）。
    剥掉外壳后，家具才会各自独立成簇 —— 这是"房间里找家具"的标准做法。
    """
    inner_lo = box_lo + margin
    inner_hi = box_hi - margin
    good = np.ones(len(local), bool)
    for d in range(3):
        if inner_hi[d] - inner_lo[d] <= margin:      # 这一维太薄，就别剥了（整个都在壳里）
            continue
        good &= local[:, d] >= inner_lo[d]
        good &= local[:, d] <= inner_hi[d]
    return good


# ---------------------------------------------------------------- 剥掉大平面（墙/地/天花）
def ransac_plane(pts: np.ndarray, *, band: float, trials: int = 300, rng=None) -> tuple[np.ndarray, float, np.ndarray] | None:
    """返回 (法向, 偏移, inlier 掩码)；找不到返回 None。"""
    if len(pts) < 50:
        return None
    rng = rng or np.random.default_rng(7)
    best = None
    best_n = 0
    for _ in range(trials):
        idx = rng.choice(len(pts), 3, replace=False)
        a, b, c = pts[idx]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n = n / ln
        d = -float(n @ a)
        inl = np.abs(pts @ n + d) < band
        k = int(inl.sum())
        if k > best_n:
            best_n, best = k, (n, d, inl)
    return best


def strip_planes(local: np.ndarray, *, band: float, max_planes: int = 8,
                 min_inlier_ratio: float = 0.08,
                 outer_ratio: float = 0.55) -> tuple[np.ndarray, list[dict]]:
    """逐个大平面剥掉（墙/地面/天花/大镜面），剩下的才是"屋里的东西"。

    为什么小空间必须这样做：厕所整间只有两三米，用"房间盒 + 边距"去剥壳等于没剥
    （实测：剥完还剩 8010/13690 点，聚出来的"家具"就是整间屋）。
    平面剥离不依赖房间盒，也不依赖曼哈顿系，对小空间更稳。
    """
    keep = np.ones(len(local), bool)
    planes = []
    work = local.copy()
    lo, hi = local.min(axis=0), local.max(axis=0)
    for _ in range(max_planes):
        cur = work[keep]
        p = ransac_plane(cur, band=band)
        if p is None:
            break
        n, d, _ = p
        inl_all = np.abs(work @ n + d) < band
        inl = inl_all & keep
        frac = float(inl.sum()) / max(int(keep.sum()), 1)
        if frac < min_inlier_ratio:
            break
        # ⚠️ 只剥**外圈**的平面（贴近房间边界的墙/地/顶）。
        #    否则会把家具自己的大面（床面、柜门）也剥掉 —— 实测：不加这条，
        #    合成房从 4/4 命中退化到 0 个盒子。
        c = work[inl].mean(axis=0)
        half = (lo + hi) / 2
        span_ = np.maximum(hi - lo, 1e-9)
        off = np.abs(c - half) / (span_ / 2)          # 0=正中心，1=贴边
        if float(off.max()) < outer_ratio:
            continue
        planes.append({"normal": [round(float(x), 4) for x in n], "offset": round(float(d), 4),
                       "inliers": int(inl.sum()), "ratio": round(frac, 3),
                       "edge": round(float(off.max()), 3)})
        keep &= ~inl_all
    return keep, planes


# ---------------------------------------------------------------- 体素连通聚类
def voxel_clusters(pts: np.ndarray, voxel: float, *, min_points: int, max_voxels: int = 400000):
    """返回簇列表；网格过大返回 None（调用方应放大体素重试）。"""
    """把点按体素做 26 邻域连通聚类，返回 [(点索引数组, 体素数)]。"""
    if len(pts) == 0:
        return []
    ijk = np.floor(pts / voxel).astype(np.int64)
    ijk -= ijk.min(axis=0)
    dims = ijk.max(axis=0) + 1
    if int(np.prod(dims)) > max_voxels * 8:
        # ⚠️ 这里**不能静默返回空**：早先版本返回 []，于是"体素太小/场景太大"看起来
        #    就像"没找到家具"，把我带偏过一次。返回 None 交给调用方放大体素重试。
        return None
    key = (ijk[:, 0].astype(np.int64) << 42) ^ (ijk[:, 1].astype(np.int64) << 21) ^ ijk[:, 2].astype(np.int64)
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    # 每个体素的中心点坐标（用点均值）
    vpos = np.zeros((len(uniq), 3))
    for d in range(3):
        vpos[:, d] = np.bincount(inv, weights=pts[:, d], minlength=len(uniq)) / counts
    vijk = np.stack([np.bincount(inv, weights=ijk[:, d], minlength=len(uniq)) / counts for d in range(3)], axis=1)
    vijk = np.rint(vijk).astype(np.int64)

    lookup = {tuple(vijk[i]): i for i in range(len(uniq))}
    seen = np.zeros(len(uniq), bool)
    clusters = []
    neigh = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
    for start in range(len(uniq)):
        if seen[start]:
            continue
        q = deque([start])
        seen[start] = True
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            base = tuple(vijk[cur])
            for da, db, dc in neigh:
                nk = (base[0] + da, base[1] + db, base[2] + dc)
                j = lookup.get(nk)
                if j is not None and not seen[j]:
                    seen[j] = True
                    q.append(j)
        if len(comp) < (2 if len(pts) < 20000 else 8):
            continue
        # 该簇包含的原始点
        mask = np.isin(inv, comp)
        idx = np.nonzero(mask)[0]
        if len(idx) >= min_points:
            clusters.append((idx, len(comp)))
    clusters.sort(key=lambda t: -len(t[0]))
    return clusters


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="从 .splat 里提取家具级线框盒子")
    ap.add_argument("--splat", required=True,
                    help="点云文件：.splat（3DGS）或 .ply（稠密/学习式重建）")
    ap.add_argument("--structure", default=None, help="结构文件（提供曼哈顿系）；没有就退回 PCA")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=0.0, help="体素边长（重建单位）；0=按场景跨度自动取（推荐）")
    ap.add_argument("--voxel-ratio", type=float, default=0.006, help="自动体素 = 场景跨度 × 该比例（默认 0.6%%）")
    ap.add_argument("--max-scale", type=float, default=0.0, help="尺度上限（重建单位）；0=按跨度自动")
    ap.add_argument("--min-alpha", type=float, default=0.02, help="不透明度下限（实测低纹理场景 alpha 中位可能只有 0.05，默认 0.02）")
    ap.add_argument("--max-scale-ratio", type=float, default=0.03, help="自动尺度上限 = 跨度 × 该比例")
    ap.add_argument("--floor-snap", type=float, default=0.0,
                    help="底面离地面在这个体素倍数内就吸附到地面；0=自动（剥壳边距+1.5 体素）")
    ap.add_argument("--scale-pct", type=float, default=75.0,
                    help="自动尺度上限还会取该点云尺度的这个分位数（默认 P75），避免把整份点云滤空")
    ap.add_argument("--min-size-ratio", type=float, default=0.012, help="自动最小边 = 跨度 × 该比例")
    ap.add_argument("--max-size-ratio", type=float, default=0.60, help="自动最大边 = 跨度 × 该比例")
    ap.add_argument("--max-size-metres", type=float, default=0.0,
                    help="绝对最大边（米）；>0 时优先用它（重建尺度≈米时更直观）")
    ap.add_argument("--strip-shell", dest="strip_shell", action="store_true", default=True,
                    help="（默认开）先剥掉贴房间六面的点，否则地板会把所有家具连成一坨")
    ap.add_argument("--no-strip-shell", dest="strip_shell", action="store_false")
    ap.add_argument("--shell-margin", type=float, default=3.0, help="剥壳边距 = 体素 × 该值")
    ap.add_argument("--dump-clusters", action="store_true", help="打印所有候选簇（诊断用）")
    ap.add_argument("--strip-planes", dest="strip_planes", action="store_true", default=True,
                    help="（默认开）再剥掉墙/地/天花这些大平面（小空间必须）")
    ap.add_argument("--no-strip-planes", dest="strip_planes", action="store_false")
    ap.add_argument("--plane-band", type=float, default=2.0, help="平面内点带宽 = 体素 × 该值")
    ap.add_argument("--cam-radius-factor", type=float, default=2.5,
                    help="无可信房间盒时：现场范围 = 相机群半径 × 该倍数")
    ap.add_argument("--min-points", type=int, default=300, help="每个家具簇的最少高斯数")
    ap.add_argument("--min-size", type=float, default=0.0, help="家具最小边；0=按跨度自动")
    ap.add_argument("--max-size", type=float, default=0.0, help="家具最大边；0=按跨度自动")
    args = ap.parse_args()

    splat = Path(args.splat)
    if not splat.exists():
        print(f"找不到 {splat}")
        return 1
    pos, smax, alpha = load_cloud(splat)
    print(f"高斯基数: {len(pos)}")

    structure = None
    if args.structure:
        p = Path(args.structure)
        if p.exists():
            structure = json.loads(p.read_text(encoding="utf-8"))
    origin, u, v, up = manhattan_frame(pos, structure)
    frame_src = "structure.json" if structure else "PCA"
    print(f"曼哈顿系来源: {frame_src}  up = {np.round(up, 3)}")

    # 场景跨度先算出来（后面所有阈值都按它取，避免"写死米数"的错误）
    span = float(np.percentile(np.linalg.norm(pos - np.median(pos, axis=0), axis=1), 95)) * 2
    # 尺度上限：不能只按"场景跨度的百分比"取 —— 实测 3DGS 的 PLY 里高斯普遍 ~1.1 单位，
    # 按 3% 跨度取（0.69）会把**所有**点滤掉（过滤后 0 个）。所以取
    #   min(跨度 × 比例, 该点云自身尺度的 P75)
    # 没有尺度信息的点云（COLMAP MVS / VGGT 的 PLY，smax 全 0）则不做尺度过滤。
    has_scale = float(np.percentile(smax, 90)) > 0
    if args.max_scale > 0:
        max_scale = args.max_scale
    elif has_scale:
        # 取**两者里更宽的那个**：跨度×比例可能比点云自身尺度还紧（实测 3DGS 的 PLY
        # 高斯普遍 ~1.1 单位，而跨度×3% 只有 0.69，取 min 会把整份点云滤空）。
        # 上限再用跨度的 50% 兜底，免得把远处的巨大"雾"高斯全放进来。
        max_scale = min(span * 0.5, max(span * args.max_scale_ratio,
                                      float(np.percentile(smax, args.scale_pct))))
    else:
        max_scale = float("inf")
    min_size = args.min_size if args.min_size > 0 else span * args.min_size_ratio
    max_size = args.max_size if args.max_size > 0 else (
        args.max_size_metres if args.max_size_metres > 0 else span * args.max_size_ratio)
    print(f"场景跨度(P95×2) = {span:.3f} 单位；阈值：尺度≤{max_scale:.3f}，边长 {min_size:.3f}~{max_size:.3f}")
    # 过滤"不像表面"的高斯
    keep = (smax <= max_scale) & (alpha >= args.min_alpha)
    if not has_scale:
        print("（该点云没有尺度信息：跳过尺度过滤）")
    pts_world = pos[keep]
    print(f"过滤后: {len(pts_world)} 个高斯（尺度≤{max_scale:.3f}，alpha≥{args.min_alpha}）")
    if len(pts_world) < 500:
        print("剩下的点太少，放弃")
        return 1

    # 转到房间系，并在房间系里聚类
    # ⚠️ 体素尺寸必须按**场景自身的尺度**给：COLMAP 重建的"1 单位"不是 1 米，
    #    写死 0.08 会得到离谱的网格（实测 0 个簇，就是这么踩的）。
    local = np.stack([(pts_world - origin) @ u, (pts_world - origin) @ v, (pts_world - origin) @ up], axis=1)
    voxel = args.voxel if args.voxel > 0 else max(span * args.voxel_ratio, 1e-6)

    # 地面高度（up 轴上的低分位）：后面用来把"贴地的盒子"底面吸附到地面。
    # 为什么要吸附：家具底面常年贴着地板，会被剥壳/剥平面一起去掉，
    # 于是盒高普遍偏短（实测合成房：床 0.50 → 0.44，柜 0.52 → 0.45，短 8~12%）。
    floor_local = float(np.percentile(local[:, 2], 1.0))

    # ---- 剥壳：先确定"房间盒"，再剔掉贴着六面的点 -------------------------
    shell_mode = "none"
    if args.strip_shell:
        room = (structure or {}).get("room") or {}
        size = room.get("size")
        sane = (structure or {}).get("sane") is not False
        room_ok = bool(size and len(size) == 3 and sane
                       and max(size) < 30 and min(size) > 0.5)
        traj = ((structure or {}).get("trajectory") or {}).get("points") or []
        if not room_ok and len(traj) >= 8:
            # ⚠️ 房间盒不可信时（实测厕所：只找到 1 面墙、自检不过、盒子 10.9×15.4×5.5 m），
            #    改用**相机轨迹**圈定现场范围：相机都在屋里，离相机群半径太远的点就是噪声/室外。
            C = np.asarray(traj, float)
            cc = C.mean(axis=0)
            c_local = np.array([(cc - origin) @ u, (cc - origin) @ v, (cc - origin) @ up])
            cam_r = float(np.median(np.linalg.norm(C - cc, axis=1)))
            half = max(cam_r * args.cam_radius_factor, 0.5)
            box_lo = c_local - half
            box_hi = c_local + half
            src = f"相机轨迹（半径 {cam_r:.3f} × {args.cam_radius_factor}）"
            shell_mode = "cam"
        elif size and len(size) == 3:
            # 结构文件的房间盒是"重力+墙对齐"的，比点云包围盒可信
            # ⚠️ 盒心必须用**角点均值**：`room.origin` 在结构文件里是"参考系原点"，
            #    常常是地面中心甚至是角点（实测：我当时拿它当盒心，盒子一半在地下一半在上，
            #    于是墙只剥掉一半，剩下的点又连成一坨 → 0 个家具盒）。
            corners = room.get("corners") or []
            if len(corners) == 8:
                cw = np.asarray(corners, float).mean(axis=0)
            else:
                cw = np.asarray(room.get("origin") or [0, 0, 0], float)
            c_local = np.array([
                (cw - origin) @ u, (cw - origin) @ v, (cw - origin) @ up,
            ])
            box_lo = c_local - np.asarray(size, float) / 2
            box_hi = c_local + np.asarray(size, float) / 2
            src = "structure.json 的房间盒"
            shell_mode = "room"
        else:
            box_lo = np.percentile(local, 1, axis=0)
            box_hi = np.percentile(local, 99, axis=0)
            src = "点云稳健包围盒"
            shell_mode = "bbox"
        margin = voxel * args.shell_margin
        keep_in = strip_shell(local, box_lo, box_hi, margin)
        print(f"剥壳（{src}，边距 {margin:.3f}）：{int(keep_in.sum())} / {len(local)} 点留下")
        local_all, local = local, local[keep_in]
        world_all, pts_world = pts_world, pts_world[keep_in]
        if len(local) < 500:
            print("剥壳后点太少（这个现场可能本来就没什么家具），继续但结果会很空")
    print(f"场景跨度(P95×2) = {span:.3f} 单位 → 体素 = {voxel:.4f}")
    clusters = None
    for attempt in range(7):
        clusters = voxel_clusters(local, voxel, min_points=args.min_points)
        if clusters is not None:
            break
        voxel *= 1.6
        print(f"  网格过大，体素放大到 {voxel:.4f} 重试（第 {attempt + 2} 次）")
    if clusters is None:
        print("网格始终过大，放弃")
        return 1
    print(f"体素聚类: {len(clusters)} 个候选簇（体素 {voxel:.4f}）")

    # 平面剥离（小空间的关键一步）：把墙/地/天花这些大平面也去掉，剩下的才是洁具/家具
    # ⚠️ 只有**房间盒不可信**时才剥平面：房间盒可信时剥壳已经够了，
    #    再剥平面会把家具自己的大面（床面、柜门）当墙剥掉 ——
    #    实测：合成房从 4/4 退化到 0 个盒子，而它恰恰是房间盒可信的那种。
    if args.strip_planes and shell_mode != "room" and len(local) > 500:
        band = voxel * args.plane_band
        keep_pl, planes = strip_planes(local, band=band)
        print(f"平面剥离：去掉 {int((~keep_pl).sum())} 点，剩下 {int(keep_pl.sum())} 点；"
              f"最大 3 个平面内点占比 {[p['ratio'] for p in planes[:3]]}")
        if int(keep_pl.sum()) >= 300:
            local = local[keep_pl]
            pts_world = pts_world[keep_pl]
            clusters = None
            for attempt in range(5):
                clusters = voxel_clusters(local, voxel, min_points=args.min_points)
                if clusters is not None:
                    break
                voxel *= 1.6
            clusters = clusters or []
            print(f"  平面剥离后重新聚类: {len(clusters)} 个候选簇（体素 {voxel:.4f}）")
            out_planes = planes
        else:
            out_planes = []
    else:
        out_planes = []

    axes = np.stack([u, v, up], axis=1)
    objects = []
    for idx, nvox in clusters:
        if args.dump_clusters:
            q = local[idx]
            lo_, hi_ = q.min(axis=0), q.max(axis=0)
            print(f"  [簇] 点 {len(idx):5d} 体素 {nvox:4d} 盒尺寸 {np.round(hi_-lo_, 3)}")
        p = local[idx]
        lo, hi = p.min(axis=0), p.max(axis=0)
        size = hi - lo
        if size.min() < min_size or size.max() > max_size:
            continue
        # 底面贴近地面的，吸附到地面（修"高度偏短"）
        gap = float(lo[2] - floor_local)
        # 阈值 = 剥壳边距（被削掉的那条带）+ 1.5 体素：底面最多只会比地面高出一个边距。
        snap = args.floor_snap if args.floor_snap > 0 else (args.shell_margin + 1.5)
        if -voxel * 0.75 <= gap <= voxel * snap:
            hi = hi.copy()
            lo = lo.copy()
            lo[2] = floor_local
            size = hi - lo
        center_local = (lo + hi) / 2
        center_world = origin + axes @ center_local
        # 粗糙的"填充率"：点数 / 盒体积，用来区分"实心家具"与"薄板/管线"
        vol = float(np.prod(np.maximum(size, 1e-6)))
        objects.append({
            "center": [round(float(x), 4) for x in center_world],
            "size": [round(float(x), 4) for x in size],
            "points": int(len(idx)),
            "voxels": int(nvox),
            "fillPerM3": round(float(len(idx) / vol), 1),
        })
    objects.sort(key=lambda o: -o["points"])
    # 太扁的（薄板/地板）先不当作家具：至少两个方向 ≥ 0.3 m 且第三方向 ≥ 0.15 m 已经由 min_size 保证

    out = {
        "source": str(splat),
        "frame": {
            "origin": [round(float(x), 4) for x in origin],
            "axisU": [round(float(x), 6) for x in u],
            "axisV": [round(float(x), 6) for x in v],
            "up": [round(float(x), 6) for x in up],
            "source": frame_src,
        },
        "params": {"voxel": args.voxel, "maxScale": args.max_scale, "minAlpha": args.min_alpha,
                   "minPoints": args.min_points, "minSize": round(min_size, 4),
                   "maxSize": round(max_size, 4), "maxScale": round(max_scale, 4), "span": round(span, 4)},
        "gaussians": int(len(pos)),
        "kept": int(len(pts_world)),
        "objects": objects,
        "note": "每个 objects[i] 是房间系里的轴对齐盒：center/size 单位为该重建的单位（COLMAP 尺度，非米）",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n找到 {len(objects)} 个家具级盒子：")
    for i, o in enumerate(objects[:12]):
        print(f"  {i+1:2d}. 尺寸 {o['size']}  中心 {o['center']}  点数 {o['points']:6d}  填充率 {o['fillPerM3']}")
    print(f"\n已写出 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
