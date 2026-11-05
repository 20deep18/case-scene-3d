#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对已有的 .splat 做**表面外点去噪**（治「墙面法向的云雾」）+ 贡献度剪枝。

为什么单独做一个工具而不是塞进 to_splat.py：
  1) 它能直接作用于**已有的模型文件** —— 不用重训、不用原始 PLY，立刻见效；
  2) 不碰生产转换路径，风险小、好回滚。

两个过滤器（可单独开关）：
  A. 表面外点（--mad-k，默认 3.0）
     现有浮尘剔除是按「到场景中心的距离」删的，对**贴着墙飘的雾**无效 ——
     它径向很近，但法向上离墙面很远。这里对每个高斯取 k 近邻做 PCA，
     量它到「邻居拟合平面」的距离，用**中位数 + k×MAD** 判定离群并删除。
  B. 贡献度剪枝（--contribution-quantile，默认 0 = 关）
     按「alpha × 体积」排序，砍掉贡献最小的尾部（文档第五节建议）。

用法：
  python tools/denoise_splat.py --in "data/models/X.splat" --out "data/models/X-denoised.splat"
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(ROOT / "pipeline"))

GAUSS_BYTES = 32


def load_splat(path: Path):
    raw = path.read_bytes()
    n = len(raw) // GAUSS_BYTES
    f = np.frombuffer(raw[: n * GAUSS_BYTES], dtype="<f4").reshape(n, 8)
    u = np.frombuffer(raw[: n * GAUSS_BYTES], dtype=np.uint8).reshape(n, GAUSS_BYTES)
    return {
        "n": n,
        "pos": f[:, 0:3].astype(np.float64),
        "scale": f[:, 3:6].astype(np.float64),
        "color": u[:, 24:28].copy(),          # RGBA
        "quat": u[:, 28:32].copy(),
        "raw_f": f,
        "raw_u": u,
    }


def surface_outlier_mask(pos: np.ndarray, *, k: int = 16, mad_k: float = 3.0,
                         sample: int = 40000, seed: int = 11) -> np.ndarray:
    """返回 True=保留。见模块文档 A。"""
    n = len(pos)
    keep = np.ones(n, dtype=bool)
    if n < k + 5 or mad_k <= 0:
        return keep
    rng = np.random.default_rng(seed)
    idx = (rng.choice(n, size=sample, replace=False) if n > sample else np.arange(n))
    sub = pos[idx]
    m = len(sub)
    kk = min(k, m - 1)
    if kk < 4:
        return keep

    resid = np.zeros(m)
    block = max(1, min(2000, m))
    for i in range(0, m, block):
        b = sub[i : i + block]
        d = np.linalg.norm(b[:, None, :] - sub[None, :, :], axis=2)
        d[np.arange(len(b)), np.arange(i, i + len(b))] = np.inf
        part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        for r in range(len(b)):
            pts = sub[part[r]]
            c = pts.mean(axis=0)
            q = pts - c
            try:
                _u, _s, vt = np.linalg.svd(q, full_matrices=False)
                normal = vt[-1]
            except np.linalg.LinAlgError:
                continue
            resid[i + r] = abs(float((sub[i + r] - c) @ normal))

    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    thr = float(np.percentile(resid, 95)) if mad <= 1e-12 else med + mad_k * 1.4826 * mad
    bad = resid > thr
    keep[idx[bad]] = False
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--mad-k", type=float, default=3.0, help="表面外点判定倍数（0=关闭，默认 3.0）")
    ap.add_argument("--knn", type=int, default=16, help="局部平面拟合的邻居数（默认 16）")
    ap.add_argument("--contribution-quantile", type=float, default=0.0,
                    help="按 alpha×体积 砍掉贡献最低的该比例（0=关闭，例如 0.1 表示砍 10%%）")
    ap.add_argument("--json", default=None, help="同时更新配套的元数据（默认同名 .json）")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    d = load_splat(src)
    n = d["n"]
    print(f"读入 {src.name}：{n} 个高斯")

    keep = surface_outlier_mask(d["pos"], k=args.knn, mad_k=args.mad_k)
    dropped_surface = int((~keep).sum())
    print(f"表面外点滤波：删除 {dropped_surface} 个（{dropped_surface / max(n,1):.1%}）")

    if args.contribution_quantile > 0:
        alpha = d["color"][:, 3].astype(np.float64) / 255.0
        vol = np.prod(np.maximum(d["scale"], 1e-9), axis=1)
        contrib = alpha * vol
        # 只在**已保留**的点里按贡献砍尾部
        alive = np.nonzero(keep)[0]
        q = float(np.clip(args.contribution_quantile, 0.0, 0.9))
        if len(alive) > 10 and q > 0:
            thr = np.quantile(contrib[alive], q)
            cut = alive[contrib[alive] <= thr]
            keep[cut] = False
            print(f"贡献度剪枝：再删除 {len(cut)} 个（alpha×体积 最低的 {q:.0%}）")

    sel = np.nonzero(keep)[0]
    if len(sel) == 0:
        print("[错误] 全被删光了，请放宽 --mad-k")
        return 1

    # 写出 .splat：直接按行取**原始 32 字节**，不做任何重新打包 ——
    # 第一版用 f.view(uint8).reshape(n,24) 再拼颜色，把每行 32 字节当成 24 字节，
    # 直接 ValueError（自己踩的坑，记在这里）。
    out = d["raw_u"][sel].copy()
    dst.write_bytes(out.tobytes())
    print(f"写出 {dst.name}：{len(sel)} 个高斯（{dst.stat().st_size} 字节）")

    # 同步 .shcoeff.bin（每高斯 45 个 float32）
    src_sh = src.with_suffix(".shcoeff.bin")
    if src_sh.is_file():
        sh = np.fromfile(src_sh, dtype="<f4")
        per = sh.size // n if n else 0
        if per > 0:
            sh = sh.reshape(n, per)[sel]
            dst_sh = dst.with_suffix(".shcoeff.bin")
            sh.astype("<f4").tofile(dst_sh)
            print(f"写出 {dst_sh.name}：{len(sel)}×{per}")

    # 更新元数据（数量 + 稳健几何）
    src_json = Path(args.json) if args.json else src.with_suffix(".json")
    if src_json.is_file():
        meta = json.loads(src_json.read_text(encoding="utf-8"))
        meta["count"] = int(len(sel))
        meta["gaussianCount"] = int(len(sel))
        p = d["pos"][sel]
        lo, hi = p.min(axis=0), p.max(axis=0)
        meta["bounds"] = {
            "min": [round(float(v), 4) for v in lo],
            "max": [round(float(v), 4) for v in hi],
            "center": [round(float(v), 4) for v in (lo + hi) / 2],
            "size": [round(float(v), 4) for v in (hi - lo)],
            "radius": round(float(np.linalg.norm(hi - lo) / 2), 4),
        }
        meta.setdefault("filters", {})["denoise"] = {
            "surfaceOutliersRemoved": dropped_surface,
            "madK": args.mad_k,
            "knn": args.knn,
            "contributionQuantile": args.contribution_quantile,
            "source": src.name,
        }
        dst.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"写出 {dst.with_suffix('.json').name}（count={meta['count']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
