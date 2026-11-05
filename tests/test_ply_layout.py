#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回归：train_gsplat.save_ply 写出的 PLY 必须是「记录主序」，且各字段物理合理。

为什么需要这个测试
------------------
曾经 save_ply 按**块主序**写数据（所有高斯的 x、再所有高斯的 y、……），
整个文件对任何标准 PLY 读取器都是错位的。
它在**训练进程内完全看不出来**（内存里的参数是对的），
只有把文件写出去、再用另一套代码读回来才会暴露 ——
当时的症状是在浏览器里加载 .splat 后显示成一堆霓虹色大色块。

本测试用「结构化 dtype 逐字段检查」来卡住这个回归：
  1) 字节数必须与 header 声明的属性数吻合
  2) 位置跨度必须与输入一致（错位会立刻破坏这一点）
  3) 四元数分量必须在 [-1, 1]
  4) 不透明度必须是 logit（本测试给的是 -2.2 / -4.6 等）
  5) 尺度必须是线性米（本测试给的是 0.01~0.05）
  6) 颜色直流分量必须约等于 (rgb - 0.5)/SH_C0
  7) 各字段统计量**不能雷同**（雷同 = 错位的典型指纹）

用法: python tests/test_ply_layout.py
退出码 0 通过 / 1 失败
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import torch  # noqa: E402

import train_gsplat  # noqa: E402

SH_C0 = 0.28209479177387814
passed = 0
failed = 0


def check(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [OK]   {msg}")
    else:
        failed += 1
        print(f"  [FAIL] {msg}")


def read_ply_strict(path):
    """用 end_header 字节偏移 + 结构化 dtype 严格读取。"""
    data = Path(path).read_bytes()
    i = data.find(b"end_header")
    assert i >= 0, "找不到 end_header"
    j = i + len(b"end_header")
    while data[j : j + 1] in (b"\n", b"\r"):
        j += 1
    hdr = data[:i].decode("ascii", "replace")
    names = [ln.split()[-1] for ln in hdr.splitlines() if ln.startswith("property")]
    n = int([ln for ln in hdr.splitlines() if ln.startswith("element vertex")][0].split()[-1])
    dt = np.dtype([(x, "<f4") for x in names])
    body = data[j:]
    return names, n, dt, body


def main():
    global passed, failed
    print("=" * 70)
    print("PLY 记录主序回归测试")
    print("=" * 70)

    # 造一组**每个字段都明显不同**的输入，这样错位一定会被抓到
    count = 500
    torch.manual_seed(0)
    # 位置：x 在 [-2,2]，y 在 [10,12]，z 在 [-50,-48] —— 三个轴范围故意差很大
    means = torch.stack(
        [
            torch.rand(count) * 4 - 2,
            torch.rand(count) * 2 + 10,
            torch.rand(count) * 2 - 50,
        ],
        dim=1,
    )
    # 线性尺度 0.01~0.05；save_ply 内部会取 exp(scales_log)，所以这里传 log
    scale_lin = torch.rand(count, 3) * 0.04 + 0.01
    scales_log = torch.log(scale_lin)
    # 四元数：必须是单位化的合法值
    quats = torch.randn(count, 4)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    # 不透明度 logit：-2.2 / -4.6 两个值
    opac_logit = torch.where(torch.rand(count) > 0.5, -2.2, -4.6)
    # 颜色：rgb 在 [0,1]
    rgb = torch.rand(count, 3)
    dc = (rgb - 0.5) / SH_C0
    features = torch.cat([dc.reshape(count, 1, 3), torch.zeros(count, 15, 3)], dim=1)

    # 注意：不能用 tempfile.TemporaryDirectory —— 沙箱给它的目录 ACL 不允许写入/删除。
    # 和其他测试一致，用工作区内的固定子目录。
    tmp = ROOT / ".buildtmp" / "ply-layout-test"
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    if True:
        out = tmp / "t.ply"
        written = train_gsplat.save_ply(
            out,
            means=means,
            scales_log=scales_log,
            quats=quats,
            opacities_logit=opac_logit,
            features=features,
        )
        check(written == count, f"save_ply 返回数量 {written}（期望 {count}）")

        names, n, dt, body = read_ply_strict(out)
        check(n == count, f"header 声明顶点 {n}（期望 {count}）")
        check(len(names) == 62, f"属性数 {len(names)}（期望 62 = 6+3+45+1+3+4）")
        check(
            len(body) == n * dt.itemsize,
            f"数据字节 {len(body)} == {n}×{dt.itemsize} = {n*dt.itemsize}",
        )
        arr = np.frombuffer(body, dtype=dt)

        # --- 位置 ---
        for k, (label, want_lo, want_hi) in enumerate(
            (("x", -2.0, 2.0), ("y", 10.0, 12.0), ("z", -50.0, -48.0))
        ):
            got = arr[label].astype(np.float64)
            got_sorted = np.sort(got)
            want_sorted = np.sort(means[:, k].numpy().astype(np.float64))
            # 必须是同一组数（错位会立刻破坏：x 列会混入 y/z 的值）
            same = np.allclose(got_sorted, want_sorted, atol=1e-4)
            check(
                same and got.min() >= want_lo - 1e-3 and got.max() <= want_hi + 1e-3,
                f"位置 {label} 与输入一致且在 [{want_lo}, {want_hi}]（实际 {got.min():.3f}~{got.max():.3f}）",
            )

        # --- 尺度：按标准约定写的是**线性**值 ---
        for k in range(3):
            v = arr[f"scale_{k}"].astype(np.float64)
            ok = np.allclose(np.sort(v), np.sort(scale_lin[:, k].numpy().astype(np.float64)), atol=1e-5)
            check(ok, f"scale_{k} 是线性尺度（{v.min():.5f}~{v.max():.5f}，期望 0.01~0.05）")
        check(
            float(arr["scale_0"].max()) < 0.051,
            f"scale_0 最大值 {float(arr['scale_0'].max()):.5f} < 0.051（没有被 exp 两次）",
        )

        # --- 四元数 ---
        q = np.stack([arr[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
        check(np.all(np.abs(q) <= 1.0 + 1e-5), f"四元数分量都在 [-1,1]（max |q| = {np.abs(q).max():.4f}）")
        norms = np.linalg.norm(q, axis=1)
        check(np.allclose(norms, 1.0, atol=1e-4), f"四元数近似单位化（范数 {norms.min():.5f}~{norms.max():.5f}）")

        # --- 不透明度：必须是 logit ---
        op = arr["opacity"].astype(np.float64)
        vals = set(np.round(np.unique(op), 3).tolist())
        check(
            vals <= {-4.6, -2.2} or all(abs(v + 4.6) < 1e-3 or abs(v + 2.2) < 1e-3 for v in vals),
            f"opacity 是原始 logit（取值集合 {sorted(vals)[:4]}，期望含 -4.6/-2.2）",
        )

        # --- 颜色直流 ---
        for k in range(3):
            v = arr[f"f_dc_{k}"].astype(np.float64)
            want = dc[:, k].numpy().astype(np.float64)
            check(
                np.allclose(np.sort(v), np.sort(want), atol=1e-4),
                f"f_dc_{k} 与 (rgb-0.5)/SH_C0 一致（{v.min():.3f}~{v.max():.3f}）",
            )

        # --- 法向占位必须是 0 ---
        for label in ("nx", "ny", "nz"):
            check(float(np.abs(arr[label]).max()) < 1e-6, f"法向占位 {label} 全为 0")

        # --- 关键：各字段统计量不能雷同（错位的指纹）---
        stats = {}
        for label in ("x", "y", "z", "f_dc_0", "opacity", "scale_0", "rot_0"):
            v = arr[label].astype(np.float64)
            stats[label] = (float(v.min()), float(v.max()))
        spans = {k: v[1] - v[0] for k, v in stats.items()}
        distinct = len({round(s, 3) for s in spans.values()})
        check(
            distinct >= 5,
            "各字段跨度互不相同（雷同=错位指纹）："
            + ", ".join(f"{k}={s:.2f}" for k, s in spans.items()),
        )

        # --- 反向验证：故意构造块主序，确认真正的检测手段（列内容比对）能抓到 ---
        #
        # 注意：不能用「字段跨度是否雷同」当判据 —— 当各轴取值范围本来就不同时，
        # 块主序的跨度也会各不相同（这里就是 7 个不同值），那个启发式会失效。
        # 真正可靠的是**列内容比对**：记录主序下 x 列恰好等于输入的 x 数组；
        # 块主序下 x 列会混进 y/z 的值。
        blocks = [
            means.numpy().astype("<f4"),
            np.zeros((count, 3), np.float32),
            dc.numpy().astype("<f4"),
            np.zeros((count, 45), np.float32),
            opac_logit.numpy().reshape(count, 1).astype("<f4"),
            scale_lin.numpy().astype("<f4"),
            quats.numpy().astype("<f4"),
        ]
        bad = np.concatenate([b.reshape(-1) for b in blocks]).astype("<f4").tobytes()
        bad_arr = np.frombuffer(bad, dtype=dt)
        bad_x = bad_arr["x"].astype(np.float64)
        want_x = means[:, 0].numpy().astype(np.float64)
        bad_matches = np.allclose(np.sort(bad_x), np.sort(want_x), atol=1e-4)
        good_matches = np.allclose(
            np.sort(arr["x"].astype(np.float64)), np.sort(want_x), atol=1e-4
        )
        check(
            good_matches and not bad_matches,
            "自检有效：列内容比对能区分记录主序（通过）与块主序（不通过）",
        )
        # 块主序下 x 列会被 y/z（量纲差 25 倍）污染，跨度必然变得很大
        check(
            float(bad_x.max() - bad_x.min()) > 10.0,
            f"自检有效：块主序的 x 列跨度 {float(bad_x.max()-bad_x.min()):.2f} > 10（被 y/z 污染）",
        )

    print()
    print(f"结果：{passed} 通过 / {failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
