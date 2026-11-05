"""/test_train_gpu.py —— GPU 训练链路自测

覆盖：
  1. 用 tools/make_test_dataset.py 生成一个小型合成 COLMAP 数据集
  2. 跑 pipeline/train_gpu.py 少量迭代
  3. 校验导出 PLY 的属性齐全、数值有限、包围盒合理
  4. 用 pipeline/to_splat.py 转成 .splat，校验 32 字节/高斯与元数据

没有 CUDA 时自动跳过（返回码 0 并说明原因）。

运行： pyruntime312\python.exe tests\test_train_gpu.py
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
WORK = ROOT / "output" / "_test_train"

passed = 0
failed = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [OK]   {name} {extra}")
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}")


def run(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def read_ply_header(path: Path):
    """返回 (顶点数, 属性名列表, 数据起始字节偏移)。

    注意：必须把 end_header 之后的那个换行符也跳过，否则数据整体错位一字节，
    读出来的浮点数会全是 nan（本项目踩过这个坑，所以这里显式处理）。
    """
    with open(path, "rb") as fh:
        raw = fh.read(65536)
    marker = raw.index(b"end_header")
    offset = raw.index(b"\n", marker) + 1
    header = raw[:offset].decode("ascii", "replace")
    props = []
    count = 0
    for line in header.splitlines():
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
        elif line.startswith("property float"):
            props.append(line.split()[-1])
    return count, props, offset


def main() -> int:
    print("=" * 70)
    print("GPU 训练链路自测")
    print("=" * 70)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("未安装 PyTorch，跳过本测试")
        return 0
    import torch

    if not torch.cuda.is_available():
        print("没有可用的 CUDA 设备，跳过本测试")
        return 0
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)

    data_dir = WORK / "scene"
    out_dir = WORK / "trained"

    # ---- 1. 生成小型合成数据集（图像小、点少，快速验证）
    print("\n[1/4] 生成合成 COLMAP 数据集 …")
    r = run([PY, "tools/make_test_dataset.py", "--out", str(data_dir), "--views", "8", "--width", "320", "--height", "180", "--points", "30000"])
    check("数据集生成成功", r.returncode == 0, f"(exit={r.returncode})")
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        return 1
    check("images 目录非空", len(list((data_dir / "images").glob("*.png"))) == 8)
    check("sparse/0/cameras.bin 存在", (data_dir / "sparse" / "0" / "cameras.bin").exists())
    check("sparse/0/images.bin 存在", (data_dir / "sparse" / "0" / "images.bin").exists())
    check("sparse/0/points3D.bin 存在", (data_dir / "sparse" / "0" / "points3D.bin").exists())

    # ---- 2. 训练
    print("\n[2/4] 跑 GPU 训练（60 次迭代）…")
    r = run(
        [
            PY,
            "pipeline/train_gpu.py",
            "--data",
            str(data_dir),
            "--out",
            str(out_dir),
            "--iterations",
            "60",
            "--log-every",
            "30",
            "--downscale",
            "1",
            "--max-gaussians",
            "20000",
        ]
    )
    check("训练进程正常退出", r.returncode == 0, f"(exit={r.returncode})")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        return 1

    summary_path = out_dir / "train_summary.json"
    check("train_summary.json 已生成", summary_path.exists())
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    check("摘要含迭代历史", isinstance(summary.get("history"), list) and len(summary["history"]) > 0)
    if summary.get("history"):
        first = summary["history"][0]["loss"]
        last = summary["history"][-1]["loss"]
        check("损失下降", last < first, f"{first:.4f} -> {last:.4f}")

    # ---- 3. 校验 PLY
    print("\n[3/4] 校验导出的 3DGS PLY …")
    ply = out_dir / "point_cloud" / "iteration_60" / "point_cloud.ply"
    check("PLY 已生成", ply.exists())
    if not ply.exists():
        return 1
    count, props, offset = read_ply_header(ply)
    check("PLY 顶点数 > 0", count > 0, f"{count}")
    need = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    missing = need - set(props)
    check("PLY 属性齐全（标准 3DGS 格式）", not missing, f"缺少 {missing}" if missing else "")
    check("属性数量正确", len(props) == len(need), f"{len(props)} 个属性")

    # 逐条读取少量数据校验数值有限、包围盒合理
    stride = len(props) * 4
    with open(ply, "rb") as fh:
        fh.seek(offset)
        raw = fh.read(min(count, 5000) * stride)
    idx = {p: i for i, p in enumerate(props)}
    finite_ok = True
    mins = [math.inf] * 3
    maxs = [-math.inf] * 3
    for vi in range(len(raw) // stride):
        vals = struct.unpack_from("<" + "f" * len(props), raw, vi * stride)
        for k, name in enumerate(("x", "y", "z")):
            v = vals[idx[name]]
            if not math.isfinite(v):
                finite_ok = False
            mins[k] = min(mins[k], v)
            maxs[k] = max(maxs[k], v)
    check("位置坐标全部有限", finite_ok)
    extent = max(maxs[i] - mins[i] for i in range(3))
    check("包围盒尺度合理（< 100m）", extent < 100, f"最大跨度 {extent:.2f}m")

    # ---- 4. 转 .splat
    print("\n[4/4] 转成 Web 用的 .splat …")
    splat = WORK / "scene.splat"
    meta = WORK / "scene.json"
    r = run([PY, "pipeline/to_splat.py", "--ply", str(ply), "--out", str(splat), "--json", str(meta)])
    check("to_splat 正常退出", r.returncode == 0, f"(exit={r.returncode})")
    if splat.exists():
        size = splat.stat().st_size
        check("字节数 = 高斯数 × 32", size % 32 == 0 and size > 0, f"{size} 字节 = {size // 32} 个高斯")
        info = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        check("元数据含 count", "count" in info, f"count={info.get('count')}")
        check("元数据含 bounds", "bounds" in info)
        check("元数据标注单位为米", info.get("units") == "meters")
        print(f"        .splat 文件：{splat}  ({size / 1024:.1f} KB)")

    print(f"\n结果：{passed} 通过 / {failed} 失败")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
