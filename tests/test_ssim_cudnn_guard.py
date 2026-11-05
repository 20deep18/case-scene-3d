#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回归：SSIM 的卷积**不许**默认走 cuDNN（`cudnn_graph64_9.dll` 原生崩溃）。

事故记录（2026-10-06）
--------------------
用户在 `pipeline/train_gpu.py` 上看到：

    python.exe 中处有未经处理的异常: 0xC0000005: 写入位置 0x0 时发生访问冲突
        (cudnn_graph64_9.dll)

现场证据（都已核对）：
  · 崩溃进程 PID 12264 = `python.exe pipeline/train_gpu.py ... --iterations 60`，
    CPU 卡在 3.03s 不再推进；
  · Visual Studio 的窗口标题是 `python (正在调试) - Microsoft Visual Studio`
    —— JIT 调试器挂上去了，所以该进程**杀不掉**（Access denied）；
  · devenv 自己随后也崩了两次（19:41:30 / 19:41:36，有 .dmp）；
  · 它占着 4.6 GB 提交内存，直接把整机提交推到 38.3/40.6 GB，
    之后连 `import torch` 都报 `WinError 1455 页面文件太小`。

根因面：两个训练器里**只有 SSIM 的 5 次分组卷积**会走 cuDNN
（高斯泼溅光栅化是本项目自研实现）。cuDNN 9 的 graph 引擎会为分组卷积现场
挑/编译执行引擎，在 sm_120 + 驱动 610.62 这套很新的组合上不稳定。

修法：把 SSIM 的卷积放进 `torch.backends.cudnn.flags(enabled=_SSIM_USE_CUDNN)`，
默认 `enabled=False`（`set DSH_SSIM_CUDNN=1` 可开回来做对照）。
这样那段代码**根本不会被调用**，属于"移除崩溃面"，不是"降低概率"。

本测试守三件事：
  1. 静态：两个训练器的 conv2d 确实被 cudnn.flags 包住、默认关闭；
  2. 动态（有 GPU 才算）：开/关 cuDNN 的 SSIM 数值必须一致；
  3. 动态：调用之后全局 `torch.backends.cudnn.enabled` 必须**没被改掉**
     （用上下文管理器而不是全局关，避免影响别的代码）。

用法: pyruntime312\\python.exe tests/test_ssim_cudnn_guard.py
退出码 0 通过 / 1 失败（环境不可用时动态部分会明确报 SKIP，不算失败）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "pipeline", ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

passed = 0
failed = 0
skipped = 0


def check(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [OK]   {msg}")
    else:
        failed += 1
        print(f"  [FAIL] {msg}")


def skip(msg):
    global skipped
    skipped += 1
    print(f"  [SKIP] {msg}")


def main() -> int:
    global passed, failed
    print("=" * 72)
    print("SSIM 卷积的 cuDNN 护栏（cudnn_graph64_9.dll 崩溃不再复发）")
    print("=" * 72)

    src_gpu = (ROOT / "pipeline" / "train_gpu.py").read_text(encoding="utf-8")
    src_gs = (ROOT / "pipeline" / "train_gsplat.py").read_text(encoding="utf-8")

    # ---- 1) 静态：默认关、且确实包住了 conv2d ------------------------------
    print("\n[1] 静态检查：两个训练器的 SSIM 卷积默认不走 cuDNN")
    for name, src in (("train_gpu.py", src_gpu), ("train_gsplat.py", src_gs)):
        check('_SSIM_USE_CUDNN = os.environ.get("DSH_SSIM_CUDNN", "0") == "1"' in src,
              f"{name}：默认关闭，且可用 DSH_SSIM_CUDNN=1 打开（对照实验）")
        check("torch.backends.cudnn.flags(enabled=_SSIM_USE_CUDNN)" in src,
              f"{name}：卷积被 cudnn.flags 上下文管理器包住")
        # conv2d 必须在 with 块里：取 with 之后的缩进块，确认 conv2d 在其中
        m = re.search(r"with torch\.backends\.cudnn\.flags\(enabled=_SSIM_USE_CUDNN\):\n((?:\s+.*\n)+)", src)
        check(bool(m) and m.group(1).count("conv2d") >= 5,
              f"{name}：5 次 conv2d 全都在 with 块内（实际 {m.group(1).count('conv2d') if m else 0} 次）")
    check("cudnn_graph64_9.dll" in src_gpu and "cudnn_graph64_9.dll" in src_gs,
          "两个文件都写明了崩溃模块，便于事后追查")

    # ---- 2) 动态：数值等价 + 不污染全局开关 --------------------------------
    print("\n[2] 动态检查（需要能 import torch 且 CUDA 可用）")
    try:
        import os  # noqa: PLC0415

        import torch  # noqa: PLC0415

        import train_gpu as tg  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        skip(f"环境不可用，跳过动态检查：{type(exc).__name__}: {exc}")
        print("         （这正是崩溃事故的余波：卡住的进程占满提交内存时，import torch 都会失败）")
        print("\n" + "=" * 72)
        print(f"结果：{passed} 通过 / {failed} 失败 / {skipped} 跳过")
        print("=" * 72)
        return 1 if failed else 0

    if not torch.cuda.is_available():
        skip("没有可用 CUDA，跳过动态检查")
    else:
        dev = torch.device("cuda")
        torch.manual_seed(0)
        a = torch.rand(1, 3, 96, 64, device=dev)
        b = torch.rand(1, 3, 96, 64, device=dev)

        # 默认（关闭 cuDNN）
        before = torch.backends.cudnn.enabled
        tg._SSIM_USE_CUDNN = False
        off = float(tg.ssim_loss(a, b))
        after = torch.backends.cudnn.enabled
        check(before == after,
              f"调用后全局 cudnn.enabled 未被改动（{before} → {after}）")

        # 打开 cuDNN 做对照
        tg._SSIM_USE_CUDNN = True
        try:
            on = float(tg.ssim_loss(a, b))
            diff = abs(on - off)
            check(diff < 1e-4,
                  f"开/关 cuDNN 的 SSIM 数值一致（on={on:.6f} off={off:.6f} 差={diff:.2e}）")
        except Exception as exc:  # noqa: BLE001
            # 若恰好复现了崩溃，那就是"关掉 cuDNN 是对的"最直接的证据
            check(False, f"打开 cuDNN 时崩了（这正说明默认关闭是对的）：{type(exc).__name__}: {exc}")
        finally:
            tg._SSIM_USE_CUDNN = False

    print("\n" + "=" * 72)
    print(f"结果：{passed} 通过 / {failed} 失败 / {skipped} 跳过")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
