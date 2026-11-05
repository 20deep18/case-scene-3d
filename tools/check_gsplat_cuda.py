#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
尝试导入 gsplat 的 CUDA 后端 —— 这会触发 PyTorch JIT 编译全部 CUDA 内核。
必须先处于 msvc-env.bat 设置好的环境（v143 + ninja + TMP 可写）中运行。
"""
import os
import sys
import time
import traceback

print("=" * 72)
print("gsplat CUDA 后端编译 / 导入自检")
print("=" * 72)
print("python   :", sys.version.split()[0])
print("TMP      :", os.environ.get("TMP"))
print("ARCH_LIST:", os.environ.get("TORCH_CUDA_ARCH_LIST"))
print("MAX_JOBS :", os.environ.get("MAX_JOBS"))

t0 = time.time()
try:
    import gsplat

    print("gsplat 版本:", getattr(gsplat, "__version__", "?"))
    print("gsplat 路径:", os.path.dirname(gsplat.__file__))
except Exception:
    print("导入 gsplat 失败：")
    traceback.print_exc()
    sys.exit(1)

print("\n开始触发 CUDA 后端编译（首次会编译全部 .cu，可能数分钟到十几分钟）…")
try:
    from gsplat.cuda._backend import _C  # noqa: F401

    print("\n✔ gsplat CUDA 后端已就绪")
except Exception:
    print("\n✘ gsplat CUDA 后端编译失败：")
    traceback.print_exc()
    sys.exit(2)

elapsed = time.time() - t0
print(f"\n总耗时 {elapsed:.1f} 秒")

# 真正调用一次内核，确认在 GPU 上能跑
try:
    import torch
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci

    dev = torch.device("cuda")
    n = 8
    quats = torch.zeros(n, 4, device=dev)
    quats[:, 0] = 1.0  # 单位四元数
    scales = torch.ones(n, 3, device=dev) * 0.1
    covars, precis = quat_scale_to_covar_preci(quats, scales, True, True, True)
    torch.cuda.synchronize()
    print("quat_scale_to_covar_preci -> covars", tuple(covars.shape), "precis", tuple(precis.shape))
    print("covars[0]:", covars[0].tolist())
    print("\n✔ gsplat CUDA 内核在 RTX 5060 上实际运行成功")
except Exception:
    print("\n✘ gsplat 内核调用失败：")
    traceback.print_exc()
    sys.exit(3)
