#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 PyTorch JIT 能否在本机编译 CUDA 扩展（gsplat 走的就是这条路）。
必须先设置好 v143 工具集环境（见 tools/msvc-env.ps1）。
"""
import os
import sys
import time

import torch

print("=" * 70)
print("PyTorch JIT CUDA 扩展编译自检")
print("=" * 70)
print("python      :", sys.version.split()[0])
print("torch       :", torch.__version__)
print("cuda 可用   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("算力        : sm_%d%d" % cap)
print("arch 列表   :", torch.cuda.get_arch_list())

import torch.utils.cpp_extension as ce

print("CUDA_HOME   :", ce.CUDA_HOME)
print("cl.exe      :", __import__("shutil").which("cl") or "(PATH 里没有 cl)")

# 关键：显式告诉 torch 只编译本机算力，避免默认编一堆架构（慢且易失败）
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "%d.%d" % cap)
print("TORCH_CUDA_ARCH_LIST =", os.environ.get("TORCH_CUDA_ARCH_LIST"))

src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_jit_probe_src")
os.makedirs(src, exist_ok=True)

with open(os.path.join(src, "probe.cu"), "w", encoding="utf-8") as f:
    f.write(
        r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scale_kernel(float* x, float k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= k;
}

torch::Tensor scale(torch::Tensor x, double k) {
    auto out = x.clone();
    int n = (int)out.numel();
    if (n == 0) return out;
    float* p = out.data_ptr<float>();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(p, (float)k, n);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kernel 启动失败: ", cudaGetErrorString(err));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale", &scale, "multiply by scalar");
}
"""
    )

build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_jit_probe_build")
os.makedirs(build_dir, exist_ok=True)
print("build 目录  :", build_dir)

t0 = time.time()
try:
    ext = ce.load(
        name="dsh_jit_probe",
        sources=[os.path.join(src, "probe.cu")],
        build_directory=build_dir,
        verbose=True,
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )
except Exception as exc:  # noqa: BLE001
    print("\n" + "=" * 70)
    print("✘ JIT 编译失败：", type(exc).__name__)
    print(str(exc)[-4000:])
    sys.exit(1)

elapsed = time.time() - t0
print("\n编译完成，耗时 %.1f 秒" % elapsed)

if torch.cuda.is_available():
    x = torch.ones(4096, device="cuda") * 2.0
    y = ext.scale(x, 3.0)
    torch.cuda.synchronize()
    ok = bool(torch.allclose(y, torch.full_like(y, 6.0)))
    print("GPU 运行结果:", y[0].item(), "（期望 6.0）", "✔" if ok else "✘")
    if not ok:
        sys.exit(2)
else:
    print("（没有可用 GPU，只验证了编译）")

print("\n✔ PyTorch JIT CUDA 扩展编译链路可用")
