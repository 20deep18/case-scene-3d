#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查 gsplat 在本环境下的可安装性与依赖情况。"""
import importlib
import sys

print("python:", sys.version.split()[0])
print()
print("=== gsplat 运行期需要的模块 ===")
need = {
    "torch": "核心",
    "rich": "gsplat/cuda/_backend.py 的状态条输出（必需）",
    "packaging": "gsplat/cuda/_backend.py 版本比较（必需）",
    "numpy": "数值计算",
}
missing = []
for m, why in need.items():
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"  [有] {m:12s} {ver:12s} {why}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [缺] {m:12s} {'':12s} {why}   -> {type(exc).__name__}")
        missing.append(m)

print()
print("=== gsplat 是否已安装 ===")
try:
    import gsplat  # noqa: F401

    print("  gsplat 已可导入:", getattr(gsplat, "__version__", "?"))
except Exception as exc:  # noqa: BLE001
    print("  gsplat 未安装:", type(exc).__name__, exc)

print()
print("缺失模块:", ", ".join(missing) if missing else "无")
