#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""打印 torch 2.7.1 的 _jit_compile / load 签名，用于修正 gsplat 的调用。"""
import inspect

import torch.utils.cpp_extension as ce

print("torch:", __import__("torch").__version__)
print()
for name in ("_jit_compile", "load", "_get_build_directory", "_import_module_from_library"):
    fn = getattr(ce, name, None)
    if fn is None:
        print(f"{name}: (不存在)")
        continue
    try:
        print(f"{name}{inspect.signature(fn)}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: 无法取签名 {exc}")
    print()

print("SUBPROCESS_DECODE_ARGS =", getattr(ce, "SUBPROCESS_DECODE_ARGS", "(无)"))
