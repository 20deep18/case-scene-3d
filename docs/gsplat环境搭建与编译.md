# gsplat 环境搭建与编译记录（方案A / M1）

> 本机实测：**gsplat 1.5.3** 在 **RTX 5060 Laptop (sm_120) + CUDA 12.8 + PyTorch 2.7.1+cu128 + MSVC v143**
> 上完成源码编译，CUDA 内核实际运行成功。
> 编译耗时 **103 秒**。

---

## 一、结论速查

| 里程碑 | 状态 | 证据 |
| --- | --- | --- |
| **M1 环境就绪** | ✅ | `gsplat: CUDA extension has been set up successfully in 103.08 seconds` |
| **M2 数据就绪** | ✅ | COLMAP 已能产出 `sparse/0` 三件套（见 [COLMAP安装与验证.md](COLMAP安装与验证.md)） |
| **M3 训练通过** | ✅ | 7000 步训练完成，留出视角 PSNR 19.78 dB / SSIM 0.687（详见 [gsplat训练结果.md](gsplat训练结果.md)） |
| **M4 交付通过** | ✅ | `.ply → .splat →` 浏览器渲染链路早已验证 |

**关键命令**（必须在 `tools\msvc-env.bat` 环境里跑）：

```bat
call tools\msvc-env.bat
python -c "from gsplat.cuda._backend import _C; print('gsplat CUDA ready')"
```

---

## 二、蓝图里必须修正的两处（会导致必然失败）

蓝图文档写的是「CUDA 12.1 + cu121 PyTorch」，这在 **RTX 50 系（sm_120）上不可能工作**：

| 蓝图写的 | 问题 | 实际应该用 |
| --- | --- | --- |
| CUDA Toolkit **12.1** | CUDA 12.1 **不支持 sm_120**，`nvcc --list-gpu-arch` 里根本没有 `compute_120` | **CUDA 12.8**（本机已装，`compute_120` 在列） |
| PyTorch **cu121** | cu121 轮子只编到 `sm_90`，运行必报 `no kernel image is available` | **cu128**（本机 torch 2.7.1+cu128，arch 列表含 `sm_120`） |

> 这两条不是"建议"，是**硬性不兼容**。蓝图是按通用/较老硬件写的。

---

## 三、预编译轮子为什么用不了（已用 cuobjdump 验证）

官方 GitHub Releases 的 Windows 轮子最高只到 `pt24cu124 / cp310`。本工作区里就有一个
`wheels/gsplat-1.5.3+pt24cu124-cp310-cp310-win_amd64.whl`，用 CUDA 自带的 `cuobjdump` 查看它内含的架构：

```
ELF file 1: csrc.1.sm_70.cubin
ELF file 5: csrc.5.sm_90.cubin
...
出现的 sm 架构: 70, 75, 80, 86, 90
No PTX file found to extract        ← 连 JIT 回退都没有
```

**结论：只有 `sm_70`~`sm_90`，没有 `sm_120`，也没有 PTX。**
在 RTX 5060（sm_120）上加载必然报 `no kernel image is available for execution on the device`。

PyPI 上的 `gsplat-1.5.3-py3-none-any.whl` 是**纯源码包**（6.2 MB，含 20 个 `.cu` + 216 个 `.cpp`，
以及 glm 头文件 1662 个），首次 `import` 时由 PyTorch JIT 现场编译。

**所以：唯一出路是源码编译，必须要有 MSVC。**

---

## 四、编译过程中踩到的 6 个坑（全部已解决）

### 坑 1：CUDA 12.8 不接受 VS 2026 的 MSVC 14.51

CUDA 12.8 的 `crt/host_config.h` 只接受到 VS 2022（`_MSC_VER < 1950`）：

```
fatal error C1189: #error:  -- unsupported Microsoft Visual Studio version!
Only the versions between 2017 and 2022 (inclusive) are supported!
```

加 `-allow-unsupported-compiler` 绕过版本检查后更糟：

```
nvcc error : 'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION)
```

**解决**：VS 2026 同时装了 **v143 工具集（MSVC 14.44）**，用它就正常：

```bat
call "...\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
```

> 注意：NVIDIA 官网上那张"MSVC 195x / Visual Studio 2026 ✅"的支持表是 **CUDA 13** 的文档，
> 不适用于 CUDA 12.8。这个坑靠**实际试编译**才暴露出来，不能只看文档。

### 坑 2：嵌入版 Python 没有开发文件

本机 `pyruntime312` 是 python.org 的 **embeddable** 包，只有 `python312.dll` 和 `.pyd`，
**没有 `Include/`**，也**没有 `libs/`**。于是：

```
nvcc: fatal error C1083: 无法打开包括文件: "Python.h"
link: fatal error LNK1104: 无法打开文件 "python312.lib"
```

**解决**：`tools/fetch-python-headers.mjs` 从 python.org 的 nuget 包取回：

- `tools/include/` → `pyruntime312/Include/`（76 个头文件，含 `internal/`）
- `tools/libs/python312.lib` → `pyruntime312/libs/`

### 坑 3：构建目录与临时目录不可写

- `%LOCALAPPDATA%\torch_extensions` → `PermissionError: [WinError 5] 拒绝访问`
- 沙箱给的 `TMP` 是**每次调用都不同**的受限目录 → `nvcc fatal: Could not open output file`

**解决**：在 `tools/msvc-env.bat` 里把两个目录都指到工作区内：

```bat
set "TORCH_EXTENSIONS_DIR=...\.buildtmp\torch_ext"
set "TMP=...\.buildtmp"
set "TEMP=...\.buildtmp"
```

### 坑 4：缺 `packaging` / `rich`，而这个 Python 没有 pip

```
ModuleNotFoundError: No module named 'packaging'
```

**解决**：`tools/fetch-wheels-direct.mjs` —— 查 PyPI JSON API → 下载 `py3-none-any` 轮子 →
用 Node 自己解 zip（内联实现，只依赖 `zlib.inflateRawSync`）。装好
`packaging`、`rich`（及其依赖 `mdurl`、`markdown-it-py`、`pygments`）。

### 坑 5：gsplat 1.5.3 与 torch 2.7.1 的 `_jit_compile` 签名不匹配

```
TypeError: _jit_compile() missing 2 required positional arguments: 'verbose' and 'with_sycl'
```

torch 2.7.1 在 `extra_cuda_cflags` 和 `extra_ldflags` **之间**插入了 `extra_sycl_cflags`：

```python
_jit_compile(name, sources, extra_cflags, extra_cuda_cflags, extra_sycl_cflags,
             extra_ldflags, extra_include_paths, build_directory, verbose,
             with_cuda, with_sycl, is_python_module, is_standalone, keep_intermediates=True)
```

gsplat 按老顺序位置传参，`verbose` 落到 `extra_sycl_cflags` 上；它的重试分支同样错位。

**解决**：改成 `inspect.signature()` 读真实签名 + 关键字传参（见 `_backend.py` 的 `load_extension`）。
新旧 torch 都兼容。

### 坑 6：gsplat 给 MSVC 传了 GCC 风格参数

```
cl: 命令行 error D8021 :无效的数值参数"/Wno-attributes"
```

`_backend.py` 里 `extra_cflags = [opt_level, "-Wno-attributes"]` 是无条件的，
于是**所有 `.cpp`（Null/Adam/Intersect/Projection/ext…）全部编译失败**。

**解决**：改成 `if os.name != "nt": extra_cflags.append("-Wno-attributes")`。

---

## 五、最终可用的环境

```
MSVC      : VS 2026 Community，强制使用 v143 工具集（MSVC 14.44.35207）
            ← 14.51 与 CUDA 12.8 不兼容（cudafe++ 崩溃）
CUDA      : 12.8 (V12.8.61)，支持 compute_120
PyTorch   : 2.7.1+cu128，arch 列表含 sm_120
Ninja     : 1.13.2（VS 2026 自带，无需另装）
CMake     : 4.3.1（VS 2026 自带）
Python    : 3.12.10（embeddable，已补 Include/ 与 libs/）
gsplat    : 1.5.3（源码 JIT 编译成功）
```

**环境入口脚本**：`tools/msvc-env.bat`

```bat
call tools\msvc-env.bat
REM 这会设置好 v143 / CUDA 12.8 / ninja / TORCH_CUDA_ARCH_LIST=12.0
REM MAX_JOBS=6 / TORCH_EXTENSIONS_DIR / TMP
```

---

## 六、已验证的自检脚本

| 脚本 | 作用 |
| --- | --- |
| `tools/msvc-env.bat` | 进入可编译 CUDA 的环境（一切的前提） |
| `tools/check_jit_cuda.py` | 编译一个最小 CUDA 扩展并在 GPU 上运行，验证整条 JIT 链路 |
| `tools/check_gsplat_cuda.py` | 触发 gsplat 全部内核编译，并实际调用 `quat_scale_to_covar_preci` |
| `tools/fetch-python-headers.mjs` | 补齐 embeddable Python 的 `Include/` 与 `libs/` |
| `tools/fetch-wheels-direct.mjs` | 无 pip 情况下从 PyPI 直接取纯 Python 包 |
| `tools/check_gsplat_env.py` | 报告 gsplat 依赖齐备情况 |

---

## 七、后续

- **M3 正式训练**：接续训练器（官方 `simple_trainer.py` 依赖较重：`torch==2.9.1`、`pycolmap`、
  `fused-ssim`、`fused-bilagrid`、`vis` 等，多数需要另行编译）。
- **显存策略**：8 GB 上限，先用 `--downsample_factor 2` 起步，确认无 OOM 再逐步提高。
- **`_backend.py` 的两处本地补丁**在重装 gsplat 后会丢失，需要重新应用
  （见第四节坑 5、坑 6）。
