"""pipeline/setup_env.py —— 一条命令式的环境自检与安装引导器（Windows 优先）。

命令行：
    python pipeline/setup_env.py [--check] [--install-torch]
        [--mirror https://pypi.tuna.tsinghua.edu.cn/simple] [--dry-run] [--yes] [--python <解释器>]

用法说明：
    --check            只做环境自检，打印一份带 ✅/❌/⚠️ 的表格与下一步命令，然后退出；
    （默认）           打印自检表 + 安装计划；如果 wheels/ 里有匹配目标解释器的离线轮子
                       且目标解释器带 pip，就直接离线安装这些轻量依赖（几秒钟、不联网）；
    --install-torch    额外把 torch / torchvision 的安装命令加入计划；配合 --yes 才真正下载安装
                       （约 2.5GB+，本机必须用 cu128 轮子，见下）；
    --mirror           指定 pip 镜像源（默认清华源）；
    --python           指定要安装到哪个解释器（默认优先仓库内 venv/env，其次当前解释器）；
    --dry-run          只打印，不执行任何命令。

本机实测约束（重要）：
  * 显卡是 RTX 5060 Laptop（Blackwell 架构，算力 sm_120）。
    必须使用 CUDA ≥ 12.8 的 PyTorch，即 torch ≥ 2.7 的 cu128 版本，否则会出现
    「no kernel image is available for execution on the device」或 CUDA 不可用。
  * gsplat 官方只为 Linux 提供预编译轮子；Windows 上需要源码编译
    （需要 MSVC 生成工具 + CUDA Toolkit ≥ 12.8），或者改用手机 App 的替代路线。

退出码：0 环境就绪；1 有缺失项需要处理；2 参数错误；3 关键步骤执行失败。
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_PIPELINE_DIR = Path(__file__).resolve().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

import config  # noqa: E402

DEFAULT_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

#: 离线安装时优先使用的轻量依赖（wheels/ 里有什么装什么）
LIGHT_PACKAGES = ["numpy", "pillow", "imageio", "imageio-ffmpeg", "opencv-python", "plyfile", "tqdm"]

#: 依赖 wheel 文件名前缀 → 包名（用于筛选离线轮子）
WHEEL_PREFIXES = {
    "numpy": "numpy",
    "pillow": "Pillow",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "imageio": "imageio",
    "opencv_python": "opencv-python",
    "plyfile": "plyfile",
    "tqdm": "tqdm",
}


def status_mark(level: str) -> str:
    return {"ok": "✅", "bad": "❌", "warn": "⚠️", "info": "ℹ️"}.get(level, "•")


# ---------------------------------------------------------------------------
# 解释器探测
# ---------------------------------------------------------------------------


def probe_python(exe: str) -> tuple[int, int]:
    """探测指定解释器的版本，失败返回 (0, 0)。"""
    try:
        result = config.run_command([exe, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"], echo=False, timeout=60)
    except Exception:  # noqa: BLE001
        return (0, 0)
    if result.returncode != 0:
        return (0, 0)
    text = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    parts = text.split(".")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return (0, 0)
    return int(parts[0]), int(parts[1])


def has_pip(exe: str) -> bool:
    """目标解释器是否带 pip。"""
    result = config.run_command([exe, "-m", "pip", "--version"], echo=False, timeout=120)
    return result.returncode == 0


def pick_target_python(explicit: str | None) -> tuple[str, str]:
    """挑选安装目标解释器。

    优先挑「版本与 wheels/ 里轮子标签一致」的解释器，避免出现
    「解释器是 3.13 但离线轮子是 cp310」这种装不上的情况。
    """
    if explicit:
        return explicit, "命令行 --python 指定"

    wanted: tuple[int, int] | None = None
    tags = wheel_python_tags()
    if tags:
        best = max(tags, key=lambda key: tags[key])
        wanted = (int(best[2]), int(best[3:]))

    candidates: list[tuple[Path, str]] = []
    for name in ("env", "venv", ".venv", "env310", "venv310", ".venv310", "python310"):
        for sub in ("Scripts/python.exe", "bin/python"):
            candidate = config.PROJECT_ROOT / name / sub
            if candidate.is_file():
                candidates.append((candidate, f"仓库内虚拟环境 {name}"))
    if all(Path(sys.executable) != path for path, _ in candidates):
        candidates.append((Path(sys.executable), "当前解释器"))
    if not candidates:
        return sys.executable, "当前解释器"

    if wanted is not None:
        for path, desc in candidates:
            if probe_python(str(path)) == wanted:
                return str(path), f"{desc}（版本与 wheels/ 里的 cp{wanted[0]}{wanted[1]} 轮子匹配）"
        path, desc = candidates[0]
        return str(path), f"{desc}（注意：wheels/ 里主要是 cp{wanted[0]}{wanted[1]} 轮子，可能不匹配）"
    return str(candidates[0][0]), candidates[0][1]


def wheel_python_tags() -> dict[str, int]:
    """统计 wheels/ 里出现的 cpXY 标签及数量（排除 abi3 与纯 Python 轮子）。"""
    counts: dict[str, int] = {}
    for wheel in config.WHEELS_DIR.glob("*.whl"):
        name = wheel.name.lower()
        if "abi3" in name:
            continue
        match = re.search(r"-cp(3\d+)-", name)
        if match:
            key = f"cp{match.group(1)}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def analyze_local_big_wheels(gpus: Sequence[dict[str, Any]]) -> list[CheckRow]:
    """分析 wheels/ 里的 torch / gsplat 离线轮子，并判断与显卡算力是否匹配。"""
    rows: list[CheckRow] = []
    capability = str(gpus[0].get("computeCap") or "") if gpus else ""
    is_blackwell = capability.startswith("12")

    for pattern, label in (("torch-*.whl", "离线 torch 轮子"), ("gsplat-*.whl", "离线 gsplat 轮子")):
        for wheel in sorted(config.WHEELS_DIR.glob(pattern)):
            size = wheel.stat().st_size
            name = wheel.name
            cuda = re.search(r"cu(\d+)", name)
            torch_tag = re.search(r"pt(\d)(\d)", name)
            py_tag = re.search(r"-(cp\d+)-", name)
            details = [py_tag.group(1) if py_tag else "版本未知"]
            if cuda:
                details.append(f"CUDA {cuda.group(1)[:2]}.{cuda.group(1)[2:]}")
            if torch_tag:
                details.append(f"对应 torch {torch_tag.group(1)}.{torch_tag.group(2)}")
            if size == 0:
                rows.append(
                    CheckRow(label, "warn", f"{name}（0 字节，下载未完成）",
                             "该文件是空壳，不能用于安装：请重新下载或删除", required=False)
                )
                continue
            level = "ok"
            advice = f"可用：pip install --no-index \"{wheel}\""
            if cuda and is_blackwell:
                cuda_version = (int(cuda.group(1)[:2]), int(cuda.group(1)[2:]))
                if cuda_version < (12, 8):
                    level = "warn"
                    advice = (
                        f"本机显卡算力 {capability}（sm_120 / Blackwell），而该轮子是 cu{cuda.group(1)}；"
                        f"cu12x（<12.8）通常不含 sm_120 的 SASS 内核，可能出现 "
                        "'no kernel image is available'。优先改用 cu128（torch ≥ 2.7）："
                        f"pip install --index-url {TORCH_CUDA_INDEX} torch torchvision"
                    )
            rows.append(CheckRow(label, level, f"{name}（{size / 1024 / 1024:.0f} MB，{', '.join(details)}）", advice, required=False))
    return rows


def print_local_wheel_section(gpus: Sequence[dict[str, Any]]) -> None:
    """打印 wheels/ 里的大件轮子分析结果。"""
    rows = analyze_local_big_wheels(gpus)
    if not rows:
        return
    print()
    config.log_rule("wheels/ 里的 torch / gsplat 离线轮子")
    print(config.format_table(["状态", "轮子", "说明"], [[status_mark(r.level), r.name, r.value] for r in rows]))
    for row in rows:
        if row.level != "ok":
            print(f"  {status_mark(row.level)} {row.name}：{row.advice}")


# ---------------------------------------------------------------------------
# 环境自检
# ---------------------------------------------------------------------------


@dataclass
class CheckRow:
    """自检表中的一行。"""

    name: str
    level: str  # ok / bad / warn / info
    value: str
    advice: str
    required: bool = True


def _cuda_version_tuple(text: str | None) -> tuple[int, int]:
    if not text:
        return (0, 0)
    parts = str(text).split(".")
    try:
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except (ValueError, IndexError):
        return (0, 0)


def collect_checks(target_python: str) -> list[CheckRow]:
    """收集环境自检结果。"""
    rows: list[CheckRow] = []
    major, minor = probe_python(target_python)
    tags = wheel_python_tags()
    expected: tuple[int, int] | None = None
    if tags:
        best = max(tags, key=lambda key: tags[key])
        expected = (int(best[2]), int(best[3:]))

    current = sys.version_info
    if target_python == sys.executable:
        rows.append(
            CheckRow(
                "Python",
                "ok" if (current.major, current.minor) >= (3, 10) else "bad",
                f"{current.major}.{current.minor}.{current.micro}（当前解释器）",
                "3.10+ 即可；注意 wheels/ 里的 numpy/pillow 是 cp310 轮子，如需离线安装要用 Python 3.10",
                required=True,
            )
        )
    else:
        version_text = f"{major}.{minor}" if major else "无法探测"
        level = "ok" if major else "bad"
        advice = "建议用 Python 3.10 建环境：tools\\dist\\python-3.10.11-amd64.exe 已在本仓库"
        if major and expected and (major, minor) != expected:
            level = "warn"
            advice = (
                f"该解释器是 Python {major}.{minor}，但 wheels/ 里的离线轮子主要是 "
                f"cp{expected[0]}{expected[1]}；离线安装会失败。请先建对应版本的环境："
                f"tools\\dist\\python-{expected[0]}.{expected[1]}*.exe -m venv env"
            )
        rows.append(
            CheckRow("Python（安装目标）", level, f"{version_text} → {target_python}", advice, required=True)
        )

    rows.append(
        CheckRow("numpy", "ok" if config.has_numpy() else "bad", "已安装" if config.has_numpy() else "未安装",
                 "pip install numpy（或离线装 wheels\\numpy-*.whl）", required=True)
    )
    rows.append(
        CheckRow("Pillow / imageio", "ok" if (config.try_import("PIL") or config.try_import("imageio")) else "warn",
                 "可用" if (config.try_import("PIL") or config.try_import("imageio")) else "未安装",
                 "训练阶段读图需要：pip install pillow imageio", required=False)
    )
    rows.append(
        CheckRow("OpenCV", "ok" if config.has_cv2() else "warn", "已安装" if config.has_cv2() else "未安装",
                 "可选：pip install opencv-python（模糊检测走 cv2 会更快）", required=False)
    )

    info = config.torch_env_info()
    if not info["available"]:
        rows.append(CheckRow("PyTorch", "bad", "未安装", f"pip install --index-url {TORCH_CUDA_INDEX} torch torchvision", required=True))
    else:
        cuda = _cuda_version_tuple(info.get("cudaVersion"))
        rows.append(
            CheckRow(
                "PyTorch",
                "ok",
                f"{info['torchVersion']}（CUDA {info.get('cudaVersion') or '无'}）",
                "版本已安装" if cuda >= (12, 8) else f"建议升级到 cu128：pip install --index-url {TORCH_CUDA_INDEX} torch torchvision",
                required=True,
            )
        )

    gpus = config.detect_gpus()
    if gpus:
        for gpu in gpus:
            cap = str(gpu.get("computeCap") or "")
            memory = gpu.get("memoryMB")
            memory_text = f"{memory} MB" if memory else "显存未知"
            rows.append(
                CheckRow(
                    "GPU",
                    "ok",
                    f"{gpu['name']} / {memory_text} / 驱动 {gpu.get('driver')} / 算力 {cap}",
                    "Blackwell（sm_120）必须用 CUDA ≥ 12.8 的 PyTorch" if cap.startswith("12") else "算力信息已获取",
                    required=False,
                )
            )
    else:
        rows.append(CheckRow("GPU", "warn", "未检测到 NVIDIA 显卡（nvidia-smi 不可用）",
                             "可用手机 App（Polycam / Luma AI）导出 PLY 后走 to_splat.py 的替代路线", required=False))

    if info["available"]:
        rows.append(
            CheckRow(
                "CUDA 可用",
                "ok" if info["cudaAvailable"] else "bad",
                f"{'是' if info['cudaAvailable'] else '否'}"
                + (f"（运行时 {info.get('cudaRuntime')}）" if info.get("cudaRuntime") else ""),
                "若为「否」：确认装的是 cu12x 版 torch、驱动 ≥ 527、且没有在用 CPU 版 torch",
                required=True,
            )
        )
        if info["cudaAvailable"] and info.get("capability") == "sm_120":
            cuda = _cuda_version_tuple(info.get("cudaVersion"))
            if cuda < (12, 8):
                rows.append(
                    CheckRow(
                        "sm_120 兼容性",
                        "warn",
                        f"torch 的 CUDA 为 {info.get('cudaVersion')}，低于 12.8",
                        f"RTX 5060 需要 cu128：pip install --index-url {TORCH_CUDA_INDEX} torch torchvision",
                        required=True,
                    )
                )
            else:
                rows.append(CheckRow("sm_120 兼容性", "ok", f"CUDA {info.get('cudaVersion')} ≥ 12.8", "满足 RTX 5060 要求", required=True))

    rows.append(
        CheckRow("gsplat", "ok" if config.has_gsplat() else "bad", "已安装" if config.has_gsplat() else "未安装",
                 "官方只有 Linux 轮子；Windows 需源码编译：pip install git+https://github.com/nerfstudio-project/gsplat.git", required=True)
    )
    rows.append(
        CheckRow("pycolmap", "ok" if config.has_pycolmap() else "warn", "已安装" if config.has_pycolmap() else "未安装",
                 "可选：pip install pycolmap（装了就不需要 colmap 命令行）", required=False)
    )
    colmap_exe = config.find_colmap()
    rows.append(
        CheckRow("COLMAP 命令行", "ok" if colmap_exe else "warn", colmap_exe or "未找到",
                 "下载 Windows 版 COLMAP 并把目录加入 PATH，或设置环境变量 COLMAP_EXE", required=not config.has_pycolmap())
    )
    ffmpeg = config.find_ffmpeg()
    rows.append(
        CheckRow("ffmpeg", "ok" if ffmpeg else "bad", ffmpeg or "未找到",
                 "pip install imageio-ffmpeg（自带二进制，最省事）；或 winget install Gyan.FFmpeg", required=True)
    )
    wheels = sorted(config.WHEELS_DIR.glob("*.whl"))
    rows.append(
        CheckRow("离线轮子 wheels/", "ok" if wheels else "warn", f"{len(wheels)} 个文件",
                 "没有离线轮子时会回退到 pip + 镜像在线安装", required=False)
    )
    return rows


def print_checks(rows: Sequence[CheckRow]) -> tuple[int, int]:
    """打印自检表，返回 (必需项缺失数, 警告数)。"""
    table_rows = [[status_mark(row.level), row.name, row.value] for row in rows]
    print(config.format_table(["状态", "组件", "检测结果"], table_rows))
    print()
    missing_required = [row for row in rows if row.required and row.level in ("bad", "warn")]
    warnings = [row for row in rows if row.level == "warn"]
    if missing_required:
        print("需要处理的项目与建议命令：")
        for index, row in enumerate(missing_required, start=1):
            print(f"  {index}. {row.name}：{row.advice}")
        print()
    return len(missing_required), len(warnings)


# ---------------------------------------------------------------------------
# 安装计划
# ---------------------------------------------------------------------------


def _wheel_compatible(filename: str, major: int, minor: int) -> bool:
    """判断轮子文件名是否与目标 Python 版本兼容。"""
    lower = filename.lower()
    abi3 = re.search(r"-cp3(\d+)-abi3-", lower)
    if abi3:  # abi3 轮子向下兼容 3.x 及其之后的版本
        return int(abi3.group(1)) <= minor
    if "-py3-" in lower or "-py2.py3-" in lower:
        return True
    exact = re.search(r"-cp(\d)(\d+)-", lower)
    if exact:
        return int(exact.group(1)) == major and int(exact.group(2)) == minor
    return True


def matching_wheels(target_python: str) -> list[Path]:
    """挑出与目标解释器兼容的离线轮子（每个包只取一个）。"""
    major, minor = probe_python(target_python)
    if not major:
        major, minor = sys.version_info[0], sys.version_info[1]
    picks: dict[str, Path] = {}
    for wheel in sorted(config.WHEELS_DIR.glob("*.whl")):
        name = wheel.name.lower()
        if not _wheel_compatible(name, major, minor):
            continue
        if "win_amd64" not in name and "none-any" not in name:
            continue
        for prefix, package in WHEEL_PREFIXES.items():
            if name.startswith(prefix):
                picks[package] = wheel
    return list(picks.values())


def print_light_plan(target_python: str, mirror: str) -> list[Path]:
    wheels = matching_wheels(target_python)
    if wheels:
        config.log_info("离线安装轻量依赖（不联网）：")
        config.log_info("  " + " ".join(f'"{w}"' for w in wheels))
        config.log_info("  → 等价命令：pip install --no-index " + " ".join(f'"{w}"' for w in wheels))
    else:
        config.log_warn(f"wheels/ 里没有与 {target_python} 匹配的轮子，将走在线安装。")
        config.log_info(f'  "{target_python}" -m pip install -i {mirror} ' + " ".join(LIGHT_PACKAGES))
    return wheels


def print_torch_plan(target_python: str, mirror: str, gpus: Sequence[dict[str, Any]]) -> None:
    cap = str(gpus[0].get("computeCap") or "") if gpus else ""
    config.log_info("PyTorch 安装命令（默认只打印，加 --install-torch --yes 才会真正执行）：")
    if cap.startswith("12"):
        config.log_info(f'  "{target_python}" -m pip install --index-url {TORCH_CUDA_INDEX} torch torchvision')
        config.log_info("  ↑ RTX 5060（sm_120 / Blackwell）必须用 CUDA ≥ 12.8 的轮子；torch 需 ≥ 2.7")
    else:
        config.log_info(f'  "{target_python}" -m pip install -i {mirror} torch torchvision')
        config.log_info(f"  （CUDA 版可改用：--index-url {TORCH_CUDA_INDEX}，本机显卡算力 {cap or '未知'}）")
    config.log_info("gsplat 安装（Windows 需要源码编译，耗时 10~40 分钟）：")
    config.log_info(f'  "{target_python}" -m pip install git+https://github.com/nerfstudio-project/gsplat.git')
    config.log_info("  前置条件：Visual Studio 生成工具（含 MSVC v143）、CUDA Toolkit ≥ 12.8，并设置 CUDA_HOME")


def print_fetch_wheels_plan(target_python: str, packages: Sequence[str]) -> None:
    fetcher = config.TOOLS_DIR / "fetch-wheels.mjs"
    if not fetcher.is_file():
        return
    major, minor = probe_python(target_python)
    tag = f"cp{major}{minor}" if major else "cp310"
    config.log_info("需要联网补齐轮子时，可用仓库自带的下载器（绕开 pip 的临时目录问题）：")
    config.log_info(f'  $env:PY_TAG="{tag}"; node "{fetcher}" ' + " ".join(packages))


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="setup_env.py",
        description="三维重建流水线环境自检与安装引导（Windows 优先，中文进度与错误处理）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python pipeline/setup_env.py --check\n"
            "  python pipeline/setup_env.py --dry-run\n"
            "  python pipeline/setup_env.py --install-torch --yes\n"
        ),
    )
    parser.add_argument("--check", action="store_true", help="只做环境自检并打印表格，不安装任何东西")
    parser.add_argument("--install-torch", action="store_true", help="把 torch/torchvision 的安装加入计划（配合 --yes 才真正执行）")
    parser.add_argument("--mirror", default=DEFAULT_MIRROR, help=f"pip 镜像源（默认 {DEFAULT_MIRROR}）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行任何命令")
    parser.add_argument("--yes", action="store_true", help="不再询问，直接执行计划中的安装命令")
    parser.add_argument("--python", default=None, help="指定要安装到哪个 Python 解释器")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    config.ensure_utf8_stdio()
    config.require_python()
    args = build_parser().parse_args(argv)

    config.log_rule("案发现场三维重建流水线 · 环境自检")
    target, reason = pick_target_python(args.python)
    config.log_info(f"安装目标解释器：{target}（{reason}）")
    config.log_info(f"项目根目录：{config.PROJECT_ROOT}")
    config.log_info(f"离线轮子目录：{config.WHEELS_DIR}")

    rows = collect_checks(target)
    print()
    missing, warnings = print_checks(rows)
    print_local_wheel_section(config.detect_gpus())

    if args.check:
        config.log_info(f"自检完成：{len([r for r in rows if r.level == 'bad'])} 项缺失，{warnings} 项警告。")
        if missing:
            config.log_info("执行 python pipeline/setup_env.py 可以看到完整的安装计划。")
            return 1
        config.log_ok("环境已就绪，可以直接跑 pipeline/make_test_scene.py 与 tests/test_pipeline.py。")
        return 0

    # ---- 安装计划 ----
    print()
    config.log_rule("安装计划")
    wheels = print_light_plan(target, args.mirror)
    print()
    print_torch_plan(target, args.mirror, config.detect_gpus())
    print()
    print_fetch_wheels_plan(target, LIGHT_PACKAGES)

    if args.install_torch:
        config.log_warn("已启用 --install-torch：torch 下载约 2.5GB+，请确保网络稳定；Windows 上 gsplat 仍需源码编译。")

    if args.dry_run:
        config.log_ok("dry-run：以上命令均未执行。")
        return 0

    # ---- 轻量依赖：有条件地真正安装 ----
    pip_ready = has_pip(target)
    if not pip_ready:
        config.log_warn(f"目标解释器没有可用的 pip：{target}")
        config.log_info("先执行：")
        config.log_info(f'  "{target}" -m ensurepip --upgrade')
        config.log_info(f'  "{target}" -m pip install -i {args.mirror} pip --upgrade')
        if not args.yes:
            config.log_info("（加 --yes 可在修复 pip 后自动继续安装）")
            return 1
        config.log_info("正在尝试引导 pip…")
        boot = config.run_command([target, "-m", "ensurepip", "--upgrade"], echo=False, timeout=600)
        if not boot.ok:
            config.log_error(f"引导 pip 失败：{boot.tail(6)}")
            config.log_info("可改为使用系统 Python（自带 pip）：python -m pip install ... 或重新创建虚拟环境。")
            return 3
        pip_ready = has_pip(target)
        if not pip_ready:
            config.log_error("引导后仍然没有 pip，请手动处理。")
            return 3
        config.log_ok("pip 已就绪。")

    if not wheels:
        config.log_warn("没有可用的离线轮子，未执行安装。")
        config.log_info("如需在线安装，请手动执行上面打印的 pip 命令（或加 --yes 让脚本执行，但本脚本默认不联网装大包）。")
        return 1

    if not args.yes:
        config.log_info("检测到可离线安装的轻量依赖。要立即执行安装，请加 --yes，例如：")
        config.log_info("  python pipeline/setup_env.py --yes")
        config.log_info("（不加 --yes 时本脚本只打印计划，避免未经确认就改动环境。）")
        return 1 if missing else 0

    config.log_rule("开始离线安装轻量依赖")
    command = [target, "-m", "pip", "install", "--no-index", "--no-cache-dir", *[str(w) for w in wheels]]
    config.log_info("执行：" + " ".join(f'"{part}"' if " " in part else part for part in command))
    started = time.time()
    result = config.run_command(command, echo=False, timeout=3600)
    if not result.ok:
        config.log_error("离线安装失败。排查建议：")
        config.log_error("  1) 换用与轮子匹配的解释器（wheels/ 里是 cp310 轮子，需要 Python 3.10）；")
        config.log_error("  2) 加 --no-cache-dir 已在命令里；如仍失败可加 --force-reinstall；")
        config.log_error("  3) 网络可用时改用镜像在线安装：" + args.mirror)
        config.log_error("详细输出：\n" + result.tail(12))
        return 3
    config.log_ok(f"离线安装完成，耗时 {config.format_duration(time.time() - started)}")
    if result.stdout.strip():
        config.log_info(result.stdout.strip().splitlines()[-1])

    if args.install_torch:
        config.log_rule("开始安装 PyTorch（cu128）")
        torch_command = [target, "-m", "pip", "install", "--index-url", TORCH_CUDA_INDEX, "torch", "torchvision"]
        config.log_info("执行：" + " ".join(torch_command))
        result = config.run_command(torch_command, echo=False, timeout=7200)
        if not result.ok:
            config.log_error("PyTorch 安装失败。排查建议：")
            config.log_error(f"  1) 网络不通时换镜像：-i {args.mirror}（但 cu128 轮子通常只在官方源有）；")
            config.log_error("  2) 确认 Python 版本有对应轮子（cp310/cp311/cp312 均有 cu128 轮子）；")
            config.log_error("  3) 磁盘至少预留 10GB 空间；")
            config.log_error("详细输出：\n" + result.tail(12))
            return 3
        config.log_ok("PyTorch 安装完成。建议重新运行 --check 确认 CUDA 可用。")
        config.log_info("gsplat 还需要源码编译（MSVC + CUDA Toolkit ≥ 12.8）：")
        config.log_info("  pip install git+https://github.com/nerfstudio-project/gsplat.git")

    print()
    config.log_rule("下一步")
    config.log_info("1) 自检：python pipeline/setup_env.py --check")
    config.log_info("2) 生成合成场景验证前端链路：python pipeline/make_test_scene.py")
    config.log_info("3) 跑自测：python tests/test_pipeline.py")
    config.log_info("4) 有真实视频后：extract_frames.py → colmap_recon.py → train_gsplat.py → to_splat.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
