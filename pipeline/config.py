"""案发现场三维重建流水线 —— 公共配置与依赖探测模块。

本模块提供：
  * 项目根目录探测与全流水线共享的路径常量；
  * 第三方依赖探测（numpy / cv2 / torch / gsplat / pycolmap / colmap / ffmpeg）；
  * 统一的中文日志（带时间戳与级别）；
  * 目录创建、命令执行、表格格式化等公共工具。

设计原则：
  1. 本模块只依赖 Python 标准库，任何第三方库都用惰性探测，绝不在此处硬导入；
  2. 缺依赖时返回 None / False，由调用方给出中文安装指引，不允许直接崩溃；
  3. 所有输出走 UTF-8（可通过环境变量 PIPELINE_STDIO_ENCODING 覆盖）。
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "PROJECT_ROOT",
    "PIPELINE_DIR",
    "TOOLS_DIR",
    "WHEELS_DIR",
    "ENV_DIR",
    "VENV_PYTHON",
    "DATA_DIR",
    "MODELS_DIR",
    "PROJECTS_DIR",
    "TESTS_DIR",
    "ensure_dirs",
    "ensure_utf8_stdio",
    "has_numpy",
    "has_cv2",
    "has_torch",
    "has_gsplat",
    "has_pycolmap",
    "has_imageio_ffmpeg",
    "find_colmap",
    "find_ffmpeg",
    "try_import",
    "torch_env_info",
    "detect_gpus",
    "log",
    "log_debug",
    "log_info",
    "log_ok",
    "log_warn",
    "log_error",
    "log_step",
    "log_rule",
    "now_iso",
    "format_duration",
    "human_bytes",
    "format_table",
    "run_command",
    "require_python",
]

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------


def _detect_project_root() -> Path:
    """探测项目根目录。

    规则：本文件位于 <root>/pipeline/config.py，因此上一级目录即项目根；
    若该目录缺少预期标志（app/ 或 data/），则继续向上查找，最后回退到上一级。
    """
    here = Path(__file__).resolve()
    candidates: list[Path] = [here.parent.parent]
    candidates.extend(here.parent.parent.parents[:3])
    for cand in candidates:
        if (cand / "app").is_dir() and (cand / "pipeline").is_dir():
            return cand
    for cand in candidates:
        if (cand / "data").is_dir() or (cand / "pipeline").is_dir():
            return cand
    return here.parent.parent


PROJECT_ROOT: Path = _detect_project_root()
"""项目根目录（默认 C:\\Users\\HP\\Documents\\deepseek-harness\\default-workspace）。"""

PIPELINE_DIR: Path = PROJECT_ROOT / "pipeline"
TOOLS_DIR: Path = PROJECT_ROOT / "tools"
WHEELS_DIR: Path = PROJECT_ROOT / "wheels"
ENV_DIR: Path = PROJECT_ROOT / "env"
VENV_PYTHON: Path = ENV_DIR / "Scripts" / "python.exe"
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = DATA_DIR / "models"
PROJECTS_DIR: Path = DATA_DIR / "projects"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

#: 合成测试场景的默认输出路径
DEFAULT_TEST_SPLAT: Path = MODELS_DIR / "test-scene.splat"
DEFAULT_TEST_JSON: Path = MODELS_DIR / "test-scene.json"
DEFAULT_TEST_CAMERAS: Path = MODELS_DIR / "test-scene-cameras.json"
DEFAULT_TEST_OBSERVATIONS: Path = MODELS_DIR / "test-scene-observations.json"


def ensure_dirs(*extra: Path | str) -> list[Path]:
    """创建流水线需要的目录（幂等），返回实际创建/确认的目录列表。"""
    targets: list[Path] = [DATA_DIR, MODELS_DIR, PROJECTS_DIR]
    targets.extend(Path(p) for p in extra)
    created: list[Path] = []
    for path in targets:
        try:
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
        except OSError as exc:  # pragma: no cover - 权限/磁盘异常
            log_error(f"创建目录失败：{path}（{exc}）")
    return created


# ---------------------------------------------------------------------------
# 标准输出编码
# ---------------------------------------------------------------------------


def ensure_utf8_stdio() -> None:
    """把 stdout/stderr 切到 UTF-8，避免 Windows GBK 控制台打印 ✅/❌ 时崩溃。

    Windows 上 Python 默认用 ANSI 代码页（本机为 gbk）编码标准输出，
    打印 ✅/❌/⚠️ 一类字符会抛 UnicodeEncodeError。这里统一改为 UTF-8 并
    忽略无法编码的字符；如需恢复系统行为可设置环境变量
    PIPELINE_STDIO_ENCODING=gbk（或 system）。
    """
    wanted = os.environ.get("PIPELINE_STDIO_ENCODING", "utf-8").strip()
    if wanted.lower() in {"system", "default", "none", ""}:
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding=wanted, errors="replace")
        except (ValueError, OSError):  # pragma: no cover - 非文本流
            pass


# ---------------------------------------------------------------------------
# 统一日志（中文）
# ---------------------------------------------------------------------------

_LEVEL_ORDER: dict[str, int] = {"调试": 10, "信息": 20, "成功": 25, "警告": 30, "错误": 40}
_MIN_LEVEL: int = _LEVEL_ORDER["信息"]


def set_log_level(level: str) -> None:
    """设置最低输出级别（调试 / 信息 / 成功 / 警告 / 错误）。"""
    global _MIN_LEVEL
    if level in _LEVEL_ORDER:
        _MIN_LEVEL = _LEVEL_ORDER[level]


def log(message: str, level: str = "信息", *, stream: Any = None) -> None:
    """打印一行带时间戳与级别的中文日志。"""
    if _LEVEL_ORDER.get(level, 20) < _MIN_LEVEL:
        return
    stamp = time.strftime("%H:%M:%S")
    target = stream if stream is not None else sys.stdout
    try:
        print(f"[{stamp}] [{level}] {message}", file=target, flush=True)
    except UnicodeEncodeError:  # pragma: no cover - 极端控制台
        print(f"[{stamp}] [{level}] {message.encode('utf-8', 'replace').decode('utf-8')}".encode("ascii", "replace").decode("ascii"), file=target)


def log_debug(message: str) -> None:
    log(message, "调试")


def log_info(message: str) -> None:
    log(message, "信息")


def log_ok(message: str) -> None:
    log(message, "成功")


# 别名：历史上 train_gsplat.py 误写过 config.log_success(...)，
# 而 config 里只有 log_ok，于是「画质越好越会崩」（PSNR ≥ 24dB 时抛 AttributeError，
# 评估跑完但 PLY 还没写，整个建模任务以退出码 5 失败）。
# 调用点已经改成 log_ok，这里再留一个别名兜底，避免同类拼写错误再次炸掉长任务。
log_success = log_ok


def log_warn(message: str) -> None:
    log(message, "警告", stream=sys.stderr)


def log_error(message: str) -> None:
    log(message, "错误", stream=sys.stderr)


def log_step(index: int, total: int, title: str) -> None:
    """打印步骤分隔行，例如：[1/4] 抽帧。"""
    log(f"—— 步骤 {index}/{total}：{title} ——")


def log_rule(title: str = "") -> None:
    """打印一条分隔线，可选标题。"""
    line = "=" * 64
    if title:
        log(f"{line}\n{title}\n{line}")
    else:
        log(line)


# ---------------------------------------------------------------------------
# 时间 / 体积 / 表格工具
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """当前本地时间的 ISO 8601 字符串（秒精度）。"""
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def format_duration(seconds: float) -> str:
    """把秒数格式化为「1小时2分3.4秒」这类中文可读文本。"""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}秒"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}分{sec:.1f}秒"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}小时{minutes}分{sec:.0f}秒"


def human_bytes(num: float) -> str:
    """把字节数格式化为人类可读文本。"""
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024.0
    return f"{value:.2f} TB"


def format_table(headers: Sequence[str], rows: Iterable[Sequence[Any]], *, indent: str = "  ") -> str:
    """生成等宽对齐的纯文本表格（中文字符按两格宽计算），用于环境自检输出。"""
    def width(text: str) -> int:
        total = 0
        for ch in text:
            code = ord(ch)
            if 0xFE00 <= code <= 0xFE0F:  # 变体选择符（如 ⚠️ 后面的 VS16）不占宽度
                continue
            total += 2 if (code > 0x2E7F or 0x2600 <= code <= 0x27BF) else 1
        return total

    str_rows = [[("" if cell is None else str(cell)) for cell in row] for row in rows]
    all_rows = [[str(h) for h in headers]] + str_rows
    widths = [0] * len(headers)
    for row in all_rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], width(cell))

    def render(row: Sequence[str]) -> str:
        cells = []
        for i, cell in enumerate(row):
            pad = widths[i] - width(cell) if i < len(widths) else 0
            cells.append(cell + " " * pad)
        return indent + "  ".join(cells).rstrip()

    lines = [render(all_rows[0]), indent + "  ".join("-" * w for w in widths)]
    lines.extend(render(row) for row in str_rows)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 依赖探测（全部惰性、带缓存）
# ---------------------------------------------------------------------------


def try_import(module_name: str) -> Any | None:
    """尝试导入模块，失败返回 None（不抛异常，用于可选的第三方依赖）。"""
    try:
        return importlib.import_module(module_name)
    except BaseException:  # noqa: BLE001 - 第三方库可能抛任意异常（含 SystemExit）
        return None


@lru_cache(maxsize=None)
def _has_module(module_name: str) -> bool:
    """判断模块是否可导入（仅查 spec，不真正导入，速度快且无副作用）。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except BaseException:  # noqa: BLE001 - 半损坏的包可能抛异常
        return False


def has_numpy() -> bool:
    """是否安装了 numpy（模糊检测、向量化 PLY 转换会用到）。"""
    return _has_module("numpy")


def has_cv2() -> bool:
    """是否安装了 OpenCV（模糊检测的快速路径）。"""
    return _has_module("cv2")


def has_torch() -> bool:
    """是否安装了 PyTorch。"""
    return _has_module("torch")


def has_gsplat() -> bool:
    """是否安装了 gsplat（CUDA 光栅化库）。"""
    return _has_module("gsplat")


def has_pycolmap() -> bool:
    """是否安装了 pycolmap（COLMAP 的 Python 绑定）。"""
    return _has_module("pycolmap")


def has_imageio_ffmpeg() -> bool:
    """是否安装了 imageio-ffmpeg（提供自带 ffmpeg 可执行文件）。"""
    return _has_module("imageio_ffmpeg")


def _first_existing(paths: Iterable[Path]) -> str | None:
    for path in paths:
        try:
            if path.is_file():
                return str(path)
        except OSError:
            continue
    return None


def find_ffmpeg() -> str | None:
    """查找可用的 ffmpeg，返回可执行文件路径或 None。

    查找顺序：
      1. 环境变量 DSH_FFMPEG / FFMPEG_EXE 指定的路径；
      2. imageio_ffmpeg.get_ffmpeg_exe()（随 wheel 分发的静态二进制）；
      3. 项目 tools/ 下的常见位置；
      4. 系统 PATH 中的 ffmpeg。
    """
    for env_name in ("DSH_FFMPEG", "FFMPEG_EXE", "FFMPEG"):
        value = os.environ.get(env_name)
        if value and Path(value).is_file():
            return value

    module = try_import("imageio_ffmpeg")
    if module is not None:
        getter = getattr(module, "get_ffmpeg_exe", None)
        if callable(getter):
            try:
                exe = getter()
                if exe and Path(exe).is_file():
                    return str(exe)
            except BaseException:  # noqa: BLE001 - 下载/权限失败时继续兜底
                pass

    local = _first_existing(
        [
            TOOLS_DIR / "ffmpeg" / "bin" / "ffmpeg.exe",
            TOOLS_DIR / "dist" / "ffmpeg" / "bin" / "ffmpeg.exe",
            TOOLS_DIR / "ffmpeg.exe",
            ENV_DIR / "Scripts" / "ffmpeg.exe",
        ]
    )
    if local:
        return local

    return shutil.which("ffmpeg")


def find_colmap() -> str | None:
    """查找 COLMAP 命令行可执行文件，返回路径或 None。

    查找顺序：环境变量 COLMAP_EXE / COLMAP → 系统 PATH → 常见安装目录。
    """
    for env_name in ("COLMAP_EXE", "COLMAP", "DSH_COLMAP"):
        value = os.environ.get(env_name)
        if value and Path(value).is_file():
            return value

    found = shutil.which("colmap")
    if found:
        return found

    candidates = [
        Path(r"C:\Program Files\COLMAP\COLMAP.bat"),
        Path(r"C:\Program Files\COLMAP\colmap.exe"),
        Path(r"C:\Program Files (x86)\COLMAP\COLMAP.bat"),
        Path(r"C:\COLMAP\bin\colmap.exe"),
        Path(r"C:\COLMAP\COLMAP.bat"),
        TOOLS_DIR / "colmap" / "bin" / "colmap.exe",
        TOOLS_DIR / "colmap" / "COLMAP.bat",
        Path.home() / "COLMAP" / "COLMAP.bat",
    ]
    return _first_existing(candidates)


def torch_env_info() -> dict[str, Any]:
    """收集 PyTorch / CUDA 环境信息，未安装时返回 available=False。"""
    info: dict[str, Any] = {
        "available": False,
        "torchVersion": None,
        "cudaAvailable": False,
        "cudaVersion": None,
        "cudaRuntime": None,
        "deviceName": None,
        "capability": None,
    }
    torch = try_import("torch")
    if torch is None:
        return info
    info["available"] = True
    info["torchVersion"] = getattr(torch, "__version__", "未知")
    info["cudaVersion"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        info["cudaAvailable"] = bool(torch.cuda.is_available())
    except BaseException:  # noqa: BLE001
        info["cudaAvailable"] = False
    if info["cudaAvailable"]:
        try:
            info["deviceName"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            info["capability"] = f"sm_{cap[0]}{cap[1]}"
            info["cudaRuntime"] = getattr(torch.version, "cuda", None)
        except BaseException:  # noqa: BLE001
            pass
    return info


def detect_gpus() -> list[dict[str, Any]]:
    """用 nvidia-smi 探测 NVIDIA 显卡（名称、显存、驱动、算力），失败返回空列表。"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    query = "name,memory.total,driver_version,compute_cap"
    try:
        proc = subprocess.run(
            [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpus.append(
            {
                "name": parts[0],
                "memoryMB": _safe_int(parts[1]),
                "driver": parts[2],
                "computeCap": parts[3],
            }
        )
    return gpus


def _safe_int(text: str) -> int | None:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def python_version_str() -> str:
    """当前解释器版本，例如 3.13.2 (CPython)。"""
    return f"{sys.version.split()[0]} ({sys.implementation.name})"


def require_python(min_major: int = 3, min_minor: int = 10) -> None:
    """检查 Python 版本，太低时打印中文提示并退出（退出码 5）。"""
    if sys.version_info < (min_major, min_minor):
        log_error(
            f"当前 Python 版本为 {python_version_str()}，本流水线要求 Python "
            f"{min_major}.{min_minor}+。请改用 env\\Scripts\\python.exe 或安装新版 Python。"
        )
        raise SystemExit(5)


# ---------------------------------------------------------------------------
# 子进程执行
# ---------------------------------------------------------------------------


@dataclass
class CommandResult:
    """一次外部命令执行的结果。"""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def tail(self, lines: int = 12) -> str:
        """返回 stderr（为空则 stdout）的最后若干行，便于打印报错。"""
        text = self.stderr.strip() or self.stdout.strip()
        if not text:
            return ""
        return "\n".join(text.splitlines()[-lines:])


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path | str | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    echo: bool = True,
) -> CommandResult:
    """执行外部命令并捕获输出。

    某些受限沙箱禁止子进程使用管道，这里在遇到 PermissionError 时自动
    退化为「重定向到临时文件」，保证脚本在受控环境中依然可用。
    """
    argv_list = [str(a) for a in argv]
    if echo:
        log_debug("执行命令：" + " ".join(_quote(a) for a in argv_list))
    started = time.time()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    try:
        proc = subprocess.run(
            argv_list,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=merged_env,
        )
        return CommandResult(argv_list, proc.returncode, proc.stdout or "", proc.stderr or "", time.time() - started)
    except FileNotFoundError:
        return CommandResult(argv_list, 127, "", f"找不到可执行文件：{argv_list[0]}", time.time() - started)
    except PermissionError:
        log_warn("管道捕获被沙箱拒绝，改用临时文件重定向重试。")
        return _run_command_via_files(argv_list, cwd=cwd, timeout=timeout, env=merged_env)
    except subprocess.TimeoutExpired:
        return CommandResult(argv_list, 124, "", f"命令超时（>{timeout} 秒）", time.time() - started)
    except OSError as exc:
        return CommandResult(argv_list, 126, "", f"启动命令失败：{exc}", time.time() - started)


def _run_command_via_files(
    argv: list[str],
    *,
    cwd: Path | str | None,
    timeout: float | None,
    env: dict[str, str],
) -> CommandResult:
    """把子进程输出写入临时文件再读回（沙箱禁用管道时的兜底路径）。"""
    import tempfile

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="pipeline-cmd-") as tmp:
        out_path = Path(tmp) / "stdout.txt"
        err_path = Path(tmp) / "stderr.txt"
        try:
            with out_path.open("wb") as out_f, err_path.open("wb") as err_f:
                code = subprocess.call(argv, cwd=str(cwd) if cwd else None, stdout=out_f, stderr=err_f, env=env, timeout=timeout)
        except FileNotFoundError:
            return CommandResult(argv, 127, "", f"找不到可执行文件：{argv[0]}", time.time() - started)
        except subprocess.TimeoutExpired:
            return CommandResult(argv, 124, "", f"命令超时（>{timeout} 秒）", time.time() - started)
        except OSError as exc:
            return CommandResult(argv, 126, "", f"启动命令失败：{exc}", time.time() - started)
        stdout = out_path.read_text("utf-8", errors="replace") if out_path.exists() else ""
        stderr = err_path.read_text("utf-8", errors="replace") if err_path.exists() else ""
    return CommandResult(argv, code, stdout, stderr, time.time() - started)


def _quote(text: str) -> str:
    return f'"{text}"' if " " in text else text


def dump_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    """写出 UTF-8 JSON（保留中文，不转义），返回实际路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")
    return target


def load_json(path: Path | str) -> Any:
    """读取 UTF-8 JSON。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover - 手工自检入口
    ensure_utf8_stdio()
    log_rule("pipeline/config.py 自检")
    log_info(f"项目根目录：{PROJECT_ROOT}")
    log_info(f"Python：{python_version_str()}（{sys.executable}）")
    log_info(f"numpy={has_numpy()} cv2={has_cv2()} torch={has_torch()} gsplat={has_gsplat()} pycolmap={has_pycolmap()}")
    log_info(f"ffmpeg：{find_ffmpeg() or '未找到'}")
    log_info(f"colmap：{find_colmap() or '未找到'}")
    for gpu in detect_gpus():
        log_info(f"GPU：{gpu['name']} / {gpu['memoryMB']} MB / 驱动 {gpu['driver']} / {gpu['computeCap']}")
    ensure_dirs()
    log_ok("目录检查完成。")
