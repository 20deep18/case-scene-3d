"""pipeline/extract_frames.py —— 用 ffmpeg 抽帧，并用拉普拉斯方差剔除模糊帧。

命令行：
    python pipeline/extract_frames.py --video <视频> --out <目录> [--fps 2] [--long-side 1600]
                                      [--max-frames 400] [--blur-threshold 60]

流程：
  1. 调用 ffmpeg 把视频按固定帧率抽成 frame_0001.png 序列；
     同时用一路 256×144 的灰度原始流做「模糊分析流」，避免在 Python 里解码大尺寸 PNG；
  2. 用拉普拉斯方差（numpy 可用时向量化；否则纯标准库实现）给每帧打分；
     分数低于 --blur-threshold 的帧移动到**同级目录** <out>-rejected/（不放在 <out> 里面，
  否则 COLMAP 递归扫描会把它们又读进去 —— 实测踩过）；
  3. 写出 <out>/frames.json，供 COLMAP 与后续步骤使用。

退出码：0 成功；2 参数错误；3 视频不存在；4 缺少 ffmpeg；5 抽帧失败；6 全部帧都被判为模糊。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Sequence

_PIPELINE_DIR = Path(__file__).resolve().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

import config  # noqa: E402

#: 模糊分析用的固定分辨率（灰度）。固定尺寸让阈值在不同视频之间可比。
ANALYSIS_WIDTH: int = 256
ANALYSIS_HEIGHT: int = 144

#: 抽帧文件名模板
FRAME_PATTERN = "frame_%04d.png"

#: 模糊分析流文件名
ANALYSIS_FILE = "_analysis.gray"


# ---------------------------------------------------------------------------
# 图像读取（纯标准库：PNG / PGM / PPM）
# ---------------------------------------------------------------------------


class ImageError(Exception):
    """图像解码错误。"""


def read_pgm(path: Path | str) -> tuple[int, int, bytes]:
    """读取 PGM/PPM（P2/P3 ASCII 与 P5/P6 二进制），统一返回灰度 (宽, 高, 字节)。"""
    data = Path(path).read_bytes()
    if not data.startswith(b"P"):
        raise ImageError("不是 PGM/PPM 文件（缺少 P1-P6 魔数）。")
    magic = data[:2]
    if magic in (b"P1", b"P4"):
        raise ImageError("不支持 PBM（P1/P4）格式，请改用 PGM(P5) 或 PNG。")
    pos = 2
    values: list[int] = []
    need = 4 if magic in (b"P2", b"P3") else 3  # 二进制格式只需要 width/height/maxval

    def next_token() -> bytes:
        nonlocal pos
        while pos < len(data):
            ch = data[pos : pos + 1]
            if ch.isspace():
                pos += 1
                continue
            if ch == b"#":
                while pos < len(data) and data[pos : pos + 1] not in (b"\n", b"\r"):
                    pos += 1
                continue
            break
        start = pos
        while pos < len(data) and not data[pos : pos + 1].isspace():
            pos += 1
        return data[start:pos]

    for _ in range(need):
        token = next_token()
        if not token:
            raise ImageError("PGM/PPM 头信息不完整。")
        values.append(int(token))
    pos += 1  # 跳过后面的单个空白字符
    width, height = values[0], values[1]
    maxval = values[2]
    channels = 3 if magic in (b"P3", b"P6") else 1

    if magic in (b"P2", b"P3"):
        tokens = data[pos:].split()
        if len(tokens) < width * height * channels:
            raise ImageError("ASCII PGM/PPM 数据不足。")
        pixels = [int(t) for t in tokens[: width * height * channels]]
    else:
        expected = width * height * channels
        raw = data[pos : pos + expected]
        if len(raw) < expected:
            raise ImageError("二进制 PGM/PPM 数据不足。")
        pixels = list(raw)
    if maxval != 255:
        scale = 255.0 / maxval
        pixels = [int(min(255.0, value * scale + 0.5)) for value in pixels]
    if channels == 3:
        gray = bytes(
            0 if False else int(0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2] + 0.5)
            for i in range(0, len(pixels), 3)
        )
    else:
        gray = bytes(pixels)
    return width, height, gray


def png_to_gray(data: bytes) -> tuple[int, int, bytes]:
    """把 8 位非隔行 PNG 解码成灰度 (宽, 高, 字节)。

    只依赖标准库 zlib；支持灰度/RGB/灰度+Alpha/RGBA/调色板 5 种常见颜色类型。
    注意：大图（> 200 万像素）会比较慢，抽帧脚本正常情况下走 ffmpeg 的原始灰度流，
    不会用到这里；本函数主要服务于小图与自测。
    """
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ImageError("不是 PNG 文件。")
    pos = 8
    width = height = bit_depth = color_type = interlace = 0
    palette = b""
    idat = bytearray()
    while pos + 8 <= len(data):
        length = struct.unpack_from(">I", data, pos)[0]
        ctype = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filt, interlace = struct.unpack(">IIBBBBB", body[:13])
        elif ctype == b"PLTE":
            palette = bytes(body)
        elif ctype == b"IDAT":
            idat.extend(body)
        elif ctype == b"IEND":
            break
    if width == 0 or height == 0:
        raise ImageError("PNG 缺少 IHDR 信息。")
    if bit_depth != 8:
        raise ImageError(f"仅支持 8 位 PNG，当前为 {bit_depth} 位。")
    if interlace != 0:
        raise ImageError("不支持隔行（Adam7）PNG。")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ImageError(f"不支持的 PNG 颜色类型：{color_type}")
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    expected = (stride + 1) * height
    if len(raw) < expected:
        raise ImageError("PNG 数据不完整。")

    out = bytearray(stride * height)
    prev = bytearray(stride)
    offset = 0
    for row in range(height):
        filter_type = raw[offset]
        offset += 1
        line = bytearray(raw[offset : offset + stride])
        offset += stride
        if filter_type == 1:  # Sub
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif filter_type == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif filter_type == 3:  # Average
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif filter_type == 4:  # Paeth
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 0xFF
        elif filter_type != 0:
            raise ImageError(f"未知的 PNG 行过滤类型：{filter_type}")
        out[row * stride : (row + 1) * stride] = line
        prev = line

    if color_type == 0 or color_type == 4:
        return width, height, bytes(out[0::channels]) if channels != 1 else bytes(out)
    if color_type == 3:
        gray = bytearray(width * height)
        for i in range(width * height):
            index = out[i] * 3
            if index + 2 < len(palette):
                gray[i] = int(0.299 * palette[index] + 0.587 * palette[index + 1] + 0.114 * palette[index + 2] + 0.5)
        return width, height, bytes(gray)
    gray = bytearray(width * height)
    for i in range(width * height):
        base = i * channels
        gray[i] = int(0.299 * out[base] + 0.587 * out[base + 1] + 0.114 * out[base + 2] + 0.5)
    return width, height, bytes(gray)


def read_png_gray(path: Path | str) -> tuple[int, int, bytes]:
    """读取 PNG 并转成灰度 (宽, 高, 字节)。"""
    return png_to_gray(Path(path).read_bytes())


def read_gray_image(path: Path | str) -> tuple[int, int, bytes]:
    """按扩展名自动选择 PGM/PPM 或 PNG 解码器。"""
    suffix = Path(path).suffix.lower()
    if suffix in (".pgm", ".ppm", ".pnm"):
        return read_pgm(path)
    return read_png_gray(path)


def read_raw_frames(path: Path | str, width: int, height: int) -> list[bytes]:
    """读取 ffmpeg 输出的连续灰度原始流，切成每帧一份。"""
    data = Path(path).read_bytes()
    frame_bytes = width * height
    if frame_bytes <= 0:
        return []
    return [data[i : i + frame_bytes] for i in range(0, len(data) - frame_bytes + 1, frame_bytes)]


# ---------------------------------------------------------------------------
# 模糊度打分：拉普拉斯方差
# ---------------------------------------------------------------------------


def laplacian_variance(gray: bytes | bytearray | Sequence[int], width: int, height: int, numpy: Any | None = None) -> float:
    """计算灰度图的 3x3 拉普拉斯卷积方差（数值越大越清晰）。

    使用核 [0 1 0; 1 -4 1; 0 1 0]，只在图像内部（去掉最外一圈）统计。
    """
    if width < 3 or height < 3:
        return 0.0
    if numpy is None and config.has_numpy():
        numpy = config.try_import("numpy")
    if numpy is not None:
        img = numpy.frombuffer(bytes(gray), dtype=numpy.uint8)[: width * height].reshape(height, width).astype(numpy.float64)
        lap = 4.0 * img[1:-1, 1:-1] - img[:-2, 1:-1] - img[2:, 1:-1] - img[1:-1, :-2] - img[1:-1, 2:]
        return float(lap.var())

    values = list(gray) if not isinstance(gray, (bytes, bytearray)) else list(gray)
    total = 0.0
    total_sq = 0.0
    count = 0
    stride = width
    for y in range(1, height - 1):
        row = y * stride
        up = row - stride
        down = row + stride
        # 逐行用切片 + zip 计算，避免 Python 层三重循环
        for value in [
            4.0 * values[i] - values[up + i] - values[down + i] - values[row + i - 1] - values[row + i + 1]
            for i in range(1, width - 1)
        ]:
            total += value
            total_sq += value * value
            count += 1
    if count == 0:
        return 0.0
    mean = total / count
    return max(0.0, total_sq / count - mean * mean)


def image_blur_score(path: Path | str, numpy: Any | None = None) -> tuple[float, int, int]:
    """对单张图片打分，返回 (分数, 宽, 高)。"""
    width, height, gray = read_gray_image(path)
    return laplacian_variance(gray, width, height, numpy), width, height


# ---------------------------------------------------------------------------
# ffmpeg 抽帧
# ---------------------------------------------------------------------------


def find_ffmpeg_or_hint() -> str | None:
    """查找 ffmpeg；找不到时打印中文安装指引。"""
    exe = config.find_ffmpeg()
    if exe:
        return exe
    config.log_error("未找到 ffmpeg，无法抽帧。")
    config.log_info("安装方式（任选其一，装完重新运行即可）：")
    config.log_info("  1) pip install imageio-ffmpeg        # 自带 ffmpeg 二进制，无需配置 PATH")
    config.log_info("  2) winget install Gyan.FFmpeg        # 安装到系统 PATH")
    config.log_info("  3) 下载 https://www.gyan.dev/ffmpeg/builds/ 的 release-essentials 压缩包，解压后把 bin 目录加入 PATH")
    config.log_info("  4) 也可以设置环境变量 DSH_FFMPEG 指向 ffmpeg.exe 的绝对路径")
    config.log_info("提示：先运行 python pipeline/setup_env.py --check 可以看到完整环境自检表。")
    return None


def build_vf_chain(fps: float, long_side: int) -> str:
    """构造抽帧用的 -vf 滤镜链。"""
    chain = f"fps={fps:g}"
    if long_side > 0:
        # 长边限制到 long_side（短边 -2 表示按比例自动取偶数）；逗号在单引号内不会被当作滤镜分隔符
        chain += (
            f",scale='if(gt(iw,ih),min(iw,{long_side}),-2)'"
            f":'if(gt(iw,ih),-2,min(ih,{long_side}))'"
        )
    return chain


def extract_frames(
    video: Path,
    out_dir: Path,
    *,
    fps: float,
    long_side: int,
    max_frames: int,
    ffmpeg: str,
) -> tuple[int, bool]:
    """调用 ffmpeg 抽帧，返回 (抽取帧数, 是否生成了模糊分析流)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / FRAME_PATTERN
    analysis_path = out_dir / ANALYSIS_FILE
    if analysis_path.exists():
        analysis_path.unlink()

    vf = build_vf_chain(fps, long_side)
    argv = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        vf,
        "-frames:v",
        str(max_frames),
        "-start_number",
        "1",
        "-progress",
        "pipe:1",
        "-nostats",
        str(pattern),
        "-vf",
        f"fps={fps:g},scale={ANALYSIS_WIDTH}:{ANALYSIS_HEIGHT},format=gray",
        "-frames:v",
        str(max_frames),
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        str(analysis_path),
    ]
    config.log_info(f"开始抽帧（fps={fps:g}，长边上限 {long_side} 像素，最多 {max_frames} 帧）…")
    result = config.run_command(argv, timeout=None, echo=False)
    frames = sorted(out_dir.glob("frame_*.png"))
    if not result.ok and not frames:
        config.log_warn("首次抽帧命令失败，尝试去掉缩放滤镜后重试（可能是不支持该 scale 表达式）。")
        config.log_warn("ffmpeg 输出：" + (result.tail(8) or "（无输出）"))
        argv_fallback = list(argv)
        vf_index = argv_fallback.index("-vf")
        argv_fallback[vf_index + 1] = f"fps={fps:g}"
        result = config.run_command(argv_fallback, timeout=None, echo=False)
        frames = sorted(out_dir.glob("frame_*.png"))
    if not result.ok and not frames:
        config.log_error("ffmpeg 抽帧失败。常见原因：视频编码不支持、文件损坏、路径含特殊字符。")
        config.log_error("ffmpeg 输出：" + (result.tail(10) or "（无输出）"))
        return 0, False

    for line in result.stdout.splitlines():
        if line.startswith("frame=") or line.startswith("out_time="):
            pass  # 进度信息已经由 ffmpeg 写出，这里只做占位，最终帧数以目录扫描为准
    has_analysis = analysis_path.is_file() and analysis_path.stat().st_size >= ANALYSIS_WIDTH * ANALYSIS_HEIGHT
    if not has_analysis:
        config.log_warn("没有拿到模糊分析流（ffmpeg 未输出 rawvideo），将回退到逐张解码 PNG 打分（大图会较慢）。")
    return len(frames), has_analysis


# ---------------------------------------------------------------------------
# 模糊过滤
# ---------------------------------------------------------------------------


def blur_filter(
    out_dir: Path,
    *,
    threshold: float,
    has_analysis: bool,
    numpy: Any | None,
    progress_every: int = 25,
) -> dict[str, Any]:
    """按拉普拉斯方差过滤模糊帧，返回统计信息。"""
    frames = sorted(out_dir.glob("frame_*.png"))
    # ⚠️ 2026-10-07 修：剔除的帧**不能放在图目录里面**。
    #    COLMAP 的 image_path 会**递归扫描子目录**，于是 `_rejected/` 里的模糊帧
    #    照样被特征提取与重建读进去 —— 模糊过滤等于白做。
    #    实测（用户 2026-10-07 的房间视频，fps 3）：227 张保留 + 45 张剔除，
    #    数据库里却是 **272 张**，其中 45 张来自 `_rejected/`；
    #    而重建断链的帧区间（72~85、118~125）正好就是这些模糊帧。
    #    早先文档里已经记过同一现象（"COLMAP 把 8 张模糊帧也注册了，占 3% 位姿"），
    #    这次量到 16.5%，所以直接挪到**同级目录**去。
    rejected_dir = out_dir.parent / f"{out_dir.name}-rejected"
    # ⚠️ 清掉上一轮的剔除帧：否则新的一轮会把旧的混进来，
    #    而 frames.json 里的 rejected 计数会与实际目录内容对不上（自测会看不出来）。
    if rejected_dir.is_dir():
        for old in rejected_dir.glob("*"):
            if old.is_file():
                old.unlink()
    kept: list[str] = []
    rejected: list[str] = []
    scores: list[float] = []
    started = time.time()

    analysis_frames: list[bytes] = []
    if has_analysis:
        analysis_frames = read_raw_frames(out_dir / ANALYSIS_FILE, ANALYSIS_WIDTH, ANALYSIS_HEIGHT)
        if len(analysis_frames) < len(frames):
            config.log_warn(
                f"分析流只有 {len(analysis_frames)} 帧，少于实际抽出的 {len(frames)} 帧，"
                "多出的帧将回退到逐张解码。"
            )

    for index, frame in enumerate(frames):
        if index < len(analysis_frames):
            score = laplacian_variance(analysis_frames[index], ANALYSIS_WIDTH, ANALYSIS_HEIGHT, numpy)
        else:
            score, _w, _h = image_blur_score(frame, numpy)
        scores.append(score)
        if threshold > 0 and score < threshold:
            rejected_dir.mkdir(parents=True, exist_ok=True)
            target = rejected_dir / frame.name
            if target.exists():
                target.unlink()
            shutil.move(str(frame), str(target))
            rejected.append(frame.name)
        else:
            kept.append(frame.name)
        if progress_every and (index + 1) % progress_every == 0:
            config.log_info(f"模糊检测进度 {index + 1}/{len(frames)}：已保留 {len(kept)}，剔除 {len(rejected)}")

    analysis_path = out_dir / ANALYSIS_FILE
    if analysis_path.exists():
        analysis_path.unlink()

    stats: dict[str, Any] = {
        "kept": len(kept),
        "rejected": len(rejected),
        "elapsedSec": round(time.time() - started, 3),
    }
    if scores:
        ordered = sorted(scores)
        stats["scoreMin"] = round(ordered[0], 2)
        stats["scoreMedian"] = round(ordered[len(ordered) // 2], 2)
        stats["scoreMax"] = round(ordered[-1], 2)
    return {"kept": kept, "rejected": rejected, "stats": stats}


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_frames.py",
        description="用 ffmpeg 从视频抽帧，并用拉普拉斯方差剔除模糊帧（输出 frame_0001.png 序列与 frames.json）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python pipeline/extract_frames.py --video data/projects/case01/source.mp4 --out data/projects/case01/frames --fps 2\n"
            "  python pipeline/extract_frames.py --video raw.mp4 --out frames --fps 1 --long-side 2000 --blur-threshold 40\n"
        ),
    )
    parser.add_argument("--video", required=True, help="输入视频文件（mp4/mov/mkv 等 ffmpeg 支持的格式）")
    parser.add_argument("--out", required=True, help="输出目录（会生成 frame_0001.png 与 frames.json）")
    parser.add_argument("--fps", type=float, default=2.0, help="抽帧帧率（默认 2，即每秒 2 帧；视频场景建议 1~3）")
    parser.add_argument("--long-side", type=int, default=1600, help="输出图片长边上限（默认 1600，0 表示不缩放）")
    parser.add_argument("--max-frames", type=int, default=400, help="最多抽取多少帧（默认 400）")
    parser.add_argument("--blur-threshold", type=float, default=60.0, help="拉普拉斯方差阈值，低于该值判为模糊并移出图目录（默认 60，0 表示不过滤）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    config.ensure_utf8_stdio()
    config.require_python()
    args = build_parser().parse_args(argv)

    if args.fps <= 0:
        config.log_error("--fps 必须大于 0。")
        return 2
    if args.max_frames <= 0:
        config.log_error("--max-frames 必须大于 0。")
        return 2
    if args.long_side < 0:
        config.log_error("--long-side 不能为负数。")
        return 2
    if args.blur_threshold < 0:
        config.log_error("--blur-threshold 不能为负数。")
        return 2

    video = Path(args.video)
    out_dir = Path(args.out)
    config.log_rule("视频抽帧与模糊过滤")
    config.log_info(f"视频：{video}")
    config.log_info(f"输出目录：{out_dir}")

    if not video.is_file():
        config.log_error(f"视频文件不存在：{video}")
        config.log_info("请检查路径是否正确；如果路径含空格或中文，请用英文引号包起来。")
        return 3

    ffmpeg = find_ffmpeg_or_hint()
    if not ffmpeg:
        return 4
    config.log_info(f"ffmpeg：{ffmpeg}")

    numpy = config.try_import("numpy")
    config.log_info(f"numpy：{'可用（向量化打分）' if numpy is not None else '不可用（纯标准库打分，速度稍慢）'}")

    started = time.time()
    count, has_analysis = extract_frames(
        video, out_dir, fps=args.fps, long_side=args.long_side, max_frames=args.max_frames, ffmpeg=ffmpeg
    )
    if count == 0:
        config.log_error("抽帧结果为空：没有生成任何 PNG。")
        config.log_info("排查建议：")
        config.log_info(f"  1) 先手动确认视频可解码：\"{ffmpeg}\" -i \"{video}\"");
        config.log_info("  2) 视频可能是 0 秒的损坏文件，或使用了非标准编码；")
        config.log_info("  3) 也可以先用其它工具把视频转成 H.264 mp4 再重试。")
        return 5
    config.log_ok(f"抽帧完成：共 {count} 帧")

    if args.blur_threshold > 0:
        config.log_info(f"开始模糊检测（拉普拉斯方差阈值 {args.blur_threshold}）…")
        result = blur_filter(out_dir, threshold=args.blur_threshold, has_analysis=has_analysis, numpy=numpy)
    else:
        frames = sorted(path.name for path in out_dir.glob("frame_*.png"))
        result = {"kept": frames, "rejected": [], "stats": {"kept": len(frames), "rejected": 0, "elapsedSec": 0.0}}

    kept = result["kept"]
    rejected = result["rejected"]
    if not kept:
        config.log_error("所有帧都被判为模糊，没有可用帧。")
        config.log_info(f"建议：降低阈值（例如 --blur-threshold {max(5.0, args.blur_threshold / 3):.0f}）或先用三脚架稳定重拍；")
        config.log_info(f"被判为模糊的帧已移出图目录，放在：{rejected_dir}（可以先看一眼再决定）")
        return 6

    payload = {
        "video": str(video),
        "fps": args.fps,
        "longSide": args.long_side,
        "kept": len(kept),
        "rejected": len(rejected),
        "frames": kept,
        "blurThreshold": args.blur_threshold,
        "rejectedFrames": rejected,
        "imageSize": {"width": ANALYSIS_WIDTH, "height": ANALYSIS_HEIGHT, "note": "模糊分析流分辨率；实际 PNG 分辨率以文件为准"},
        "blurScore": {k: v for k, v in result["stats"].items() if k.startswith("score")},
        "elapsedSec": round(time.time() - started, 3),
        "createdAt": config.now_iso(),
        "generator": "pipeline/extract_frames.py",
    }
    json_path = out_dir / "frames.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = result["stats"]
    config.log_ok(
        f"模糊过滤完成：保留 {len(kept)} 帧，剔除 {len(rejected)} 帧"
        f"（拉普拉斯方差 min={stats.get('scoreMin')} 中位={stats.get('scoreMedian')} max={stats.get('scoreMax')}）"
    )
    config.log_ok(f"已写出 {json_path}")
    config.log_info(f"总耗时 {config.format_duration(time.time() - started)}")
    config.log_info(
        "下一步：python pipeline/colmap_recon.py --images "
        f"\"{out_dir}\" --workspace \"{out_dir.parent / 'colmap'}\""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
