"""tests/test_pipeline.py —— 三维重建流水线的自测脚本（不需要任何第三方库，有 numpy 更好）。

运行：
    python tests/test_pipeline.py

覆盖内容：
  1. to_splat.py：内置极简 PLY 写入器生成带 f_dc_/f_rest_/opacity/scale_/rot_ 的小 PLY，
     用命令行跑转换，校验「字节数 = 数量 × 32」、逐字段解码与输入一致（四元数误差 ≤ 1/128）、
     min-opacity 过滤生效、.shcoeff.bin 尺寸正确、JSON 字段齐全；
  2. make_test_scene.py：生成的 .splat 可解析、包围盒合理、6 个三维点的重投影误差 < 1px；
  3. 端到端一致性：make_test_scene 的 --ply 经过 to_splat.py 往返后与直接生成的 .splat 一致；
  4. extract_frames.py：拉普拉斯方差能把模糊图判为模糊（PGM 与 PNG 两种容器都测）；
  5. colmap_recon.py：自己实现的 COLMAP 二进制读写器与「按官方格式手写的字节」往返一致。

退出码：全部通过 0，有失败 1。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Sequence

_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TEST_DIR.parent
_PIPELINE_DIR = _PROJECT_ROOT / "pipeline"
for _path in (str(_PIPELINE_DIR), str(_PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import colmap_recon  # noqa: E402
import config  # noqa: E402
import extract_frames  # noqa: E402
import to_splat  # noqa: E402

PYTHON = sys.executable
SH_C0 = 0.28209479177387814
GAUSSIAN_BYTES = 32


# ---------------------------------------------------------------------------
# 极简测试框架
# ---------------------------------------------------------------------------

_TESTS: list[tuple[str, Callable[[Path], None]]] = []
_RESULTS: list[tuple[str, bool, str, float]] = []


def test(func: Callable[[Path], None]) -> Callable[[Path], None]:
    """注册一个测试用例；用例接收一个可写的临时目录。"""
    _TESTS.append((func.__name__, func))
    return func


def _fail(message: str) -> None:
    raise AssertionError(message)


def check(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def almost(a: float, b: float, tol: float, message: str) -> None:
    if not (abs(a - b) <= tol):
        _fail(f"{message}（期望 {b}，实际 {a}，容差 {tol}）")


def run_script(script: str, args: Sequence[str]) -> tuple[int, str, str]:
    """用子进程运行 pipeline 下的脚本，返回 (退出码, stdout, stderr)。"""
    result = config.run_command([PYTHON, "-X", "utf8", str(_PIPELINE_DIR / script), *args], echo=False)
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# 独立实现的小工具（刻意不复用 to_splat 的编码函数，避免自证）
# ---------------------------------------------------------------------------


def u8_half_up(value: float) -> int:
    """四舍五入到 0..255（.5 进位），与 to_splat.py 的约定一致。"""
    if not (value > 0.0):
        return 0
    if value >= 255.0:
        return 255
    return int(math.floor(value + 0.5))


def expect_rgb(dc: Sequence[float]) -> tuple[int, int, int]:
    return tuple(u8_half_up((0.5 + SH_C0 * dc[i]) * 255.0) for i in range(3))  # type: ignore[return-value]


def expect_alpha(opacity_raw: float) -> int:
    return u8_half_up(1.0 / (1.0 + math.exp(-opacity_raw)) * 255.0)


def expect_quat(q: Sequence[float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(v * v for v in q))
    return tuple(v / norm for v in q)  # type: ignore[return-value]


def write_minimal_ply(path: Path, records: list[dict[str, Any]], rest_per_channel: int = 15) -> None:
    """测试内置的极简 3DGS PLY 写入器（二进制小端，独立实现）。"""
    props = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
    props += [f"f_rest_{i}" for i in range(rest_per_channel * 3)]
    props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\ncomment test fixture\n"
    header += f"element vertex {len(records)}\n"
    header += "".join(f"property float {name}\n" for name in props)
    header += "end_header\n"
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        for record in records:
            values = list(record["xyz"]) + [0.0, 0.0, 0.0] + list(record["dc"])
            values += list(record["rest"])
            values += [record["opacity"]]
            values += list(record["scale"])
            values += list(record["rot"])
            handle.write(struct.pack(f"<{len(values)}f", *values))


def make_records(count: int) -> list[dict[str, Any]]:
    """构造一批可预期的测试记录（含应被过滤掉的脏数据）。"""
    records: list[dict[str, Any]] = []
    for i in range(count):
        records.append(
            {
                "xyz": (0.1 * i - 0.5, -0.05 * i, 0.02 * i),
                "dc": (((i % 5) - 2) * 0.4, ((i % 3) - 1) * 0.35, 0.15 * (i % 4) - 0.2),
                "rest": [0.01 * ((i + k) % 7) - 0.03 for k in range(45)],
                "opacity": 4.0 - 0.2 * (i % 4),
                # scale 一律用**线性尺度**（标准 3DGS PLY 约定，单位米）
                "scale": (0.05 + 0.01 * (i % 3), 0.08, 0.018),
                "rot": (1.0, 0.2 * (i % 3), -0.1 * (i % 2), 0.05 * i),
            }
        )
    return records


def read_splat_raw(path: Path) -> list[tuple[Any, ...]]:
    """按 .splat 的 32 字节布局解码（独立实现）。"""
    data = path.read_bytes()
    check(len(data) % GAUSSIAN_BYTES == 0, f".splat 长度 {len(data)} 不是 32 的整数倍")
    struct_format = struct.Struct("<6f8B")
    return [struct_format.unpack_from(data, offset) for offset in range(0, len(data), GAUSSIAN_BYTES)]


# ---------------------------------------------------------------------------
# 测试 1：to_splat.py 的核心正确性
# ---------------------------------------------------------------------------


@test
def test_to_splat_roundtrip(work: Path) -> None:
    """命令行转换 + 逐字段解码校验 + 过滤生效 + JSON 字段齐全。"""
    records = make_records(20)
    # 第 3 条：opacity 极低，必须被丢弃
    records[2]["opacity"] = -8.0
    # 第 5 条：尺度 2.0 m，超过 --max-scale 1.0，必须被丢弃（线性尺度，标准约定）
    records[5]["scale"] = (2.0, 0.08, 0.018)
    expected = [record for index, record in enumerate(records) if index not in (2, 5)]

    ply = work / "fixture.ply"
    write_minimal_ply(ply, records)
    out = work / "fixture.splat"
    meta_path = work / "fixture.json"
    code, stdout, stderr = run_script(
        "to_splat.py",
        [
            "--ply", str(ply), "--out", str(out), "--json", str(meta_path),
            "--sh-degree", "1",
            # 显式指定阈值：本用例校验的是**过滤逻辑本身**，不该依赖默认策略。
            # 默认已改为 auto（按本模型尺度分布的 P99.5 自适应）——那是有意的产品行为，
            # 拿它当断言的隐含前提，会让「策略调整」误报成「过滤逻辑坏了」。
            "--max-scale", "1.0",
        ],
    )
    check(code == 0, f"to_splat.py 退出码应为 0，实际 {code}\nstdout={stdout}\nstderr={stderr}")

    raw = read_splat_raw(out)
    check(len(raw) == len(expected), f"高斯数量应为 {len(expected)}，实际 {len(raw)}")
    check(out.stat().st_size == len(expected) * GAUSSIAN_BYTES, "输出字节数应为 数量 × 32")

    for index, (record, decoded) in enumerate(zip(expected, raw)):
        x, y, z, sx, sy, sz, r, g, b, a, qw, qx, qy, qz = decoded
        almost(x, record["xyz"][0], 1e-6, f"第 {index} 条 x 不一致")
        almost(y, record["xyz"][1], 1e-6, f"第 {index} 条 y 不一致")
        almost(z, record["xyz"][2], 1e-6, f"第 {index} 条 z 不一致")
        for axis, value in enumerate((sx, sy, sz)):
            # PLY 里 scale_* 就是线性尺度，转换器不再取 exp
            almost(value, record["scale"][axis], 1e-6 * max(1.0, abs(value)), f"第 {index} 条 scale_{axis} 不一致")
        want_rgb = expect_rgb(record["dc"])
        check((r, g, b) == want_rgb, f"第 {index} 条颜色不一致：期望 {want_rgb}，实际 {(r, g, b)}")
        check(a == expect_alpha(record["opacity"]), f"第 {index} 条 alpha 不一致：期望 {expect_alpha(record['opacity'])}，实际 {a}")
        want_q = expect_quat(record["rot"])
        got_q = ((qw - 128) / 128.0, (qx - 128) / 128.0, (qy - 128) / 128.0, (qz - 128) / 128.0)
        norm = math.sqrt(sum(v * v for v in got_q))
        for axis in range(4):
            almost(got_q[axis] / norm, want_q[axis], 1.0 / 128.0, f"第 {index} 条四元数分量 {axis} 超出 1/128")

    sh_path = out.with_suffix(".shcoeff.bin")
    check(sh_path.is_file(), "缺少 .shcoeff.bin")
    per_gaussian = 3 * 3  # K=3（1 阶球谐每通道 3 个系数）× 3 通道
    check(
        sh_path.stat().st_size == len(expected) * per_gaussian * 4,
        f".shcoeff.bin 大小应为 {len(expected) * per_gaussian * 4}，实际 {sh_path.stat().st_size}",
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    required = {
        "format",
        "bytesPerGaussian",
        "count",
        "shDegree",
        "shRestCount",
        "sourcePly",
        "bounds",
        "aabbMeters",
        "createdAt",
        "units",
        "axes",
    }
    missing = required - set(meta)
    check(not missing, f"JSON 缺少字段：{sorted(missing)}")
    check(meta["format"] == "splat", "format 应为 splat")
    check(meta["bytesPerGaussian"] == 32, "bytesPerGaussian 应为 32")
    check(meta["count"] == len(expected), "count 与文件不一致")
    check(meta["shDegree"] == 1, "shDegree 应为 1")
    check(meta["shRestCount"] == 9, f"shRestCount 应为 9，实际 {meta['shRestCount']}")
    check(meta["units"] == "meters", "units 应为 meters")
    check(meta["axes"]["system"] == "COLMAP 3DGS", "axes.system 应为 COLMAP 3DGS")
    check(meta["axes"]["y"] == "down", "axes.y 应为 down")
    bounds = meta["bounds"]
    for key in ("min", "max", "center", "size", "radius"):
        check(key in bounds, f"bounds 缺少 {key}")
    xs = [record["xyz"][0] for record in expected]
    almost(bounds["min"][0], min(xs), 1e-5, "bounds.min.x 不正确")
    almost(bounds["max"][0], max(xs), 1e-5, "bounds.max.x 不正确")
    almost(bounds["center"][0], (min(xs) + max(xs)) / 2.0, 1e-5, "bounds.center.x 不正确")
    check(meta["filters"]["droppedByOpacity"] == 1, "应记录 1 个被 opacity 过滤掉的高斯")
    check(meta["filters"]["droppedByScale"] == 1, "应记录 1 个被尺度过滤掉的高斯")


@test
def test_min_opacity_and_decimate(work: Path) -> None:
    """min-opacity 过滤与 decimate 抽稀的精确条数。"""
    records = make_records(4)
    records[0]["opacity"] = 10.0  # 0.99995
    records[1]["opacity"] = 1.0  # 0.731
    records[2]["opacity"] = -1.0  # 0.269
    records[3]["opacity"] = -10.0  # 0.0000454
    ply = work / "opacity.ply"
    write_minimal_ply(ply, records)

    out = work / "keep2.splat"
    code, _stdout, stderr = run_script(
        "to_splat.py", ["--ply", str(ply), "--out", str(out), "--sh-degree", "0", "--min-opacity", "0.5"]
    )
    check(code == 0, f"退出码应为 0，实际 {code}，stderr={stderr}")
    kept = read_splat_raw(out)
    check(len(kept) == 2, f"--min-opacity 0.5 应保留 2 个，实际 {len(kept)}")
    check(out.stat().st_size == 2 * GAUSSIAN_BYTES, "输出大小应为 64 字节")

    out_half = work / "half.splat"
    code, _stdout, stderr = run_script(
        "to_splat.py", ["--ply", str(ply), "--out", str(out_half), "--sh-degree", "0", "--decimate", "0.5"]
    )
    check(code == 0, f"退出码应为 0，实际 {code}，stderr={stderr}")
    half = read_splat_raw(out_half)
    check(len(half) == 2, f"--decimate 0.5 应保留 2 个，实际 {len(half)}")
    # 保留的应该是 opacity 最高的两条（records[0]、records[1]）
    opacities = [record["opacity"] for record in sorted(records, key=lambda r: r["opacity"], reverse=True)[:2]]
    for decoded in half:
        check(
            any(abs(decoded[1] - record["xyz"][1]) < 1e-6 for record in records if record["opacity"] in opacities),
            "抽稀保留的不是 opacity 最高的高斯",
        )
    check(not out.with_suffix(".shcoeff.bin").exists() or out.with_suffix(".shcoeff.bin").stat().st_size == 0, "sh-degree 0 不应写高阶系数")


# ---------------------------------------------------------------------------
# 测试 2：make_test_scene.py
# ---------------------------------------------------------------------------


@test
def test_make_test_scene(work: Path) -> None:
    """合成场景可解析、包围盒合理、观测文件重投影误差 < 1px。"""
    splat = work / "test-scene.splat"
    meta_path = work / "test-scene.json"
    obs_path = work / "test-scene-observations.json"
    poses_path = work / "test-scene-cameras.json"
    code, stdout, stderr = run_script(
        "make_test_scene.py",
        [
            "--out", str(splat),
            "--json", str(meta_path),
            "--cameras", str(obs_path),
            "--camera-poses", str(poses_path),
            "--count", "4000",
        ],
    )
    check(code == 0, f"make_test_scene.py 退出码应为 0，实际 {code}\nstdout={stdout}\nstderr={stderr}")
    check(splat.stat().st_size == 4000 * GAUSSIAN_BYTES, f"文件大小应为 {4000 * GAUSSIAN_BYTES}，实际 {splat.stat().st_size}")

    records = list(to_splat.iter_splat_records(splat))
    check(len(records) == 4000, f"应能解析出 4000 个高斯，实际 {len(records)}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bounds = meta["bounds"]
    check(meta["count"] == 4000, "元数据 count 不正确")
    check(meta["axes"]["y"] == "down", "坐标系约定应为 y 向下")
    check(meta["synthetic"] is True, "应标记 synthetic=True")

    # 包围盒合理性：房间是 6×2.8×5 米，y 轴向下（地面 y=0，天花板 y=-2.8）
    check(-3.2 <= bounds["min"][0] and bounds["max"][0] <= 3.2, f"x 包围盒超出房间：{bounds}")
    check(-2.9 <= bounds["min"][1] and bounds["max"][1] <= 0.1, f"y 包围盒超出房间：{bounds}")
    check(-2.6 <= bounds["min"][2] and bounds["max"][2] <= 2.6, f"z 包围盒超出房间：{bounds}")
    check(bounds["size"][0] > 4.0 and bounds["size"][1] > 2.0 and bounds["size"][2] > 3.0, f"房间尺寸不合理：{bounds['size']}")
    check(4.0 <= bounds["radius"] <= 6.0, f"包围球半径不合理：{bounds['radius']}")

    colors = {(record.rgba[0], record.rgba[1], record.rgba[2]) for record in records}
    check(len(colors) > 20, f"颜色过于单一（只有 {len(colors)} 种），光照/物体区分可能失效")
    alphas = {record.rgba[3] for record in records}
    check(all(200 <= a <= 255 for a in alphas), f"alpha 超出合理范围：{sorted(alphas)[:5]}")

    poses = json.loads(poses_path.read_text(encoding="utf-8"))
    check(len(poses["cameras"]) == 8, f"应有 8 个环绕相机，实际 {len(poses['cameras'])}")
    for camera in poses["cameras"]:
        check(len(camera["position"]) == 3 and len(camera["target"]) == 3, "相机位姿字段不完整")
        check(len(camera["c2w"]) == 16 and len(camera["w2c"]) == 16, "c2w/w2c 应为 16 个元素的 4×4 矩阵")

    # 用自己实现的针孔投影校验观测数据
    observations = json.loads(obs_path.read_text(encoding="utf-8"))
    intrinsics = observations["intrinsics"]
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    points = {p["id"]: p["xyz"] for p in observations["points3D"]}
    cameras = {c["id"]: c for c in observations["cameras"]}
    check(len(points) == 6, f"应有 6 个已知三维点，实际 {len(points)}")
    check(len(observations["observations"]) >= 6, "观测条目太少")
    check(len(cameras) >= 3, f"观测文件至少应涉及 3 个相机，实际 {len(cameras)}")

    per_point: dict[str, int] = {pid: 0 for pid in points}
    max_error = 0.0
    for observation in observations["observations"]:
        pid = observation["pointId"]
        camera = cameras[observation["cameraId"]]
        per_point[pid] += 1
        xyz = points[pid]
        r = camera["rotation"]
        t = camera["translation"]
        dx = xyz[0] - camera["position"][0]
        dy = xyz[1] - camera["position"][1]
        dz = xyz[2] - camera["position"][2]
        x_cam = r[0] * dx + r[1] * dy + r[2] * dz
        y_cam = r[3] * dx + r[4] * dy + r[5] * dz
        z_cam = r[6] * dx + r[7] * dy + r[8] * dz
        check(z_cam > 0, f"点 {pid} 在相机 {observation['cameraId']} 后方，观测数据有误")
        u = fx * x_cam / z_cam + cx
        v = fy * y_cam / z_cam + cy
        error = math.hypot(u - observation["uv"][0], v - observation["uv"][1])
        max_error = max(max_error, error)
        check(error < 1.0, f"点 {pid} 在相机 {observation['cameraId']} 的重投影误差 {error:.4f}px ≥ 1px")
        # 内参一致性：t 必须等于 -R·C
        tx = -(r[0] * camera["position"][0] + r[1] * camera["position"][1] + r[2] * camera["position"][2])
        almost(t[0], tx, 1e-6, "translation 与 position/rotation 不一致")
    for pid, count in per_point.items():
        check(count >= 1, f"三维点 {pid} 没有任何观测")
    print(f"      最大重投影误差 = {max_error:.6f} px（{len(observations['observations'])} 条观测）", flush=True)


@test
def test_ply_roundtrip_matches_direct_splat(work: Path) -> None:
    """make_test_scene 的 --ply 经 to_splat.py 往返后应与直接生成的 .splat 一致。"""
    direct = work / "direct.splat"
    ply = work / "direct.ply"
    code, stdout, stderr = run_script(
        "make_test_scene.py",
        [
            "--out", str(direct),
            "--json", str(work / "direct.json"),
            "--cameras", str(work / "direct-obs.json"),
            "--camera-poses", str(work / "direct-poses.json"),
            "--count", "2000",
            "--ply", str(ply),
        ],
    )
    check(code == 0, f"make_test_scene.py 失败：{code}\n{stdout}\n{stderr}")
    back = work / "back.splat"
    code, stdout, stderr = run_script(
        "to_splat.py",
        [
            "--ply", str(ply), "--out", str(back), "--sh-degree", "0",
            # 本用例校验的是**往返一致性**（逐字段比对），不是过滤策略。
            # 必须关掉尺度过滤，否则随机生成的测试场景里那些大尺度高斯
            # 会被默认策略裁掉，数量就对不上了（实测 2000 → 1990）。
            "--max-scale", "0",
        ],
    )
    check(code == 0, f"to_splat.py 失败：{code}\n{stdout}\n{stderr}")

    first = read_splat_raw(direct)
    second = read_splat_raw(back)
    check(len(first) == len(second), f"往返后数量不一致：{len(first)} vs {len(second)}")
    for index, (a, b) in enumerate(zip(first, second)):
        for field in (0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13):  # 位置 / RGBA / 四元数必须完全相同
            check(a[field] == b[field], f"第 {index} 条第 {field} 个字段往返后不一致：{a[field]} vs {b[field]}")
        for axis in (3, 4, 5):  # 尺度经过 log/exp 往返，允许 1 个 float32 ulp 的差别
            almost(b[axis], a[axis], 1e-5 * max(1.0, abs(a[axis])), f"第 {index} 条尺度往返差异过大")


# ---------------------------------------------------------------------------
# 测试 3：抽帧的模糊度算法
# ---------------------------------------------------------------------------


def write_pgm(path: Path, width: int, height: int, data: bytes) -> None:
    path.write_bytes(b"P5\n%d %d\n255\n" % (width, height) + data)


def write_png_gray(path: Path, width: int, height: int, data: bytes, filter_type: int = 0) -> None:
    """写一张 8 位灰度 PNG（用于验证内置 PNG 解码器）。"""
    rows = bytearray()
    previous = bytes(width)
    for y in range(height):
        line = bytearray(data[y * width : (y + 1) * width])
        if filter_type == 1:  # Sub：减去左侧像素
            for i in range(len(line) - 1, 0, -1):
                line[i] = (line[i] - line[i - 1]) & 0xFF
        elif filter_type == 2:  # Up：减去上方像素
            for i in range(len(line)):
                line[i] = (line[i] - previous[i]) & 0xFF
        rows.append(filter_type)
        rows.extend(line)
        previous = data[y * width : (y + 1) * width]

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(rows))) + chunk(b"IEND", b"")
    path.write_bytes(png)


def checkerboard(width: int, height: int, cell: int = 4) -> bytes:
    return bytes(
        240 if ((x // cell) + (y // cell)) % 2 == 0 else 15
        for y in range(height)
        for x in range(width)
    )


def blur_image(width: int, height: int, data: bytes, radius: int = 16) -> bytes:
    """可分离方框模糊，制造「不清晰」的测试图（半径大于纹理周期才能有效低通）。"""
    horizontal = bytearray(width * height)
    for y in range(height):
        base = y * width
        total = sum(data[base : base + radius + 1])
        count = radius + 1
        for x in range(width):
            horizontal[base + x] = total // count
            add = base + x + radius + 1
            drop = base + x - radius
            if add < base + width:
                total += data[add]
                count += 1
            if drop >= base:
                total -= data[drop]
                count -= 1
    out = bytearray(width * height)
    for x in range(width):
        total = sum(horizontal[y * width + x] for y in range(radius + 1))
        count = radius + 1
        for y in range(height):
            out[y * width + x] = total // count
            add = (y + radius + 1) * width + x
            drop = (y - radius) * width + x
            if add < width * height:
                total += horizontal[add]
                count += 1
            if drop >= 0:
                total -= horizontal[drop]
                count -= 1
    return bytes(out)


@test
def test_blur_detection(work: Path) -> None:
    """拉普拉斯方差：清晰图得分高、模糊图得分低（PGM 与 PNG 两条解码路径）。"""
    width, height = 160, 120
    sharp = checkerboard(width, height)
    blurred = blur_image(width, height, sharp, radius=16)

    sharp_pgm = work / "sharp.pgm"
    blur_pgm = work / "blur.pgm"
    sharp_png = work / "sharp.png"
    blur_png = work / "blur.png"
    sub_png = work / "sharp_sub.png"
    write_pgm(sharp_pgm, width, height, sharp)
    write_pgm(blur_pgm, width, height, blurred)
    write_png_gray(sharp_png, width, height, sharp, filter_type=0)
    write_png_gray(blur_png, width, height, blurred, filter_type=2)
    write_png_gray(sub_png, width, height, sharp, filter_type=1)

    sharp_score, _w, _h = extract_frames.image_blur_score(sharp_pgm)
    blur_score, _w2, _h2 = extract_frames.image_blur_score(blur_pgm)
    check(sharp_score > blur_score, f"清晰图得分（{sharp_score:.1f}）应高于模糊图（{blur_score:.1f}）")
    threshold = 60.0
    check(sharp_score >= threshold, f"清晰图得分 {sharp_score:.1f} 应 ≥ 阈值 {threshold}")
    check(blur_score < threshold, f"模糊图得分 {blur_score:.1f} 应 < 阈值 {threshold}")

    # PNG 解码路径（含 Sub/Up 行过滤）必须与 PGM 得到同一幅图
    png_sharp, _, _ = extract_frames.image_blur_score(sharp_png)
    png_blur, _, _ = extract_frames.image_blur_score(blur_png)
    png_sub, _, _ = extract_frames.image_blur_score(sub_png)
    almost(png_sharp, sharp_score, 1e-6, "PNG(过滤器 0) 与 PGM 解码结果不一致")
    almost(png_blur, blur_score, 1e-6, "PNG(过滤器 2/Up) 与 PGM 解码结果不一致")
    almost(png_sub, sharp_score, 1e-6, "PNG(过滤器 1/Sub) 与 PGM 解码结果不一致")

    # 纯色图（完全无纹理）得分应为 0
    flat = bytes([128] * (width * height))
    flat_pgm = work / "flat.pgm"
    write_pgm(flat_pgm, width, height, flat)
    flat_score, _, _ = extract_frames.image_blur_score(flat_pgm)
    almost(flat_score, 0.0, 1e-9, "纯色图的拉普拉斯方差应为 0")
    print(f"      清晰={sharp_score:.1f} 模糊={blur_score:.1f} 纯色={flat_score:.1f}（阈值 {threshold}）", flush=True)


# ---------------------------------------------------------------------------
# 测试 4：COLMAP 二进制解析器
# ---------------------------------------------------------------------------


@test
def test_colmap_binary_roundtrip(work: Path) -> None:
    """自己写的读写器往返一致，且能解析「按官方格式手工拼出的字节」。"""
    model_dir = work / "model"
    camera_a = colmap_recon.ColmapCamera(1, 4, 1920, 1080, [1250.0, 1248.5, 960.0, 541.5, 0.012, -0.003, 0.0002, -0.0001])
    camera_b = colmap_recon.ColmapCamera(2, 0, 800, 600, [520.0, 400.0, 300.0])
    rotation = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    image_a = colmap_recon.ColmapImage(
        7, colmap_recon.rotmat_to_qvec(rotation), (0.25, -0.5, 1.75), 1, "frame_0001.png",
        [(10.5, 20.25), (30.0, 40.0)], [3, colmap_recon.INVALID_POINT3D_ID],
    )
    image_b = colmap_recon.ColmapImage(
        9, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 2, "frame_0002.png", [(1.0, 2.0)], [5],
    )
    point = colmap_recon.ColmapPoint3D(3, (1.5, -2.25, 3.75), (10, 200, 255), 0.375, [7, 9], [0, 1])
    model = colmap_recon.ColmapModel({1: camera_a, 2: camera_b}, {7: image_a, 9: image_b}, {3: point})
    colmap_recon.write_model(model_dir, model)
    loaded = colmap_recon.read_model(model_dir)

    check(set(loaded.cameras) == {1, 2}, "相机数量不一致")
    check(loaded.cameras[1] == camera_a, "相机 A 往返不一致")
    check(loaded.cameras[2] == camera_b, "相机 B 往返不一致")
    check(loaded.images[7].qvec == image_a.qvec, "qvec 往返不一致")
    check(loaded.images[7].name == "frame_0001.png", "图片名往返不一致")
    check(loaded.images[7].xys == image_a.xys, "二维点往返不一致")
    check(loaded.images[7].point3D_ids == [3, colmap_recon.INVALID_POINT3D_ID], "point3D_ids 往返不一致")
    # 负数 point3D_id 应被规范化成 uint64 哨兵
    normalized = work / "normalized"
    colmap_recon.write_model(
        normalized,
        colmap_recon.ColmapModel({1: camera_b}, {1: colmap_recon.ColmapImage(1, (1, 0, 0, 0), (0, 0, 0), 1, "a.png", [(0.0, 0.0)], [-1])}, {}),
    )
    check(
        colmap_recon.read_model(normalized).images[1].point3D_ids == [colmap_recon.INVALID_POINT3D_ID],
        "-1 的 point3D_id 应写成 uint64 哨兵",
    )
    check(loaded.points3D[3] == point, "三维点往返不一致")
    almost(loaded.mean_reprojection_error, 0.375, 1e-9, "平均重投影误差不正确")
    almost(loaded.mean_track_length, 2.0, 1e-9, "平均 track 长度不正确")
    check(loaded.cameras[1].intrinsics() == (1250.0, 1248.5, 960.0, 541.5), "OPENCV 内参解析不正确")
    check(loaded.cameras[2].intrinsics() == (520.0, 520.0, 400.0, 300.0), "SIMPLE_PINHOLE 内参解析不正确")

    # 相机光心：R = I 时 C = -t
    almost(loaded.images[9].camera_center()[2], 0.0, 1e-9, "光心计算不正确")

    # 完全按 COLMAP 官方二进制格式手工拼接一份文件，验证读取器不是「只认自己写的格式」
    manual = work / "manual"
    manual.mkdir()
    numbers: list[bytes] = []
    numbers.append(struct.pack("<Q", 1))  # 1 台相机
    numbers.append(struct.pack("<iiQQ", 42, 1, 640, 480))  # PINHOLE，4 个参数
    numbers.append(struct.pack("<4d", 500.0, 501.0, 320.0, 240.0))
    (manual / "cameras.bin").write_bytes(b"".join(numbers))

    payload: list[bytes] = [struct.pack("<Q", 1)]  # 1 张图片
    payload.append(struct.pack("<idddddddi", 42, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 42))
    payload.append(b"cam42.png\x00")  # 以 \0 结尾的文件名
    payload.append(struct.pack("<Q", 2))  # 2 个二维点
    payload.append(struct.pack("<ddQ", 12.5, 34.5, 1))
    payload.append(struct.pack("<ddQ", 56.5, 78.5, 18446744073709551615))  # 无效点用 uint64 最大值
    (manual / "images.bin").write_bytes(b"".join(payload))

    point_bytes: list[bytes] = [struct.pack("<Q", 1)]
    point_bytes.append(struct.pack("<QdddBBB", 1, 0.5, -0.5, 1.5, 255, 0, 128))
    point_bytes.append(struct.pack("<d", 0.75))
    point_bytes.append(struct.pack("<Q", 1))
    point_bytes.append(struct.pack("<ii", 42, 0))
    (manual / "points3D.bin").write_bytes(b"".join(point_bytes))

    parsed = colmap_recon.read_model(manual)
    check(parsed.cameras[42].model == "PINHOLE", "手工 cameras.bin 的模型解析错误")
    check(parsed.cameras[42].width == 640 and parsed.cameras[42].height == 480, "手工 cameras.bin 的尺寸解析错误")
    check(parsed.images[42].name == "cam42.png", "手工 images.bin 的文件名解析错误")
    check(parsed.images[42].xys == [(12.5, 34.5), (56.5, 78.5)], "手工 images.bin 的二维点解析错误")
    check(parsed.images[42].point3D_ids[1] == 18446744073709551615, "无效点 id（uint64 最大值）解析错误")
    check(parsed.points3D[1].rgb == (255, 0, 128), "手工 points3D.bin 的颜色解析错误")
    almost(parsed.points3D[1].error, 0.75, 1e-9, "手工 points3D.bin 的误差解析错误")


# ---------------------------------------------------------------------------
# 测试 5：流水线其它组件的冒烟检查
# ---------------------------------------------------------------------------


@test
def test_dependency_probes_and_helpers(work: Path) -> None:
    """依赖探测、日志与工具函数在缺依赖环境下也不应抛异常。"""
    for probe in (config.has_numpy, config.has_cv2, config.has_torch, config.has_gsplat, config.has_pycolmap, config.has_imageio_ffmpeg):
        value = probe()
        check(isinstance(value, bool), f"{probe.__name__} 应返回 bool")
    check(config.find_ffmpeg() is None or isinstance(config.find_ffmpeg(), str), "find_ffmpeg 返回值类型错误")
    check(config.find_colmap() is None or isinstance(config.find_colmap(), str), "find_colmap 返回值类型错误")
    check(config.human_bytes(2048) == "2.00 KB", f"human_bytes 结果异常：{config.human_bytes(2048)}")
    check("秒" in config.format_duration(3.5), "format_duration 应为中文")
    table = config.format_table(["名称", "状态"], [["torch", "❌ 未安装"], ["numpy", "✅ 就绪"]])
    check("torch" in table and "numpy" in table, "format_table 输出异常")
    check(config.dump_json(work / "a.json", {"中文": "值"}).is_file(), "dump_json 失败")
    check(config.load_json(work / "a.json")["中文"] == "值", "load_json 往返失败")
    check((_PROJECT_ROOT / "pipeline").is_dir(), "项目根目录探测错误")
    check(config.MODELS_DIR.name == "models" and config.MODELS_DIR.parent.name == "data", "MODELS_DIR 应为 data/models")


@test
@test
def test_auto_max_scale_keeps_body(work: Path) -> None:
    """`--max-scale auto` 必须只裁**尾部离群**大高斯，不碰主体（Bug A 回归）。

    背景（真实事故）：导出脚本硬编码 `--max-scale 0.5`，而训练器 `_clamp_scales()`
    把线性尺度夹在 `prune_scale × spatial_scale`（某任务 = 1.1185 m）。
    导出阈值比训练器上限还紧 2.24 倍，于是训练器**刻意保留**的
    「铺墙铺地的大高斯」被整批删掉 —— 实测丢了 51.5%（12338 → 5036），
    模型因此发空发虚。

    这里用「一堆正常尺度 + 一条巨大离群」的夹具守住两点：
      1) auto 不能把主体砍掉（数量必须保住）；
      2) 真正的离群大高斯要被裁掉；
      3) 尺度丢弃率 < 5%（文档第七节验收标准 1）。
    """
    records = make_records(200)
    for i, rec in enumerate(records):
        rec["scale"] = (0.02 + (i % 10) * 0.008, 0.02, 0.02)  # 0.02~0.092 m，正常房间尺度
    # 一条明显离群的大高斯（模拟未收敛的漂浮团）
    records.append(dict(records[0]))
    records[-1]["xyz"] = (50.0, 50.0, 50.0)
    records[-1]["scale"] = (400.0, 400.0, 400.0)

    ply = work / "auto.ply"
    write_minimal_ply(ply, records)
    out = work / "auto.splat"
    meta_path = work / "auto.json"
    code, stdout, stderr = run_script(
        "to_splat.py",
        ["--ply", str(ply), "--out", str(out), "--json", str(meta_path), "--sh-degree", "0",
         "--max-scale", "auto", "--floater-factor", "0"],
    )
    check(code == 0, f"to_splat.py 退出码应为 0，实际 {code}\nstdout={stdout}\nstderr={stderr}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    filters = meta.get("filters") or {}
    kept = int(meta.get("count") or 0)
    dropped = int(filters.get("droppedByScale") or 0)
    total = kept + dropped + int(filters.get("droppedByOpacity") or 0) + int(filters.get("droppedInvalid") or 0)

    check(
        filters.get("maxScaleMeters") is not None and float(filters["maxScaleMeters"]) > 0,
        f"元数据应记录实际使用的尺度上限（在 filters 里），实际 {filters.get('maxScaleMeters')}",
    )
    check(kept >= 200, f"auto 不应裁掉主体：200 条正常高斯应全部保留，实际只留 {kept}")
    check(dropped >= 1, f"那条 400 m 的离群大高斯应被裁掉，实际 droppedByScale={dropped}")
    ratio = dropped / total if total else 1.0
    check(ratio < 0.05, f"尺度丢弃率应 < 5%（文档验收标准），实际 {ratio:.1%}")


def test_cli_error_paths(work: Path) -> None:
    """缺文件/参数错误时的退出码与中文提示。"""
    code, _stdout, stderr = run_script("to_splat.py", ["--ply", str(work / "missing.ply"), "--out", str(work / "x.splat")])
    check(code == 3, f"PLY 不存在时应返回 3，实际 {code}")
    check("找不到 PLY" in stderr or "找不到 PLY" in _stdout, "缺少中文错误提示")

    code, _stdout, _stderr = run_script("extract_frames.py", ["--video", str(work / "nope.mp4"), "--out", str(work / "frames")])
    check(code == 3, f"视频不存在时应返回 3，实际 {code}")

    code, _stdout, _stderr = run_script("colmap_recon.py", ["--images", str(work / "noimages"), "--workspace", str(work / "ws")])
    check(code == 3, f"图片目录为空时应返回 3，实际 {code}")

    code, _stdout, _stderr = run_script("train_gsplat.py", ["--data", str(work / "noscene"), "--out", str(work / "train")])
    check(code == 2, f"数据目录不存在时应返回 2，实际 {code}")

    bad = work / "bad.ply"
    bad.write_bytes(b"not a ply at all\n")
    code, _stdout, _stderr = run_script("to_splat.py", ["--ply", str(bad), "--out", str(work / "bad.splat")])
    check(code == 4, f"损坏的 PLY 应返回 4，实际 {code}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def make_workdir() -> tuple[Path, Callable[[], None]]:
    """准备临时工作目录。

    优先用仓库内的 .piptmp/pipeline-tests（受限沙箱下系统临时目录常常
    只允许创建顶层目录，不允许再建子目录）；不可用时再退回系统临时目录。
    """
    fallback = _PROJECT_ROOT / ".piptmp" / "pipeline-tests"
    try:
        shutil.rmtree(fallback, ignore_errors=True)
        fallback.mkdir(parents=True, exist_ok=True)
        probe = fallback / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return fallback, lambda: shutil.rmtree(fallback, ignore_errors=True)
    except OSError:
        pass
    tmp = tempfile.mkdtemp(prefix="pipeline-test-")
    return Path(tmp), lambda: shutil.rmtree(tmp, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    config.ensure_utf8_stdio()
    config.log_rule("三维重建流水线自测")
    config.log_info(f"Python：{sys.version.split()[0]}（{PYTHON}）")
    config.log_info(f"项目根目录：{_PROJECT_ROOT}")
    config.log_info(
        f"可选依赖：numpy={'有' if config.has_numpy() else '无'} "
        f"cv2={'有' if config.has_cv2() else '无'} "
        f"torch={'有' if config.has_torch() else '无'} "
        f"gsplat={'有' if config.has_gsplat() else '无'} "
        f"pycolmap={'有' if config.has_pycolmap() else '无'}"
    )
    config.log_info(f"共 {len(_TESTS)} 个用例。\n")

    work, cleanup = make_workdir()
    failed = 0
    try:
        for index, (name, func) in enumerate(_TESTS, start=1):
            started = time.time()
            case_dir = work / f"{index:02d}-{name}"
            case_dir.mkdir(parents=True, exist_ok=True)
            try:
                func(case_dir)
            except Exception as exc:  # noqa: BLE001 - 任何异常都算失败
                elapsed = time.time() - started
                _RESULTS.append((name, False, f"{type(exc).__name__}: {exc}", elapsed))
                failed += 1
                print(f"[{index}/{len(_TESTS)}] ❌ {name}  ({elapsed:.2f}s)", flush=True)
                print(f"        失败原因：{type(exc).__name__}: {exc}", flush=True)
                if os.environ.get("PIPELINE_TEST_TRACEBACK"):
                    import traceback  # noqa: PLC0415

                    traceback.print_exc()
            else:
                elapsed = time.time() - started
                _RESULTS.append((name, True, "", elapsed))
                print(f"[{index}/{len(_TESTS)}] ✅ {name}  ({elapsed:.2f}s)", flush=True)
    finally:
        cleanup()

    passed = len(_RESULTS) - failed
    config.log_rule()
    for name, ok, message, elapsed in _RESULTS:
        mark = "✅ 通过" if ok else "❌ 失败"
        line = f"  {mark}  {name}  ({elapsed:.2f}s)"
        if not ok:
            line += f"\n        {message}"
        print(line, flush=True)
    print(f"\n{passed} 通过 / {failed} 失败", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
