"""pipeline/to_splat.py —— 把 3DGS 训练输出的二进制 PLY 转成 Web 端可直接加载的 .splat。

.splat（antimatter15 流式格式）每个高斯固定 32 字节，小端：
    offset  0..11  float32  x, y, z            世界坐标（米）
    offset 12..23  float32  scaleX, scaleY, scaleZ
                              线性尺度 = exp(scale_raw)，不是 log 值
    offset 24..27  uint8    R, G, B, A
                              RGB = clamp(0.5 + 0.28209479177387814 * f_dc, 0, 1) * 255
                              A   = clamp(sigmoid(opacity), 0, 1) * 255
    offset 28..31  uint8    四元数 (w, x, y, z)
                              单位化后按 round(q * 128 + 128) 编码，钳制到 0..255。
                              PLY 的 rot_0..3 本身就是 (w, x, y, z)，写入顺序也必须是 w, x, y, z。

高度方向 / 坐标系约定：COLMAP / 3DGS 世界系，**y 轴向下**，单位米。转换只做数据搬运，
不做任何坐标系翻转；Web 端的 up 向量由前端的 world 设置决定。

用法：
    python pipeline/to_splat.py --ply point_cloud.ply --out scene.splat --json scene.json

本脚本在**只有标准库**的环境下即可运行（自行解析 PLY 头 + struct 解包）；
检测到 numpy 时自动启用向量化快速路径，并在启动阶段用合成数据把两条路径对比校验，
一旦不一致就自动退回纯 Python 路径，保证结果永远正确。
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Sequence

_PIPELINE_DIR = Path(__file__).resolve().parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

import config  # noqa: E402  （同目录模块，需先把 pipeline/ 加进 sys.path）

# ---------------------------------------------------------------------------
# 常量与基础数学
# ---------------------------------------------------------------------------

#: 球谐 0 阶（直流项）系数，与 3DGS 官方实现一致
SH_C0: float = 0.28209479177387814

#: 雾护栏：单条高斯的最大线性尺度上限（米），`--max-scale auto` 会与它取小。
#:
#: 依据实测（`tools/analyze_model_fog.py` 跑遍模型库）：
#:   健康模型 >0.5 m 的高斯占 **0%**，尺度中位 0.05~0.21 m；
#:   发雾模型 >0.5 m 的高斯占 **56%**，尺度中位 0.59 m，且这批高斯明显更白更淡
#:   （亮度 139~149 vs 108，alpha 0.10 vs 0.16）—— 它们叠起来就是那层白雾。
#: 0.5 m 是两类**分得最开**的位置：健康模型完全不受影响，发雾模型被削掉。
#: 这不是拍脑袋的魔数，而是「室内场景里单个超过半米的高斯不可能是表面元素、
#: 只能是没收敛的雾团」这一物理判断。
FOG_SCALE_CAP: float = 0.5

#: 每个高斯在 .splat 中的字节数
GAUSSIAN_BYTES: int = 32

#: .splat 单条记录：6 个 float32（位置 + 尺度）+ 8 个 uint8（RGBA + 四元数）
SPLAT_RECORD = struct.Struct("<6f8B")

#: 每块处理的记录条数（流式读写，内存占用与总高斯数无关）
CHUNK_RECORDS: int = 60000

#: 位置绝对值超过该值即视为非法（同时排除 NaN / Inf）
POSITION_LIMIT: float = 1e7

#: PLY 类型名 → struct 格式字符
PLY_TO_STRUCT: dict[str, str] = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}

#: PLY 类型名 → numpy dtype（小端）
PLY_TO_NUMPY: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


class PlyError(Exception):
    """PLY 解析/布局相关的错误。"""


def sigmoid(x: float) -> float:
    """数值稳定的 sigmoid。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def to_u8_half_up(value: float) -> int:
    """按「四舍五入（.5 进位）」转 uint8 并钳制到 0..255；NaN 记为 0。

    刻意用 floor(v + 0.5) 而非 Python 的 round()（银行家舍入），
    以保证 numpy 向量化路径与纯 Python 路径逐字节一致。
    """
    if not (value > 0.0):  # 同时覆盖 <= 0 与 NaN
        return 0
    if value >= 255.0:
        return 255
    return int(math.floor(value + 0.5))


def encode_dc_to_rgb(dc0: float, dc1: float, dc2: float) -> tuple[int, int, int]:
    """由 SH 直流项计算 RGB（0..255）。"""
    return (
        to_u8_half_up((0.5 + SH_C0 * dc0) * 255.0),
        to_u8_half_up((0.5 + SH_C0 * dc1) * 255.0),
        to_u8_half_up((0.5 + SH_C0 * dc2) * 255.0),
    )


def encode_opacity_to_alpha(raw_opacity: float) -> int:
    """由 logit 形式的 opacity 计算 0..255 的 alpha。"""
    return to_u8_half_up(sigmoid(raw_opacity) * 255.0)


def encode_quaternion(qw: float, qx: float, qy: float, qz: float) -> tuple[int, int, int, int]:
    """单位化四元数并编码为 4 个 uint8，返回顺序固定为 (w, x, y, z)。"""
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or norm <= 1e-12:
        return (255, 128, 128, 128)  # 退化输入视为单位旋转 w=1
    return (
        to_u8_half_up(qw / norm * 128.0 + 128.0),
        to_u8_half_up(qx / norm * 128.0 + 128.0),
        to_u8_half_up(qy / norm * 128.0 + 128.0),
        to_u8_half_up(qz / norm * 128.0 + 128.0),
    )


def decode_u8_to_quaternion(qw: int, qx: int, qy: int, qz: int) -> tuple[float, float, float, float]:
    """把 .splat 里的 4 个 uint8 还原成归一化前的四元数（便于测试比对）。"""
    return ((qw - 128) / 128.0, (qx - 128) / 128.0, (qy - 128) / 128.0, (qz - 128) / 128.0)


# ---------------------------------------------------------------------------
# PLY 头解析
# ---------------------------------------------------------------------------


@dataclass
class PlyProperty:
    """PLY 元素的一个属性。"""

    name: str
    type_name: str
    is_list: bool = False
    count_type: str | None = None


@dataclass
class PlyElement:
    """PLY 的一个元素（vertex / face ...）。"""

    name: str
    count: int
    properties: list[PlyProperty] = field(default_factory=list)

    @property
    def has_list(self) -> bool:
        return any(p.is_list for p in self.properties)

    @property
    def record_size(self) -> int:
        """每条记录的字节数；含 list 属性时不可预知，返回 0。"""
        if self.has_list:
            return 0
        return sum(struct.calcsize(PLY_TO_STRUCT[p.type_name]) for p in self.properties)

    @property
    def property_names(self) -> list[str]:
        return [p.name for p in self.properties]


@dataclass
class PlyHeader:
    """PLY 文件头。"""

    format: str
    elements: list[PlyElement]
    data_offset: int
    comments: list[str] = field(default_factory=list)

    @property
    def is_binary(self) -> bool:
        return self.format.startswith("binary")

    @property
    def little_endian(self) -> bool:
        return self.format == "binary_little_endian"

    def element(self, name: str) -> PlyElement | None:
        for el in self.elements:
            if el.name == name:
                return el
        return None

    def vertex(self) -> PlyElement:
        el = self.element("vertex")
        if el is None:
            raise PlyError("PLY 文件里没有 vertex 元素，无法转换。")
        return el


def read_ply_header(source: str | Path | BinaryIO) -> PlyHeader:
    """读取 PLY 文件头，返回 PlyHeader（含二进制数据起始偏移）。"""
    if isinstance(source, (str, Path)):
        handle = open(source, "rb")
        should_close = True
    else:
        handle = source
        should_close = False
    try:
        first = handle.readline()
        if first.strip() != b"ply":
            raise PlyError("不是合法的 PLY 文件（首行不是 'ply'）。")
        offset = len(first)
        fmt: str | None = None
        elements: list[PlyElement] = []
        comments: list[str] = []
        current: PlyElement | None = None
        while True:
            line = handle.readline()
            if not line:
                raise PlyError("PLY 头没有以 end_header 结束，文件可能被截断。")
            offset += len(line)
            text = line.decode("ascii", "replace").strip()
            if not text:
                continue
            parts = text.split()
            key = parts[0].lower()
            if key in ("comment", "obj_info"):
                comments.append(text)
            elif key == "format":
                fmt = parts[1].lower()
            elif key == "element":
                if len(parts) < 3:
                    raise PlyError(f"PLY 头 element 行不完整：{text}")
                current = PlyElement(name=parts[1], count=int(parts[2]))
                elements.append(current)
            elif key == "property":
                if current is None:
                    raise PlyError("PLY 头里 property 出现在 element 之前。")
                if parts[1].lower() == "list":
                    if len(parts) < 5:
                        raise PlyError(f"PLY 头 list 属性不完整：{text}")
                    current.properties.append(
                        PlyProperty(name=parts[4], type_name=parts[3].lower(), is_list=True, count_type=parts[2].lower())
                    )
                else:
                    if len(parts) < 3:
                        raise PlyError(f"PLY 头 property 行不完整：{text}")
                    current.properties.append(PlyProperty(name=parts[2], type_name=parts[1].lower()))
            elif key == "end_header":
                break
        if fmt is None:
            raise PlyError("PLY 头缺少 format 行。")
        if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
            raise PlyError(f"不支持的 PLY 格式：{fmt}")
        for el in elements:
            for prop in el.properties:
                if prop.type_name not in PLY_TO_STRUCT:
                    raise PlyError(f"不支持的 PLY 属性类型：{prop.type_name}（属性 {prop.name}）")
        return PlyHeader(format=fmt, elements=elements, data_offset=offset, comments=comments)
    finally:
        if should_close:
            handle.close()


def vertex_data_offset(header: PlyHeader) -> int:
    """计算 vertex 元素二进制数据的起始偏移（跳过排在它前面的定长元素）。"""
    offset = header.data_offset
    for el in header.elements:
        if el.name == "vertex":
            return offset
        if el.has_list:
            raise PlyError(
                f"PLY 里 vertex 之前的元素 {el.name} 含 list 属性，无法定位 vertex 数据；"
                "请使用 COLMAP / 3DGS 标准导出的 PLY。"
            )
        offset += el.record_size * el.count
    raise PlyError("PLY 文件里没有 vertex 元素。")


# ---------------------------------------------------------------------------
# 字段定位
# ---------------------------------------------------------------------------


@dataclass
class FieldMap:
    """把 PLY 属性名映射到记录元组下标。"""

    positions: tuple[int, int, int]
    colors_dc: tuple[int, int, int] | None
    colors_rgb: tuple[int, int, int] | None
    opacity: int | None
    scales: tuple[int, int, int] | None
    rotations: tuple[int, int, int, int]
    rest: list[int]


def resolve_fields(header: PlyHeader, quat_order: str = "wxyz") -> FieldMap:
    """在 vertex 元素里定位位置/颜色/不透明度/尺度/旋转/高阶 SH 的下标。"""
    element = header.vertex()
    if element.has_list:
        raise PlyError("vertex 元素含 list 属性，本转换器不支持这种布局。")
    index = {name: i for i, name in enumerate(element.property_names)}

    def pick(*names: str) -> int | None:
        for name in names:
            if name in index:
                return index[name]
        return None

    def pick3(prefix: str) -> tuple[int, int, int] | None:
        got = [pick(f"{prefix}{i}") for i in range(3)]
        if any(g is None for g in got):
            return None
        return (int(got[0]), int(got[1]), int(got[2]))  # type: ignore[arg-type]

    def pick4(prefix: str) -> tuple[int, int, int, int] | None:
        got = [pick(f"{prefix}{i}") for i in range(4)]
        if any(g is None for g in got):
            return None
        return (int(got[0]), int(got[1]), int(got[2]), int(got[3]))  # type: ignore[arg-type]

    px, py, pz = pick("x"), pick("y"), pick("z")
    if px is None or py is None or pz is None:
        raise PlyError("PLY 的 vertex 元素缺少 x/y/z 属性，无法确定高斯位置。")
    positions = (px, py, pz)

    colors_dc = pick3("f_dc_")
    colors_rgb: tuple[int, int, int] | None = None
    red, green, blue = pick("red", "r"), pick("green", "g"), pick("blue", "b")
    if red is not None and green is not None and blue is not None:
        colors_rgb = (red, green, blue)
    if colors_dc is None and colors_rgb is None:
        raise PlyError("PLY 里既没有 f_dc_0..2（3DGS 球谐直流项）也没有 red/green/blue 属性，无法还原颜色。")

    rotations = pick4("rot_")
    if rotations is None:
        raise PlyError("PLY 缺少 rot_0..3 四元数属性，无法转换。")
    if quat_order == "xyzw":
        rotations = (rotations[1], rotations[2], rotations[3], rotations[0])

    rest: list[int] = []
    i = 0
    while f"f_rest_{i}" in index:
        rest.append(index[f"f_rest_{i}"])
        i += 1

    return FieldMap(
        positions=positions,
        colors_dc=colors_dc,
        colors_rgb=colors_rgb,
        opacity=pick("opacity", "alpha"),
        scales=pick3("scale_"),
        rotations=rotations,
        rest=rest,
    )


def max_supported_degree(rest_properties: int) -> int:
    """由 f_rest_* 属性数量推算 PLY 里实际存到几阶球谐（最高 3 阶）。"""
    if rest_properties <= 0 or rest_properties % 3 != 0:
        return 0
    per_channel = rest_properties // 3
    degree = 3
    while degree > 0 and (degree + 1) ** 2 - 1 > per_channel:
        degree -= 1
    return degree if (degree + 1) ** 2 - 1 == per_channel else 0


# ---------------------------------------------------------------------------
# .splat 读写工具
# ---------------------------------------------------------------------------


class SplatWriter:
    """流式写 .splat 文件（攒够 1MB 再落盘）。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: BinaryIO = open(self.path, "wb")
        self.count = 0
        self._buffer: list[bytes] = []
        self._buffer_size = 0

    def write_raw(self, data: bytes) -> None:
        """写入已经编码好的原始字节。"""
        if not data:
            return
        self._buffer.append(data)
        self._buffer_size += len(data)
        if self._buffer_size >= 1 << 20:
            self.flush()

    def write_gaussian(
        self,
        position: Sequence[float],
        scale: Sequence[float],
        rgba: Sequence[int],
        quat: Sequence[int],
    ) -> None:
        """写入一条 32 字节记录；scale 为线性尺度，quat 顺序必须是 (w, x, y, z)。"""
        self.write_raw(
            SPLAT_RECORD.pack(
                position[0], position[1], position[2],
                scale[0], scale[1], scale[2],
                rgba[0], rgba[1], rgba[2], rgba[3],
                quat[0], quat[1], quat[2], quat[3],
            )
        )
        self.count += 1

    def flush(self) -> None:
        if self._buffer:
            self._handle.write(b"".join(self._buffer))
            self._buffer.clear()
            self._buffer_size = 0

    def close(self) -> None:
        self.flush()
        self._handle.close()

    def __enter__(self) -> "SplatWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class SplatRecord:
    """一条解码后的 .splat 记录（尺度为线性值，四元数为编码后的 uint8）。"""

    x: float
    y: float
    z: float
    scale: tuple[float, float, float]
    rgba: tuple[int, int, int, int]
    quat: tuple[int, int, int, int]

    @property
    def quat_normalized(self) -> tuple[float, float, float, float]:
        """把 uint8 四元数还原到大约的归一化区间 [-1, 1]。"""
        return decode_u8_to_quaternion(*self.quat)


def iter_splat_records(path: Path | str, limit: int | None = None) -> Iterator[SplatRecord]:
    """逐条读取 .splat（供测试、校验与 Web 端等价性对照使用）。"""
    read = 0
    with open(path, "rb") as handle:
        while True:
            data = handle.read(GAUSSIAN_BYTES)
            if len(data) < GAUSSIAN_BYTES:
                if data:
                    raise PlyError(f".splat 文件尾部有 {len(data)} 字节残缺记录（文件长度应为 32 的整数倍）。")
                return
            rec = SPLAT_RECORD.unpack(data)
            yield SplatRecord(rec[0], rec[1], rec[2], (rec[3], rec[4], rec[5]), (rec[6], rec[7], rec[8], rec[9]), (rec[10], rec[11], rec[12], rec[13]))
            read += 1
            if limit is not None and read >= limit:
                return


def splat_file_info(path: Path | str) -> dict[str, Any]:
    """.splat 文件基本信息（条数、字节数、是否按 32 字节对齐）。"""
    target = Path(path)
    size = target.stat().st_size
    return {
        "path": str(target),
        "bytes": size,
        "count": size // GAUSSIAN_BYTES,
        "aligned": size % GAUSSIAN_BYTES == 0,
    }


# ---------------------------------------------------------------------------
# 转换参数与主流程
# ---------------------------------------------------------------------------


@dataclass
class ConvertOptions:
    """转换参数。"""

    sh_degree: int = 3
    min_opacity: float = 0.005
    max_scale: float = 1.0
    decimate: float = 1.0
    quat_order: str = "wxyz"
    #: PLY 里的 scale_* 是否为 log 值。
    #: 标准 3DGS PLY 存的是**线性**尺度，所以默认 False。
    #: 只有早期版本的 train_gsplat.py 产物才是 log 值（那时需设为 True）。
    scales_are_log: bool = False
    #: 离群浮尘（floaters）剔除倍数：距离稳健中心超过 `该倍数 × P95 距离` 的高斯丢弃。
    #: 0 表示不剔除。
    #:
    #: 为什么必须做：实测一个 175 帧的真实案件模型里，**43 个高斯（0.10%）**
    #: 把 min/max 包围球半径从真实的 10.6 m 撑到 **79.78 m**（max/P95 = 11.9×）。
    #: 而前端就是拿这个半径给相机取景的 —— 相机被推到 80 米外，
    #: 10 米的房间在屏幕上缩成一个点，于是用户看到的是「一团雾，看不出房间」。
    #: 极少数野点足以毁掉整个观感，所以这里按「稳健中心 + P95 距离」把野点切掉。
    floater_factor: float = 2.0
    #: 计算稳健统计量时的抽样步长（1 = 全部参与；越大越快）。
    floater_stride: int = 1


@dataclass
class ChunkResult:
    """一块 PLY 记录编码后的结果。"""

    records: bytes = b""
    positions: Any = None  # (n, 3) float64 数组（numpy）或 list[tuple]
    raw_opacity: Any = None  # (n,) float64 数组（numpy）或 list[float]
    sh_bytes: bytes = b""
    dropped_invalid: int = 0
    dropped_scale: int = 0
    dropped_opacity: int = 0


def _record_struct(header: PlyHeader) -> struct.Struct:
    """按 PLY 头顺序构建整条记录的 struct（用于 iter_unpack）。"""
    prefix = "<" if header.format == "binary_little_endian" else ">"
    chars = "".join(PLY_TO_STRUCT[p.type_name] for p in header.vertex().properties)
    return struct.Struct(prefix + chars)


def _numpy_dtype(element: PlyElement) -> Any:
    """构建与该 PLY vertex 记录逐字节对应的 numpy 结构化 dtype。"""
    numpy = config.try_import("numpy")
    if numpy is None:  # pragma: no cover - 调用方已保证 numpy 可用
        raise PlyError("numpy 不可用。")
    dtype = numpy.dtype([(p.name, PLY_TO_NUMPY[p.type_name]) for p in element.properties])
    if dtype.itemsize != element.record_size:
        raise PlyError(f"内部错误：numpy dtype 大小 {dtype.itemsize} 与 PLY 记录大小 {element.record_size} 不一致。")
    return dtype


def _iter_binary_chunks(path: Path, offset: int, record_size: int, total: int, chunk: int) -> Iterator[bytes]:
    """流式读取 vertex 二进制数据块。"""
    with open(path, "rb") as handle:
        handle.seek(offset)
        remaining = total
        while remaining > 0:
            take = min(chunk, remaining)
            data = handle.read(take * record_size)
            if len(data) < take * record_size:
                raise PlyError(f"PLY 数据段被截断：期望 {take * record_size} 字节，实际读到 {len(data)} 字节。")
            yield data
            remaining -= take


def _iter_ascii_records(path: Path, offset: int, total: int, n_props: int) -> Iterator[list[float]]:
    """流式读取 ASCII PLY 的 vertex 记录。"""
    read = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        for line in handle:
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) < n_props:
                raise PlyError("ASCII PLY 的数据行字段数少于表头声明。")
            yield [float(t) for t in tokens[:n_props]]
            read += 1
            if read >= total:
                return
        if read < total:
            raise PlyError("ASCII PLY 的数据行数少于表头声明的 vertex 数量。")


# ---------------------------------------------------------------------------
# 纯 Python 编解码（权威实现，numpy 路径必须与之一致）
# ---------------------------------------------------------------------------


def classify_record(rec: Sequence[float], fields: FieldMap, opts: ConvertOptions) -> str:
    """过滤判定，返回 'ok' / 'invalid' / 'scale' / 'opacity'。

    判定顺序固定，numpy 路径必须保持同样的优先级才能保证统计口径一致。
    """
    px, py, pz = rec[fields.positions[0]], rec[fields.positions[1]], rec[fields.positions[2]]
    if not (-POSITION_LIMIT < px < POSITION_LIMIT and -POSITION_LIMIT < py < POSITION_LIMIT and -POSITION_LIMIT < pz < POSITION_LIMIT):
        return "invalid"  # NaN / Inf / 极端值都在这里被排除
    if fields.scales is not None:
        # 3DGS 官方 PLY 的 scale_* 是**线性尺度**（已激活），直接比较即可。
        # 早期本项目的 train_gsplat.py 误写成 log 值，这里为了兼容那种旧产物，
        # 用 --scales-are-log 开关（默认关）。
        raw_scales = [rec[i] for i in fields.scales]
        linear = [math.exp(v) if opts.scales_are_log else v for v in raw_scales]
        if not all(value == value for value in linear):  # NaN 检测
            return "invalid"
        if any(value > opts.max_scale for value in linear):
            return "scale"
    if fields.opacity is not None:
        raw = rec[fields.opacity]
        if raw != raw:
            return "invalid"
        if sigmoid(raw) < opts.min_opacity:
            return "opacity"
    return "ok"


def encode_record_python(
    rec: Sequence[float], fields: FieldMap, opts: ConvertOptions, sh_indices: list[int]
) -> tuple[bytes, list[float]]:
    """把一条通过过滤的 PLY 记录编码成 .splat 字节（+ 高阶 SH 系数）。"""
    px, py, pz = float(rec[fields.positions[0]]), float(rec[fields.positions[1]]), float(rec[fields.positions[2]])
    if fields.scales is not None:
        if opts.scales_are_log:
            sx, sy, sz = (math.exp(rec[i]) for i in fields.scales)
        else:
            sx, sy, sz = (float(rec[i]) for i in fields.scales)
    else:
        sx = sy = sz = 0.01
    if fields.colors_dc is not None:
        r8, g8, b8 = encode_dc_to_rgb(float(rec[fields.colors_dc[0]]), float(rec[fields.colors_dc[1]]), float(rec[fields.colors_dc[2]]))
    else:
        assert fields.colors_rgb is not None
        r8 = to_u8_half_up(float(rec[fields.colors_rgb[0]]))
        g8 = to_u8_half_up(float(rec[fields.colors_rgb[1]]))
        b8 = to_u8_half_up(float(rec[fields.colors_rgb[2]]))
    if fields.opacity is not None:
        a8 = to_u8_half_up(sigmoid(float(rec[fields.opacity])) * 255.0)
    else:
        a8 = 255
    qw, qx, qy, qz = encode_quaternion(
        float(rec[fields.rotations[0]]),
        float(rec[fields.rotations[1]]),
        float(rec[fields.rotations[2]]),
        float(rec[fields.rotations[3]]),
    )
    payload = SPLAT_RECORD.pack(px, py, pz, sx, sy, sz, r8, g8, b8, a8, qw, qx, qy, qz)
    sh_values = [float(rec[i]) for i in sh_indices] if sh_indices else []
    return payload, sh_values


def encode_chunk_python(
    chunk: bytes,
    record_struct: struct.Struct,
    fields: FieldMap,
    opts: ConvertOptions,
    sh_indices: list[int],
) -> ChunkResult:
    """纯 Python 路径：编码一整块 PLY 记录。"""
    result = ChunkResult(positions=[], raw_opacity=[])
    out = bytearray()
    sh_out = bytearray()
    for rec in record_struct.iter_unpack(chunk):
        status = classify_record(rec, fields, opts)
        if status == "invalid":
            result.dropped_invalid += 1
            continue
        if status == "scale":
            result.dropped_scale += 1
            continue
        if status == "opacity":
            result.dropped_opacity += 1
            continue
        payload, sh_values = encode_record_python(rec, fields, opts, sh_indices)
        out.extend(payload)
        result.positions.append((float(rec[fields.positions[0]]), float(rec[fields.positions[1]]), float(rec[fields.positions[2]])))
        result.raw_opacity.append(float(rec[fields.opacity]) if fields.opacity is not None else 12.0)
        if sh_values:
            sh_out.extend(struct.pack(f"<{len(sh_values)}f", *sh_values))
    result.records = bytes(out)
    result.sh_bytes = bytes(sh_out)
    return result


def encode_chunk_numpy(
    chunk: bytes,
    element: PlyElement,
    fields: FieldMap,
    opts: ConvertOptions,
    numpy: Any,
    sh_indices: list[int],
) -> ChunkResult:
    """numpy 向量化路径：编码一整块 PLY 记录。

    所有中间量都用 float64 计算，最后一次性转 float32，
    与纯 Python 路径（math.exp + struct.pack('<f')）逐字节一致。
    """
    names = element.property_names
    arr = numpy.frombuffer(chunk, dtype=_numpy_dtype(element))

    px = arr[names[fields.positions[0]]].astype(numpy.float64)
    py = arr[names[fields.positions[1]]].astype(numpy.float64)
    pz = arr[names[fields.positions[2]]].astype(numpy.float64)
    pos_ok = (
        numpy.isfinite(px) & numpy.isfinite(py) & numpy.isfinite(pz)
        & (numpy.abs(px) < POSITION_LIMIT) & (numpy.abs(py) < POSITION_LIMIT) & (numpy.abs(pz) < POSITION_LIMIT)
    )

    if fields.scales is not None:
        _sx = arr[names[fields.scales[0]]].astype(numpy.float64)
        _sy = arr[names[fields.scales[1]]].astype(numpy.float64)
        _sz = arr[names[fields.scales[2]]].astype(numpy.float64)
        if opts.scales_are_log:
            sx, sy, sz = numpy.exp(_sx), numpy.exp(_sy), numpy.exp(_sz)
        else:
            sx, sy, sz = _sx, _sy, _sz
    else:
        sx = sy = sz = numpy.full(len(arr), 0.01)
    scale_finite = numpy.isfinite(sx) & numpy.isfinite(sy) & numpy.isfinite(sz)
    scale_bad = scale_finite & ((sx > opts.max_scale) | (sy > opts.max_scale) | (sz > opts.max_scale))

    if fields.opacity is not None:
        raw_op = arr[names[fields.opacity]].astype(numpy.float64)
        op_finite = numpy.isfinite(raw_op)
        alpha = 1.0 / (1.0 + numpy.exp(-numpy.where(op_finite, raw_op, 0.0)))
        opacity_bad = op_finite & (alpha < opts.min_opacity)
    else:
        raw_op = numpy.full(len(arr), 12.0)
        op_finite = numpy.ones(len(arr), dtype=bool)
        alpha = numpy.ones(len(arr))
        opacity_bad = numpy.zeros(len(arr), dtype=bool)

    invalid = ~(pos_ok & scale_finite & op_finite)
    scale_bad = scale_bad & ~invalid
    opacity_bad = opacity_bad & ~invalid & ~scale_bad
    passed = ~(invalid | scale_bad | opacity_bad)

    result = ChunkResult(
        dropped_invalid=int(invalid.sum()),
        dropped_scale=int(scale_bad.sum()),
        dropped_opacity=int(opacity_bad.sum()),
    )
    sel = numpy.nonzero(passed)[0]
    if len(sel) == 0:
        result.positions = numpy.zeros((0, 3), dtype=numpy.float64)
        result.raw_opacity = numpy.zeros(0, dtype=numpy.float64)
        return result

    px, py, pz = px[sel], py[sel], pz[sel]
    sx, sy, sz = sx[sel], sy[sel], sz[sel]
    alpha = alpha[sel]

    if fields.colors_dc is not None:
        dc = numpy.stack(
            [
                arr[names[fields.colors_dc[0]]].astype(numpy.float64),
                arr[names[fields.colors_dc[1]]].astype(numpy.float64),
                arr[names[fields.colors_dc[2]]].astype(numpy.float64),
            ],
            axis=1,
        )[sel]
        col = numpy.nan_to_num(0.5 + SH_C0 * dc, nan=0.0, posinf=1.0, neginf=0.0)
        rgb_u8 = numpy.floor(numpy.clip(col, 0.0, 1.0) * 255.0 + 0.5).astype(numpy.uint8)
    else:
        assert fields.colors_rgb is not None
        rgb = numpy.stack(
            [
                arr[names[fields.colors_rgb[0]]].astype(numpy.float64),
                arr[names[fields.colors_rgb[1]]].astype(numpy.float64),
                arr[names[fields.colors_rgb[2]]].astype(numpy.float64),
            ],
            axis=1,
        )[sel]
        rgb_u8 = numpy.where(numpy.isfinite(rgb), numpy.floor(numpy.clip(rgb, 0.0, 255.0) + 0.5), 0.0).astype(numpy.uint8)

    alpha_u8 = numpy.floor(numpy.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(numpy.uint8)

    q = numpy.stack(
        [
            arr[names[fields.rotations[0]]].astype(numpy.float64),
            arr[names[fields.rotations[1]]].astype(numpy.float64),
            arr[names[fields.rotations[2]]].astype(numpy.float64),
            arr[names[fields.rotations[3]]].astype(numpy.float64),
        ],
        axis=1,
    )[sel]
    norm = numpy.sqrt((q * q).sum(axis=1))
    degenerate = (~numpy.isfinite(norm)) | (norm <= 1e-12)
    norm = numpy.where(degenerate, 1.0, norm)
    q = q / norm[:, None]
    quat = numpy.floor(q * 128.0 + 128.0 + 0.5)
    quat[degenerate] = (255.0, 128.0, 128.0, 128.0)
    quat_u8 = numpy.clip(quat, 0, 255).astype(numpy.uint8)

    out = numpy.empty(
        len(sel),
        dtype=numpy.dtype([("pos", "<f4", (3,)), ("scale", "<f4", (3,)), ("rgba", "u1", (4,)), ("quat", "u1", (4,))]),
    )
    out["pos"] = numpy.stack([px, py, pz], axis=1).astype("<f4")
    out["scale"] = numpy.stack([sx, sy, sz], axis=1).astype("<f4")
    out["rgba"] = numpy.concatenate([rgb_u8, alpha_u8[:, None]], axis=1)
    out["quat"] = quat_u8

    result.records = out.tobytes()
    result.positions = numpy.stack([px, py, pz], axis=1)
    result.raw_opacity = raw_op[sel]
    if sh_indices:
        sh = numpy.stack([arr[names[i]].astype("<f4") for i in sh_indices], axis=1)[sel]
        result.sh_bytes = numpy.ascontiguousarray(sh).tobytes()
    return result


# ---------------------------------------------------------------------------
# numpy 路径自检（用合成数据对齐两条实现，避免静默产生错误文件）
# ---------------------------------------------------------------------------


def _synthetic_chunk(element: PlyElement, count: int, numpy: Any) -> bytes:
    """构造一块含有正常值/边界值/NaN/Inf 的合成 PLY 记录数据。"""
    records: list[bytes] = []
    prefix = "<"
    fmt = prefix + "".join(PLY_TO_STRUCT[p.type_name] for p in element.properties)
    st = struct.Struct(fmt)
    for i in range(count):
        values: list[Any] = []
        for p in element.properties:
            name = p.name
            if name == "x":
                values.append(0.1 * i)
            elif name == "y":
                values.append(-0.25 * i)
            elif name == "z":
                values.append(0.05 * i - 1.0)
            elif name.startswith("f_dc_"):
                values.append(((i % 7) - 3) * 0.6)
            elif name.startswith("f_rest_"):
                values.append(0.01 * (i % 11) - 0.05)
            elif name == "opacity":
                values.append([8.0, 2.5, -4.0, 0.0, 12.0, -9.9, 3.3][i % 7])
            elif name.startswith("scale_"):
                values.append([-3.0, -1.2, 0.0, -5.5, 0.7, -2.2, 1.4][i % 7])
            elif name.startswith("rot_"):
                values.append([0.0, 0.5, -0.5, 1.0][i % 4])
            elif p.type_name in ("float", "float32", "double", "float64"):
                values.append(0.0)
            else:
                values.append(0)
        rec = st.pack(*values)
        records.append(rec)
    data = bytearray(b"".join(records))
    # 人为插入几个非法值：第 2 条记录位置为 NaN，第 3 条记录 scale_0 为 +Inf
    if count >= 4:
        nan = struct.pack("<f", float("nan"))
        inf = struct.pack("<f", float("inf"))
        names = element.property_names
        offsets: list[int] = []
        acc = 0
        for p in element.properties:
            offsets.append(acc)
            acc += struct.calcsize(PLY_TO_STRUCT[p.type_name])
        stride = element.record_size
        x_off = offsets[names.index("x")]
        data[1 * stride + x_off : 1 * stride + x_off + 4] = nan
        if "scale_0" in names:
            s_off = offsets[names.index("scale_0")]
            data[2 * stride + s_off : 2 * stride + s_off + 4] = inf
    return bytes(data)


def verify_numpy_path(header: PlyHeader, fields: FieldMap, opts: ConvertOptions, sh_indices: list[int], numpy: Any) -> bool:
    """用合成数据对比 numpy 路径与纯 Python 路径是否逐字节一致。"""
    element = header.vertex()
    try:
        chunk = _synthetic_chunk(element, 64, numpy)
        got = encode_chunk_numpy(chunk, element, fields, opts, numpy, sh_indices)
        expect = encode_chunk_python(chunk, _record_struct(header), fields, opts, sh_indices)
    except Exception as exc:  # noqa: BLE001 - 自检失败即退回慢速路径
        config.log_warn(f"numpy 快速路径自检异常（{exc}），改用纯 Python 路径。")
        return False
    same = (
        got.records == expect.records
        and got.sh_bytes == expect.sh_bytes
        and got.dropped_invalid == expect.dropped_invalid
        and got.dropped_scale == expect.dropped_scale
        and got.dropped_opacity == expect.dropped_opacity
    )
    if not same:
        config.log_warn("numpy 快速路径自检未通过，自动改用纯 Python 路径（结果不受影响，仅速度略慢）。")
    return same


# ---------------------------------------------------------------------------
# 转换主流程
# ---------------------------------------------------------------------------


def convert_ply_to_splat(
    ply_path: Path | str,
    out_path: Path | str,
    options: ConvertOptions | None = None,
    *,
    json_path: Path | str | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """执行 PLY → .splat 转换，返回写进 JSON 的元数据字典。"""
    opts = options or ConvertOptions()
    ply = Path(ply_path)
    out = Path(out_path)
    if not ply.is_file():
        raise FileNotFoundError(f"找不到 PLY 文件：{ply}")

    with open(ply, "rb") as handle:
        header = read_ply_header(handle)
    fields = resolve_fields(header, opts.quat_order)
    element = header.vertex()
    total = element.count
    if total <= 0:
        raise PlyError("PLY 的 vertex 数量为 0，没有可转换的高斯。")

    available_degree = max_supported_degree(len(fields.rest))
    wanted_degree = max(0, min(3, int(opts.sh_degree)))
    degree = min(wanted_degree, available_degree)
    if wanted_degree > 0 and degree < wanted_degree:
        config.log_warn(
            f"PLY 里只有 {len(fields.rest)} 个 f_rest_* 属性（最多支持 {available_degree} 阶），"
            f"请求 {wanted_degree} 阶，自动降为 {degree} 阶。"
        )
    k_per_channel = (degree + 1) ** 2 - 1 if degree > 0 else 0
    rest_per_channel = len(fields.rest) // 3 if len(fields.rest) % 3 == 0 else 0
    sh_indices: list[int] = []
    if k_per_channel > 0 and rest_per_channel >= k_per_channel:
        for channel in range(3):
            base = channel * rest_per_channel
            sh_indices.extend(fields.rest[base + k] for k in range(k_per_channel))
    else:
        k_per_channel = 0
        degree = 0

    # ---- --max-scale：先把 'auto' 解析成**具体数字**，再谈读取路径 ----
    #
    # ⚠️ 顺序很重要：必须在 `verify_numpy_path()` **之前**解析。
    #    否则 numpy 自检拿到的是字符串，会报
    #    "ufunc 'greater' did not contain a loop ... (Float64DType, StrDType)"
    #    然后被迫退回纯 Python 路径（第一版就踩了这个坑）。
    #
    # 为什么不再硬编码：训练器 `_clamp_scales()` 把线性尺度夹在
    # `prune_scale × spatial_scale`（某真实任务 = 0.1 × 11.185 = 1.1185 m），
    # 而导出脚本曾硬编码 0.5 m —— 比训练器上限还紧 2.24 倍，
    # 于是训练器**刻意保留**的大高斯（铺墙、铺地、覆盖大面积平滑区的主力）
    # 被整批当成噪点删掉。实测同一份 PLY：
    #     阈值 0.5     → 尺度丢弃 6358 个（51.5%），导出 5036 个
    #     阈值 1.1185  → 尺度丢弃 0 个，导出 11090 个（与阈值 999 结果完全一致）
    # 默认改为「本模型尺度分布的 99.5 百分位」：只裁真正的尾部离群大高斯。
    if isinstance(opts.max_scale, str):
        token = opts.max_scale.strip().lower()
        if token in ("auto", ""):
            resolved = resolve_auto_max_scale(ply, header, fields, opts, None, False)
            # ---- 雾护栏（fog cap）----------------------------------------------
            # 实测（tools/analyze_model_fog.py）：
            #   健康模型 room-real      尺度中位 0.138 m，>0.5 m 占 **0%**
            #   健康模型 sample-room    尺度中位 0.049 m，>0.5 m 占 0%
            #   用户认可的小视频模型      尺度中位 0.208 m，>0.5 m 占 **0%**
            #   发雾模型（白云）          尺度中位 0.59 m，>0.5 m 占 **56%**
            # 且大高斯明显更白更淡（亮度 139~149 vs 108，alpha 0.10 vs 0.16）——
            # **它们就是那朵云，把有颜色的细节盖住了。**
            #
            # 所以 auto 不能只看 P99.5：训练器 `_clamp_scales()` 的上限是
            # `prune_scale × spatial_scale`（某任务 = 1.1185 m），优化器会把大量高斯
            # 顶到那个上限，于是 P99.5 也接近 1.1 m —— 照单全收等于把雾留下。
            # 这里再压一道「物理上不可能是表面元素」的天花板。
            # 健康模型本来就没有超过它的高斯，因此**对好模型零影响**。
            if resolved > FOG_SCALE_CAP:
                config.log_warn(
                    f"尺度上限 auto 求得 {resolved:.3f} m，超过雾护栏 {FOG_SCALE_CAP:.2f} m，"
                    f"已按护栏截断。"
                )
                config.log_warn(
                    "  这通常说明**训练时尺度失控**（大量高斯被顶到 prune_scale×spatial_scale 的上限）："
                )
                config.log_warn(
                    "  那些又大又白又淡的高斯叠起来就是一层白雾，会盖住有颜色的细节。"
                )
                config.log_warn(
                    "  截断只是让**这一份**产物可用；根治要在训练侧收紧尺度上限并重训。"
                )
                resolved = FOG_SCALE_CAP
            if resolved > 0:
                config.log_info(
                    f"尺度上限 auto → {resolved:.4f} m（本模型尺度分布 P99.5 与雾护栏"
                    f" {FOG_SCALE_CAP:.2f} m 取小；只裁尾部离群与雾团，不碰真实表面元素）"
                )
            else:
                config.log_info("尺度上限 auto → 不限制（样本不足，交给训练器自身的钳位）")
            opts.max_scale = resolved
        else:
            try:
                opts.max_scale = float(token)
            except ValueError as exc:
                raise PlyError(f"--max-scale 只接受数字或 auto，收到 {opts.max_scale!r}") from exc
    if opts.max_scale <= 0:
        # ⚠️ 不能只是"记一句日志就完事"：过滤比较是 `sx > opts.max_scale`，
        #    阈值 0 会把**所有**正尺度高斯全部判为超限 → 一个都不剩 →
        #    报「过滤后没有任何高斯剩余」（实测踩到，test_pipeline 因此失败）。
        #    所以这里归一成一个"事实上不限制"的大数。
        #    不用 float('inf') 是因为它会让元数据写出 JSON 的 `Infinity`（非法 JSON），
        #    前端 JSON.parse 会直接炸。
        config.log_info("尺度上限 <= 0：本次不做尺度过滤。")
        opts.max_scale = 1e9

    # 选择读取路径：ASCII 一律走纯 Python；numpy 必须通过自检
    numpy: Any | None = None
    if header.format != "ascii" and config.has_numpy():
        candidate = config.try_import("numpy")
        if candidate is not None and verify_numpy_path(header, fields, opts, sh_indices, candidate):
            numpy = candidate
        elif candidate is not None:
            config.log_info("本次转换使用纯 Python 路径。")
    use_numpy = numpy is not None

    decimate = max(0.0, min(1.0, float(opts.decimate)))
    threshold: float | None = None
    tie_slots = 0
    if decimate < 1.0:
        scan = _scan_opacities(ply, header, fields, opts, numpy, use_numpy)
        if scan["kept"] == 0:
            raise PlyError("所有高斯都被过滤掉了（--min-opacity / --max-scale 太严格？），没有内容可输出。")
        keep_target = max(1, min(scan["kept"], int(scan["kept"] * decimate + 0.5)))
        threshold = scan["threshold_at"](keep_target)
        n_greater = scan["count_greater"](threshold)
        tie_slots = max(0, keep_target - n_greater)
        config.log_info(
            f"抽稀：过滤后 {scan['kept']} 个高斯，按 opacity 保留 {keep_target} 个"
            f"（阈值 opacity={sigmoid(threshold):.5f}，同分名额 {tie_slots}）。"
        )

    # 离群浮尘（floaters）剔除：用稳健统计先定一条切线，再在写出时按距离过滤。
    # 这一步直接决定前端取景半径，是「看不出房间」的头号原因（见 ConvertOptions.floater_factor）。
    floater = scan_floater_cutoff(ply, header, fields, opts, numpy, use_numpy)
    if floater["cutoff"] > 0:
        config.log_info(
            f"稳健统计：中心 ({floater['center'][0]:.2f}, {floater['center'][1]:.2f}, {floater['center'][2]:.2f})，"
            f"P95 距离 {floater['p95']:.2f} m → 浮尘剔除半径 {floater['cutoff']:.2f} m"
            f"（{floater['factor']:g}×P95，抽样 {floater['sampled']}/{floater['total']}）"
        )
    else:
        config.log_info("浮尘剔除：已关闭（--floater-factor 0）。")

    started = time.time()
    report = _write_splat(
        ply, out, header, fields, opts, numpy, use_numpy, sh_indices, k_per_channel, degree,
        threshold, tie_slots, progress, floater
    )
    elapsed = time.time() - started

    min_b = report["_min"]
    max_b = report["_max"]
    center = [round((min_b[i] + max_b[i]) / 2.0, 6) for i in range(3)]
    size = [round(max_b[i] - min_b[i], 6) for i in range(3)]
    radius = round(0.5 * math.sqrt(sum(v * v for v in size)), 6)
    bounds = {
        "min": [round(v, 6) for v in min_b],
        "max": [round(v, 6) for v in max_b],
        "center": center,
        "size": size,
        "radius": radius,
    }

    sh_path = out.with_suffix(".shcoeff.bin")
    metadata: dict[str, Any] = {
        "format": "splat",
        "bytesPerGaussian": GAUSSIAN_BYTES,
        "count": report["count"],
        "shDegree": degree,
        "shRestCount": k_per_channel * 3,
        "shRestPerChannel": k_per_channel,
        "sourcePly": str(ply),
        "sourcePlyBytes": ply.stat().st_size,
        "sourceGaussians": total,
        "outputBytes": out.stat().st_size,
        "bounds": bounds,
        "aabbMeters": {"min": bounds["min"], "max": bounds["max"], "size": bounds["size"]},
        # 稳健统计（浮尘剔除的依据）：即便 bounds 因为个别残留野点偏大，
        # 前端/下游也能用这个半径取景，不会再被一个点把相机推到几十米外。
        "robust": {
            "center": [round(v, 6) for v in report["floater"]["center"]] if report["floater"]["center"] else None,
            "p95Distance": round(report["floater"]["p95Distance"], 6) if report["floater"]["p95Distance"] else None,
            "radius": round(report["floater"]["cutoffRadius"], 6) if report["floater"]["cutoffRadius"] else None,
        },
        "floaterRemoval": report["floater"],
        "createdAt": config.now_iso(),
        "units": "meters",
        "axes": {"system": "COLMAP 3DGS", "x": "right", "y": "down", "z": "forward"},
        "filters": report["filters"],
        "elapsedSec": round(elapsed, 3),
        "usedNumpy": use_numpy,
        "generator": "pipeline/to_splat.py",
    }
    if report["shFloats"] > 0 and sh_path.is_file():
        metadata["shCoeffFile"] = sh_path.name
        metadata["shCoeffBytes"] = sh_path.stat().st_size
        metadata["shCoeffOrder"] = "每个高斯按通道主序排列：R 的 K 个系数 → G 的 K 个 → B 的 K 个（float32 小端）"

    if json_path:
        config.dump_json(json_path, metadata)
        config.log_ok(f"已写出元数据：{json_path}")

    return metadata


def scan_floater_cutoff(
    ply: Path,
    header: PlyHeader,
    fields: FieldMap,
    opts: ConvertOptions,
    numpy: Any | None,
    use_numpy: bool,
) -> dict[str, Any]:
    """第一遍扫描：用**稳健统计**算出「离群浮尘」的剔除半径。

    返回 `{center, p95, cutoff, sampled, total}`；`cutoff <= 0` 表示不剔除。

    为什么不用 min/max：min/max 完全由**最极端的那个点**决定。
    实测一个真实案件模型：43 个浮尘（占 0.10%）把半径从 10.6 m 撑到 79.78 m，
    前端据此取景 → 相机退到 80 米外 → 10 米的房间在屏幕上就是一个点。
    所以这里用「中位数中心 + P95 距离」，再乘一个倍数作为切线。
    """
    factor = float(opts.floater_factor or 0.0)
    if factor <= 0:
        return {"center": None, "p95": 0.0, "cutoff": 0.0, "sampled": 0, "total": int(header.vertex().count)}

    element = header.vertex()
    record_struct = _record_struct(header)
    offset = vertex_data_offset(header)
    stride = max(1, int(opts.floater_stride or 1))

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    seen = 0
    index = 0
    for chunk in _iter_binary_chunks(ply, offset, element.record_size, element.count, CHUNK_RECORDS):
        if use_numpy:
            positions = encode_chunk_numpy(chunk, element, fields, opts, numpy, []).positions
            for row in positions:
                index += 1
                if index % stride:
                    continue
                xs.append(float(row[0])); ys.append(float(row[1])); zs.append(float(row[2]))
            seen += len(positions)
        else:
            for rec in record_struct.iter_unpack(chunk):
                index += 1
                if classify_record(rec, fields, opts) != "ok":
                    continue
                seen += 1
                if index % stride:
                    continue
                xs.append(float(rec[fields.positions[0]]))
                ys.append(float(rec[fields.positions[1]]))
                zs.append(float(rec[fields.positions[2]]))

    if not xs:
        return {"center": None, "p95": 0.0, "cutoff": 0.0, "sampled": 0, "total": seen}

    if use_numpy:
        arr = numpy.stack([numpy.asarray(xs), numpy.asarray(ys), numpy.asarray(zs)], axis=1)
        center = numpy.median(arr, axis=0)
        dist = numpy.linalg.norm(arr - center, axis=1)
        p95 = float(numpy.percentile(dist, 95))
        center_list = [float(v) for v in center]
    else:
        xs.sort(); ys.sort(); zs.sort()
        center_list = [xs[len(xs) // 2], ys[len(ys) // 2], zs[len(zs) // 2]]
        d = sorted(
            math.sqrt((x - center_list[0]) ** 2 + (y - center_list[1]) ** 2 + (z - center_list[2]) ** 2)
            for x, y, z in zip(xs, ys, zs)
        )
        p95 = float(d[int(0.95 * (len(d) - 1))])

    cutoff = p95 * factor
    return {
        "center": center_list,
        "p95": p95,
        "cutoff": cutoff,
        "factor": factor,
        "sampled": len(xs),
        "total": seen,
    }


def _scan_opacities(
    ply: Path,
    header: PlyHeader,
    fields: FieldMap,
    opts: ConvertOptions,
    numpy: Any | None,
    use_numpy: bool,
) -> dict[str, Any]:
    """第一遍扫描：收集通过其它过滤条件的高斯的 opacity，用于抽稀阈值。"""
    element = header.vertex()
    record_struct = _record_struct(header)
    offset = vertex_data_offset(header)
    collected: list[float] = []
    if use_numpy:
        buffer = numpy.empty(element.count, dtype=numpy.float64)
    kept = 0
    for chunk in _iter_binary_chunks(ply, offset, element.record_size, element.count, CHUNK_RECORDS):
        if use_numpy:
            result = encode_chunk_numpy(chunk, element, fields, opts, numpy, [])
            n = len(result.raw_opacity)
            buffer[kept : kept + n] = result.raw_opacity
            kept += n
        else:
            for rec in record_struct.iter_unpack(chunk):
                if classify_record(rec, fields, opts) != "ok":
                    continue
                collected.append(float(rec[fields.opacity]) if fields.opacity is not None else 12.0)
                kept += 1

    if use_numpy:
        values = buffer[:kept]
        order = numpy.argsort(values)[::-1]
        sorted_values = values[order]

        def threshold_at(k: int) -> float:
            return float(sorted_values[k - 1])

        def count_greater(thr: float) -> int:
            return int((values > thr).sum())
    else:
        sorted_values = sorted(collected, reverse=True)

        def threshold_at(k: int) -> float:
            return float(sorted_values[k - 1])

        def count_greater(thr: float) -> int:
            lo, hi = 0, len(sorted_values)
            while lo < hi:  # 二分找第一个 <= thr 的位置
                mid = (lo + hi) // 2
                if sorted_values[mid] > thr:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

    return {"kept": kept, "threshold_at": threshold_at, "count_greater": count_greater}


def resolve_auto_max_scale(
    ply: Path,
    header: PlyHeader,
    fields: FieldMap,
    opts: ConvertOptions,
    numpy: Any | None,
    use_numpy: bool,
    *,
    percentile: float = 99.5,
    sample: int = 200_000,
) -> float:
    """`--max-scale auto` 的阈值：本模型线性尺度 max(sx,sy,sz) 的 P99.5。

    文档第四节做法 B。拿不到训练侧尺度信息时，用**百分位**而不是绝对值，
    这样只裁尾部离群大高斯（天空、远处噪点、未收敛的漂浮团），
    保住几何覆盖最大的主体。

    返回 0.0 表示「不限制」（没有样本）；调用方据此跳过尺度过滤。
    """
    if fields.scales is None:
        return 0.0
    element = header.vertex()
    record_struct = _record_struct(header)
    offset = vertex_data_offset(header)
    stride = max(1, int(element.count // sample)) if element.count > sample else 1
    vals: list[float] = []
    index = 0
    for chunk in _iter_binary_chunks(ply, offset, element.record_size, element.count, CHUNK_RECORDS):
        for rec in record_struct.iter_unpack(chunk):
            index += 1
            if index % stride:
                continue
            raw = [rec[i] for i in fields.scales]
            if not all(v == v for v in raw):  # NaN 检测（v != v 即 NaN）
                continue
            linear = [math.exp(v) if opts.scales_are_log else v for v in raw]
            if not all(math.isfinite(v) for v in linear):
                continue
            vals.append(max(linear))
    if not vals:
        return 0.0
    vals.sort()
    k = min(len(vals) - 1, max(0, int(round(percentile / 100.0 * (len(vals) - 1)))))
    return float(vals[k])


def surface_outlier_mask(
    pos: numpy.ndarray,
    scale: numpy.ndarray,
    *,
    k: int = 16,
    mad_k: float = 3.0,
    max_sample: int = 40000,
) -> numpy.ndarray:
    """局部平面外点滤波：删掉**不在任何表面上**的浮点（治「法向云雾」）。

    为什么需要：现有的浮尘剔除是按「到场景中心的距离」（径向）删的，
    对**贴着墙飘的那层雾**完全无效 —— 它径向很近，但在法向上离墙面很远。
    用户反馈「墙面切向的云雾没有了，但法向的云雾还是很多」说的就是这批点。

    做法（点云去噪的标准手段）：
      1) 对每个高斯取 k 个最近邻；
      2) 用邻居做 PCA，最小特征值方向即局部法向；
      3) 高斯到「邻居拟合平面」的距离记为 d；
      4) 用**中位数 + k×MAD** 判定离群（MAD 比标准差抗野点），超过就删。

    只对采样点算（默认 4 万），再按下标映射回全体；大数据集也不会卡。
    返回 True = 保留。
    """
    n = len(pos)
    keep = numpy.ones(n, dtype=bool)
    if n < k + 5 or mad_k <= 0:
        return keep

    if n > max_sample:
        idx = numpy.linspace(0, n - 1, max_sample).astype(numpy.int64)
    else:
        idx = numpy.arange(n)
    sub = pos[idx]
    m = len(sub)
    kk = min(k, m - 1)
    if kk < 4:
        return keep

    # 分块算 k 近邻，控制内存
    dist = numpy.empty((m, kk))
    nbr = numpy.empty((m, kk), dtype=numpy.int64)
    block = max(1, min(2000, m))
    for i in range(0, m, block):
        b = sub[i : i + block]
        d = numpy.linalg.norm(b[:, None, :] - sub[None, :, :], axis=2)
        d[numpy.arange(len(b)), numpy.arange(i, i + len(b))] = numpy.inf
        part = numpy.argpartition(d, kk - 1, axis=1)[:, :kk]
        rows = numpy.arange(len(b))[:, None]
        dist[i : i + len(b)] = d[rows, part]
        nbr[i : i + len(b)] = part

    # 每个点：到邻居拟合平面的距离（PCA 最小特征值方向的投影）
    resid = numpy.zeros(m)
    for i in range(m):
        pts = sub[nbr[i]]
        c = pts.mean(axis=0)
        q = pts - c
        try:
            _, _, vt = numpy.linalg.svd(q, full_matrices=False)
            normal = vt[-1]
        except numpy.linalg.LinAlgError:
            continue
        resid[i] = abs(float((sub[i] - c) @ normal))

    med = float(numpy.median(resid))
    mad = float(numpy.median(numpy.abs(resid - med)))
    if mad <= 1e-12:
        # MAD 为 0（几乎共面）：用 P95 兜底，避免把所有点判成离群
        thr = float(numpy.percentile(resid, 95))
    else:
        thr = med + mad_k * 1.4826 * mad
    bad = resid > thr
    keep[idx[bad]] = False
    return keep


def _write_splat(
    ply: Path,
    out: Path,
    header: PlyHeader,
    fields: FieldMap,
    opts: ConvertOptions,
    numpy: Any | None,
    use_numpy: bool,
    sh_indices: list[int],
    k_per_channel: int,
    degree: int,
    threshold: float | None,
    tie_slots: int,
    progress: bool,
    floater: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """第二遍扫描并写出 .splat（以及可选的 .shcoeff.bin）。"""
    element = header.vertex()
    total = element.count
    offset = vertex_data_offset(header)
    record_struct = _record_struct(header)
    process_sh = degree > 0 and k_per_channel > 0 and bool(sh_indices)

    # 浮尘切线（稳健统计得出；<=0 表示不剔除）
    floater = floater or {}
    _fcenter = floater.get("center")
    _fcut = float(floater.get("cutoff") or 0.0)
    _fcut2 = _fcut * _fcut if (_fcenter and _fcut > 0) else 0.0
    dropped_floater = 0

    def floater_ok(x: float, y: float, z: float) -> bool:
        if _fcut2 <= 0:
            return True
        dx = x - _fcenter[0]
        dy = y - _fcenter[1]
        dz = z - _fcenter[2]
        return dx * dx + dy * dy + dz * dz <= _fcut2

    dropped_invalid = dropped_scale = dropped_opacity = dropped_decimate = 0
    kept = 0
    reported_blocks = 0
    sh_floats = 0
    slots_left = tie_slots
    min_b = [float("inf")] * 3
    max_b = [float("-inf")] * 3

    def update_bounds(px: float, py: float, pz: float) -> None:
        if px < min_b[0]:
            min_b[0] = px
        if px > max_b[0]:
            max_b[0] = px
        if py < min_b[1]:
            min_b[1] = py
        if py > max_b[1]:
            max_b[1] = py
        if pz < min_b[2]:
            min_b[2] = pz
        if pz > max_b[2]:
            max_b[2] = pz

    sh_handle: BinaryIO | None = open(out.with_suffix(".shcoeff.bin"), "wb") if process_sh else None
    try:
        with SplatWriter(out) as writer:
            for chunk in _iter_binary_chunks(ply, offset, element.record_size, total, CHUNK_RECORDS):
                if use_numpy:
                    result = encode_chunk_numpy(chunk, element, fields, opts, numpy, sh_indices)
                    dropped_invalid += result.dropped_invalid
                    dropped_scale += result.dropped_scale
                    dropped_opacity += result.dropped_opacity
                    positions = result.positions
                    records = result.records
                    sh_bytes = result.sh_bytes
                    if threshold is not None and len(positions):
                        raw = result.raw_opacity
                        keep_mask = raw > threshold
                        ties = numpy.nonzero(raw == threshold)[0]
                        if len(ties):
                            allowed = ties[:slots_left]
                            slots_left -= int(allowed.size)
                            keep_mask[allowed] = True
                        dropped_decimate += int((~keep_mask).sum())
                        if not keep_mask.all():
                            count = int(keep_mask.sum())
                            if count == 0:
                                continue
                            arr = numpy.frombuffer(records, dtype=numpy.uint8).reshape(-1, GAUSSIAN_BYTES)
                            records = arr[keep_mask].tobytes()
                            positions = positions[keep_mask]
                            if sh_bytes:
                                sh_arr = numpy.frombuffer(sh_bytes, dtype="<f4").reshape(-1, len(sh_indices))
                                sh_bytes = numpy.ascontiguousarray(sh_arr[keep_mask]).tobytes()
                    # 浮尘剔除：距离稳健中心超过切线的点直接丢掉
                    if _fcut2 > 0 and len(positions):
                        d = numpy.linalg.norm(positions - numpy.asarray(_fcenter, dtype=numpy.float64), axis=1)
                        fmask = d <= _fcut
                        dropped_floater += int((~fmask).sum())
                        if not fmask.all():
                            if not fmask.any():
                                continue
                            arr = numpy.frombuffer(records, dtype=numpy.uint8).reshape(-1, GAUSSIAN_BYTES)
                            records = arr[fmask].tobytes()
                            positions = positions[fmask]
                            if sh_bytes:
                                sh_arr = numpy.frombuffer(sh_bytes, dtype="<f4").reshape(-1, len(sh_indices))
                                sh_bytes = numpy.ascontiguousarray(sh_arr[fmask]).tobytes()
                    writer.write_raw(records)
                    if sh_handle is not None and sh_bytes:
                        sh_handle.write(sh_bytes)
                        sh_floats += len(sh_bytes) // 4
                    n = len(positions)
                    if n:
                        kept += n
                        for row in positions.tolist():
                            update_bounds(row[0], row[1], row[2])
                else:
                    result = encode_chunk_python(chunk, record_struct, fields, opts, sh_indices)
                    dropped_invalid += result.dropped_invalid
                    dropped_scale += result.dropped_scale
                    dropped_opacity += result.dropped_opacity
                    positions = result.positions
                    raw_values = result.raw_opacity
                    payloads: list[bytes] = [result.records[i : i + GAUSSIAN_BYTES] for i in range(0, len(result.records), GAUSSIAN_BYTES)]
                    sh_all = result.sh_bytes
                    sh_stride = len(sh_indices) * 4
                    selected: list[int] = []
                    if threshold is not None:
                        for i, raw in enumerate(raw_values):
                            if raw > threshold:
                                selected.append(i)
                            elif raw == threshold and slots_left > 0:
                                selected.append(i)
                                slots_left -= 1
                        dropped_decimate += len(raw_values) - len(selected)
                    else:
                        selected = list(range(len(raw_values)))
                    out_bytes = bytearray()
                    written = 0
                    for i in selected:
                        px, py, pz = positions[i]
                        # 浮尘剔除（纯 Python 路径）
                        if not floater_ok(px, py, pz):
                            dropped_floater += 1
                            continue
                        out_bytes.extend(payloads[i])
                        if sh_handle is not None and sh_stride:
                            sh_handle.write(sh_all[i * sh_stride : (i + 1) * sh_stride])
                            sh_floats += len(sh_indices)
                        update_bounds(px, py, pz)
                        written += 1
                    writer.write_raw(bytes(out_bytes))
                    kept += written
                if progress and kept and kept // 250000 > reported_blocks:
                    reported_blocks = kept // 250000
                    config.log_info(f"已转换 {kept} / {total} 个高斯…")
    finally:
        if sh_handle is not None:
            sh_handle.close()

    if kept == 0:
        raise PlyError("过滤后没有任何高斯剩余，请放宽 --min-opacity / --max-scale 或检查 PLY 内容。")

    return {
        "count": kept,
        "shFloats": sh_floats,
        "_min": min_b,
        "_max": max_b,
        "floater": {
            "enabled": _fcut2 > 0,
            "factor": floater.get("factor"),
            "center": _fcenter,
            "p95Distance": floater.get("p95"),
            "cutoffRadius": _fcut,
            "droppedByFloater": dropped_floater,
            "sampled": floater.get("sampled"),
            "scanned": floater.get("total"),
        },
        "filters": {
            "shDegree": degree,
            "shCoefficientsPerChannel": k_per_channel,
            "minOpacity": opts.min_opacity,
            "maxScaleMeters": opts.max_scale,
            "decimate": opts.decimate,
            "droppedByOpacity": dropped_opacity,
            "droppedByScale": dropped_scale,
            "droppedInvalid": dropped_invalid,
            "droppedByDecimate": dropped_decimate,
            "droppedByFloater": dropped_floater,
            "kept": kept,
        },
    }


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="to_splat.py",
        description="把 3DGS 训练输出的二进制 PLY 转成 Web 端使用的 .splat（每个高斯 32 字节）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python pipeline/to_splat.py --ply data/models/scene.ply --out data/models/scene.splat --json data/models/scene.json\n"
            "  python pipeline/to_splat.py --ply scene.ply --out scene.splat --min-opacity 0.01 --max-scale 0.5 --decimate 0.6\n"
        ),
    )
    parser.add_argument("--ply", required=True, help="输入：3DGS 训练输出的二进制小端 PLY")
    parser.add_argument("--out", required=True, help="输出：.splat 文件路径")
    parser.add_argument("--sh-degree", type=int, default=3, choices=[0, 1, 2, 3], help="写入 .shcoeff.bin 的球谐阶数（0 表示不写，默认 3）")
    parser.add_argument("--min-opacity", type=float, default=0.005, help="低于该不透明度的高斯将被丢弃（默认 0.005）")
    parser.add_argument(
        "--max-scale",
        default="auto",
        help=(
            "线性尺度超过该值（米）的高斯视为漂浮噪点丢弃。"
            "**默认 auto**：按尺度分布的 99.5 百分位自适应截断（只裁尾部离群大高斯），"
            "避免硬编码绝对值把「铺墙铺地的大高斯」整批删掉。"
            "传具体数字则按该数字过滤；传 0 或负数表示不限制。"
            "（实测：某模型硬编码 0.5 时丢掉 51.5%% 的高斯、导出只剩 5036 个；"
            "训练器自身的钳位是 prune_scale×spatial_scale=1.1185，用同源阈值后丢弃 0%%）"
        ),
    )
    parser.add_argument(
        "--scales-are-log",
        action="store_true",
        help="PLY 里的 scale_* 是 log 值（早期 train_gsplat.py 产物）。标准 3DGS PLY 是线性尺度，默认不加此开关",
    )
    parser.add_argument("--decimate", type=float, default=1.0, help="抽稀比例，0.5 表示按 opacity 排序保留一半（默认 1.0）")
    parser.add_argument(
        "--floater-factor",
        type=float,
        default=2.0,
        help=(
            "离群浮尘剔除倍数：距离稳健中心超过「该倍数 × P95 距离」的高斯丢弃（默认 2.0；0 表示不剔除）。"
            "实测 0.1%% 的野点就能把取景半径从 10.6 m 撑到 79.8 m，进而让房间在画面里缩成一个点。"
            "想要更干净的隔离感可以调到 1.2~1.5，代价是可能削掉走廊等真实远端结构。"
        ),
    )
    parser.add_argument(
        "--floater-stride",
        type=int,
        default=1,
        help="估算稳健统计量时的抽样步长（1=全部参与；高斯数极大时可调大以加快首遍扫描）",
    )
    parser.add_argument("--json", default=None, help="输出元数据 JSON 的路径")
    parser.add_argument("--quat-order", default="wxyz", choices=["wxyz", "xyzw"], help="PLY 中 rot_0..3 的排列顺序（默认 wxyz，即 3DGS 约定）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    config.ensure_utf8_stdio()
    config.require_python()
    args = build_parser().parse_args(argv)

    config.log_rule("PLY → .splat 转换")
    config.log_info(f"输入 PLY：{args.ply}")
    config.log_info(f"输出 .splat：{args.out}")
    config.log_info(
        f"参数：sh-degree={args.sh_degree} min-opacity={args.min_opacity} "
        f"max-scale={args.max_scale} decimate={args.decimate} quat-order={args.quat_order} "
        f"floater-factor={args.floater_factor}"
    )
    config.log_info(f"numpy：{'可用（向量化路径，启动时自检）' if config.has_numpy() else '不可用（纯标准库路径）'}")

    options = ConvertOptions(
        sh_degree=args.sh_degree,
        min_opacity=args.min_opacity,
        max_scale=args.max_scale,
        decimate=args.decimate,
        quat_order=args.quat_order,
        scales_are_log=getattr(args, "scales_are_log", False),
        floater_factor=args.floater_factor,
        floater_stride=args.floater_stride,
    )
    try:
        metadata = convert_ply_to_splat(args.ply, args.out, options, json_path=args.json)
    except FileNotFoundError as exc:
        config.log_error(str(exc))
        return 3
    except PlyError as exc:
        config.log_error(f"PLY 解析失败：{exc}")
        config.log_info(
            "排查建议：确认该 PLY 是 3DGS 训练输出（属性应包含 f_dc_0..2 / opacity / scale_0..2 / rot_0..3），"
            "用文本编辑器打开文件头即可看到 property 列表。"
        )
        return 4
    except MemoryError:
        config.log_error("内存不足：请减少高斯数量（训练时降低 --max-gaussians）或先用 --decimate 抽稀。")
        return 6
    except OSError as exc:
        config.log_error(f"文件读写失败：{exc}")
        return 3

    filt = metadata["filters"]
    config.log_ok(
        f"转换完成：{metadata['count']} 个高斯，{config.human_bytes(metadata['outputBytes'])}"
        f"（{metadata['outputBytes']} 字节 = {metadata['count']} × 32），耗时 {metadata['elapsedSec']} 秒"
    )
    config.log_info(f"包围盒 min={metadata['bounds']['min']} max={metadata['bounds']['max']} radius={metadata['bounds']['radius']}")
    config.log_info(
        f"过滤统计：opacity 丢弃 {filt['droppedByOpacity']}，尺度丢弃 {filt['droppedByScale']}，"
        f"非法值丢弃 {filt['droppedInvalid']}，抽稀丢弃 {filt['droppedByDecimate']}"
    )
    if metadata.get("shCoeffFile"):
        config.log_info(f"高阶球谐：{metadata['shCoeffFile']}（每高斯 {metadata['shRestCount']} 个 float32）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
