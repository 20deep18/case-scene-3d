/**
 * 3D Gaussian Splatting 数据加载器
 * 支持：
 *   - .splat   （antimatter15 / gsplat 社区通用 32 字节/高斯 流式格式）
 *   - .ply     （原始 3DGS 训练输出，含 scale/opacity/rot/SH 系数）
 *   - .bin/.json 元数据（由本系统 Python 流水线导出的高斯统计信息）
 *
 * 输出统一为 InterleavedBuffer 友好的 Float32/Uint8 数组：
 *   position : Float32Array(n*3)
 *   scale    : Float32Array(n*3)  线性尺度（已是 exp(scale) 后的值）
 *   quat     : Float32Array(n*4)  xyzw 单位四元数
 *   color    : Uint8Array(n*4)    RGBA（已应用 sigmoid）
 *   sh       : Float32Array(n*3*k) 可选，球谐高阶系数（DC 之外）
 */

// ------------------------------------------------------------------ .splat

export function parseSplat(buffer, onProgress) {
  const buf = buffer instanceof ArrayBuffer ? buffer : buffer.buffer;
  const rowBytes = 32;
  const count = Math.floor(buf.byteLength / rowBytes);
  const f32 = new Float32Array(buf, 0, Math.floor(buf.byteLength / 4));

  const position = new Float32Array(count * 3);
  const scale = new Float32Array(count * 3);
  const quat = new Float32Array(count * 4);
  const color = new Uint8Array(count * 4);
  const u8 = new Uint8Array(buf);

  for (let i = 0; i < count; i++) {
    const f = i * 8; // float index
    const b = i * 32; // byte index
    position[i * 3] = f32[f];
    position[i * 3 + 1] = f32[f + 1];
    position[i * 3 + 2] = f32[f + 2];
    scale[i * 3] = f32[f + 3];
    scale[i * 3 + 1] = f32[f + 4];
    scale[i * 3 + 2] = f32[f + 5];
    color[i * 4] = u8[b + 24];
    color[i * 4 + 1] = u8[b + 25];
    color[i * 4 + 2] = u8[b + 26];
    color[i * 4 + 3] = u8[b + 27];
    const a = (u8[b + 28] - 128) / 128.0;
    const qb = (u8[b + 29] - 128) / 128.0;
    const c = (u8[b + 30] - 128) / 128.0;
    const d = (u8[b + 31] - 128) / 128.0;
    quat[i * 4] = a;
    quat[i * 4 + 1] = qb;
    quat[i * 4 + 2] = c;
    quat[i * 4 + 3] = d;
    if (onProgress && (i & 0xffff) === 0) onProgress(i / count);
  }
  return { count, position, scale, quat, color, sh: null, shDegree: 0, format: 'splat' };
}

export function encodeSplat({ position, scale, quat, color }) {
  const count = position.length / 3;
  const out = new ArrayBuffer(count * 32);
  const f32 = new Float32Array(out);
  const u8 = new Uint8Array(out);
  for (let i = 0; i < count; i++) {
    const f = i * 8;
    const b = i * 32;
    f32[f] = position[i * 3];
    f32[f + 1] = position[i * 3 + 1];
    f32[f + 2] = position[i * 3 + 2];
    f32[f + 3] = scale[i * 3];
    f32[f + 4] = scale[i * 3 + 1];
    f32[f + 5] = scale[i * 3 + 2];
    u8[b + 24] = color[i * 4];
    u8[b + 25] = color[i * 4 + 1];
    u8[b + 26] = color[i * 4 + 2];
    u8[b + 27] = color[i * 4 + 3];
    u8[b + 28] = Math.max(0, Math.min(255, quat[i * 4] * 128 + 128));
    u8[b + 29] = Math.max(0, Math.min(255, quat[i * 4 + 1] * 128 + 128));
    u8[b + 30] = Math.max(0, Math.min(255, quat[i * 4 + 2] * 128 + 128));
    u8[b + 31] = Math.max(0, Math.min(255, quat[i * 4 + 3] * 128 + 128));
  }
  return out;
}

// ------------------------------------------------------------------ .ply

const SH_C0 = 0.28209479177387814;

const PLY_TYPE_SIZE = {
  char: 1, uchar: 1, int8: 1, uint8: 1,
  short: 2, ushort: 2, int16: 2, uint16: 2,
  int: 4, uint: 4, float: 4, int32: 4, uint32: 4, float32: 4,
  double: 8, float64: 8,
};

const PLY_READERS = {
  char: (dv, o) => dv.getInt8(o),
  int8: (dv, o) => dv.getInt8(o),
  uchar: (dv, o) => dv.getUint8(o),
  uint8: (dv, o) => dv.getUint8(o),
  short: (dv, o) => dv.getInt16(o, true),
  int16: (dv, o) => dv.getInt16(o, true),
  ushort: (dv, o) => dv.getUint16(o, true),
  uint16: (dv, o) => dv.getUint16(o, true),
  int: (dv, o) => dv.getInt32(o, true),
  int32: (dv, o) => dv.getInt32(o, true),
  uint: (dv, o) => dv.getUint32(o, true),
  uint32: (dv, o) => dv.getUint32(o, true),
  float: (dv, o) => dv.getFloat32(o, true),
  float32: (dv, o) => dv.getFloat32(o, true),
  double: (dv, o) => dv.getFloat64(o, true),
  float64: (dv, o) => dv.getFloat64(o, true),
};

export function parsePlyHeader(bytes) {
  // 头部为 ASCII，逐行读取直到 end_header
  let text = '';
  let headerEnd = -1;
  const decoder = new TextDecoder('ascii');
  const step = 4096;
  for (let off = 0; off < Math.min(bytes.length, 1 << 22); off += step) {
    text += decoder.decode(bytes.subarray(off, Math.min(off + step, bytes.length)));
    const idx = text.indexOf('end_header');
    if (idx >= 0) {
      const nl = text.indexOf('\n', idx);
      headerEnd = nl + 1;
      text = text.slice(0, headerEnd);
      break;
    }
  }
  if (headerEnd < 0) throw new Error('PLY 头部未找到 end_header');
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length);
  if (lines[0].trim() !== 'ply') throw new Error('不是合法的 PLY 文件');
  const header = { format: null, elements: [] };
  let current = null;
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].trim().split(/\s+/);
    if (parts[0] === 'format') header.format = parts[1];
    else if (parts[0] === 'element') {
      current = { name: parts[1], count: Number(parts[2]), properties: [] };
      header.elements.push(current);
    } else if (parts[0] === 'property' && current) {
      if (parts[1] === 'list') {
        current.properties.push({ name: parts[4], list: true, countType: parts[2], type: parts[3] });
      } else {
        current.properties.push({ name: parts[2], type: parts[1], list: false });
      }
    }
  }
  return { header, headerEnd };
}

export function parsePlyFromArrayBuffer(arrayBuffer, onProgress) {
  const bytes = new Uint8Array(arrayBuffer);
  const { header, headerEnd } = parsePlyHeader(bytes);
  const vertex = header.elements.find((e) => e.name === 'vertex');
  if (!vertex) throw new Error('PLY 缺少 vertex 元素');
  const fmt = header.format ?? '';
  const isBinaryLE = fmt.includes('binary_little_endian');
  const isBinaryBE = fmt.includes('binary_big_endian');
  if (!isBinaryLE && !isBinaryBE) throw new Error(`暂不支持 PLY 格式：${fmt}`);

  const count = vertex.count;
  // 计算每行字节偏移
  let stride = 0;
  const offsets = new Map();
  for (const prop of vertex.properties) {
    offsets.set(prop.name, stride);
    stride += PLY_TYPE_SIZE[prop.type] ?? 0;
  }

  const get = (view, offset, type) => {
    const reader = PLY_READERS[type];
    if (!reader) throw new Error(`不支持的属性类型 ${type}`);
    return isBinaryBE ? beReader(view, offset, type) : reader(view, offset);
  };

  const dataStart = headerEnd;
  const dv = new DataView(arrayBuffer, dataStart, count * stride);

  const position = new Float32Array(count * 3);
  const scale = new Float32Array(count * 3);
  const quat = new Float32Array(count * 4);
  const color = new Uint8Array(count * 4);

  const has = (n) => offsets.has(n);
  const scaleNames = ['scale_0', 'scale_1', 'scale_2'].filter(has);
  const rotNames = ['rot_0', 'rot_1', 'rot_2', 'rot_3'].filter(has);

  const shRestNames = [];
  for (const key of offsets.keys()) {
    if (/^f_rest_\d+$/.test(key)) shRestNames.push(key);
  }
  shRestNames.sort((a, b) => Number(a.slice(8)) - Number(b.slice(8)));
  const shRestCount = shRestNames.length;
  const shPerChannel = shRestCount / 3;
  const shDegree = shPerChannel >= 15 ? 3 : shPerChannel >= 8 ? 2 : shPerChannel >= 3 ? 1 : 0;
  const sh = shRestCount ? new Float32Array(count * shRestCount) : null;

  for (let i = 0; i < count; i++) {
    const base = i * stride;
    position[i * 3] = get(dv, base + offsets.get('x'), 'float');
    position[i * 3 + 1] = get(dv, base + offsets.get('y'), 'float');
    position[i * 3 + 2] = get(dv, base + offsets.get('z'), 'float');

    if (scaleNames.length === 3) {
      scale[i * 3] = Math.exp(get(dv, base + offsets.get('scale_0'), 'float'));
      scale[i * 3 + 1] = Math.exp(get(dv, base + offsets.get('scale_1'), 'float'));
      scale[i * 3 + 2] = Math.exp(get(dv, base + offsets.get('scale_2'), 'float'));
    } else {
      scale[i * 3] = scale[i * 3 + 1] = scale[i * 3 + 2] = 0.01;
    }

    if (rotNames.length === 4) {
      quat[i * 4] = get(dv, base + offsets.get('rot_0'), 'float');
      quat[i * 4 + 1] = get(dv, base + offsets.get('rot_1'), 'float');
      quat[i * 4 + 2] = get(dv, base + offsets.get('rot_2'), 'float');
      quat[i * 4 + 3] = get(dv, base + offsets.get('rot_3'), 'float');
    } else {
      quat[i * 4 + 3] = 1;
    }

    const fdc = ['f_dc_0', 'f_dc_1', 'f_dc_2'];
    if (offsets.has('f_dc_0')) {
      for (let c = 0; c < 3; c++) {
        const v = get(dv, base + offsets.get(fdc[c]), 'float');
        const linear = 0.5 + SH_C0 * v;
        color[i * 4 + c] = Math.max(0, Math.min(255, Math.round(linear * 255)));
      }
    } else if (offsets.has('red')) {
      color[i * 4] = Math.max(0, Math.min(255, Math.round(get(dv, base + offsets.get('red'), 'float'))));
      color[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(get(dv, base + offsets.get('green'), 'float'))));
      color[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(get(dv, base + offsets.get('blue'), 'float'))));
    } else {
      color[i * 4] = color[i * 4 + 1] = color[i * 4 + 2] = 200;
    }

    if (offsets.has('opacity')) {
      const o = get(dv, base + offsets.get('opacity'), 'float');
      const a = 1 / (1 + Math.exp(-o)); // sigmoid
      color[i * 4 + 3] = Math.max(0, Math.min(255, Math.round(a * 255)));
    } else {
      color[i * 4 + 3] = 255;
    }

    if (sh) {
      for (let k = 0; k < shRestCount; k++) {
        sh[i * shRestCount + k] = get(dv, base + offsets.get(shRestNames[k]), 'float');
      }
    }
    if (onProgress && (i & 0x3ffff) === 0) onProgress(i / count);
  }

  return { count, position, scale, quat, color, sh, shDegree, shRestCount, format: 'ply' };
}

function beReader(dv, offset, type) {
  switch (type) {
    case 'float':
    case 'float32':
      return dv.getFloat32(offset, false);
    case 'double':
    case 'float64':
      return dv.getFloat64(offset, false);
    case 'int':
    case 'int32':
      return dv.getInt32(offset, false);
    case 'uint':
    case 'uint32':
      return dv.getUint32(offset, false);
    case 'short':
      return dv.getInt16(offset, false);
    case 'ushort':
      return dv.getUint16(offset, false);
    default:
      return PLY_READERS[type](dv, offset);
  }
}

// ------------------------------------------------------------------ 统一入口

export async function loadGaussians(url, { onProgress, onStage } = {}) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`加载失败 ${res.status}：${url}`);
  const total = Number(res.headers.get('content-length') ?? 0);
  let buffer;
  if (!total || typeof res.body?.getReader !== 'function') {
    buffer = await res.arrayBuffer();
  } else {
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      onProgress?.(got / total, got, total);
    }
    const merged = new Uint8Array(got);
    let off = 0;
    for (const c of chunks) {
      merged.set(c, off);
      off += c.length;
    }
    buffer = merged.buffer;
  }

  onStage?.('decoding');
  const lower = url.toLowerCase();
  if (lower.endsWith('.ply')) {
    return parsePlyFromArrayBuffer(buffer, (p) => onStage?.('decoding', p));
  }
  if (lower.endsWith('.splat')) {
    return parseSplat(buffer, (p) => onStage?.('decoding', p));
  }
  // 嗅探
  const head = new Uint8Array(buffer, 0, Math.min(4, buffer.byteLength));
  if (head[0] === 0x70 && head[1] === 0x6c && head[2] === 0x79) {
    return parsePlyFromArrayBuffer(buffer, (p) => onStage?.('decoding', p));
  }
  return parseSplat(buffer, (p) => onStage?.('decoding', p));
}

/**
 * 统计高斯位置的包围盒。
 *
 * ⚠️ **默认用稳健分位数（1% / 99%）而不是 min/max**，这是一个实测出来的教训：
 *   某真实案件模型共 43,404 个高斯，其中**只有 43 个（0.10%）是浮尘**，
 *   但它们把包围球半径从真实的 **10.6 m 撑到 79.78 m**（max/P95 = 11.9×）。
 *   前端正是用这个半径给相机取景的 —— 相机被推到 80 米外，
 *   10 米的房间在屏幕上缩成一个点，用户看到的就是「一团雾，看不出房间」。
 *   极少数野点没有资格决定整个场景的取景。
 *
 * @param {Float32Array} position 扁平化的 xyz
 * @param {object} [opts]
 * @param {boolean} [opts.robust=true] 是否使用分位数（false 则退回精确 min/max）
 * @param {number}  [opts.sample=200000] 估算分位数时的最大抽样数
 */
export function computeBounds(position, { robust = true, sample = 200000 } = {}) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const n = Math.floor(position.length / 3);
  const stride = Math.max(1, Math.floor(n / Math.max(1, sample)));
  const xs = [];
  const ys = [];
  const zs = [];
  let sampled = 0;
  for (let i = 0, idx = 0; i < position.length; i += 3, idx++) {
    const x = position[i];
    const y = position[i + 1];
    const z = position[i + 2];
    if (x < min[0]) min[0] = x;
    if (x > max[0]) max[0] = x;
    if (y < min[1]) min[1] = y;
    if (y > max[1]) max[1] = y;
    if (z < min[2]) min[2] = z;
    if (z > max[2]) max[2] = z;
    if (robust && idx % stride === 0) {
      xs.push(x);
      ys.push(y);
      zs.push(z);
      sampled++;
    }
  }
  const exact = {
    min: [...min],
    max: [...max],
    radius: Math.max(1e-3, 0.5 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2])),
  };

  // 用「修剪极值」替换 min/max：两侧各丢掉约 2% 的极端样本。
  // 不用严格分位数，是因为样本很小时（几十个高斯）1% 分位仍然等于最小值本身；
  // 修剪法则在大样本（实测的 0.1%~0.5% 野点）下同样成立。
  // 样本不足 100 个时不做稳健估计 —— 那点样本量算不出有意义的分位。
  let usedRobust = false;
  if (robust && sampled >= 100) {
    const trim = (arr) => {
      arr.sort((a, b) => a - b);
      const k = Math.max(1, Math.round(arr.length * 0.02));
      return [arr[k], arr[arr.length - 1 - k]];
    };
    const [x0, x1] = trim(xs);
    const [y0, y1] = trim(ys);
    const [z0, z1] = trim(zs);
    if (Number.isFinite(x0) && Number.isFinite(y0) && Number.isFinite(z0)) {
      min[0] = x0; max[0] = x1;
      min[1] = y0; max[1] = y1;
      min[2] = z0; max[2] = z1;
      usedRobust = true;
    }
  }

  const center = [0, 0, 0];
  const size = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    if (!Number.isFinite(min[k])) {
      min[k] = -1;
      max[k] = 1;
    }
    center[k] = (min[k] + max[k]) / 2;
    size[k] = max[k] - min[k];
  }

  // 取景半径用「到中心的**距离**的 98% 分位」，而不是修剪盒的半对角。
  //
  // 为什么：修剪是**逐轴**做的，三个轴联合起来会漏掉不少内容 ——
  // 实测某个真实模型落在修剪盒内的只有 **88.7%** 的高斯，
  // 拿它当取景半径会**欠取景**（画面边缘之外的模型被切掉）。
  // 距离分位则直接控制「框住多少内容」，语义更准。
  let radius = Math.max(1e-3, 0.5 * Math.hypot(size[0], size[1], size[2]));
  if (usedRobust && sampled > 0) {
    const d = new Float64Array(sampled);
    let m = 0;
    for (let i = 0; i < xs.length; i++) {
      const dx = xs[i] - center[0];
      const dy = ys[i] - center[1];
      const dz = zs[i] - center[2];
      d[m++] = Math.hypot(dx, dy, dz);
    }
    const sorted = d.subarray(0, m).slice().sort();
    const q98 = sorted[Math.min(m - 1, Math.floor(0.98 * (m - 1)))];
    if (Number.isFinite(q98) && q98 > 0) {
      radius = Math.max(1e-3, q98);
    }
  }
  return {
    min,
    max,
    center,
    size,
    radius,
    diagonal: Math.hypot(size[0], size[1], size[2]),
    // 便于排查/展示：这一份是没有任何修剪的精确范围，以及是否用了稳健分位数
    robust: usedRobust,
    exact,
    sampled,
  };
}
