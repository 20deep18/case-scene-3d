/**
 * 3D Gaussian Splatting · WebGL2 实时渲染器
 *
 * 实现要点（对齐 3DGS 论文的 EWA splatting 走样消除方案）：
 *   1. 顶点着色器把每个高斯的 3D 协方差矩阵 Σ = R S Sᵀ Rᵀ 变换到相机空间；
 *   2. 用 Jacobian J 投影到屏幕空间得到 2D 协方差，再求逆得到 conic（椭圆二次型）；
 *   3. 片元着色器用 exp(-0.5 · 二次型) 还原高斯衰减，按深度排序后 alpha 混合；
 *   4. 按视深度做分块基数排序 + 多次 drawElements（顺序无关近似正确）。
 *
 * 支持球谐 SH0；若数据带 SH1/SH2 系数可开启（useSH）。
 */
import * as THREE from 'three';

export const MAX_BATCH = 400000; // 每个 draw call 的高斯数量上限

/** 顶点属性固定槽位：与 BASE_VERT 里的 attribute 声明一一对应 */
const ATTR_SLOTS = {
  position: 0,
  scale: 1,
  quat: 2,
  color: 3,
  sh1a: 4,
  sh1b: 5,
  sh1c: 6,
  sh2a: 7,
  sh2b: 8,
  sh2c: 9,
  sh2d: 10,
  sh2e: 11,
};

const BASE_VERT = /* glsl */ `
precision highp float;
precision highp int;

attribute vec3 position;
attribute vec3 scale;
attribute vec4 quat;
attribute vec4 color;
#ifdef USE_SH
attribute vec3 sh1a;
attribute vec3 sh1b;
attribute vec3 sh1c;
attribute vec3 sh2a;
attribute vec3 sh2b;
attribute vec3 sh2c;
attribute vec3 sh2d;
attribute vec3 sh2e;
#endif

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;
uniform vec2 viewport;
uniform float focalX;
uniform float focalY;
uniform float shDegree;
uniform vec3 uCameraPosition;
uniform float uSizeScale;
uniform float uAlphaGain;    // 不透明度倍率（滑杆「不透明度」）
uniform float uExposure;     // 曝光倍率（滑杆「曝光」）

varying vec4 vColor;
varying vec2 vPos;
varying float vAlpha;
varying vec3 vConic;

vec3 quatRotate(vec4 q, vec3 v) {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

#ifndef USE_SH
vec3 computeColor(vec4 c) {
  return c.rgb / 255.0;
}
#else
// 说明：这里刻意不使用「常量数组构造」写法（const float C2[5] = float[5](...)）。
// WebGL2 走 GLSL ES 3.00，ANGLE（Chrome/Edge 的 GL 后端）对函数内的常量数组构造
// 支持不稳定，会直接报「顶点着色器编译失败」。改用独立的 const 标量最稳妥。
// 注意：注释里不要出现反引号，否则会提前结束 JS 模板字符串。
const float SH_C0 = 0.28209479177387814;
const float SH_C1 = 0.4886025119029199;
const float SH_C2_0 = 1.0925484305920792;
const float SH_C2_1 = -1.0925484305920792;
const float SH_C2_2 = 0.31539156525252005;
const float SH_C2_3 = -1.0925484305920792;
const float SH_C2_4 = 0.5462742152960396;

vec3 computeColor(vec4 c) {
  // 注意：三个正交方向的基向量要与 splat-loader.js / to_splat.py 的 SH 约定一致
  vec3 dir = normalize((modelViewMatrix * vec4(position, 1.0)).xyz);
  float x = dir.x, y = dir.y, z = dir.z;
  vec3 dc = (c.rgb / 255.0 - 0.5) / SH_C0;
  vec3 result = SH_C0 * dc;
  if (shDegree > 0.5) {
    result += -SH_C1 * y * sh1a + SH_C1 * z * sh1b - SH_C1 * x * sh1c;
  }
  if (shDegree > 1.5) {
    result +=
      SH_C2_0 * x * y * sh2a +
      SH_C2_1 * y * z * sh2b +
      SH_C2_2 * (2.0 * z * z - x * x - y * y) * sh2c +
      SH_C2_3 * x * z * sh2d +
      SH_C2_4 * (x * x - y * y) * sh2e;
  }
  return max(result + 0.5, 0.0);
}
#endif

void main() {
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vec4 camPos = projectionMatrix * mvPosition;

  if (camPos.w <= 0.0001) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vAlpha = 0.0;
    vColor = vec4(0.0);
    vPos = vec2(0.0);
    vConic = vec3(0.0);
    return;
  }

  mat3 S = mat3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z);
  mat3 R = mat3(1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z), 2.0 * (quat.x * quat.y + quat.w * quat.z), 2.0 * (quat.x * quat.z - quat.w * quat.y),
                2.0 * (quat.x * quat.y - quat.w * quat.z), 1.0 - 2.0 * (quat.x * quat.x + quat.z * quat.z), 2.0 * (quat.y * quat.z + quat.w * quat.x),
                2.0 * (quat.x * quat.z + quat.w * quat.y), 2.0 * (quat.y * quat.z - quat.w * quat.x), 1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y));
  // Σ = M·Mᵀ，M = S·R
  // 注意：**GLSL ES 3.00 没有 transpose() 函数**（WebGL2 编译会直接报
  //     "no matching overloaded function found"）。
  // 桌面 GLSL 有 transpose，但 ES 3.00 规范里没有，因此这里手工转置。
  mat3 M = S * R;
  mat3 Mt = mat3(M[0][0], M[1][0], M[2][0], M[0][1], M[1][1], M[2][1], M[0][2], M[1][2], M[2][2]);
  mat3 Sigma = M * Mt;

  // 视图旋转 W = mat3(modelViewMatrix)，Vrk = W·Σ·Wᵀ
  mat3 W = mat3(modelViewMatrix);
  mat3 Wt = mat3(W[0][0], W[1][0], W[2][0], W[0][1], W[1][1], W[2][1], W[0][2], W[1][2], W[2][2]);
  mat3 Vrk = W * Sigma * Wt;

  float focal = 0.5 * (focalX + focalY);
  float invZ = 1.0 / mvPosition.z;
  float J1 = focal * invZ;
  float J2 = -J1 * mvPosition.x * invZ;
  float J3 = -J1 * mvPosition.y * invZ;
  mat3 J = mat3(J1, 0.0, J2, 0.0, J1, J3, 0.0, 0.0, 0.0);
  mat3 Jt = mat3(J[0][0], J[1][0], J[2][0], J[0][1], J[1][1], J[2][1], J[0][2], J[1][2], J[2][2]);
  mat3 T = J * Vrk * Jt;

  float cxx = T[0][0] + 0.3;
  float cyy = T[1][1] + 0.3;
  float cxy = T[0][1];
  float det = cxx * cyy - cxy * cxy;

  vec3 conic = vec3(0.0);
  if (det > 1e-8) {
    float invDet = 1.0 / det;
    conic = vec3(cyy * invDet, -cxy * invDet, cxx * invDet);
  }

  vec3 ndc = camPos.xyz / camPos.w;
  vec2 screen = (ndc.xy * 0.5 + 0.5) * viewport;

  float mid = 0.5 * (cxx + cyy);
  float lambda = max(mid, 0.1);
  // 99% 置信椭圆半径，并限制最大屏幕尺寸，避免近处高斯撑满屏幕。
  // 下限给到 2.5px：GL_POINTS 的方形点至少要盖住像素中心才可能落进片元，
  // 过小的点会在部分驱动（含 SwiftShader、部分 ANGLE 配置）上被整点剔除。
  //
  // 上限从 320 降到 192：这个上限存在的意义是限制像素填充量，不是"允许画一个
  // 比屏幕还大的方块"。被夹住的高斯现在靠片元着色器里的贴边渐隐磨掉硬边
  //（见 fragment 里的 rn 那几行），所以上限可以放小，视觉更干净。
  float radiusPx = clamp(3.0 * sqrt(lambda) * uSizeScale, 2.5, 192.0);

  float baseAlpha = clamp(color.a / 255.0, 0.0, 1.0);
  float dist = length(mvPosition.xyz);
  float distFade = clamp(1.0 - dist / 400.0, 0.05, 1.0); // 极远处淡出，避免噪点

  // 曝光直接乘在颜色上（近似线性增益，够用且开销为零）
  vColor = vec4(computeColor(color) * uExposure, 1.0);
  // 不透明度倍率：让滑杆能压掉"一片白雾"，或调亮偏暗的扫描
  vAlpha = clamp(baseAlpha * distFade * uAlphaGain, 0.0, 1.0);
  vPos = vec2(0.0);
  vConic = conic;
  gl_Position = vec4(ndc.xy, 0.0, 1.0);
  gl_PointSize = radiusPx;
}
`;

const BASE_FRAG = /* glsl */ `precision highp float;

varying vec4 vColor;
varying vec2 vPos;
varying float vAlpha;
varying vec3 vConic;

void main() {
  if (vAlpha <= 0.0) discard;
  vec2 d = gl_PointCoord - vec2(0.5);
  float power = -2.0 * (vConic.x * d.x * d.x + vConic.z * d.y * d.y) - 2.0 * vConic.y * d.x * d.y;
  if (power > 0.0) discard;
  float alpha = vAlpha * exp(power);
  // ---- 贴边渐隐：治「巨大彩色方块」------------------------------------------
  //
  // radiusPx 上面被 clamp 到上限。一个**本该比屏幕还大**的高斯被夹住后，
  // 精灵四边被切断时 exp(power) 还很大（实测边缘 alpha 可到 0.6），
  // 于是渲染出来是硬邦邦的方块 —— 用户看到的「蓝色/橙色方块污染」就是它。
  // 正常高斯在 3σ 处 power 已到 -4.5（alpha≈0.011），本来就在这一圈趋于 0，
  // 所以这里加窗**对正常高斯几乎零影响**，只把被截断的那些边缘磨掉。
  float rn = length(d) * 2.0;                 // 0=中心，1=精灵边中点
  if (rn > 0.90) alpha *= 1.0 - smoothstep(0.90, 1.0, rn);
  if (alpha < 1.0 / 255.0) discard;
  gl_FragColor = vec4(vColor.rgb, alpha);
}
`;

/**
 * 与渲染器内部完全一致的着色器源码（供自动化测试在真实浏览器中做逐段诊断）。
 * 避免两处维护漂移：渲染器编译时就用的这两个常量。
 */
export const SHADER_SOURCES = {
  get vertex() {
    return BASE_VERT;
  },
  get fragment() {
    return BASE_FRAG;
  },
};

export class SplatRenderer {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2', {
      alpha: true,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
      premultipliedAlpha: true,
    });
    if (!this.gl) throw new Error('当前浏览器不支持 WebGL2，请升级 Chrome/Edge 或开启硬件加速');
    this.options = options;
    // 3DGS 使用「按视深度排序 + alpha 混合」的前向合成，必须关闭深度写入
    this.gl.disable(this.gl.DEPTH_TEST);
    this.gl.disable(this.gl.CULL_FACE);
    this.gl.enable(this.gl.BLEND);
    // 预乘 alpha 合成：src 已乘 alpha
    this.gl.blendFuncSeparate(this.gl.ONE, this.gl.ONE_MINUS_SRC_ALPHA, this.gl.ONE, this.gl.ONE_MINUS_SRC_ALPHA);
    this.gl.clearColor(0, 0, 0, 0);

    this.chunkOrder = []; // 视深度降序排列的块索引
    this.chunks = null;
    this.fovY = (options.fov ?? 55) * (Math.PI / 180);
    this.modelMatrix = new THREE.Matrix4();
    this.program = null;
    this.sortedFor = new THREE.Vector3(NaN, NaN, NaN);
    this.sortDebt = 0;
    this.stats = { gaussians: 0, drawn: 0, sortMs: 0, lastSortAt: 0 };
    this.useSH = false;
    this.shDegree = 0;
    this.pixelRatio = Math.min(options.maxPixelRatio ?? 2, globalThis.devicePixelRatio || 1);
    this._tmp = new THREE.Vector3();
  }

  // ------------------------------------------------------------ 程序编译

  _compile(useSH) {
    const gl = this.gl;
    const vert = useSH ? '#define USE_SH 1\n' + BASE_VERT : BASE_VERT;
    // 调试注入：globalThis.__SPLAT_DEBUG_FRAG 可以整体替换片元着色器，用于定位问题
    const frag = globalThis.__SPLAT_DEBUG_FRAG ?? BASE_FRAG;
    this.lastShaderError = null;

    const compileOne = (type, src, label) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        // 把驱动给出的报错**连同带行号的源码**一起抛出，
        // 否则浏览器只显示一句「编译失败」，无法定位是哪一行
        const log = gl.getShaderInfoLog(sh) ?? '(驱动未返回信息)';
        const lines = src.split('\n');
        const numbered = lines.map((l, i) => `${String(i + 1).padStart(4)}| ${l}`).join('\n');
        const err = new Error(`${label}编译失败：${log.trim()}`);
        err.shaderLog = log;
        err.shaderSource = numbered;
        this.lastShaderError = { label, log, source: numbered };
        gl.deleteShader(sh);
        throw err;
      }
      return sh;
    };

    const vs = compileOne(gl.VERTEX_SHADER, vert, `顶点着色器(${useSH ? 'SH' : 'SH0'})`);
    let fs;
    try {
      fs = compileOne(gl.FRAGMENT_SHADER, frag, '片元着色器');
    } catch (e) {
      gl.deleteShader(vs);
      throw e;
    }

    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    // 把顶点属性位置**固定下来**再链接：
    // 否则属性位置由驱动/编译器自行分配，一旦程序重新编译（例如切换 SH 模式或再次载入模型），
    // 位置就可能变化，而 VAO 里保存的是上一次的位置，会出现「属性看似配好实际未启用」，
    // 表现就是绘制调用成功但画面几乎全黑。固定位置可以彻底消除这类隐患。
    for (const [name, slot] of Object.entries(ATTR_SLOTS)) {
      gl.bindAttribLocation(program, slot, name);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program) ?? '(驱动未返回信息)';
      this.lastShaderError = { label: '着色器链接', log, source: '' };
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteProgram(program);
      throw new Error(`着色器链接失败：${log.trim()}`);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    if (this.program) gl.deleteProgram(this.program);
    this.program = program;
    // 暴露源码，便于自动化测试在真实浏览器里对渲染管线做逐段诊断
    this.debugVertSrc = vert;
    this.debugFragSrc = frag;
    this.loc = {
      position: gl.getAttribLocation(program, 'position'),
      scale: gl.getAttribLocation(program, 'scale'),
      quat: gl.getAttribLocation(program, 'quat'),
      color: gl.getAttribLocation(program, 'color'),
      sh1a: gl.getAttribLocation(program, 'sh1a'),
      sh1b: gl.getAttribLocation(program, 'sh1b'),
      sh1c: gl.getAttribLocation(program, 'sh1c'),
      sh2a: gl.getAttribLocation(program, 'sh2a'),
      sh2b: gl.getAttribLocation(program, 'sh2b'),
      sh2c: gl.getAttribLocation(program, 'sh2c'),
      sh2d: gl.getAttribLocation(program, 'sh2d'),
      sh2e: gl.getAttribLocation(program, 'sh2e'),
      projectionMatrix: gl.getUniformLocation(program, 'projectionMatrix'),
      modelViewMatrix: gl.getUniformLocation(program, 'modelViewMatrix'),
      viewport: gl.getUniformLocation(program, 'viewport'),
      focalX: gl.getUniformLocation(program, 'focalX'),
      focalY: gl.getUniformLocation(program, 'focalY'),
      shDegree: gl.getUniformLocation(program, 'shDegree'),
      cameraPosition: gl.getUniformLocation(program, 'uCameraPosition'),
      sizeScale: gl.getUniformLocation(program, 'uSizeScale'),
    alphaGain: gl.getUniformLocation(program, 'uAlphaGain'),
    exposure: gl.getUniformLocation(program, 'uExposure'),
    };
    gl.useProgram(program);
    return program;
  }

  // ------------------------------------------------------------ 数据装载

  /**
   * @param {{count:number, position:Float32Array, scale:Float32Array, quat:Float32Array, color:Uint8Array, sh?:Float32Array|null, shDegree?:number}} gaussians
   */
  setData(gaussians, { useSH = false } = {}) {
    const gl = this.gl;
    this.clear();
    const n = gaussians.count;
    this.useSH = !!(useSH && gaussians.sh && gaussians.shDegree > 0);
    this.shDegree = gaussians.shDegree ?? 0;
    this._compile(this.useSH);

    // ---- 分块，便于排序与多次绘制；块内顺序由排序数组决定
    const chunks = [];
    for (let start = 0; start < n; start += MAX_BATCH) {
      const count = Math.min(MAX_BATCH, n - start);
      chunks.push({ start, count, capacity: count, vao: gl.createVertexArray(), order: null });
    }
    this.chunks = chunks;
    this.raw = gaussians;

    // 预计算渲染顺序矩阵（模型矩阵为单位，排序在视图空间进行，模型变换交给世界矩阵）
    gl.useProgram(this.program);

    for (const chunk of chunks) {
      const { start, count } = chunk;
      gl.bindVertexArray(chunk.vao);

      const make = (array, size, attrName, type = gl.FLOAT, normalized = false) => {
        // 必须用 `attrName == null`，不能写 `!attrName`：
        // 合法的顶点属性位置 **0** 是 falsy，用 `!attrName` 会把 position(location=0)
        // 整个跳过 —— 属性永远不启用、一直读到 0，画面几乎全黑。
        if (attrName == null || attrName < 0) return null;
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, array, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(attrName);
        gl.vertexAttribPointer(attrName, size, type, normalized, 0, 0);
        return buf;
      };

      chunk.buffers = [
        make(gaussians.position.subarray(start * 3, (start + count) * 3), 3, this.loc.position),
        make(gaussians.scale.subarray(start * 3, (start + count) * 3), 3, this.loc.scale),
        make(gaussians.quat.subarray(start * 4, (start + count) * 4), 4, this.loc.quat),
        // 注意：**不能**用 normalized=true。
        // 颜色以 0..255 的 u8 存储，而着色器里做的是 `c.rgb / 255.0`。
        // 如果这里归一化（GPU 先除以 255），着色器再除一次 255，得到的就是 0..1/255，
        // 于是 vAlpha 全部趋近于 0，片元着色器的 `alpha < 1/255` 会把每一个像素都丢弃
        // —— 这正是「三维现场已载入但画面全黑」的原因。
        make(gaussians.color.subarray(start * 4, (start + count) * 4), 4, this.loc.color, gl.UNSIGNED_BYTE, false),
      ].filter(Boolean);

      if (this.useSH) {
        const sh = gaussians.sh;
        const per = gaussians.shRestCount;
        const component = (k) => {
          const out = new Float32Array(count * 3);
          for (let i = 0; i < count; i++) {
            for (let c = 0; c < 3; c++) {
              out[i * 3 + c] = sh[(start + i) * per + c * (per / 3) + k] ?? 0;
            }
          }
          return out;
        };
        const names = ['sh1a', 'sh1b', 'sh1c', 'sh2a', 'sh2b', 'sh2c', 'sh2d', 'sh2e'];
        const want = this.shDegree >= 3 ? 8 : this.shDegree >= 2 ? 8 : 3;
        for (let k = 0; k < want; k++) {
          const name = names[k];
          const loc = this.loc[name];
          if (loc < 0) continue;
          const data = component(k);
          const buf = gl.createBuffer();
          gl.bindBuffer(gl.ARRAY_BUFFER, buf);
          gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
          gl.enableVertexAttribArray(loc);
          gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
          chunk.buffers.push(buf);
        }
      }

      // 索引缓冲：初始顺序 (0..count-1)，每次排序后回写
      chunk.order = this._identityOrder(count);
      chunk.indexBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, chunk.indexBuffer);

      // 自检：VAO 里所有顶点属性必须处于 enabled 状态，否则该属性会一直读到 0，
      // 表现为「GPU 在跑、绘制调用成功、但画面几乎全黑」。这类问题很难肉眼定位，
      // 因此这里在装配完成后立刻校验，并在控制台给出明确报错。
      const bad = [];
      for (const [name, loc] of Object.entries({
        position: this.loc.position,
        scale: this.loc.scale,
        quat: this.loc.quat,
        color: this.loc.color,
      })) {
        if (loc === undefined || loc < 0) {
          bad.push(`${name}(location=${loc})`);
          continue;
        }
        if (!gl.getVertexAttrib(loc, gl.VERTEX_ATTRIB_ARRAY_ENABLED)) {
          bad.push(`${name}(location=${loc} 未启用)`);
        }
      }
      if (bad.length) {
        console.error(`[splat-renderer] 顶点属性装配异常：${bad.join('、')}`);
        this.stats.attribError = bad.join('、');
      } else {
        this.stats.attribCheck = 'all-enabled';
      }
      this.stats.attribLocations = {
        position: this.loc.position,
        scale: this.loc.scale,
        quat: this.loc.quat,
        color: this.loc.color,
      };
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, chunk.order, gl.DYNAMIC_DRAW);

      gl.bindVertexArray(null);
    }

    this.stats.gaussians = n;
    this.chunkOrder = chunks.map((_, i) => i);
    this._forceSort = true;
  }

  _identityOrder(count) {
    // 统一使用 Uint32 索引，避免 Uint16/Uint32 混用导致 ELEMENT_ARRAY_BUFFER 类型不一致
    const arr = new Uint32Array(count);
    for (let i = 0; i < count; i++) arr[i] = i;
    return arr;
  }

  clear() {
    const gl = this.gl;
    if (this.chunks) {
      for (const c of this.chunks) {
        gl.deleteVertexArray(c.vao);
        for (const b of c.buffers) gl.deleteBuffer(b);
        if (c.indexBuffer) gl.deleteBuffer(c.indexBuffer);
      }
    }
    this.chunks = null;
    this.raw = null;
    this.stats.gaussians = 0;
  }

  // ------------------------------------------------------------ 排序

  /** 用计数排序按视深度对每个块重新排序（远 → 近） */
  _sort(cameraWorldPos) {
    const t0 = performance.now();
    const raw = this.raw;
    if (!raw) return;
    const inv = new THREE.Matrix4().copy(this.modelMatrix).invert();
    const localCam = cameraWorldPos.clone().applyMatrix4(inv);
    const { position } = raw;

    for (const chunk of this.chunks) {
      const { start, count } = chunk;
      if (!this._keyBuf || this._keyBuf.length !== chunk.capacity) {
        this._keyBuf = new Float32Array(chunk.capacity);
        this._idxBuf = new Uint32Array(chunk.capacity);
        this._sortedBuf = new Uint32Array(chunk.capacity);
      }
      const keys = this._keyBuf;
      const idx = this._idxBuf;
      for (let i = 0; i < count; i++) {
        const p = (start + i) * 3;
        const dx = position[p] - localCam.x;
        const dy = position[p + 1] - localCam.y;
        const dz = position[p + 2] - localCam.z;
        keys[i] = dx * dx + dy * dy + dz * dz;
      }
      for (let i = 0; i < count; i++) idx[i] = i;
      const sorted = radixSortDesc(keys, idx, count, this._sortedBuf);
      // 注意：索引类型必须与 VAO 中绑定的 ELEMENT_ARRAY_BUFFER 类型一致，因此回写到 chunk.order
      const order = chunk.order;
      for (let i = 0; i < count; i++) order[i] = sorted[i];

      // 注意：ELEMENT_ARRAY_BUFFER 的绑定是 **VAO 状态的一部分**。
      // 这里必须先绑定该块自己的 VAO，再 bufferSubData，
      // 否则会把排序结果写进「默认 VAO」的绑定里，而该块的 VAO 仍然指向初始化时那份
      // 未排序的索引，索引与深度顺序不匹配 —— 表现为画面几乎全黑/只剩零星亮点。
      const gl = this.gl;
      gl.bindVertexArray(chunk.vao);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, chunk.indexBuffer);
      gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, order);
      gl.bindVertexArray(null);
    }

    this.stats.sortMs = performance.now() - t0;
    this.stats.lastSortAt = performance.now();
    this._forceSort = false;
    this.sortedFor.copy(cameraWorldPos);
  }

  maybeSort(camera) {
    if (!this.chunks) return;
    const pos = camera.position ?? camera.getWorldPosition(this._tmp);
    if (this._forceSort || Number.isNaN(this.sortedFor.x)) return this._sort(pos);
    const moved = this.sortedFor.distanceTo(pos);
    const ref = Math.max(0.25, this._boundRadius ?? 1);
    if (moved > 0.02 * ref) this._sort(pos);
  }

  // ------------------------------------------------------------ 绘制

  resize(width, height) {
    const w = Math.max(1, Math.floor(width * this.pixelRatio));
    const h = Math.max(1, Math.floor(height * this.pixelRatio));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  render(camera) {
    const gl = this.gl;
    if (!this.program || !this.chunks) return;
    const width = this.canvas.width;
    const height = this.canvas.height;

    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // 必须自己更新相机矩阵：Three.js 只在 WebGLRenderer.render() 里递归更新矩阵，
    // 我们用自己的 WebGL2 渲染器，所以 matrixWorld / matrixWorldInverse 不会自动刷新，
    // 否则 modelViewMatrix 与 projectionMatrix 会一直是初始值（表现为画面全黑）。
    camera.updateMatrixWorld(true);
    camera.updateProjectionMatrix();
    if (typeof camera.updateMatrixWorldInverse === 'function') camera.updateMatrixWorldInverse();

    const fovY = camera.isPerspectiveCamera ? (camera.fov * Math.PI) / 180 : this.fovY;
    const focalY = 0.5 * height / Math.tan(0.5 * fovY);
    const focalX = focalY;

    const modelView = new THREE.Matrix4().multiplyMatrices(camera.matrixWorldInverse, this.modelMatrix);

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.loc.projectionMatrix, false, camera.projectionMatrix.elements);
    gl.uniformMatrix4fv(this.loc.modelViewMatrix, false, modelView.elements);
    gl.uniform2f(this.loc.viewport, width, height);
    gl.uniform1f(this.loc.focalX, focalX);
    gl.uniform1f(this.loc.focalY, focalY);
    gl.uniform1f(this.loc.shDegree, this.shDegree);
    if (this.loc.sizeScale) {
      gl.uniform1f(this.loc.sizeScale, this.options?.gaussScale ?? 1);
    }
    // ⚠️ 这两个 uniform 以前**不存在**：页面上的"不透明度/曝光"滑杆只写了 state.filters，
    //    渲染器从来没读过 —— 拖动滑杆画面纹丝不动（实测确认）。现在接上了。
    if (this.loc.alphaGain) {
      gl.uniform1f(this.loc.alphaGain, this.options?.opacityGain ?? 1);
    }
    if (this.loc.exposure) {
      gl.uniform1f(this.loc.exposure, this.options?.exposure ?? 1);
    }
    const camPos = camera.position ?? camera.getWorldPosition(this._tmp);
    gl.uniform3f(this.loc.cameraPosition, camPos.x, camPos.y, camPos.z);

    this.maybeSort(camera);

    let drawn = 0;
    for (const index of this.chunkOrder) {
      const chunk = this.chunks[index];
      gl.bindVertexArray(chunk.vao);
      gl.drawElements(gl.POINTS, chunk.count, gl.UNSIGNED_INT, 0);
      drawn += chunk.count;
    }
    gl.bindVertexArray(null);
    this.stats.drawn = drawn;
    this.stats.lastDrawError = gl.getError();
    this.stats.sizeScaleApplied = this.options?.gaussScale ?? 1;

    // 一次性自检：把第一个高斯的完整投影链路算出来，写入 stats.debug，
    // 便于在没有调试器的环境下（或自动化测试里）定位「绘制调用成功但画面为空」。
    if (!this.stats.debug && this.raw && this.raw.count > 0) {
      const p = this.raw.position;
      const s = this.raw.scale;
      const q = this.raw.quat;
      const v = new THREE.Vector3(p[0], p[1], p[2]);
      const mv4 = new THREE.Vector4(p[0], p[1], p[2], 1).applyMatrix4(modelView);
      const clip = new THREE.Vector4(mv4.x, mv4.y, mv4.z, 1).applyMatrix4(camera.projectionMatrix);
      const ndcX = clip.x / clip.w;
      const ndcY = clip.y / clip.w;
      this.stats.debug = {
        firstGaussian: {
          pos: [p[0], p[1], p[2]],
          scale: [s[0], s[1], s[2]],
          quat: [q[0], q[1], q[2], q[3]],
          rgba: [this.raw.color[0], this.raw.color[1], this.raw.color[2], this.raw.color[3]],
        },
        viewZ: mv4.z,
        viewW: mv4.w,
        ndc: [ndcX, ndcY, clip.z / clip.w],
        screenPx: [((ndcX + 1) / 2) * width, ((ndcY + 1) / 2) * height],
        focal: [focalX, focalY],
        viewport: [width, height],
        sizeScale: this.options?.gaussScale ?? 1,
      };
      // 清屏 + 只画一个高斯，然后读回像素：用来判断「顶点着色器是否真的产出了片元」
      const probe = new Uint8Array(4);
      gl.readPixels(Math.max(0, Math.min(width - 1, Math.round(((ndcX + 1) / 2) * width))), Math.max(0, Math.min(height - 1, Math.round(((ndcY + 1) / 2) * height))), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, probe);
      this.stats.debug.pixelAtFirstGaussian = [probe[0], probe[1], probe[2], probe[3]];
    }
  }

  dispose() {
    this.clear();
    if (this.program) {
      this.gl.deleteProgram(this.program);
      this.program = null;
    }
  }
}

/**
 * 无符号整数键的 LSD 基数排序（降序）。keys 为 float 位模式，按位比较等价于按值比较（非负）。
 * 返回排序后的索引数组（视深度远 → 近）。
 */
function radixSortDesc(keys, idx, count, scratch) {
  const RADIX = 256;
  const buckets = new Uint32Array(RADIX);
  const keyBits = new Uint32Array(count);
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);

  for (let i = 0; i < count; i++) {
    f32[0] = keys[i];
    // 非负浮点数的位模式单调，直接使用
    keyBits[i] = u32[0];
  }

  let src = idx.subarray(0, count);
  let dst = scratch && scratch.length >= count ? scratch.subarray(0, count) : new Uint32Array(count);

  for (let shift = 0; shift < 32; shift += 8) {
    buckets.fill(0);
    for (let i = 0; i < count; i++) buckets[(keyBits[src[i]] >>> shift) & 0xff]++;
    // 降序：高位桶先输出
    let sum = 0;
    const starts = new Uint32Array(RADIX);
    for (let b = RADIX - 1; b >= 0; b--) {
      starts[b] = sum;
      sum += buckets[b];
    }
    for (let i = 0; i < count; i++) {
      const v = src[i];
      dst[starts[(keyBits[v] >>> shift) & 0xff]++] = v;
    }
    const tmp = src;
    src = dst;
    dst = tmp;
  }
  return src;
}
