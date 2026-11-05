/**
 * 用 CDP（Chrome DevTools Protocol）在真实浏览器里检查渲染器状态与 WebGL 错误。
 * 用法: node tests/browser_probe.mjs [url]
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const URL_TARGET = process.argv[2] ?? 'http://127.0.0.1:8787/';
const PORT = 9222;
const PROFILE = path.join(process.cwd(), '.browsertmp', 'cdp-profile');

rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

const edge = spawn(
  EDGE,
  [
    '--headless=new',
    '--no-sandbox',
    '--disable-crash-reporter',
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${PROFILE}`,
    '--window-size=1280,800',
    URL_TARGET,
  ],
  { stdio: 'ignore', detached: false }
);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getTarget() {
  for (let i = 0; i < 40; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
      if (page) return page;
    } catch {
      /* 还没起来 */
    }
    await sleep(400);
  }
  throw new Error('无法连接到浏览器调试端口');
}

const target = await getTarget();
const ws = new WebSocket(target.webSocketDebuggerUrl);
let msgId = 0;
const pending = new Map();
const consoleLogs = [];
const exceptions = [];

await new Promise((resolve, reject) => {
  ws.onopen = resolve;
  ws.onerror = reject;
});

ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    pending.get(msg.id)(msg);
    pending.delete(msg.id);
    return;
  }
  if (msg.method === 'Runtime.consoleAPICalled') {
    consoleLogs.push(`[${msg.params.type}] ${msg.params.args.map((a) => a.value ?? a.description ?? '').join(' ')}`);
  }
  if (msg.method === 'Runtime.exceptionThrown') {
    exceptions.push(msg.params.exceptionDetails?.exception?.description ?? JSON.stringify(msg.params.exceptionDetails));
  }
};

function send(method, params = {}) {
  const id = ++msgId;
  return new Promise((resolve) => {
    pending.set(id, resolve);
    ws.send(JSON.stringify({ id, method, params }));
  });
}

await send('Runtime.enable');
await send('Page.enable');
// 等页面跑起来（含模型下载与首帧渲染）
await sleep(12000);

const probe = `
(async () => {
  const THREE = await import('/vendor/three/build/three.module.js');
  const out = {};
  const state = window.__caseSystem?.state;
  const canvas = document.getElementById('viewport');
  out.canvas = canvas ? { w: canvas.width, h: canvas.height } : null;
  const gl = canvas?.getContext('webgl2');
  out.gl = !!gl;
  if (gl) {
    out.glError = gl.getError();
    out.blend = gl.getParameter(gl.BLEND);
    out.blendSrcRGB = gl.getParameter(gl.BLEND_SRC_RGB);
    out.blendDstRGB = gl.getParameter(gl.BLEND_DST_RGB);
    out.depthTest = gl.getParameter(gl.DEPTH_TEST);
    out.cullFace = gl.getParameter(gl.CULL_FACE);
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    out.renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    out.aliasedPointSizeRange = Array.from(gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) || []);
  }
  if (state?.renderer) {
    out.rendererStats = JSON.parse(JSON.stringify(state.renderer.stats || {}));
    out.hasProgram = !!state.renderer.program;
    out.chunks = (state.renderer.chunks || []).map(c => ({ start: c.start, count: c.count, hasVao: !!c.vao, hasIbo: !!c.indexBuffer }));
    out.loc = {};
    for (const [k, v] of Object.entries(state.renderer.loc || {})) {
      out.loc[k] = v === null ? 'NULL' : v === undefined ? 'undef' : (typeof v === 'number' ? v : 'loc#' + v.toString().slice(8, 14));
    }
    const prog = state.renderer.program;
    if (prog && gl) {
      out.programInfo = {
        linkStatus: gl.getProgramParameter(prog, gl.LINK_STATUS),
        activeUniforms: gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS),
        activeAttributes: gl.getProgramParameter(prog, gl.ACTIVE_ATTRIBUTES),
        log: gl.getProgramInfoLog(prog),
      };
      const names = [];
      for (let i = 0; i < out.programInfo.activeUniforms; i++) {
        const u = gl.getActiveUniform(prog, i);
        names.push(u ? (u.name + '[' + u.size + ']') : '?');
      }
      out.activeUniformNames = names;
      // 逐个查表
      out.lookup = {};
      for (const n of ['projectionMatrix','modelViewMatrix','viewport','focalX','focalY','shDegree','uCameraPosition','uSizeScale']) {
        out.lookup[n] = gl.getUniformLocation(prog, n) === null ? 'NULL' : 'ok';
      }
    }
  }
  out.camera = state ? {
    pos: state.camera.position.toArray().map(v => +v.toFixed(3)),
    q: state.camera.quaternion.toArray().map(v => +v.toFixed(4)),
    fov: state.camera.fov,
    aspect: +state.camera.aspect.toFixed(4),
    mwi: Array.from(state.camera.matrixWorldInverse.elements).map(v => +v.toFixed(3)),
    proj: Array.from(state.camera.projectionMatrix.elements).map(v => +v.toFixed(3)),
  } : null;

  // 关键实验：把一个已知点手动走一遍着色器的数学，看看它会落到屏幕哪里
  if (state?.renderer?.raw && state?.renderer?.chunks?.length) {
    const raw = state.renderer.raw;
    const cam = state.camera;
    cam.updateMatrixWorld(true);
    const mv = new THREE.Matrix4().multiplyMatrices(cam.matrixWorldInverse, state.renderer.modelMatrix);
    const pts = [];
    for (let i = 0; i < Math.min(raw.count, 400); i += 40) {
      const v = new THREE.Vector3(raw.position[i*3], raw.position[i*3+1], raw.position[i*3+2]);
      const local = v.clone();
      const camSpace = local.clone().applyMatrix4(mv);
      const ndc = local.clone().project(cam);
      pts.push({
        i,
        world: [ +local.x.toFixed(2), +local.y.toFixed(2), +local.z.toFixed(2) ],
        camZ: +camSpace.z.toFixed(3),
        w: +new THREE.Vector4(local.x, local.y, local.z, 1).applyMatrix4(mv).w.toFixed(3),
        ndc: [ +ndc.x.toFixed(3), +ndc.y.toFixed(3), +ndc.z.toFixed(3) ],
      });
    }
    out.samplePoints = pts;
    // 检查一个高斯的 scale/quat/color 原始值是否合理
    out.firstGaussian = {
      pos: [raw.position[0], raw.position[1], raw.position[2]].map(v => +v.toFixed(4)),
      scale: [raw.scale[0], raw.scale[1], raw.scale[2]].map(v => +v.toExponential(3)),
      quat: [raw.quat[0], raw.quat[1], raw.quat[2], raw.quat[3]].map(v => +v.toFixed(4)),
      rgba: [raw.color[0], raw.color[1], raw.color[2], raw.color[3]],
    };
  }

  // 采样画布像素（多点采样，避免采到空处）
  if (gl) {
    const W = canvas.width, H = canvas.height;
    const buf = new Uint8Array(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    let nonZero = 0, maxA = 0, sumA = 0;
    for (let i = 0; i < buf.length; i += 4) {
      if (buf[i] || buf[i+1] || buf[i+2] || buf[i+3]) nonZero++;
      maxA = Math.max(maxA, buf[i+3]);
      sumA += buf[i+3];
    }
    out.pixels = { total: W*H, nonZero, maxAlpha: maxA, meanAlpha: +(sumA/(W*H)).toFixed(3) };
    // 用应用自己的渲染器重画一次并立刻读像素，避免读到下一帧清屏后的结果
    if (state?.renderer?.program && state?.renderer?.chunks?.length) {
      const r0 = state.renderer;
      const cam0 = state.camera;
      cam0.updateMatrixWorld(true);
      const mv0 = new THREE.Matrix4().multiplyMatrices(cam0.matrixWorldInverse, r0.modelMatrix);
      const focal0 = 0.5 * H / Math.tan(0.5 * cam0.fov * Math.PI / 180);
      gl.useProgram(r0.program);
      gl.uniformMatrix4fv(r0.loc.projectionMatrix, false, cam0.projectionMatrix.elements);
      gl.uniformMatrix4fv(r0.loc.modelViewMatrix, false, mv0.elements);
      gl.uniform1f(r0.loc.focalX, focal0);
      gl.uniform1f(r0.loc.focalY, focal0);
      if (r0.loc.shDegree) gl.uniform1f(r0.loc.shDegree, 0);
      if (r0.loc.cameraPosition) gl.uniform3f(r0.loc.cameraPosition, cam0.position.x, cam0.position.y, cam0.position.z);
      if (r0.loc.sizeScale) gl.uniform1f(r0.loc.sizeScale, r0.options?.gaussScale ?? 1);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.viewport(0, 0, W, H);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      for (const idx of r0.chunkOrder) {
        const ch = r0.chunks[idx];
        gl.bindVertexArray(ch.vao);
        gl.drawElements(gl.POINTS, ch.count, gl.UNSIGNED_INT, 0);
      }
      gl.bindVertexArray(null);
      const b2 = new Uint8Array(W * H * 4);
      gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, b2);
      let nz2 = 0, mx2 = 0, s2 = 0;
      for (let i = 0; i < b2.length; i += 4) {
        if (b2[i] || b2[i+1] || b2[i+2] || b2[i+3]) nz2++;
        mx2 = Math.max(mx2, b2[i+3]);
        s2 += b2[i+3];
      }
      out.pixelsAfterExplicitDraw = { nonZero: nz2, maxAlpha: mx2, meanAlpha: +(s2/(W*H)).toFixed(3) };
      out.sizeScale = r0.options?.gaussScale ?? 1;
      out.filters = state.filters;
    }
    // 找几个非零像素的位置
    const spots = [];
    for (let y = 0; y < H && spots.length < 6; y += 32) {
      for (let x = 0; x < W && spots.length < 6; x += 32) {
        const o = (y*W + x)*4;
        if (buf[o] || buf[o+1] || buf[o+2] || buf[o+3]) spots.push({ x, y, rgba: [buf[o], buf[o+1], buf[o+2], buf[o+3]] });
      }
    }
    out.pixelSpots = spots;
    // 取中心像素
    const c = ((H>>1)*W + (W>>1))*4;
    out.centerPixel = [buf[c], buf[c+1], buf[c+2], buf[c+3]];

    // ---- 决定性实验：把 gl_PointSize 强制放大，看是否只是点太小被丢弃 ----
    const r = state.renderer;
    if (r?.program && r.chunks?.length) {
      const camPos = state.camera.position;
      const mv = new THREE.Matrix4().multiplyMatrices(state.camera.matrixWorldInverse, r.modelMatrix);
      const proj = state.camera.projectionMatrix;
      gl.useProgram(r.program);
      gl.uniformMatrix4fv(r.loc.projectionMatrix, false, proj.elements);
      gl.uniformMatrix4fv(r.loc.modelViewMatrix, false, mv.elements);
      gl.uniform2f(r.loc.viewport, W, H);
      gl.uniform1f(r.loc.focalX, 0.5 * H / Math.tan(0.5 * state.camera.fov * Math.PI / 180));
      gl.uniform1f(r.loc.focalY, 0.5 * H / Math.tan(0.5 * state.camera.fov * Math.PI / 180));
      gl.uniform1f(r.loc.shDegree, 0);
      gl.uniform3f(r.loc.cameraPosition, camPos.x, camPos.y, camPos.z);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.disable(gl.BLEND);              // 关掉混合，直接看不透明输出
      gl.disable(gl.DEPTH_TEST);
      for (const idx of r.chunkOrder) {
        const ch = r.chunks[idx];
        gl.bindVertexArray(ch.vao);
        gl.drawElements(gl.POINTS, ch.count, gl.UNSIGNED_INT, 0);
      }
      gl.bindVertexArray(null);
      const gErr = gl.getError();
      const buf2 = new Uint8Array(W * H * 4);
      gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, buf2);
      let nz2 = 0, maxA2 = 0;
      for (let i = 0; i < buf2.length; i += 4) {
        if (buf2[i] || buf2[i+1] || buf2[i+2] || buf2[i+3]) nz2++;
        maxA2 = Math.max(maxA2, buf2[i+3]);
      }
      out.noBlendTest = { glError: gErr, nonZero: nz2, maxAlpha: maxA2 };
      // 恢复混合
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    }
  }
  return JSON.stringify(out);
})()
`;

const res = await send('Runtime.evaluate', { expression: probe, returnByValue: true, awaitPromise: true });
if (res.result?.exceptionDetails) {
  console.log('!!! 页面内表达式抛异常：');
  console.log(JSON.stringify(res.result.exceptionDetails, null, 2).slice(0, 1500));
}
console.log('=== 浏览器内探测结果 ===');
const val = res.result?.result?.value;
if (val) {
  try {
    console.log(JSON.stringify(JSON.parse(val), null, 2));
  } catch {
    console.log(val);
  }
} else {
  console.log(JSON.stringify(res, null, 2).slice(0, 2000));
}

console.log('\n=== 控制台输出 ===');
if (consoleLogs.length === 0) console.log('(空)');
for (const l of consoleLogs.slice(-40)) console.log('  ' + l);

console.log('\n=== 未捕获异常 ===');
if (exceptions.length === 0) console.log('(无)');
for (const e of exceptions.slice(0, 10)) console.log('  ' + String(e).split('\n').slice(0, 4).join('\n  '));

// 顺便截图
const shot = await send('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) {
  const out = path.join(process.cwd(), 'output', 'verify', 'cdp.png');
  mkdirSync(path.dirname(out), { recursive: true });
  writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log(`\n截图已保存: ${out}`);
}

ws.close();
edge.kill();
process.exit(0);
