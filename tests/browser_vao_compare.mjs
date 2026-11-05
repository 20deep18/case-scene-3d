/**
 * 对比实验：同一份数据、同一个程序，
 *   (1) 用应用自己构建的 VAO 绘制
 *   (2) 用测试现场新建的 VAO 绘制
 * 二者差异即可定位是 VAO/缓冲绑定问题还是别的。
 * 用法: node tests/browser_vao_compare.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const PORT = 9226;
const PROFILE = path.join(process.cwd(), '.browsertmp', 'vao-profile');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1280,800', 'http://127.0.0.1:8787/',
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let page = null;
for (let i = 0; i < 40 && !page; i++) {
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
  } catch { /* retry */ }
  if (!page) await sleep(400);
}
if (!page) { console.error('无法连接'); edge.kill(); process.exit(1); }

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
};
const send = (method, params = {}) => new Promise((res) => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
await send('Runtime.enable');
await sleep(11000);

const script = `
(async () => {
  const st = window.__caseSystem.state;
  const r = st.renderer;
  const cam = st.camera;
  const gl = r.gl;
  const canvas = document.getElementById('viewport');
  const W = canvas.width, H = canvas.height;
  const THREE = await import('/vendor/three/build/three.module.js');
  cam.updateMatrixWorld(true);
  const mv = new THREE.Matrix4().multiplyMatrices(cam.matrixWorldInverse, r.modelMatrix);
  const focal = 0.5 * H / Math.tan(0.5 * cam.fov * Math.PI / 180);
  const out = {};

  function setupUniforms(prog) {
    gl.useProgram(prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'projectionMatrix'), false, cam.projectionMatrix.elements);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'modelViewMatrix'), false, mv.elements);
    const vp = gl.getUniformLocation(prog, 'viewport'); if (vp) gl.uniform2f(vp, W, H);
    gl.uniform1f(gl.getUniformLocation(prog, 'focalX'), focal);
    gl.uniform1f(gl.getUniformLocation(prog, 'focalY'), focal);
    const sd = gl.getUniformLocation(prog, 'shDegree'); if (sd) gl.uniform1f(sd, 0);
    const cp = gl.getUniformLocation(prog, 'uCameraPosition'); if (cp) gl.uniform3f(cp, cam.position.x, cam.position.y, cam.position.z);
    const ss = gl.getUniformLocation(prog, 'uSizeScale'); if (ss) gl.uniform1f(ss, 1.0);
  }
  function drawCount(vaoList, count) {
    gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    for (const vao of vaoList) {
      gl.bindVertexArray(vao);
      gl.drawElements(gl.POINTS, count, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);
    const e = gl.getError();
    const b = new Uint8Array(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, b);
    let nz = 0, mx = 0;
    for (let i = 0; i < b.length; i += 4) { if (b[i] || b[i+1] || b[i+2] || b[i+3]) nz++; mx = Math.max(mx, b[i+3]); }
    return { nonZero: nz, maxAlpha: mx, glError: e };
  }

  // (1) 应用自己的 VAO
  setupUniforms(r.program);
  out.appVao = drawCount(r.chunks.map((c) => c.vao), r.raw.count);

  // (2) 现场新建 VAO（与最小验证页一致）
  const raw = r.raw;
  const n = raw.count;
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const mk = (arr, size, name, type = gl.FLOAT, norm = false) => {
    const l = gl.getAttribLocation(r.program, name);
    if (l < 0) { out['attr_' + name] = 'missing'; return; }
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(l);
    gl.vertexAttribPointer(l, size, type, norm, 0, 0);
  };
  mk(raw.position, 3, 'position');
  mk(raw.scale, 3, 'scale');
  mk(raw.quat, 4, 'quat');
  mk(raw.color, 4, 'color', gl.UNSIGNED_BYTE, false);
  const ibo = gl.createBuffer();
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
  gl.bindVertexArray(null);

  setupUniforms(r.program);
  out.freshVao = drawCount([vao], n);

  // (3) 应用 VAO + 应用当前排序索引（应当与 (1) 相同）
  out.appVaoSecond = drawCount(r.chunks.map((c) => c.vao), r.raw.count);

  // 诊断：检查应用 VAO 里各属性是否真的处于启用状态
  gl.bindVertexArray(r.chunks[0].vao);
  out.enabledAttribs = [];
  for (const nm of ['position','scale','quat','color']) {
    const l = gl.getAttribLocation(r.program, nm);
    out.enabledAttribs.push(nm + '=' + l + (l >= 0 ? (gl.getVertexAttrib(l, gl.VERTEX_ATTRIB_ARRAY_ENABLED) ? ':on' : ':OFF') : ':na'));
  }
  out.elemBufBound = gl.getParameter(gl.ELEMENT_ARRAY_BUFFER_BINDING) !== null;
  gl.bindVertexArray(null);
  out.chunkOrderLen = r.chunkOrder.length;
  out.chunkCounts = r.chunks.map((c) => c.count);
  return JSON.stringify(out);
})()
`;

const res = await send('Runtime.evaluate', { expression: script, returnByValue: true, awaitPromise: true });
if (res.result?.exceptionDetails) console.log('异常:', res.result.exceptionDetails.exception?.description);
console.log(res.result?.result?.value ?? JSON.stringify(res).slice(0, 800));

const shot = await send('Page.captureScreenshot', { format: 'png' });
if (shot.result?.data) {
  const out = path.join(process.cwd(), 'output', 'verify', 'vao-compare.png');
  writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log('截图:', out);
}
ws.close();
edge.kill();
process.exit(0);
