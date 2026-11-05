/**
 * 回归测试：真实浏览器中的三维渲染与帧率。
 *   node tests/browser_render_check.mjs
 *
 * 覆盖四类曾在真实交互中暴露的缺陷：
 *   1) 顶点属性 location=0 被 `!attrName` 误判为无效 → position 永不启用 → 画面全黑
 *   2) 颜色属性 normalized=true + 着色器再除 255 → alpha≈0 → 全部片元被丢弃
 *   3) 画布缓冲未按窗口尺寸分配（渲染器创建前调用 resize）→ 只有 300x150 且画面异常
 *   4) 帧率显示异常且没有上限
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const URL_TARGET = process.argv[2] ?? 'http://127.0.0.1:8787/';
const PORT = 9228;
const PROFILE = path.join(process.cwd(), '.browsertmp', 'render-check');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

let pass = 0;
let fail = 0;
const check = (name, ok, detail = '') => {
  if (ok) {
    pass++;
    console.log(`  [OK]   ${name}${detail ? '  ' + detail : ''}`);
  } else {
    fail++;
    console.log(`  [FAIL] ${name}${detail ? '  ' + detail : ''}`);
  }
};

const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1280,800', URL_TARGET,
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let page = null;
for (let i = 0; i < 40 && !page; i++) {
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
  } catch { /* 重试 */ }
  if (!page) await sleep(400);
}
if (!page) {
  console.error('无法连接到浏览器调试端口（需要允许浏览器启动）');
  edge.kill();
  process.exit(1);
}

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const consoleErrors = [];
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Runtime.consoleAPICalled' && (m.params.type === 'error' || m.params.type === 'warning')) {
    consoleErrors.push(m.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
  }
};
const send = (method, params = {}) => new Promise((res) => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
await send('Runtime.enable');
await sleep(12000); // 等模型下载并渲染

console.log('浏览器三维渲染回归测试\n');

const probe = `
(async () => {
  const state = window.__caseSystem?.state;
  const canvas = document.getElementById('viewport');
  const gl = canvas?.getContext('webgl2');
  const out = { ok: false };
  if (!state?.renderer || !gl) { out.reason = '渲染器未创建'; return JSON.stringify(out); }
  const r = state.renderer;
  out.canvas = { w: canvas.width, h: canvas.height, cw: canvas.clientWidth, ch: canvas.clientHeight };
  out.gaussians = r.raw?.count ?? 0;
  out.stats = JSON.parse(JSON.stringify(r.stats || {}));

  // 属性装配检查
  gl.bindVertexArray(r.chunks[0].vao);
  out.attribs = {};
  for (const nm of ['position', 'scale', 'quat', 'color']) {
    const l = gl.getAttribLocation(r.program, nm);
    out.attribs[nm] = { loc: l, enabled: l >= 0 ? !!gl.getVertexAttrib(l, gl.VERTEX_ATTRIB_ARRAY_ENABLED) : false };
  }
  gl.bindVertexArray(null);

  // 用应用自己的渲染器绘制并读回像素
  const THREE = await import('/vendor/three/build/three.module.js');
  const cam = state.camera;
  cam.updateMatrixWorld(true);
  const W = canvas.width, H = canvas.height;
  const mv = new THREE.Matrix4().multiplyMatrices(cam.matrixWorldInverse, r.modelMatrix);
  const focal = 0.5 * H / Math.tan(0.5 * cam.fov * Math.PI / 180);
  gl.useProgram(r.program);
  gl.uniformMatrix4fv(r.loc.projectionMatrix, false, cam.projectionMatrix.elements);
  gl.uniformMatrix4fv(r.loc.modelViewMatrix, false, mv.elements);
  gl.uniform1f(r.loc.focalX, focal);
  gl.uniform1f(r.loc.focalY, focal);
  if (r.loc.shDegree) gl.uniform1f(r.loc.shDegree, 0);
  if (r.loc.sizeScale) gl.uniform1f(r.loc.sizeScale, r.options?.gaussScale ?? 1);
  gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.viewport(0, 0, W, H); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
  for (const idx of r.chunkOrder) {
    const ch = r.chunks[idx];
    gl.bindVertexArray(ch.vao);
    gl.drawElements(gl.POINTS, ch.count, gl.UNSIGNED_INT, 0);
  }
  gl.bindVertexArray(null);
  const b = new Uint8Array(W * H * 4);
  gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, b);
  let nz = 0, mx = 0, colored = 0;
  for (let i = 0; i < b.length; i += 4) {
    if (b[i] || b[i+1] || b[i+2] || b[i+3]) nz++;
    if (b[i] > 24 || b[i+1] > 24 || b[i+2] > 24) colored++;
    mx = Math.max(mx, b[i+3]);
  }
  out.pixels = { total: W * H, nonZero: nz, colored, maxAlpha: mx, coverage: +(nz / (W*H)).toFixed(4) };

  // 帧率
  const t0 = performance.now();
  let frames = 0;
  await new Promise((resolve) => {
    const tick = () => { frames++; if (performance.now() - t0 >= 2500) resolve(); else requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  });
  out.rafFps = +(frames / ((performance.now() - t0) / 1000)).toFixed(1);

  // ---- 回归：场景尺度必须可读；点击标注不得把相机打成 NaN（曾导致整屏黑） ----
  const finite3 = (v) => v && [v.x, v.y, v.z].every((n) => Number.isFinite(n));
  out.boundsRadius = state.bounds?.radius ?? null;
  out.boundsIsFinite = Number.isFinite(state.bounds?.radius);
  const checkpoints = [{ label: '点击前', finite: finite3(state.camera.position) }];
  const items = document.querySelectorAll('.anno-item');
  out.annoItems = items.length;
  for (const node of items) {
    node.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    await new Promise((r) => setTimeout(r, 120));
    checkpoints.push({
      label: node.querySelector('.label')?.textContent ?? '?',
      finite: finite3(state.camera.position),
      pos: state.camera.position.toArray().map((v) => (Number.isFinite(v) ? +v.toFixed(2) : 'NaN')),
    });
    if (!finite3(state.camera.position)) break;
  }
  out.cameraCheckpoints = checkpoints;

  // 点完之后画布是否还有内容
  {
    const r0 = state.renderer;
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (r0?.program && r0.chunks?.length) {
      const cam = state.camera;
      cam.updateMatrixWorld(true);
      const mv = new THREE.Matrix4().multiplyMatrices(cam.matrixWorldInverse, r0.modelMatrix);
      const focal = 0.5 * H / Math.tan(0.5 * cam.fov * Math.PI / 180);
      gl.useProgram(r0.program);
      gl.uniformMatrix4fv(r0.loc.projectionMatrix, false, cam.projectionMatrix.elements);
      gl.uniformMatrix4fv(r0.loc.modelViewMatrix, false, mv.elements);
      gl.uniform1f(r0.loc.focalX, focal);
      gl.uniform1f(r0.loc.focalY, focal);
      if (r0.loc.shDegree) gl.uniform1f(r0.loc.shDegree, 0);
      if (r0.loc.sizeScale) gl.uniform1f(r0.loc.sizeScale, r0.options?.gaussScale ?? 1);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.viewport(0, 0, W, H);
      for (const idx of r0.chunkOrder) {
        const ch = r0.chunks[idx];
        gl.bindVertexArray(ch.vao);
        gl.drawElements(gl.POINTS, ch.count, gl.UNSIGNED_INT, 0);
      }
      gl.bindVertexArray(null);
    }
    const b3 = new Uint8Array(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, b3);
    let nz3 = 0;
    for (let i = 0; i < b3.length; i += 4) if (b3[i] || b3[i + 1] || b3[i + 2] || b3[i + 3]) nz3++;
    out.pixelsAfterClicks = nz3;
    // 诊断：0 像素到底是「相机飞走了」还是「手动渲染这条路本身没画出来」
    out.diagAfterClicks = {
      camPos: state.camera.position.toArray().map((v) => (Number.isFinite(v) ? +v.toFixed(2) : 'NaN')),
      target: state.controls?.target?.toArray?.().map((v) => +v.toFixed(2)),
      radius: state.bounds?.radius ?? null,
      chunks: r0?.chunks?.length ?? null,
      hasProgram: !!r0?.program,
      W, H,
      glError: gl.getError(),
      // 绘制列表长度：chunks 有 1 个但 chunkOrder 若是空的，循环一次都不画 →
      // 表现就是「相机正常、程序正常、却 0 像素」。必须量出来才能定性。
      // （注意：本段整体是模板字符串，注释里**不能出现反引号**，否则提前截断。）
      orderLen: r0?.chunkOrder?.length ?? null,
      chunkCount: r0?.chunks?.length ?? null,
      // 目标是否在相机前方（点积 < 0 = 在背后，那自然什么都看不到）。
      // 不用 THREE：这里是求值块的局部作用域，THREE 不一定可见（上一次就因此整体崩了）。
      targetInFront: (() => {
        try {
          const e = state.camera.matrixWorld.elements;
          const f = { x: -e[8], y: -e[9], z: -e[10] }; // 相机前方 = -Z 列
          const cx = state.controls?.target?.x ?? 0;
          const cy = state.controls?.target?.y ?? 0;
          const cz = state.controls?.target?.z ?? 0;
          const px = state.camera.position.x, py = state.camera.position.y, pz = state.camera.position.z;
          const dx = cx - px, dy = cy - py, dz = cz - pz;
          const len = Math.hypot(dx, dy, dz) || 1;
          return +((f.x * dx + f.y * dy + f.z * dz) / len).toFixed(3);
        } catch (err) {
          // 同样不能用模板字符串：外层已经是模板字符串了
          return 'err:' + err.message;
        }
      })(),
      lastCheckpoint: out.cameraCheckpoints[out.cameraCheckpoints.length - 1]?.pos ?? null,
    };
  }

  out.ok = true;
  return JSON.stringify(out);
})()
`;

const res = await send('Runtime.evaluate', { expression: probe, returnByValue: true, awaitPromise: true });
if (res.result?.exceptionDetails) {
  console.log('页面内异常:', res.result.exceptionDetails.exception?.description);
}
const data = JSON.parse(res.result?.result?.value ?? '{}');

if (!data.ok) {
  console.log('  [FAIL] 无法取得渲染器状态：' + (data.reason ?? '未知'));
  fail++;
} else {
  console.log(`画布 ${data.canvas.w}x${data.canvas.h}（CSS ${data.canvas.cw}x${data.canvas.ch}），高斯 ${data.gaussians} 个\n`);

  check('画布绘制缓冲已按窗口尺寸分配', data.canvas.w > 800 && data.canvas.h > 400,
    `${data.canvas.w}x${data.canvas.h}`);
  check('画布宽高比与 CSS 尺寸一致（无拉伸）',
    Math.abs(data.canvas.w / data.canvas.h - data.canvas.cw / data.canvas.ch) < 0.02,
    `${(data.canvas.w / data.canvas.h).toFixed(3)} vs ${(data.canvas.cw / data.canvas.ch).toFixed(3)}`);
  for (const [nm, a] of Object.entries(data.attribs)) {
    check(`顶点属性 ${nm} 已启用`, a.enabled, `location=${a.loc}`);
  }
  check('无顶点属性装配报错', !data.stats.attribError, data.stats.attribError ?? '');
  check('绘制调用无 GL 错误', data.stats.lastDrawError === 0, `glError=${data.stats.lastDrawError}`);
  check('实际绘制的高斯数量等于总数', data.stats.drawn === data.gaussians,
    `${data.stats.drawn}/${data.gaussians}`);
  check('画布确实产出像素（覆盖率 > 5%）', data.pixels.coverage > 0.05,
    `覆盖 ${(data.pixels.coverage * 100).toFixed(1)}%，${data.pixels.nonZero} 像素`);
  check('存在明亮像素（不是全黑）', data.pixels.colored > 1000,
    `${data.pixels.colored} 个较亮像素`);
  check('不透明度正常（maxAlpha > 200）', data.pixels.maxAlpha > 200, `maxAlpha=${data.pixels.maxAlpha}`);
  check('帧率不超过 60FPS 上限', data.rafFps <= 62, `${data.rafFps} FPS`);
  check('FPS 数值合理（不是几万的异常值）', data.rafFps > 0 && data.rafFps <= 200, `${data.rafFps}`);
  // 场景尺度必须可读：state.bounds 若为空，`?.radius * 0.4 ?? 1` 会算出 NaN 并把相机打成黑屏
  check('场景尺度 state.bounds 已写入且为有限数', data.boundsIsFinite === true,
    `radius=${data.boundsRadius}`);
  const bad = (data.cameraCheckpoints ?? []).filter((c) => !c.finite);
  check('点击标注列表不会把相机位置变成 NaN（黑屏根因）', bad.length === 0,
    bad.length ? `异常于「${bad[0].label}」，位置 ${JSON.stringify(bad[0].pos)}` : `检查了 ${data.annoItems} 个条目`);
  check('点击标注后三维区仍有内容', (data.pixelsAfterClicks ?? 0) > 1000,
    `${data.pixelsAfterClicks} 像素；诊断 ${JSON.stringify(data.diagAfterClicks)}`);
}

const noise = consoleErrors.filter((m) => !/Download the React|DevTools/.test(m));
check('控制台无 error 输出', noise.length === 0, noise.slice(0, 3).join(' | '));

console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
ws.close();
edge.kill();
process.exit(fail === 0 ? 0 : 1);
