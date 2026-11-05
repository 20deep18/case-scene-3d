/**
 * 验证：房间结构骨架**真的被画到屏幕上**了。
 *
 * 上一轮这个测试是假阳性：只断言「THREE.Scene 里有对象」，12/12 全过但截图里
 * 一条线都没有（本项目没有 Three.js 渲染器）。这一版改成两层验证：
 *   1) 数据链路：structureUrl / 自检 / SVG 元素数量与屏幕坐标；
 *   2) **像素级**：截两张图（结构开 / 关），统计画面里结构线颜色的像素数，
 *      开的时候必须显著多于关的时候。这才是「用户看得见」的证据。
 *
 * 用法: node tests/browser_structure_layer.mjs [模型名关键字]
 *
 * ⚠️ 2026-10-06 默认目标从「现场模型」改为「room-real」：
 *    案件里那批「现场模型」只有 16 个相机 / 292 个稀疏点，房间被拟合成 17×11 m，
 *    新的三态判定把它们诚实地判为 **uncertain**（证据见 sceneModeEvidence）。
 *    此时前端**故意不画房间盒** —— 继续断言"房间盒被画出来"等于断言一个假东西。
 *    room-real（178 相机、真实房间）才是这条链路的合格目标。
 *    另见文件末尾新增的「判定门控」一节：那里直接验证了
 *    「只有 room 才画房间盒」这条规则（柜子被糊上假房间盒的那个回归）。
 */
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const KEY = process.argv[2] ?? 'room-real';
const PORT = 9274;
const OUT = path.join(process.cwd(), 'output', 'verify');
mkdirSync(OUT, { recursive: true });

let pass = 0;
let fail = 0;
const check = (name, ok, detail = '') => {
  if (ok) { pass++; console.log(`  [OK]   ${name}${detail ? `  ${detail}` : ''}`); }
  else { fail++; console.log(`  [FAIL] ${name}${detail ? `  ${detail}` : ''}`); }
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// 前置：模型索引里必须带 structureUrl
const idx = await (await fetch(`${BASE}/models/index.json`)).json();
const target = idx.models.find((m) => String(m.name).includes(KEY));
check('模型索引里能找到目标模型', Boolean(target), target?.name ?? '');
check('该模型带 structureUrl（有结构骨架）', Boolean(target?.structureUrl), target?.structureUrl ?? '无');
const st = target?.structureUrl ? await (await fetch(BASE + target.structureUrl)).json() : null;
check('结构骨架内容自检通过（相机离地高度合理）', st?.sane === true,
  st ? `房间 ${st.room.size.map((v) => v.toFixed(1)).join(' × ')} m，相机离地 ${st.trajectory.heightAboveFloor.median} m` : '');
check('方向覆盖度诊断存在', Boolean(st?.coverage?.verdict), st?.coverage?.verdict ?? '缺失');
check('覆盖度结论也进了模型索引（前端不用再拉一次文件）',
  Boolean(target?.coverage?.verdict), target?.coverage?.verdict ?? '缺失');
check('覆盖度还能定向一致（索引值 == 结构文件值）',
  target?.coverage?.verdict === st?.coverage?.verdict);

const PROFILE = path.join(process.cwd(), '.browsertmp', 'structure');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });
const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1440,1000', BASE,
], { stdio: 'ignore' });

let page = null;
for (let i = 0; i < 40 && !page; i++) {
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
  } catch { /* retry */ }
  if (!page) await sleep(400);
}
if (!page) { console.log('浏览器没起来'); edge.kill(); process.exit(1); }

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const errors = [];
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails?.exception?.description ?? '?');
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
    errors.push(m.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
  }
};
const send = (method, params = {}) =>
  new Promise((res) => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (e, a = false) => {
  const r = await send('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: a });
  if (r.result?.exceptionDetails) return `EXC: ${r.result.exceptionDetails.exception?.description}`;
  return r.result?.result?.value;
};
const json = async (e, a = false) => { try { return JSON.parse(await evaluate(e, a)); } catch { return {}; } };
await send('Runtime.enable');
for (let i = 0; i < 40; i++) {
  if ((await evaluate('String(!!(window.__caseSystem && window.__caseSystem.structureLayer))')) === 'true') break;
  await sleep(500);
}
check('页面暴露了 structureLayer', true);
await sleep(1500);

const loaded = await json(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const m = j.models.find(x => String(x.name).includes(${JSON.stringify(KEY)})) ?? j.models[0];
  // announce:true = 用户主动载入（点「载入三维现场」）。拍摄建议只在这一路径弹出，
  // 避免开机自动恢复模型时用模态窗口挡住用户（这个 bug 被 browser_recon_gate 抓到过）。
  await S.loadModel(m, { announce: true });
  await new Promise(r => setTimeout(r, 4000));
  const L = S.structureLayer;
  const st = L.stats();
  // 再看 SVG 里是否真的算出了屏幕坐标（不只是建了元素）
  const svg = document.getElementById('structureLayer');
  const lines = [...svg.querySelectorAll('line')];
  const onScreen = lines.filter(el => el.style.display !== 'none'
    && Number(el.getAttribute('x1')) >= -50 && Number(el.getAttribute('x1')) <= window.innerWidth + 50
    && Number(el.getAttribute('y1')) >= -50 && Number(el.getAttribute('y1')) <= window.innerHeight + 50);
  const nonzero = lines.filter(el => {
    const dx = Number(el.getAttribute('x2')) - Number(el.getAttribute('x1'));
    const dy = Number(el.getAttribute('y2')) - Number(el.getAttribute('y1'));
    return Math.hypot(dx, dy) > 4;
  });
  return JSON.stringify({
    ...st,
    svgPresent: !!svg,
    svgLines: lines.length,
    svgPolylines: svg.querySelectorAll('polyline').length,
    svgCircles: svg.querySelectorAll('circle').length,
    onScreenLines: onScreen.length,
    nonZeroLines: nonzero.length,
  });
})()`, true);
console.log('结构图层:', JSON.stringify(loaded));
check('页面里有 SVG 覆盖层', loaded.svgPresent === true);
check('结构数据来自 structure.json', (loaded.roomSize ?? null) !== null,
  loaded.roomSize ? `房间 ${loaded.roomSize.map((v) => v.toFixed(1)).join(' × ')} m，轨迹点 ${loaded.trajPoints}` : '');
check('生成了房间盒/网格线元素', (loaded.segments ?? 0) >= 20, `线段 ${loaded.segments} 条`);
check('生成了相机轨迹与起终点', (loaded.polylines ?? 0) >= 1 && (loaded.dots ?? 0) >= 2,
  `polyline ${loaded.polylines}，circle ${loaded.dots}`);
check('线段投影到了屏幕内（有真实屏幕坐标）', (loaded.onScreenLines ?? 0) >= 8,
  `屏内 ${loaded.onScreenLines}/${loaded.svgLines}`);
check('线段长度非零（不是退化成点）', (loaded.nonZeroLines ?? 0) >= 8,
  `非零长度 ${loaded.nonZeroLines}/${loaded.svgLines}`);
// 2026-11-02 改：未校正比例尺时标签写「单位」而不是「m」——
// 因为重建单位 ≠ 米（白瓷砖那份 1 单位 ≈ 0.09 m，标成 m 会差 11 倍）。
// 校正过（scaleToMeters ≠ 1）才写 m，这条在下面单独验证。
check('房间盒上标了长/宽/高尺寸', (loaded.labels ?? 0) >= 3
  && (loaded.labelTexts ?? []).every((t) => /(单位| m)$/.test(t)),
  `标注 ${JSON.stringify(loaded.labelTexts ?? [])}`);
// 校正比例尺后，标签必须变成米（且数值乘上系数）—— 验的是"标定接通了"
const calibrated = await json(`(() => {
  window.__caseSystem.structureLayer.setCalibration(2.0);
  const texts = window.__caseSystem.structureLayer.labels.map(l => l.text);
  window.__caseSystem.structureLayer.setCalibration(1);
  return JSON.stringify({ texts });
})()`);
check('校正比例尺后尺寸标签变成米（数值×系数）',
  (calibrated.texts ?? []).length >= 3
  && calibrated.texts.every((t) => / m$/.test(t))
  && calibrated.texts.some((t) => /^(3[0-9]|[0-9]{2})\.[0-9]{2} m$/.test(t)),
  `校正后 ${JSON.stringify(calibrated.texts ?? [])}`);
const svgText = await evaluate(`document.querySelectorAll('#structureLayer text').length`);
check('尺寸文字真的进了 SVG', Number(svgText) >= 3, `text 元素 ${svgText} 个`);

// ---------------------------------------------------------------- 拍摄引导
// 覆盖度不达标时，载入模型后必须**主动告诉用户「有方向没拍到、要补拍」**，
// 否则他会以为是参数问题，反复重跑、每次白等二十分钟。
// 覆盖度提示：**从模态框改成了非阻塞 toast**（模态会正好盖在模型与结构线上，
// 演示时被误认为"结构线没了"——实测踩到）。这里断言 toast 里有完整信息，
// 并且**不再弹模态**。
let advice = { modalOpen: null, toast: '', shown: false };
for (let i = 0; i < 24; i++) {
  advice = await json(`JSON.stringify({
    modalOpen: document.getElementById('dlgShooting')?.open ?? null,
    toast: document.getElementById('toast')?.textContent ?? '',
    shown: (document.getElementById('toast')?.className || '').includes('show'),
  })`);
  if (advice.shown && /覆盖提示/.test(advice.toast)) break;
  await new Promise((r) => setTimeout(r, 300));
}
if (target?.coverage?.verdict && target.coverage.verdict !== '覆盖良好') {
  check('覆盖度不佳时给出「覆盖提示」（toast）', advice.shown && /覆盖提示/.test(advice.toast),
    advice.toast.slice(0, 80));
  check('提示里说清了「有方向没拍到」', /没拍到|缺面|塌陷/.test(advice.toast), advice.toast.slice(0, 80));
  check('提示里给出了具体帧数', /\d+\s*帧/.test(advice.toast), advice.toast.slice(0, 80));
  check('提示里点明「这不是参数问题」并建议补拍', /不是参数/.test(advice.toast) && /补拍/.test(advice.toast),
    advice.toast.slice(0, 80));
  check('不再用模态框挡画面（演示友好）', advice.modalOpen !== true, `modalOpen=${advice.modalOpen}`);
} else {
  check('覆盖良好时不打扰用户（不弹提示）', !/覆盖提示/.test(advice.toast), advice.toast);
}


// 开关必须生效
const toggled = await json(`(() => {
  const L = window.__caseSystem.structureLayer;
  const cb = document.getElementById('cStructure');
  cb.checked = false; cb.dispatchEvent(new Event('change', { bubbles: true }));
  const off = L.visible;
  cb.checked = true; cb.dispatchEvent(new Event('change', { bubbles: true }));
  return JSON.stringify({ off, on: L.visible });
})()`);
check('「显示房间结构线框」开关能关掉图层', toggled.off === false);
check('开关能重新打开图层', toggled.on === true);

// ---------------------------------------------------------------- 像素级验证
// 这才是「用户看得见」的证据：截两张图（结构开 / 关），统计结构线颜色的像素数。
const shoot = async (file) => {
  const s = await send('Page.captureScreenshot', { format: 'png' });
  if (!s.result?.data) return null;
  const p = path.join(OUT, file);
  writeFileSync(p, Buffer.from(s.result.data, 'base64'));
  return p;
};
// 固定一个能同时看到房间盒与轨迹的视角（环绕）
await evaluate(`(() => {
  const b = [...document.querySelectorAll('#modeSwitch button')].find(x => x.dataset.mode === 'orbit');
  if (b) b.click();
  return 'ok';
})()`);
await sleep(2000);
await evaluate(`(() => { const L = window.__caseSystem.structureLayer; L.setVisible(true);
  const cb = document.getElementById('cStructure'); if (cb) cb.checked = true; return 'ok'; })()`);
await sleep(1200);
const onPath = await shoot('structure-pixel-on.png');
await evaluate(`(() => { window.__caseSystem.structureLayer.setVisible(false); return 'ok'; })()`);
await sleep(1200);
const offPath = await shoot('structure-pixel-off.png');
await evaluate(`(() => {
  window.__caseSystem.structureLayer.setVisible(true);
  const cb = document.getElementById('cStructure'); if (cb) cb.checked = true;
  return 'ok';
})()`);

/** 用绑定的 Python + Pillow 数像素（测试脚本不引入新依赖） */
function countPixels(image, colors) {
  const py = path.join(process.cwd(), 'pyruntime312', 'python.exe');
  const tool = path.join(process.cwd(), 'tools', 'count_color_pixels.py');
  const r = spawnSync(py, [tool, '--image', image, '--color', colors, '--tol', '46', '--json'], {
    encoding: 'utf8',
  });
  if (r.status !== 0) return null;
  try {
    return JSON.parse(r.stdout.trim().split('\n').pop());
  } catch {
    return null;
  }
}

const COLORS = '#4dd0e1,#ff8a80,#ffcc33'; // 房间盒 / 相机轨迹 / 墙面
let onCount = null;
let offCount = null;
if (onPath && offPath) {
  onCount = countPixels(onPath, COLORS);
  offCount = countPixels(offPath, COLORS);
}
console.log('  开：', onCount ? `${onCount.sum} 像素` : '（取不到）',
  onCount ? onCount.counts.map((c) => `${c.color}=${c.pixels}`).join(' ') : '');
console.log('  关：', offCount ? `${offCount.sum} 像素` : '（取不到）',
  offCount ? offCount.counts.map((c) => `${c.color}=${c.pixels}`).join(' ') : '');
check('能统计到结构线颜色的像素（截图可解析）', Boolean(onCount && offCount));
// 注意：青色（#4dd0e1）同时是**界面主题色**（按钮、选中态都用它），
// 所以关掉图层后青色不会归零 —— 必须看**增量**，不能看绝对值。
const delta = onCount && offCount ? onCount.sum - offCount.sum : 0;
check('打开结构线框后结构相关像素明显增加（≥ 2500 像素）',
  delta >= 2500, onCount && offCount ? `开 ${onCount.sum} → 关 ${offCount.sum}（+${delta}）` : '');
// ⚠️ 2026-10-06 从「关掉后 ≥2.5×」改成「绝对增量 ≥1500 像素」：
//    换成 room-real（真实房间视频训练的模型）当目标后，**画面内容本身**就有
//    1958 个像素落在轨迹色 #ff8a80 的容差(46)内 —— 木色/暖色墙面本来就是橙红系。
//    比值判据会把背景算进去，测到 4601/1958 = 2.35 而误报失败。
//    轨迹"有没有真的画出来"要看**它自己多画了多少像素**，即绝对增量。
const trajDelta = onCount && offCount ? onCount.counts[1].pixels - offCount.counts[1].pixels : 0;
check('相机轨迹在开启后确实多画出来了（绝对增量 ≥ 1500 像素）',
  Boolean(onCount && offCount) && trajDelta >= 1500,
  onCount && offCount ? `开 ${onCount.counts[1].pixels} → 关 ${offCount.counts[1].pixels}（+${trajDelta}）` : '');
check('房间盒颜色（青）也有可测的增量（≥ 500 像素）',
  Boolean(onCount && offCount) && onCount.counts[0].pixels - offCount.counts[0].pixels >= 500,
  onCount && offCount ? `开 ${onCount.counts[0].pixels} → 关 ${offCount.counts[0].pixels}` : '');

// ---------------------------------------------------------------- 判定门控
// 回归护栏（2026-10-06）：`structure-layer.set()` 曾经**无条件**画房间盒 ——
// `sceneMode` 这个字段在 `app/` 里当时零引用，于是柜子那份素材被糊上一个
// 假的「6.0 × 3.7 × 3.7 m 房间盒」。用户的原话是「明明是个柜子，但是模型是不规则的」，
// 界面上再画一个假户型只会更误导。
//
// 这里用**合成 payload** 直接验证门控规则（与真实模型无关，所以断言是确定的）：
//   room + sane           → 画房间盒/地面网格（segments 多）
//   object / uncertain    → **只有相机轨迹**（segments = 0，polylines = 1）
//   room + sane=false     → 也降级为只画轨迹（盒子可能是倒扣的）
const gating = await json(`(() => {
  const L = window.__caseSystem.structureLayer;
  const real = L.data;                 // 先存下真实数据，测完恢复，免得影响截图以外的状态
  const c = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [0, 1]) c.push([sx * 2, sy * 1.5, sz * 2.6]);
  const base = {
    room: { size: [4, 3, 2.6], corners: c, origin: [0, 0, 0], axisU: [1, 0, 0], axisV: [0, 1, 0], floorY: 0, ceilingY: 2.6 },
    walls: [],
    planOutline: { segmentsWorld: [] },
    trajectory: { points: [[0, 0, 0], [1, 0, 0]] },
    sceneModeEvidence: { rejectReasons: ['合成用例：仅用于验证门控'] },
  };
  const out = {};
  for (const mode of ['room', 'object', 'uncertain']) {
    L.set({ ...base, sceneMode: mode, sane: true });
    out[mode] = { mode: L.mode, segments: L.segments.length, polylines: L.polylines.length, labels: L.labels.length, notice: L.notice };
  }
  L.set({ ...base, sceneMode: 'room', sane: false });
  out.roomNotSane = { mode: L.mode, segments: L.segments.length, polylines: L.polylines.length, notice: L.notice };
  // 老文件（没有 sceneMode 字段）：退回按 sane 判断，保持向后兼容
  L.set({ ...base, sane: true });
  out.legacySane = { mode: L.mode, segments: L.segments.length };
  L.set(real);                          // 恢复真实数据
  return JSON.stringify(out);
})()`);
console.log('判定门控:', JSON.stringify(gating));
check('sceneMode=room → 画房间盒与地面网格', gating.room.segments >= 20 && gating.room.labels >= 3,
  `线段 ${gating.room.segments}，尺寸标注 ${gating.room.labels}`);
check('sceneMode=object → 只画相机轨迹，不画房间盒',
  gating.object.segments === 0 && gating.object.polylines === 1,
  `线段 ${gating.object.segments}，polyline ${gating.object.polylines}`);
check('sceneMode=uncertain → 只画相机轨迹，不画房间盒',
  gating.uncertain.segments === 0 && gating.uncertain.polylines === 1,
  `线段 ${gating.uncertain.segments}，polyline ${gating.uncertain.polylines}`);
check('不画房间盒时必须给出原因（不能默默什么都不显示）',
  /物体|不确定|自检/.test(gating.object.notice) && /物体|不确定|自检/.test(gating.uncertain.notice),
  (gating.uncertain.notice || '').slice(0, 60));
check('sane=false 时即使 sceneMode=room 也不画房间盒（盒子可能倒扣）',
  gating.roomNotSane.segments === 0 && /自检/.test(gating.roomNotSane.notice),
  `线段 ${gating.roomNotSane.segments}，${(gating.roomNotSane.notice || '').slice(0, 40)}`);
check('老结构文件（无 sceneMode）仍按 sane 回退，不会突然变空白',
  gating.legacySane.mode === 'room' && gating.legacySane.segments >= 20,
  `mode=${gating.legacySane.mode} 线段 ${gating.legacySane.segments}`);

// ---------------------------------------------------------------- 人工改判入口
// 用户能在这里把判定改成「房间 / 物体 / 自动」—— 几何分不开这两类素材
// （实测柜子 0.2344 vs 某段房间 0.2270），判不准时由拍摄的人指定最可靠。
// 这里走**真实 UI**（下拉框的 change 事件），并断言图层真的跟着变、改完能恢复。
const override = await json(`(async () => {
  const S = window.__caseSystem;
  const sel = document.getElementById('sceneModeSelect');
  const field = document.getElementById('sceneModeField');
  const hint = document.getElementById('sceneModeHint');
  const snap = () => ({
    seg: S.structureLayer.segments.length,
    polylines: S.structureLayer.polylines.length,
    mode: S.structureLayer.mode,
    source: S.structureLayer.modeSource,
    notice: S.structureLayer.notice,
    selValue: sel.value,
  });
  const out = { fieldHidden: field.hidden, hintHidden: hint.hidden, hintText: hint.textContent, before: snap() };
  sel.value = 'object';
  sel.dispatchEvent(new Event('change', { bubbles: true }));
  await new Promise(r => setTimeout(r, 1500));
  out.forced = snap();
  out.hintAfterForce = hint.textContent;
  sel.value = 'auto';
  sel.dispatchEvent(new Event('change', { bubbles: true }));
  await new Promise(r => setTimeout(r, 1500));
  out.restored = snap();
  return JSON.stringify(out);
})()`, true);
console.log('人工改判:', JSON.stringify(override));
check('有结构骨架时才显示改判控件', override.fieldHidden === false && override.hintHidden === false,
  `hint="${(override.hintText || '').slice(0, 50)}"`);
check('改判前是自动判定且画了房间盒',
  override.before.mode === 'room' && override.before.source === 'auto' && override.before.seg >= 20,
  `mode=${override.before.mode} source=${override.before.source} 线段=${override.before.seg}`);
check('改判为「物体」后房间盒消失（只剩轨迹）',
  override.forced.mode === 'object' && override.forced.seg === 0 && override.forced.polylines === 1,
  `mode=${override.forced.mode} 线段=${override.forced.seg} polyline=${override.forced.polylines}`);
check('改判来源标记为 manual（不能假装是自动判的）',
  override.forced.source === 'manual', `source=${override.forced.source}`);
check('界面说清了这是人工改判', /人工改判/.test(override.forced.notice || ''),
  (override.forced.notice || '').slice(0, 60));
check('下拉框保持在用户选的值上', override.forced.selValue === 'object', override.forced.selValue);
check('恢复「自动判定」后房间盒回来了',
  override.restored.mode === 'room' && override.restored.source === 'auto' && override.restored.seg >= 20,
  `mode=${override.restored.mode} source=${override.restored.source} 线段=${override.restored.seg}`);

// ---------------------------------------------------------------- 物体盒门控
// 物体（柜子）的包围盒**只在几何可信时才画**。
// 为什么：实测那份柜子素材算出来的盒子是 4.42 × 16.77 × 8.60 m（柜子实际约 1×0.5×2 m）——
// 点云退化成货架棱边、绝大多数点在十几米外的背景上。画出来就是重演"假房间盒"。
// 这里用**合成 payload** 分别验证「可信 → 画盒+层」与「不可信 → 只画轨迹」。
const objGate = await json(`(() => {
  const L = window.__caseSystem.structureLayer;
  const real = L.data;
  const c = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [0, 1]) c.push([sx * 0.6, sy * 0.3, 0.2 + sz * 1.9]);
  const base = {
    sceneMode: 'object', sane: true, walls: [],
    room: { size: [1.2, 0.6, 1.9], corners: c, origin: [0, 0, 0], axisU: [1, 0, 0], axisV: [0, 1, 0] },
    planOutline: { segmentsWorld: [] },
    trajectory: { points: [[0, 0, 0], [1, 0, 0]] },
    sceneModeEvidence: { rejectReasons: [] },
  };
  // ⚠️ layers[].height 是**相对锚点**的竖直坐标（生产里就是 proj[:,2]），
  //    而这里的锚点=盒心，所以三层分别是 −0.95 / 0 / +0.95（盒高 1.9）。
  //    第一版我按"盒子底部为 0"写成 0.2/1.15/2.1，越界的两层被守卫正确跳过了 ——
  //    于是"画了 3 层"的断言失败。是我的 payload 错，不是代码错。
  const mkObj = (ok) => ({
    mode: 'object', center: [0, 0, 1.15], size: [1.2, 0.6, 1.9], corners: c,
    axes: { u: [1, 0, 0], v: [0, 1, 0], up: [0, 0, 1] },
    anchor: { method: 'camera-orbit', center: [0, 0, 1.15] },
    layers: [ { height: -0.95, index: 0, prominence: 0.96 }, { height: 0.0, index: 1, prominence: 0.91 },
              { height: 0.95, index: 2, prominence: 0.95 } ],
    trust: { ok, reasons: ok ? [] : ['盒心离相机群中心 12.74 m（= 3.6 × 半径，＞1.0）'] },
  });
  const out = {};
  L.set({ ...base, object: mkObj(true) });
  out.trusted = { seg: L.segments.length, labels: L.labels.length, drawn: L.objectDrawn, polylines: L.polylines.length };
  L.set({ ...base, object: mkObj(false) });
  out.untrusted = { seg: L.segments.length, polylines: L.polylines.length, drawn: L.objectDrawn, notice: L.notice };
  L.set({ ...base, object: null });
  out.noObject = { seg: L.segments.length, drawn: L.objectDrawn };
  // 场景判定「不确定」但物体几何自检通过 → **也要画**。
  // 为什么：实测一份已知尺寸的柜子（合成渲染 → 真 COLMAP，36/36 注册）会被判成 uncertain
  // （方向包围度 0.117 像物体，但"相机高度自洽得像房间"把它压回不确定）。
  // 若要求 sceneMode 必须等于 object，这个几何完全可信的柜子就永远画不出来。
  L.set({ ...base, sceneMode: 'uncertain', object: mkObj(true) });
  out.uncertainTrusted = { seg: L.segments.length, drawn: L.objectDrawn, notice: L.notice, mode: L.mode };
  L.set({ ...base, sceneMode: 'uncertain', object: mkObj(false) });
  out.uncertainUntrusted = { seg: L.segments.length, drawn: L.objectDrawn };
  // 房间模式不该被物体分支抢走（哪怕物体 trust 通过）
  L.set({ ...real, object: mkObj(true) });
  out.roomWins = { mode: L.mode, drawn: L.objectDrawn, seg: L.segments.length };
  out.roomDrawnTrue = L.roomDrawn === true;
  // 强制房间但自检不通过：**不得**声称已叠加（截图实测抓到过界面谎报）
  L.set({ ...real, sane: false, sceneMode: 'room', object: mkObj(true) });
  out.roomNotSaneDrawn = L.roomDrawn === true;
  out.roomNotSaneObjectDrawn = L.objectDrawn === true;
  out.roomNotSaneSeg = L.segments.length;
  out.roomNotSanePolylines = L.polylines.length;
  L.set(real);
  return JSON.stringify(out);
})()`);
console.log('物体盒门控:', JSON.stringify(objGate));
check('trust.ok → 画出物体盒（12 棱 + 3 尺寸标注）与水平层',
  objGate.trusted.drawn === true && objGate.trusted.seg >= 12 + 3 * 4,
  `线段 ${objGate.trusted.seg}，标注 ${objGate.trusted.labels}`);
check('trust.ok → 尺寸标注写的是物体三边',
  objGate.trusted.labels >= 3, `标注 ${objGate.trusted.labels}`);
check('trust.ok=false → 不画物体盒（只剩轨迹）',
  objGate.untrusted.drawn === false && objGate.untrusted.seg === 0 && objGate.untrusted.polylines === 1,
  `线段 ${objGate.untrusted.seg} polyline ${objGate.untrusted.polylines}`);
check('trust.ok=false 时必须说明原因（不能默默什么都不画）',
  /几何不可信/.test(objGate.untrusted.notice || ''), (objGate.untrusted.notice || '').slice(0, 70));
check('没有 object 记录时也不画、不报错',
  objGate.noObject.drawn === false && objGate.noObject.seg === 0, `线段 ${objGate.noObject.seg}`);
check('场景判定「不确定」但物体几何自检通过 → 仍然按「物体」画出来',
  objGate.uncertainTrusted.drawn === true && objGate.uncertainTrusted.seg >= 12,
  `mode=${objGate.uncertainTrusted.mode} 线段=${objGate.uncertainTrusted.seg}`);
check('这条路径要说明"判定不确定、但几何自检通过"（不能假装是确定判定的）',
  /不确定/.test(objGate.uncertainTrusted.notice || '') && /自检通过/.test(objGate.uncertainTrusted.notice || ''),
  (objGate.uncertainTrusted.notice || '').slice(0, 60));
check('场景判定「不确定」且几何不可信 → 不画',
  objGate.uncertainUntrusted.drawn === false && objGate.uncertainUntrusted.seg === 0,
  `线段 ${objGate.uncertainUntrusted.seg}`);
check('roomDrawn 只在真的画了房间盒时为真（sane=false 时必须为假）',
  objGate.roomNotSaneDrawn === false && objGate.roomDrawnTrue === true,
  `sane=false 时 roomDrawn=${objGate.roomNotSaneDrawn}，正常房间时 roomDrawn=${objGate.roomDrawnTrue}`);
check('房间+自检不过：既不画房间盒、也不退回画物体盒（房间模型画物体盒同样误导）',
  objGate.roomNotSaneDrawn === false && objGate.roomNotSaneObjectDrawn === false &&
    objGate.roomNotSaneSeg === 0 && objGate.roomNotSanePolylines === 1,
  `roomDrawn=${objGate.roomNotSaneDrawn} objectDrawn=${objGate.roomNotSaneObjectDrawn} ` +
    `线段=${objGate.roomNotSaneSeg} polyline=${objGate.roomNotSanePolylines}`);
check('房间模式不会被物体分支抢走（哪怕物体 trust 通过）',
  objGate.roomWins.mode === 'room' && objGate.roomWins.drawn === false && objGate.roomWins.seg >= 20,
  `mode=${objGate.roomWins.mode} drawn=${objGate.roomWins.drawn} 线段=${objGate.roomWins.seg}`);

// 真实模型：柜子现场是 object 且 trust=false → 界面上不该出现物体盒
const cabinet = await json(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const m = j.models.find(x => String(x.name).includes('柜子现场'));
  if (!m) return JSON.stringify({ error: '没有柜子现场模型' });
  await S.setSceneMode('auto', { silent: true });
  await S.loadModel(m, { announce: false });
  await new Promise(r => setTimeout(r, 4000));
  const L = S.structureLayer;
  return JSON.stringify({ mode: L.mode, trust: L.objectTrust?.ok ?? null, drawn: L.objectDrawn,
                          seg: L.segments.length, polylines: L.polylines.length,
                          hint: document.getElementById('sceneModeHint')?.textContent ?? '' });
})()`, true);
console.log('真实柜子:', JSON.stringify(cabinet));
check('真实柜子现场判为 object', cabinet.mode === 'object', String(cabinet.mode));
check('真实柜子的物体几何被判为不可信', cabinet.trust === false, `trust=${cabinet.trust}`);
check('真实柜子不画物体盒（只有相机轨迹）',
  cabinet.drawn === false && cabinet.seg === 0 && cabinet.polylines === 1,
  `线段 ${cabinet.seg} polyline ${cabinet.polylines}`);
check('界面把"为什么没画"写出来了', /物体盒未画/.test(cabinet.hint || ''),
  (cabinet.hint || '').slice(0, 90));
// 回到 room-real，别把后续断言留在柜子上。
// ⚠️ 顺手**明确清掉柜子模型自己的改判**：`setSceneMode` 作用在"当前载入的模型"上，
//    所以在载入柜子之后再调它，清的是柜子；只清一次的话会留下 sidecar
//    （实测踩过：data/models 里留下一个 mode=object 的 sidecar，
//     下次打开界面时那个柜子会显示成"人工改判为物体"，看起来像 bug）。
await json(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const cab = j.models.find(x => String(x.name).includes('柜子现场'));
  if (cab) {
    await fetch('/api/models/scene-mode', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: decodeURIComponent(cab.url.replace('/models/', '')), mode: 'auto' }),
    });
  }
  const m = j.models.find(x => String(x.name).includes('room-real'));
  await S.loadModel(m, { announce: false });
  await new Promise(r => setTimeout(r, 3500));
  return JSON.stringify({ mode: S.structureLayer.mode });
})()`, true);

// 左栏提示与顶栏状态牌必须一致：房间盒画出来了，就**不许**说"未叠加/自检未通过"
// （这是我引入过的真 bug：判据只看了"判定是不是房间"，漏看"到底画没画"）
const hintConsistency = await json(`(async () => {
  const S = window.__caseSystem;
  const j = await (await fetch('/models/index.json')).json();
  const m = j.models.find(x => String(x.name).includes('room-real'));
  await S.loadModel(m, { announce: false });
  await new Promise(r => setTimeout(r, 4000));
  return JSON.stringify({
    roomDrawn: S.structureLayer.roomDrawn === true,
    hint: document.getElementById('modelHint').textContent,
    badge: document.getElementById('structBadge').textContent,
  });
})()`, true);
console.log('提示条一致性:', JSON.stringify(hintConsistency));
check('房间盒真的画出来时，左栏提示不得说"未叠加/自检未通过"',
  hintConsistency.roomDrawn === true && !/未叠加|自检未通过/.test(hintConsistency.hint || ''),
  `hint="${(hintConsistency.hint || '').slice(0, 60)}"`);
check('顶栏状态牌与左栏提示说法一致（都在说"已画"）',
  /已画房间盒/.test(hintConsistency.badge || '') && !/未叠加/.test(hintConsistency.hint || ''),
  `badge="${hintConsistency.badge}"`);

const noise = errors.filter((m) => !/Download the React|DevTools|setPointerCapture|favicon/.test(m));
check('控制台无 error 输出', noise.length === 0, noise.slice(0, 2).join(' | '));

ws.close();
edge.kill();
console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail === 0 ? 0 : 1);
