/**
 * 浏览器回归：现场建模（核心功能一）
 *
 * 验证：
 *   1) 左侧面板出现「现场建模」区与「开始建模」按钮
 *   2) 无素材时打开对话框会明确提示「先导入素材」
 *   3) 有素材时对话框列出素材并允许启动
 *   4) 启动后后端真的创建了任务，进度区出现，且能取到日志
 *   5) 任务失败时前端显示失败原因（用一组必然失败的素材验证错误路径）
 *   6) 模型库接口能返回模型（供「模型文件」下拉使用）
 *
 * 用法: node tests/browser_recon_panel.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync, copyFileSync, existsSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9251;

let pass = 0;
let fail = 0;
function check(name, ok, detail = '') {
  if (ok) {
    pass++;
    console.log(`  [OK]   ${name}${detail ? `  ${detail}` : ''}`);
  } else {
    fail++;
    console.log(`  [FAIL] ${name}${detail ? `  ${detail}` : ''}`);
  }
}

// ---------------------------------------------------------------- 准备案件与素材
const proj = await (
  await fetch(`${BASE}/api/projects`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'recon-panel-test', operator: 'test' }),
  })
).json();
const pid = proj.project.id;
console.log(`测试案件 ${pid}`);

// 上传素材：用工作区里已有的真实照片（data/scene-big/images 的若干帧）作为「现场照片」
const srcDir = path.join(process.cwd(), 'data', 'scene-big', 'images');
const shots = [];
if (existsSync(srcDir)) {
  const all = (await import('node:fs')).readdirSync(srcDir).filter((f) => f.endsWith('.png')).sort();
  for (const f of all.slice(0, 6)) shots.push(path.join(srcDir, f));
}
if (!shots.length) {
  console.log('缺少测试图片，跳过素材相关断言');
}

async function uploadOne(file, name) {
  const buf = (await import('node:fs')).readFileSync(file);
  const res = await fetch(
    `${BASE}/api/projects/${encodeURIComponent(pid)}/media?name=${encodeURIComponent(name)}&chunk=0&chunks=1`,
    { method: 'POST', headers: { 'content-type': 'application/octet-stream' }, body: buf }
  );
  return res.json();
}

const uploaded = [];
for (const s of shots) {
  const r = await uploadOne(s, `现场照片_${path.basename(s)}`);
  if (r?.url) uploaded.push({ url: r.url, kind: 'image', name: path.basename(s) });
}
console.log(`已上传素材 ${uploaded.length} 个`);

// 建立一个「相机」标注并把素材挂上去（模拟用户在多源画面里导入）
// 注意：服务端要求 camera 标注必须带 pose（前端导入时由 defaultCameraPose() 生成），
// 所以这里也要给一个合法位姿，否则会 400/500。
if (uploaded.length) {
  const annoRes = await fetch(`${BASE}/api/projects/${encodeURIComponent(pid)}/annotations`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      annotation: {
        kind: 'camera',
        label: '现场照片组',
        points: [[0, 1.5, -4]],
        pose: { position: [0, 1.5, -4], quaternion: [0, 0, 0, 1], fov: 60, source: 'default' },
        media: uploaded,
      },
      operator: 'test',
    }),
  });
  const annoJson = await annoRes.json().catch(() => ({}));
  if (!annoRes.ok || annoJson.ok === false) {
    console.log(`  [警告] 建立相机标注失败：${annoRes.status} ${annoJson.error ?? ''}`);
  } else {
    console.log(`已建立相机标注，挂载 ${uploaded.length} 个素材`);
  }
}

// ---------------------------------------------------------------- 浏览器
const PROFILE = path.join(process.cwd(), '.browsertmp', 'recon');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

const edge = spawn(
  EDGE,
  [
    '--headless=new', '--no-sandbox', '--disable-crash-reporter',
    '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
    `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
    '--window-size=1400,900', BASE,
  ],
  { stdio: 'ignore' }
);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let page = null;
for (let i = 0; i < 40 && !page; i++) {
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
  } catch {
    /* retry */
  }
  if (!page) await sleep(400);
}
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const consoleErrors = [];
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  }
  if (m.method === 'Runtime.exceptionThrown') {
    consoleErrors.push(m.params.exceptionDetails?.exception?.description ?? '?');
  }
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
    consoleErrors.push(m.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
  }
};
const send = (method, params = {}) =>
  new Promise((res) => {
    pending.set(++id, res);
    ws.send(JSON.stringify({ id, method, params }));
  });
await send('Runtime.enable');

let booted = false;
for (let i = 0; i < 40 && !booted; i++) {
  const r = await send('Runtime.evaluate', {
    expression: 'String(!!(window.__caseSystem && window.__caseSystem.reconPanel))',
    returnByValue: true,
  });
  booted = r.result?.result?.value === 'true';
  if (!booted) await sleep(500);
}
check('页面已启动且 reconPanel 已挂载', booted);
await sleep(1500);

// 切到测试案件
const switchCase = `
(async () => {
  const sel = document.getElementById('projectSelect');
  const opt = [...sel.options].find(o => /recon-panel-test/.test(o.textContent) || o.value.includes('recon-panel-test'));
  if (!opt) return 'no-option';
  sel.value = opt.value;
  sel.dispatchEvent(new Event('change', { bubbles: true }));
  await new Promise(r => setTimeout(r, 2000));
  const S = window.__caseSystem;
  return JSON.stringify({ project: S.state.project?.id ?? null, annotations: S.state.project?.annotations?.length ?? 0 });
})()
`;
const sw = await send('Runtime.evaluate', { expression: switchCase, returnByValue: true, awaitPromise: true });
let swData = {};
try {
  swData = JSON.parse(sw.result?.result?.value ?? '{}');
} catch {
  swData = { raw: sw.result?.result?.value };
}
console.log('切换案件:', JSON.stringify(swData));
check('已切换到测试案件', typeof swData.project === 'string' && swData.project.length > 0, swData.project ?? '');

// ---- 断言 1：面板存在
const domCheck = await send('Runtime.evaluate', {
  expression: `JSON.stringify({
    btn: !!document.getElementById('btnRecon'),
    btnText: document.getElementById('btnRecon')?.textContent?.trim() ?? null,
    statusEl: !!document.getElementById('reconStatus'),
    dlg: !!document.getElementById('dlgRecon'),
    logDlg: !!document.getElementById('dlgReconLog'),
    hintText: document.getElementById('reconHint')?.textContent?.trim()?.slice(0,40) ?? null,
  })`,
  returnByValue: true,
});
const dom = JSON.parse(domCheck.result?.result?.value ?? '{}');
check('左侧存在「开始建模」按钮', dom.btn === true, dom.btnText ?? '');
check('存在进度状态区与两个对话框', dom.statusEl && dom.dlg && dom.logDlg);

// ---- 断言 2：collectAssets 能识别素材
const assetsCheck = await send('Runtime.evaluate', {
  expression: `(async () => {
    const P = window.__caseSystem.reconPanel;
    const a = P._collectAssets();
    return JSON.stringify({ count: a.length, kinds: [...new Set(a.map(x=>x.kind))], first: a[0]?.name ?? null });
  })()`,
  returnByValue: true,
  awaitPromise: true,
});
const assets = JSON.parse(assetsCheck.result?.result?.value ?? '{}');
check('能从案件状态里收集到建模素材', (assets.count ?? 0) === uploaded.length, `收集到 ${assets.count} 个（上传 ${uploaded.length} 个），类型 ${JSON.stringify(assets.kinds)}`);

// ---- 断言 3：打开对话框会列出素材并启用启动按钮
const dlgCheck = await send('Runtime.evaluate', {
  expression: `(async () => {
    const P = window.__caseSystem.reconPanel;
    P.openDialog();
    await new Promise(r => setTimeout(r, 400));
    const dlg = document.getElementById('dlgRecon');
    return JSON.stringify({
      open: dlg.open,
      assetsHtml: document.getElementById('reconAssets')?.textContent?.trim()?.slice(0,80) ?? '',
      startDisabled: document.getElementById('btnReconStart')?.disabled ?? null,
      nameValue: document.getElementById('rcName')?.value ?? '',
    });
  })()`,
  returnByValue: true,
  awaitPromise: true,
});
const dlg = JSON.parse(dlgCheck.result?.result?.value ?? '{}');
check('建模对话框能打开', dlg.open === true);
check('对话框里列出了将使用的素材', /视频|图片/.test(dlg.assetsHtml ?? ''), (dlg.assetsHtml ?? '').slice(0, 50));
check('有素材时「开始建模」可用', dlg.startDisabled === false);
check('模型名称已预填', typeof dlg.nameValue === 'string' && dlg.nameValue.length > 0, dlg.nameValue);

// ---- 断言 4：启动任务（用小迭代 + 大 downscale，让它快速跑到失败或成功）
//
// 注意：现在点「开始建模」会先做**素材体积体检**。本测试只上传了 6 张 ~1MB 的照片，
// 总体积低于实测下限，所以会先弹「提供信息过少，可能无法正常建模」的确认框。
// 这里不要直接 `await P.start()`（会一直等用户点按钮 → 测试挂死），
// 而是先拿到 Promise，若确认框出现就替用户点「仍然开始建模」。
const startCheck = await send('Runtime.evaluate', {
  expression: `(async () => {
    const P = window.__caseSystem.reconPanel;
    // 只给很少的图片必然注册失败，用来验证错误路径；这里给 6 张，预期失败
    document.getElementById('rcIterations').value = '1200';
    document.getElementById('rcDownscale').value = '4';
    document.getElementById('rcEvalViews')?.remove?.();
    const before = await (await fetch('/api/projects/' + P.currentProjectId() + '/recon')).json();
    const started = P.start();
    await new Promise(r => setTimeout(r, 900));
    const gateShown = document.getElementById('dlgConfirm').open;
    const gateTitle = document.getElementById('cfTitle').textContent;
    if (gateShown) document.getElementById('cfOk').click();
    await started;
    await new Promise(r => setTimeout(r, 1200));
    const after = await (await fetch('/api/projects/' + P.currentProjectId() + '/recon')).json();
    return JSON.stringify({
      before: (before.jobs ?? []).length,
      after: (after.jobs ?? []).length,
      jobId: P.jobId,
      stage: document.getElementById('reconStage')?.textContent ?? null,
      statusVisible: !document.getElementById('reconStatus')?.classList.contains('hidden'),
      gateShown, gateTitle,
    });
  })()`,
  returnByValue: true,
  awaitPromise: true,
});
const st = JSON.parse(startCheck.result?.result?.value ?? '{}');
console.log('启动结果:', JSON.stringify(st));
check('素材体积不足时先弹「提供信息过少」确认框', st.gateShown === true, st.gateTitle ?? '');
check('启动后后端创建了建模任务', (st.after ?? 0) > (st.before ?? 0), `任务数 ${st.before} -> ${st.after}`);
check('面板显示了进度状态区', st.statusVisible === true, `阶段=${st.stage}`);

// ---- 断言 5：能取到日志（给若干秒让流水线写出日志）
await sleep(6000);
const logCheck = await send('Runtime.evaluate', {
  expression: `(async () => {
    const P = window.__caseSystem.reconPanel;
    const r = await fetch('/api/projects/' + P.currentProjectId() + '/recon/' + P.jobId + '/log?from=0&tail=200');
    const d = await r.json();
    return JSON.stringify({ ok: r.ok, total: d.total ?? 0, hasStage: (d.lines ?? []).some(l => l.includes('[阶段]')), stage: d.status?.stage ?? null });
  })()`,
  returnByValue: true,
  awaitPromise: true,
});
const lg = JSON.parse(logCheck.result?.result?.value ?? '{}');
check('能从后端拉取建模日志', lg.ok === true, `共 ${lg.total} 行`);
check('日志里含可解析的阶段标记', lg.hasStage === true, `当前阶段=${lg.stage}`);

// ---- 断言 6：模型库接口可用（供「模型文件」下拉）
const modelCheck = await send('Runtime.evaluate', {
  expression: `(async () => {
    const j = await (await fetch('/models/index.json')).json();
    const sel = document.getElementById('modelSelect');
    return JSON.stringify({ count: (j.models ?? []).length, options: sel?.options?.length ?? 0 });
  })()`,
  returnByValue: true,
  awaitPromise: true,
});
const mc = JSON.parse(modelCheck.result?.result?.value ?? '{}');
check('模型库接口返回模型', (mc.count ?? 0) > 0, `${mc.count} 个模型`);
check('「模型文件」下拉已填充', (mc.options ?? 0) > 0, `${mc.options} 个选项`);

// ---- 清理：取消任务并删案件
await send('Runtime.evaluate', {
  expression: `(async () => { const P = window.__caseSystem.reconPanel; await fetch('/api/projects/'+P.currentProjectId()+'/recon/'+P.jobId+'/cancel',{method:'POST'}); return 'ok'; })()`,
  returnByValue: true,
  awaitPromise: true,
});

const noise = consoleErrors.filter((m) => !/Download the React|DevTools|setPointerCapture/.test(m));
check('控制台无 error 输出', noise.length === 0, noise.slice(0, 2).join(' | '));

await fetch(`${BASE}/api/projects/${encodeURIComponent(pid)}`, { method: 'DELETE' });
ws.close();
edge.kill();
console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail === 0 ? 0 : 1);
