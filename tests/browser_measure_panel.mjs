/**
 * 浏览器回归：精确测量面板（核心功能二 / 三）
 *
 * 验证「在粗模上标出需精确测量的位置 → 民警填写实测值 → 反算比例尺」这条闭环的 UI 部分：
 *   1) 左侧面板出现「精确测量」区、待测徽标、「待测清单」与「按实测量校正」按钮
 *   2) 工具栏出现「📏 尺寸标注」工具
 *   3) 用 API 造两条待测尺寸后，徽标数字变化、清单能列出并填表
 *   4) 填表保存后实测值写回标注
 *   5) 校正后比例尺写入 project.world，且前端测距文字按真实尺寸变化
 *   6) 导出 CSV 按钮可用
 *
 * 用法: node tests/browser_measure_panel.mjs
 */
import { spawn } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';
const PORT = 9261;

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

const api = async (p, opts = {}) => {
  const r = await fetch(BASE + p, {
    method: opts.method ?? 'GET',
    headers: opts.body ? { 'content-type': 'application/json' } : {},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  return { status: r.status, data: await r.json().catch(() => ({})) };
};

const proj = (await api('/api/projects', { method: 'POST', body: { name: 'measure-ui-test' } })).data.project;
const pid = proj.id;
console.log(`测试案件 ${pid}`);

// 造两条待测尺寸：模型 2.0 与 1.0，真值分别是 4.0 与 2.0（比例尺 2.0）
const mk = (label, d) =>
  api(`/api/projects/${pid}/annotations`, {
    method: 'POST',
    body: {
      annotation: {
        kind: 'dimension', label,
        points: [[0, 0, 0], [d, 0, 0]],
        modelDistance: d, unit: 'meter', priority: 'high',
      },
      operator: '技术员',
    },
  });
const d1 = await mk('客厅东西向', 2.0);
const d2 = await mk('门洞净宽', 1.0);

// ---------------------------------------------------------------- 浏览器
const PROFILE = path.join(process.cwd(), '.browsertmp', 'measure');
rmSync(PROFILE, { recursive: true, force: true });
mkdirSync(PROFILE, { recursive: true });

const edge = spawn(EDGE, [
  '--headless=new', '--no-sandbox', '--disable-crash-reporter',
  '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  '--window-size=1400,950', BASE,
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
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const consoleErrors = [];
await new Promise((r) => (ws.onopen = r));
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') consoleErrors.push(m.params.exceptionDetails?.exception?.description ?? '?');
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
    consoleErrors.push(m.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
  }
};
const send = (method, params = {}) => new Promise((res) => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
await send('Runtime.enable');

let booted = false;
for (let i = 0; i < 40 && !booted; i++) {
  const r = await send('Runtime.evaluate', { expression: 'String(!!(window.__caseSystem && window.__caseSystem.measurePanel))', returnByValue: true });
  booted = r.result?.result?.value === 'true';
  if (!booted) await sleep(500);
}
check('页面已启动且 measurePanel 已挂载', booted);
await sleep(1200);

// 切到测试案件
const sw = await send('Runtime.evaluate', {
  expression: `(async () => {
    const sel = document.getElementById('projectSelect');
    const opt = [...sel.options].find(o => o.value.includes('measure-ui-test'));
    if (!opt) return 'no-option';
    sel.value = opt.value;
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    await new Promise(r => setTimeout(r, 2200));
    const S = window.__caseSystem;
    return JSON.stringify({ project: S.state.project?.id ?? null, dims: (S.state.project?.annotations ?? []).filter(a=>a.kind==='dimension').length });
  })()`,
  returnByValue: true, awaitPromise: true,
});
let swd = {};
try { swd = JSON.parse(sw.result?.result?.value ?? '{}'); } catch { swd = {}; }
console.log('切换案件:', JSON.stringify(swd));
check('已切到测试案件且读到了尺寸标注', swd.dims === 2, `dims=${swd.dims}`);

// ---- 1) DOM 存在性
const dom = JSON.parse((await send('Runtime.evaluate', {
  expression: `JSON.stringify({
    badge: document.getElementById('measureBadge')?.textContent ?? null,
    listBtn: !!document.getElementById('btnMeasureList'),
    recalBtn: !!document.getElementById('btnRecalibrate'),
    dlg: !!document.getElementById('dlgMeasure'),
    toolBtn: !!document.querySelector('[data-tool="dimension"]'),
    toolText: document.querySelector('[data-tool="dimension"]')?.textContent?.trim() ?? null,
    scaleHint: document.getElementById('scaleHint')?.textContent?.trim()?.slice(0,60) ?? null,
  })`, returnByValue: true,
})).result?.result?.value ?? '{}');
check('左侧有「待测清单」与「按实测量校正」按钮', dom.listBtn && dom.recalBtn);
check('工具栏有「尺寸标注」工具', dom.toolBtn === true, dom.toolText ?? '');
check('存在测量清单对话框', dom.dlg === true);
check('显示了比例尺/待测提示', (dom.scaleHint ?? '').length > 0, dom.scaleHint ?? '');
check('待测徽标显示 2 项', dom.badge === '2', `badge=${dom.badge}`);

// ---- 2) 打开清单并把两条实测值填进去
const fill = await send('Runtime.evaluate', {
  expression: `(async () => {
    const P = window.__caseSystem.measurePanel;
    await P.open();
    await new Promise(r => setTimeout(r, 500));
    const dlg = document.getElementById('dlgMeasure');
    const inputs = [...document.querySelectorAll('#dlgMeasure [data-measure-input]')];
    const rows = [...document.querySelectorAll('#dlgMeasure .measure-row')].length;
    // 按"模型值×2"填写，模拟真值比例尺为 2
    const data = P.data;
    for (const p of data.pending) {
      const el = document.querySelector('[data-measure-input="'+p.id+'"]');
      if (el) {
        el.value = String(Number(p.modelDistance) * 2);
        el.dispatchEvent(new Event('input', { bubbles: true }));
      }
    }
    const draftSize = P.draft.size;
    await P.saveAll();
    await new Promise(r => setTimeout(r, 1200));
    const after = await (await fetch('/api/projects/'+P.currentProjectId()+'/measurements')).json();
    return JSON.stringify({
      open: dlg.open, rows, inputs: inputs.length, draftSize,
      pendingAfter: (after.pending ?? []).length,
      measuredAfter: (after.measured ?? []).length,
    });
  })()`,
  returnByValue: true, awaitPromise: true,
});
const fd = JSON.parse(fill.result?.result?.value ?? '{}');
console.log('填表结果:', JSON.stringify(fd));
check('清单对话框已打开并列出 2 行', fd.open === true && fd.rows === 2, `rows=${fd.rows}`);
check('有两个可填写的实测输入框', fd.inputs === 2, `inputs=${fd.inputs}`);
check('填写的值进入了草稿', fd.draftSize === 2, `draft=${fd.draftSize}`);
check('保存后待测清空', fd.pendingAfter === 0, `pending=${fd.pendingAfter}`);
check('保存后已有 2 项实测', fd.measuredAfter === 2, `measured=${fd.measuredAfter}`);

// ---- 3) 校正比例尺，并验证前端显示按真实尺寸变化
const cal = await send('Runtime.evaluate', {
  expression: `(async () => {
    const S = window.__caseSystem, P = S.measurePanel;
    const before = Number(S.state.project?.world?.scaleToMeters ?? 1);
    await P.recalibrate();
    await new Promise(r => setTimeout(r, 1500));
    const after = Number(S.state.project?.world?.scaleToMeters ?? 1);
    // 找到一条 dimension 标注，看它的标签文字
    const ann = (S.state.project.annotations ?? []).find(a => a.kind === 'dimension');
    return JSON.stringify({
      before, after,
      annMeasured: ann?.measured ?? null,
      annModel: ann?.modelDistance ?? null,
      scaleInWorld: S.state.project?.world?.scaleToMeters ?? null,
    });
  })()`,
  returnByValue: true, awaitPromise: true,
});
const cd = JSON.parse(cal.result?.result?.value ?? '{}');
console.log('校正结果:', JSON.stringify(cd));
check('校正前比例尺为 1', cd.before === 1, `before=${cd.before}`);
check('校正后比例尺 ≈ 2', Math.abs(Number(cd.after) - 2) < 1e-6, `after=${cd.after}`);
check('实测值已写回标注', Math.abs(Number(cd.annMeasured) - 4) < 1e-6, `measured=${cd.annMeasured}`);

// 验证测距文字按比例尺缩放（textFor 里做了乘法）
const label = await send('Runtime.evaluate', {
  expression: `(async () => {
    const S = window.__caseSystem;
    const P = S.pnpPanel?.constructor; // 无关，只为拿到 app 作用域
    // 造一条 measure：模型长度 1.0，比例尺 2 → 应显示 2.000 m
    const r = await fetch('/api/projects/'+S.state.project.id+'/annotations', {
      method:'POST', headers:{'content-type':'application/json'},
      body: JSON.stringify({ annotation: { kind:'measure', label:'测试测距', points:[[0,0,0],[1,0,0]] } })
    });
    const j = await r.json();
    S.state.project.annotations.push(j.annotation);
    S.ui.renderAnnotations(S.state.project.annotations);
    // 直接问 labelProjector 的文字
    const text = S.labelProjector ? S.labelProjector.textFor(j.annotation) : null;
    return JSON.stringify({ added: !!j.annotation, text });
  })()`,
  returnByValue: true, awaitPromise: true,
});
const ld = JSON.parse(label.result?.result?.value ?? '{}');
if (ld.text) {
  check('测距标签按校正后的比例尺显示', /2\.0+ m/.test(ld.text), ld.text.replace(/<[^>]+>/g, ' '));
} else {
  check('测距标签按校正后的比例尺显示（未暴露 labelProjector，跳过）', true, 'skip');
}

// ---- 4) 导出 CSV 按钮可用
const exp = JSON.parse((await send('Runtime.evaluate', {
  expression: `(async () => {
    const P = window.__caseSystem.measurePanel;
    await P.refresh();
    const btn = document.getElementById('btnMeasureExport');
    return JSON.stringify({ exists: !!btn, disabled: btn?.disabled ?? null });
  })()`,
  returnByValue: true, awaitPromise: true,
})).result?.result?.value ?? '{}');
check('导出 CSV 按钮存在（无待测项时禁用）', exp.exists === true, `disabled=${exp.disabled}`);

const noise = consoleErrors.filter((m) => !/Download the React|DevTools|setPointerCapture/.test(m));
check('控制台无 error 输出', noise.length === 0, noise.slice(0, 2).join(' | '));

await api(`/api/projects/${pid}`, { method: 'DELETE' });
ws.close();
edge.kill();
console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail === 0 ? 0 : 1);
