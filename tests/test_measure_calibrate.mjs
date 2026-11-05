/**
 * 回归测试：精确测量 + 比例尺校正（API 层）
 *
 * 验证业务闭环：
 *   ① 在粗模上标出「需精确测量」的两点 → 生成待测项
 *   ② 民警填写实测值
 *   ③ 系统反算比例尺 → 待测清单清空、模型按真实尺寸显示
 *
 * 用法: node tests/test_measure_calibrate.mjs
 */
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';

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
const api = async (path, opts = {}) => {
  const res = await fetch(BASE + path, {
    method: opts.method ?? 'GET',
    headers: opts.body ? { 'content-type': 'application/json' } : {},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  return { status: res.status, data };
};

const proj = (await api('/api/projects', { method: 'POST', body: { name: 'measure-test' } })).data.project;
const pid = proj.id;
console.log(`测试案件 ${pid}`);

// 初始状态：没有测量
let m = (await api(`/api/projects/${pid}/measurements`)).data;
check('初始状态没有待测项', (m.pending ?? []).length === 0, `pending=${(m.pending ?? []).length}`);
check('初始比例尺为 1', m.scaleToMeters === 1, `scale=${m.scaleToMeters}`);

// ---- ① 建立「需精确测量」的尺寸标注
// 模型里两点相距 2.0（模型单位）；真实应为 4.0 米 → 期望 scaleToMeters ≈ 2.0
const mk = (label, modelDistance, measured = null) =>
  api(`/api/projects/${pid}/annotations`, {
    method: 'POST',
    body: {
      annotation: {
        kind: 'dimension',
        label,
        points: [[0, 0, 0], [modelDistance, 0, 0]],
        measured,
        modelDistance,
        unit: 'meter',
        priority: 'high',
      },
      operator: '现场民警',
    },
  });

const a1 = await mk('房间东西向净宽', 2.0);
check('能创建待测尺寸标注', a1.status === 201 || a1.status === 200, `status=${a1.status}`);
check('新建的尺寸标注是「待测量」状态', a1.data.annotation?.measured === null && a1.data.annotation?.pending === true);
check('模型侧距离被正确记录', Number(a1.data.annotation?.modelDistance) === 2.0, `modelDistance=${a1.data.annotation?.modelDistance}`);

await mk('房间南北向净深', 3.0);
await mk('门洞净宽', 0.9);

m = (await api(`/api/projects/${pid}/measurements`)).data;
check('待实测清单里有 3 项', (m.pending ?? []).length === 3, `pending=${(m.pending ?? []).length}`);
check('待测项带上了模型侧距离与优先级', m.pending[0]?.modelDistance > 0 && m.pending[0]?.priority === 'high');

// 未实测时不能标定
const c0 = await api(`/api/projects/${pid}/calibrate`, { method: 'POST', body: { operator: '系统' } });
check('没有任何实测值时拒绝标定', c0.status === 400, c0.data.error?.slice(0, 24) ?? '');

// ---- ② 民警填写实测值
// 真实尺寸 = 模型尺寸 × 2.0（制造一个已知的比例尺）
const truth = 2.0;
const pending = (await api(`/api/projects/${pid}/measurements`)).data.pending;
const meas = { 0.9: 1.8, 2.0: 4.0, 3.0: 6.0 };
for (const p of pending) {
  const real = meas[p.modelDistance];
  const r = await api(`/api/projects/${pid}/annotations/${p.id}`, {
    method: 'PATCH',
    body: { patch: { measured: real, measuredBy: '民警张三' }, operator: '民警张三' },
  });
  check(`填写实测值 ${p.label}（模型 ${p.modelDistance} → 实测 ${real}）`, r.status === 200, `status=${r.status}`);
}

m = (await api(`/api/projects/${pid}/measurements`)).data;
check('待实测清单已清空', (m.pending ?? []).length === 0, `pending=${(m.pending ?? []).length}`);
check('已实测清单有 3 项', (m.measured ?? []).length === 3);
check('记录下了测量人', m.measured?.[0]?.measuredBy === '民警张三', m.measured?.[0]?.measuredBy ?? '');

// ---- ③ 反算比例尺
const cal = await api(`/api/projects/${pid}/calibrate`, { method: 'POST', body: { operator: '系统' } });
check('标定接口返回成功', cal.status === 200 && cal.data.ok === true, `status=${cal.status}`);
const s = cal.data.scaleToMeters;
check(
  `反算出的比例尺 ≈ ${truth}`,
  Math.abs(s - truth) < 1e-6,
  `scaleToMeters=${s}`
);
check('标定结果逐条残差接近 0', (cal.data.rows ?? []).every((r) => Math.abs(r.error) < 1e-6),
  `RMS=${cal.data.rms?.toExponential?.(2)}`);
check('可信度为 high', cal.data.confidence === 'high', `meanAbsRel=${cal.data.meanAbsRel?.toExponential?.(2)}`);

// 比例尺写回了 project.world
const fresh = (await api(`/api/projects/${pid}`)).data.project;
check('比例尺已写入 project.world.scaleToMeters', Math.abs((fresh.world?.scaleToMeters ?? 0) - truth) < 1e-6,
  `world.scaleToMeters=${fresh.world?.scaleToMeters}`);
check('记录了标定时间与操作人', !!fresh.world?.calibratedAt && fresh.world?.calibratedBy === '系统',
  `${fresh.world?.calibratedBy} @ ${fresh.world?.calibratedAt}`);

// ---- 含噪声的测量：最小二乘应当给出合理折中，并降低可信度
await mk('含噪测量A', 1.0, null);
const all2 = (await api(`/api/projects/${pid}/measurements`)).data.pending;
const noisy = all2[0];
// 故意给一个明显错误的实测值（模型 1.0 却量成 1.5，真值应是 2.0）
// 这样平均相对偏差必然超过 2%，可信度应当降级
await api(`/api/projects/${pid}/annotations/${noisy.id}`, {
  method: 'PATCH',
  body: { patch: { measured: 1.5 }, operator: '民警李四' },
});
const cal2 = await api(`/api/projects/${pid}/calibrate`, { method: 'POST', body: { operator: '系统', apply: false } });
check('含噪声数据仍能标定', cal2.status === 200, `scale=${cal2.data.scaleToMeters?.toFixed?.(4)}`);
const s2 = cal2.data.scaleToMeters;
// 手算最小二乘：Σ(dm*dr)/Σ(dm²)
const expected = (0.9 * 1.8 + 2 * 4 + 3 * 6 + 1 * 1.5) / (0.81 + 4 + 9 + 1);
check('最小二乘结果与手算一致', Math.abs(s2 - expected) < 1e-9, `got ${s2} want ${expected}`);
check('apply=false 时不写回 world', (await api(`/api/projects/${pid}`)).data.project.world?.scaleToMeters === truth);
check(
  '含噪数据的可信度下降',
  cal2.data.confidence !== 'high',
  `confidence=${cal2.data.confidence} meanAbsRel=${cal2.data.meanAbsRel?.toFixed(4)}`
);

// ---- 幂等性：重复标定结果一致
const cal3 = await api(`/api/projects/${pid}/calibrate`, { method: 'POST', body: { operator: '系统', apply: false } });
check('重复标定结果稳定', Math.abs(cal3.data.scaleToMeters - s2) < 1e-12);

// ---- 维度标注的其它字段
const bad = await api(`/api/projects/${pid}/annotations`, {
  method: 'POST',
  body: { annotation: { kind: 'dimension', label: '只有一点', points: [[0, 0, 0]] } },
});
check('只有一个端点时拒绝创建', bad.status >= 400, `${bad.status} ${bad.data.error?.slice(0, 30) ?? ''}`);

const neg = await api(`/api/projects/${pid}/annotations`, {
  method: 'POST',
  body: { annotation: { kind: 'dimension', label: '负值', points: [[0, 0, 0], [1, 0, 0]], measured: -5 } },
});
check('负的实测值被当作「待测」而不是生效', neg.data.annotation?.measured === null, `measured=${neg.data.annotation?.measured}`);

await api(`/api/projects/${pid}`, { method: 'DELETE' });
console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail === 0 ? 0 : 1);
