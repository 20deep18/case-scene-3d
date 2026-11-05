/**
 * 回归测试：标注幂等性（「只导入一个视频却出现两个相机对象」）
 * 用法: node tests/test_annotation_idempotent.mjs
 */
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';

let pass = 0;
let fail = 0;
const check = (name, ok, detail = '') => {
  if (ok) { pass++; console.log(`  [OK]   ${name}${detail ? '  ' + detail : ''}`); }
  else { fail++; console.log(`  [FAIL] ${name}${detail ? '  ' + detail : ''}`); }
};

const proj = await (
  await fetch(`${BASE}/api/projects`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'idempotent-test', operator: 'test' }),
  })
).json();
const pid = proj.project.id;

const cameras = async () => {
  const p = await (await fetch(`${BASE}/api/projects/${pid}`)).json();
  return p.project.annotations.filter((a) => a.kind === 'camera');
};
const post = async (annotation) =>
  (
    await fetch(`${BASE}/api/projects/${pid}/annotations`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ annotation, operator: 'test' }),
    })
  ).json();
const patch = async (id, p) =>
  (
    await fetch(`${BASE}/api/projects/${pid}/annotations/${id}`, {
      method: 'PATCH',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ patch: p, operator: 'test' }),
    })
  ).json();

console.log('标注幂等性回归测试\n');

const media = [{ url: `/media/${pid}/media/demo.mp4`, kind: 'video', name: 'demo.mp4' }];
const first = await post({
  kind: 'camera',
  points: [[0, 1, 0]],
  pose: { source: 'manual', position: [0, 1, 0], target: [0, 1, 1], fovY: 45, aspect: 1.777 },
  media,
  label: 'demo.mp4',
});
check('创建相机后服务端有 1 个相机', (await cameras()).length === 1);

// 模拟前端 updateAnnotation 打 media 补丁
await patch(first.annotation.id, { media });
check('PATCH 后仍是 1 个相机（不新增）', (await cameras()).length === 1);

// 模拟重复提交（网络重试 / 协同回显 / 对话框二次触发）
const again = await post({ ...first.annotation, media, label: 'demo.mp4（重复提交）' });
check('带相同 id 再次 POST 后仍是 1 个相机', (await cameras()).length === 1);
check('重复提交被识别为更新而非新增', again.annotation?.id === first.annotation.id);
check('重复提交未改变标注 id', (await cameras())[0].id === first.annotation.id);
check('重复提交更新了标签', (await cameras())[0].label === 'demo.mp4（重复提交）');
check('media 数组没有重复累积', (await cameras())[0].media.length === 1);

// 不同 id 应当能正常新增
await post({
  kind: 'camera',
  points: [[1, 1, 1]],
  pose: { source: 'manual', position: [1, 1, 1], target: [0, 0, 0], fovY: 45, aspect: 1.777 },
  media: [],
  label: '第二台相机',
});
check('不同 id 能正常新增第 2 个相机', (await cameras()).length === 2);

// 事件日志应记录 upsert
const detail = await (await fetch(`${BASE}/api/projects/${pid}`)).json();
const upserts = (detail.project.events ?? []).filter((e) => e.type === 'annotation.upsert');
check('事件日志记录了 annotation.upsert', upserts.length >= 1, `${upserts.length} 条`);

await fetch(`${BASE}/api/projects/${pid}`, { method: 'DELETE' });
console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail === 0 ? 0 : 1);
