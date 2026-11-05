/**
 * 回归：场景判定的**人工改判**（前端唯一入口背后的接口）。
 *
 * 为什么需要它
 * ------------
 * 几何本身分不开「贴着物体拍」与「以旋转为主拍的房间」——
 * 实测柜子素材的方向包围度 0.2344，而某段房间素材（sequential 匹配、只注册 61/175）
 * 是 0.2270。判不准时不该硬猜，**由拍摄的人指定**最可靠。
 * 后端 `tools/scene_structure.py --scene-mode` 早就有这个能力，本接口是它在前端的入口。
 *
 * 这个接口会**写文件**，所以要守住三件事：
 *   1. 只认 data/models 下的真实模型名（挡掉路径穿越 —— 否则就是一个任意文件读写洞）；
 *   2. mode 只接受 auto / room / object；
 *   3. auto = **真的删掉** sidecar（回到自动判定），而不是留个空壳。
 * 另外要确认改判是**落盘并进索引**的（多人共享），不是只存在某个浏览器里。
 *
 * 用法: node tests/test_scene_mode_override.mjs
 * 退出码 0 通过 / 1 失败
 */
const BASE = process.env.BASE ?? 'http://127.0.0.1:8787';

let pass = 0;
let fail = 0;
const check = (name, ok, detail = '') => {
  if (ok) {
    pass++;
    console.log(`  [OK]   ${name}${detail ? `  ${detail}` : ''}`);
  } else {
    fail++;
    console.log(`  [FAIL] ${name}${detail ? `  ${detail}` : ''}`);
  }
};

const post = async (body) => {
  const res = await fetch(`${BASE}/api/models/scene-mode`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  let data = {};
  try {
    data = await res.json();
  } catch {
    /* 忽略 */
  }
  return { status: res.status, data };
};

const modelEntry = async (key) => {
  const idx = await (await fetch(`${BASE}/models/index.json`)).json();
  return (idx.models ?? []).find((m) => String(m.name).includes(key)) ?? null;
};

const target = await modelEntry('room-real');
check('找到一个带结构骨架的模型作为被测对象', Boolean(target?.structureUrl), target?.name ?? '无');
if (!target?.structureUrl) {
  console.log('\n结果：0 通过 / 1 失败（缺少可测模型）');
  process.exit(1);
}
const file = decodeURIComponent(target.url.replace(/^\/models\//, ''));
console.log(`  被测模型：${target.name}（${file}），结构文件里的自动判定 = ${target.sceneMode}`);

// 先清干净，避免上一次残留影响断言
await post({ model: file, mode: 'auto' });

console.log('\n[1] 输入校验（这个接口会写文件，必须守住边界）');
const bad = await post({ model: file, mode: 'hack' });
check('非法 mode 被拒绝（400）', bad.status === 400, `${bad.status} ${bad.data.error ?? ''}`);
const unknown = await post({ model: 'definitely-not-a-model.splat', mode: 'room' });
check('未知模型被拒绝（400）', unknown.status === 400, unknown.data.error ?? '');
const traversal = await post({ model: '../index.json', mode: 'room' });
check('路径穿越被拒绝（400）', traversal.status === 400, traversal.data.error ?? '');
const traversal2 = await post({ model: '../../server/index.js', mode: 'room' });
check('带目录的路径同样被拒绝（400）', traversal2.status === 400, traversal2.data.error ?? '');

console.log('\n[2] 改判要落盘、要进索引（多人共享，不是只存在某个浏览器里）');
const set = await post({ model: file, mode: 'object', operator: '回归测试' });
check('设为 object 成功（200）', set.status === 200 && set.data.sceneModeOverride?.mode === 'object',
  JSON.stringify(set.data.sceneModeOverride ?? set.data));
check('记录里带上了操作者（可追溯）', set.data.sceneModeOverride?.operator === '回归测试',
  set.data.sceneModeOverride?.operator ?? '');
const afterSet = await modelEntry('room-real');
check('模型索引带出了改判结果', afterSet?.sceneModeOverride?.mode === 'object',
  JSON.stringify(afterSet?.sceneModeOverride ?? null));
check('索引里的**自动判定**没有被改判污染（两者分开存）',
  afterSet?.sceneMode === target.sceneMode,
  `改判前 ${target.sceneMode} → 改判后 ${afterSet?.sceneMode}`);

console.log('\n[3] 改判为 room 同样生效');
const setRoom = await post({ model: file, mode: 'room', operator: '回归测试' });
check('设为 room 成功（200）', setRoom.status === 200 && setRoom.data.sceneModeOverride?.mode === 'room',
  setRoom.data.sceneModeOverride?.mode ?? '');
const afterRoom = await modelEntry('room-real');
check('索引随之变为 room', afterRoom?.sceneModeOverride?.mode === 'room',
  afterRoom?.sceneModeOverride?.mode ?? '');

console.log('\n[4] auto = 清除改判，回到自动判定');
const clear = await post({ model: file, mode: 'auto' });
check('清除成功（200 且返回 null）', clear.status === 200 && clear.data.sceneModeOverride === null,
  JSON.stringify(clear.data));
const afterClear = await modelEntry('room-real');
check('索引里不再有改判', afterClear?.sceneModeOverride === null,
  JSON.stringify(afterClear?.sceneModeOverride ?? null));
check('自动判定仍然是它本来的值', afterClear?.sceneMode === target.sceneMode,
  `${afterClear?.sceneMode}`);

console.log('\n[5] GET 查询与 POST 状态一致');
const got = await (await fetch(`${BASE}/api/models/scene-mode?model=${encodeURIComponent(file)}`)).json();
check('GET 查询可用且与索引一致', got.ok === true && got.sceneModeOverride === null, JSON.stringify(got));
const gotBad = await fetch(`${BASE}/api/models/scene-mode?model=../x`);
check('GET 也挡路径穿越（400）', gotBad.status === 400, String(gotBad.status));

console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
process.exit(fail === 0 ? 0 : 1);
