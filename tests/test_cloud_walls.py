"""cloud_walls.py 的回归测试。

锁住两件事：
1. **准确时真准**：合成房（真值 4.6×3.6×2.8）误差 <5%
2. **不准时如实说**：部分视角（宿舍）必须标 `estimated=True`，不能装作可信

用法：<python> tests/test_cloud_walls.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
TOOL = ROOT / "tools" / "cloud_walls.py"

passed = 0
failed = 0


def check(name: str, ok: bool, extra: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}")


def run_tool(cloud: Path, out: Path, poses: Path | None = None) -> dict:
    cmd = [PY, str(TOOL), "--cloud", str(cloud), "--out", str(out)]
    if poses is not None:
        cmd += ["--poses", str(poses)]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    if p.returncode != 0:
        print(p.stdout[-800:], p.stderr[-800:])
    return json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}


def main() -> int:
    print("cloud_walls 回归测试")
    tmp = Path(tempfile.mkdtemp(prefix="cw-"))

    # ---- 1) 合成房：真值 4.6 × 3.6 × 2.8 ----
    demo = ROOT / "output" / "furniture" / "demo-room-noscale.ply"
    if demo.exists():
        st = run_tool(demo, tmp / "demo.json")
        size = st.get("room", {}).get("size") or []
        check("合成房有房间盒", len(size) == 3, str(size))
        if len(size) == 3:
            got = sorted(size)
            truth = [2.8, 3.6, 4.6]
            err = [abs(g - t) / t * 100 for g, t in zip(got, truth)]
            check("合成房尺寸误差均 <5%", max(err) < 5.0,
                  f"got={[round(v, 2) for v in got]} err={[round(e, 1) for e in err]}")
            check("合成房高度不含竖轴错误（2.8 而非 4.6）", abs(got[0] - 2.8) / 2.8 < 0.05,
                  f"最低维 {got[0]:.2f}")
        check("合成房标为可信", st.get("sane") is True and st.get("estimated") is False,
              f"sane={st.get('sane')} estimated={st.get('estimated')}")
    else:
        print("  [SKIP] 合成房点云不存在")

    # ---- 2) 宿舍（部分视角）：必须标 estimated ----
    dorm = ROOT / "output" / "dorm" / "auto" / "scene.ply"
    dorm_poses = ROOT / "output" / "dorm" / "auto" / "scene.poses.json"
    if dorm.exists() and dorm_poses.exists():
        st = run_tool(dorm, tmp / "dorm.json", poses=dorm_poses)
        check("宿舍用相机朝向定重力", st.get("evidence", {}).get("up", {}).get("source") == "camera-up",
              str(st.get("evidence", {}).get("up")))
        check("宿舍（部分视角）标为 estimated 且不可信",
              st.get("estimated") is True and st.get("sane") is False,
              f"estimated={st.get('estimated')} sane={st.get('sane')}")
        check("宿舍给出覆盖方向的说明", "方向" in str(st.get("evidence", {}).get("wallNote")),
              str(st.get("evidence", {}).get("wallNote")))
    else:
        print("  [SKIP] 宿舍点云不存在")

    print(f"\n结果：{passed} 通过 / {failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
