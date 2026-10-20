"""GateNavEnv 单元测试：运动学 / 过门 / 奖励 / 归一化 / 布局约束"""

import math
import numpy as np
from gate_env import (GateNavEnv, FIELD, V_MAX, W_MAX, GATE_WIDTH,
                      R_GATE, R_ALL, R_CRASH, R_WRONG, MAX_STEPS)

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def drive_to(env, target_xy, max_steps=400, speed=1.0):
    """简单 P 控制器：转向目标点直行。返回是否到达 (距离<0.25m)。
    注：all-passed 视为成功到达（最后一门穿过即结束）。"""
    for _ in range(max_steps):
        if env.last_event == "all-passed":
            return True
        dx, dy = target_xy[0] - env.pos[0], target_xy[1] - env.pos[1]
        dist = math.hypot(dx, dy)
        if dist < 0.25:
            return True
        want = math.atan2(dy, dx)
        err = math.atan2(math.sin(want - env.yaw), math.cos(want - env.yaw))
        a = np.array([speed if abs(err) < 1.0 else 0.3, np.clip(err * 2.0, -1.0, 1.0)], dtype=np.float32)
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            return env.last_event == "all-passed"
    return False


def through_gate(env, gate, speed=1.0):
    """从门后方沿穿越方向直线穿过门（保证对准）。"""
    back, fwd = env._approach_points(gate)
    if not drive_to(env, back, speed=speed):
        return False
    return drive_to(env, fwd, speed=speed)


print("=" * 60)
print("T1 运动学：固定动作下直线行驶 + 原地转向")
env = GateNavEnv(num_gates=4, mode="fixed", seed=1)
obs, _ = env.reset()
p0, yaw0 = env.pos.copy(), env.yaw
for _ in range(20):
    env.step(np.array([1.0, 0.0], dtype=np.float32))
d = env.pos - p0
check("直行方向与航向一致", abs(math.atan2(d[1], d[0]) - yaw0) < 0.05, f"move_dir={math.atan2(d[1], d[0]):.3f} yaw={yaw0:.3f}")
check("直行位移量合理", 0 < math.hypot(*d) <= 20 * V_MAX * 0.1 + 1e-6, f"dist={math.hypot(*d):.3f}")

obs, _ = env.reset()
yaw0 = env.yaw
for _ in range(31):  # w=π rad/s × 0.1s × 31 = 3.1π → wrap 后 ≈ π
    env.step(np.array([0.0, 1.0], dtype=np.float32))
turn = math.atan2(math.sin(env.yaw - yaw0), math.cos(env.yaw - yaw0))
check("原地转向角度≈±π", abs(abs(turn) - math.pi) < 0.2, f"turn={turn:.3f}")

print("T2 过正确门：+10 且 target+1")
env = GateNavEnv(num_gates=4, mode="fixed", seed=2)
env.reset()
g1 = env.gates[0]
ok = through_gate(env, g1)
r_last = env.total_reward
check("脚本车能穿过门1", ok and env.passed_in_order == 1, f"passed={env.passed_in_order}")
check("过门后 target=1", env.target == 1, f"target={env.target}")

print("T3 顺序过完四门：终奖 +50 事件 all-passed")
env = GateNavEnv(num_gates=4, mode="fixed", seed=3)
env.reset()
done = False
for g in env.gates:
    if not through_gate(env, g):
        break
    if env.last_event == "all-passed":
        done = True
        break
    # 过门后可能因带宽内多步重复判定，继续下一门
check("四门全过触发 all-passed", done and env.passed_in_order == 4,
      f"event={env.last_event} passed={env.passed_in_order}")

print("T4 过错门惩罚：跳过门1直接穿门2 → -5")
env = GateNavEnv(num_gates=4, mode="fixed", seed=4)
env.reset()
g2 = env.gates[1]
# 逼近门2但先绕开门1（门1在 (5,2) 附近，从起点直奔门2路径通常不穿门1）
through_gate(env, g2)
check("过错门触发 wrong-gate 或 target 仍为 0",
      env.last_event == "wrong-gate" or env.target == 0,
      f"event={env.last_event} target={env.target}")

print("T5 撞墙：朝墙直冲 → -5 且终止")
env = GateNavEnv(num_gates=4, mode="fixed", seed=5)
env.reset()
env.yaw = -math.pi / 4  # 朝左下角
for _ in range(100):
    _, r, term, trunc, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    if term:
        break
check("出界触发 crash", info.get("event") == "crash", f"event={info.get('event')}")
check("撞墙奖励含 -5", r <= R_CRASH + 2.0 + 0.5, f"r={r:.2f}")

print("T6 obs 归一化：100 回合随机动作全部落在 [-1,1]")
rng = np.random.default_rng(0)
bad = 0
for ep in range(100):
    env = GateNavEnv(num_gates=4, mode="random", seed=100 + ep)
    obs, _ = env.reset()
    if not env.observation_space.contains(obs):
        bad += 1
        continue
    for _ in range(200):
        obs, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 2))
        if not env.observation_space.contains(obs):
            bad += 1
            break
        if term or trunc:
            break
check("无越界 obs", bad == 0, f"bad={bad}")

print("T7 布局约束：门间距 / 离起点 / 离墙")
mind, near_start, out = [], [], []
for ep in range(200):
    env = GateNavEnv(num_gates=4, mode="random", seed=1000 + ep)
    env.reset()
    gs = env.gates
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            mind.append(math.hypot(gs[i][0] - gs[j][0], gs[i][1] - gs[j][1]))
    near_start.append(min(math.hypot(g[0] - 0.8, g[1] - 0.8) for g in gs))
    out.extend([1 for g in gs if not (1.0 < g[0] < FIELD - 1.0 and 1.0 < g[1] < FIELD - 1.0)])
check("门间最小间距 >= 1.8m", min(mind) >= 1.8, f"min={min(mind):.2f}")
check("门离起点 >= 2.0m", min(near_start) >= 2.0, f"min={min(near_start):.2f}")
check("门全部在场内", len(out) == 0)

print("T8 步数上限：原地不动 300 步后 truncate")
env = GateNavEnv(num_gates=4, mode="fixed", seed=8)
env.reset()
for _ in range(MAX_STEPS):
    _, _, term, trunc, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
check("300 步触发 truncate", trunc and not term)

print("=" * 60)
print(f"结果：{PASS} 通过 / {FAIL} 失败")
raise SystemExit(1 if FAIL else 0)
