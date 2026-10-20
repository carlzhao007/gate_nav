"""
Step 0 仿真环境：小车按顺序过四道门（Gymnasium 版）
====================================================
场地：10m x 10m 平面，边界 [0,10]x[0,10]
小车：差速底盘运动学模型（线速度 v + 角速度 w），一阶电机惯性
门：4 道，每道是垂直于穿越方向的窄线段（宽 1.2m），
    车中心落在门线段带宽内且航向与穿越方向夹角 < 60° 判为“过门”

obs（20 维，全部归一化到 ~[-1,1]）：
    [0:2]   车位置 (x/5-1, y/5-1)
    [2:4]   航向 (cos yaw, sin yaw)
    [4:6]   速度 (v/V_MAX, w/W_MAX)
    [6:8]   目标门相对坐标 (dx/5, dy/5)
    [8:10]  目标门穿越方向差
    [10]    目标门编号 (target+1)/num_gates - 0.5
    [11:15] 下一门相对坐标 + 穿越方向差（无则 0）
    [15:19] 其余两道门相对坐标 (dx/5, dy/5) x2

act（2 维连续）：
    a[0] → 线速度指令 v = a[0]*V_MAX
    a[1] → 角速度指令 w = a[1]*W_MAX

奖励（密集引导 + 稀疏里程碑）：
    -0.01                       时间惩罚
    +2.0*(d_prev - d_now)       进度奖励（朝目标门靠近）
    +0.5*cos(航向对准角)         对准奖励
    +10.0                       按顺序过正确门，target += 1
    +50.0                       四门全过终奖（结束）
    -5.0                        出界撞墙（结束）
    -5.0                        过错门（不结束，可继续）

episode：最长 300 步（dt=0.1s = 30 秒模拟时间）
课程：mode='fixed' 基准布局小扰动 / 'random' 全场地随机
"""

import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------- 物理参数 ----------------
FIELD = 10.0          # 场地边长 (m)
V_MAX = 1.5           # 最大线速度 (m/s)
W_MAX = math.pi       # 最大角速度 (rad/s)
DT = 0.1              # 控制周期 (s)
GATE_WIDTH = 1.2      # 门宽 (m)
GATE_BAND = 0.3       # 过门判定带宽（穿越方向厚度）(m)
CAR_RAD = 0.15        # 车半径 (m)
MAX_STEPS = 400
ALIGN_TOL = math.radians(60)   # 过门航向容差

# 奖励
R_TIME = -0.02
R_PROG_K = 2.0
R_ALIGN_K = 0.5
R_GATE = 10.0
R_ALL = 50.0
R_CRASH = -5.0
R_WRONG = -5.0

# 基准门布局（课程阶段用）：位置 + 门线朝向（穿越方向 = 朝向 + 90°）
BASE_LAYOUT = [
    (5.0, 2.0, 0.0),    # 门1：门线沿 x 轴，向北穿越
    (8.0, 5.0, math.pi / 2),  # 门2：门线沿 y 轴，向东穿越
    (5.0, 8.0, 0.0),    # 门3：向北穿越
    (2.0, 5.0, math.pi / 2),  # 门4：向东穿越
]


class GateNavEnv(gym.Env):
    """小车按 1→2→3→4 顺序过门。"""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, num_gates=4, mode="random", jitter=2.0, seed=None):
        super().__init__()
        self.num_gates = int(num_gates)
        self.mode = mode            # 'fixed' | 'random'
        self.jitter = jitter        # fixed 模式下基准点扰动半径
        self._rng = random.Random(seed)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 运行时状态
        self.pos = np.zeros(2)
        self.prev_pos = np.zeros(2)
        self.yaw = 0.0
        self.v = 0.0
        self.w = 0.0
        self.gates = []            # [(gx, gy, heading, normal), ...]
        self.target = 0
        self.steps = 0
        self.total_reward = 0.0
        self.passed_in_order = 0
        self.last_event = "reset"

    # ---------------- 布局生成 ----------------
    def _gen_layout(self):
        if self.mode == "random":
            gates = []
            for _ in range(self.num_gates):
                gx = self._rng.uniform(1.5, FIELD - 1.5)
                gy = self._rng.uniform(1.5, FIELD - 1.5)
                heading = self._rng.uniform(0.0, math.pi)
                gates.append([gx, gy, heading])
        else:
            gates = []
            for i in range(self.num_gates):
                bx, by, bh = BASE_LAYOUT[i]
                ang = self._rng.uniform(0.0, 2.0 * math.pi)
                r = self._rng.uniform(0.0, self.jitter)
                gx = min(max(bx + r * math.cos(ang), 1.5), FIELD - 1.5)
                gy = min(max(by + r * math.sin(ang), 1.5), FIELD - 1.5)
                heading = bh + self._rng.uniform(-0.3, 0.3)
                gates.append([gx, gy, heading])
        gates = self._separate(gates)
        return [(gx, gy, h, h + math.pi / 2.0) for gx, gy, h in gates]

    @staticmethod
    def _separate(gates, min_dist=2.0):
        """推开间距过近的门，避免布局死局。每轮迭代后夹回边界。"""
        gs = [list(g) for g in gates]
        for _ in range(80):
            for g in gs:
                g[0] = min(max(g[0], 1.5), FIELD - 1.5)
                g[1] = min(max(g[1], 1.5), FIELD - 1.5)
            moved = False
            for i in range(len(gs)):
                for j in range(i + 1, len(gs)):
                    dx, dy = gs[j][0] - gs[i][0], gs[j][1] - gs[i][1]
                    dist = math.hypot(dx, dy)
                    if dist < min_dist:
                        push = (min_dist - dist) / 2.0 + 0.01
                        if dist < 1e-6:
                            dx, dy, dist = 0.01, 0.0, 0.01
                        nx_, ny_ = dx / dist, dy / dist
                        gs[i][0] -= nx_ * push
                        gs[i][1] -= ny_ * push
                        gs[j][0] += nx_ * push
                        gs[j][1] += ny_ * push
                        moved = True
            if not moved:
                break
        for g in gs:
            g[0] = min(max(g[0], 1.5), FIELD - 1.5)
            g[1] = min(max(g[1], 1.5), FIELD - 1.5)
        return gs

    # ---------------- 过门几何 ----------------
    def _approach_points(self, g):
        """门两侧沿穿越方向各 1.5m 的逼近点；贴墙时取在场内的那侧。"""
        gx, gy, _, normal = g
        b = np.array([gx - math.cos(normal) * 1.5, gy - math.sin(normal) * 1.5])
        f = np.array([gx + math.cos(normal) * 1.5, 1.0])  # 占位
        f = np.array([gx + math.cos(normal) * 1.5, gy + math.sin(normal) * 1.5])
        pad = 0.4
        def clamp(p):
            return np.array([min(max(p[0], pad), FIELD - pad),
                             min(max(p[1], pad), FIELD - pad)])
        return clamp(b), clamp(f)

    def _on_gate(self, g):
        """车中心是否落在门线段判定带内。"""
        gx, gy, heading, _ = g
        dx, dy = self.pos[0] - gx, self.pos[1] - gy
        along = dx * math.cos(heading) + dy * math.sin(heading)      # 沿门线方向
        cross = -dx * math.sin(heading) + dy * math.cos(heading)     # 沿穿越方向
        return abs(along) <= GATE_WIDTH / 2.0 and abs(cross) <= GATE_BAND

    def _aligned(self, g):
        """车航向与门穿越方向夹角是否在容差内。"""
        _, _, _, normal = g
        err = math.atan2(math.sin(self.yaw - normal), math.cos(self.yaw - normal))
        return abs(err) < ALIGN_TOL

    # ---------------- Gym API ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        # 起点固定左下角，布局重抽直到与起点保持距离
        start = np.array([0.8, 0.8])
        for _ in range(50):
            self.gates = self._gen_layout()
            if all(math.hypot(g[0] - start[0], g[1] - start[1]) > 2.0 for g in self.gates):
                break
        self.pos = start.copy()
        self.prev_pos = start.copy()
        self.yaw = math.pi / 4.0     # 朝场地内
        self.v = 0.0
        self.w = 0.0
        self.target = 0
        self.steps = 0
        self.total_reward = 0.0
        self.passed_in_order = 0
        self.last_event = "reset"
        return self._obs(), {}

    def _obs(self):
        x, y = self.pos
        obs = np.zeros(20, dtype=np.float32)
        obs[0] = x / (FIELD / 2.0) - 1.0
        obs[1] = y / (FIELD / 2.0) - 1.0
        obs[2] = math.cos(self.yaw)
        obs[3] = math.sin(self.yaw)
        obs[4] = self.v / V_MAX
        obs[5] = self.w / W_MAX

        if self.target < len(self.gates):
            g = self.gates[self.target]
            dx, dy = g[0] - x, g[1] - y
            obs[6] = dx / (FIELD / 2.0)
            obs[7] = dy / (FIELD / 2.0)
            dth = g[3] - self.yaw
            obs[8] = math.cos(dth)
            obs[9] = math.sin(dth)
            obs[10] = (self.target + 1) / self.num_gates - 0.5
            if self.target + 1 < len(self.gates):
                gn = self.gates[self.target + 1]
                dxn, dyn = gn[0] - x, gn[1] - y
                obs[11] = dxn / (FIELD / 2.0)
                obs[12] = dyn / (FIELD / 2.0)
                dthn = gn[3] - self.yaw
                obs[13] = math.cos(dthn)
                obs[14] = math.sin(dthn)
        # 其余门概览（目标/下一门之外的两道）
        rest = self.gates[self.target + 2:] + self.gates[:max(self.target - 1, 0)]
        rest = rest[:2]
        for k, g in enumerate(rest):
            obs[15 + 2 * k] = (g[0] - x) / (FIELD / 2.0)
            obs[16 + 2 * k] = (g[1] - y) / (FIELD / 2.0)
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def step(self, action):
        self.prev_pos = self.pos.copy()
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # 一阶电机惯性
        self.v += 0.6 * (a[0] * V_MAX - self.v)
        self.w += 0.6 * (a[1] * W_MAX - self.w)
        # 差速运动学积分
        self.yaw = math.atan2(math.sin(self.yaw + self.w * DT), math.cos(self.yaw + self.w * DT))
        self.pos = self.pos + np.array([self.v * math.cos(self.yaw), self.v * math.sin(self.yaw)]) * DT
        self.steps += 1

        reward = R_TIME
        terminated = False
        truncated = False
        self.last_event = "normal"

        # ---- 出界 ----
        m = CAR_RAD
        if not (m <= self.pos[0] <= FIELD - m and m <= self.pos[1] <= FIELD - m):
            reward += R_CRASH
            terminated = True
            self.last_event = "crash"
            self.total_reward += reward
            return self._obs(), float(reward), terminated, truncated, {"event": self.last_event}

        # ---- 进度 + 对准奖励 ----
        if self.target < len(self.gates):
            g = self.gates[self.target]
            d_now = math.hypot(g[0] - self.pos[0], g[1] - self.pos[1])
            d_prev = math.hypot(g[0] - self.prev_pos[0], g[1] - self.prev_pos[1])
            prog = d_prev - d_now
            reward += R_PROG_K * prog
            # 对准奖励：速度门控 + 只在真正接近时给（防绕门盘旋刷分）
            err = math.atan2(math.sin(self.yaw - g[3]), math.cos(self.yaw - g[3]))
            reward += R_ALIGN_K * math.cos(err) * (abs(self.v) / V_MAX) * max(prog, 0.0) * 20.0

            # ---- 过正确门 ----
            if self._on_gate(g) and self._aligned(g):
                reward += R_GATE
                self.target += 1
                self.passed_in_order += 1
                self.last_event = "pass"
                if self.target >= self.num_gates:
                    reward += R_ALL
                    terminated = True
                    self.last_event = "all-passed"
                    self.total_reward += reward
                    return self._obs(), float(reward), terminated, truncated, {"event": self.last_event}

        # ---- 过错门（非当前目标的其他门） ----
        if not terminated and self.last_event != "pass":
            for idx in range(self.num_gates):
                if idx == self.target or idx == self.target - 1:
                    continue
                gg = self.gates[idx]
                if self._on_gate(gg) and self._aligned(gg):
                    # 反向穿越也算过错门
                    reward += R_WRONG
                    self.last_event = "wrong-gate"
                    break

        if self.steps >= MAX_STEPS:
            truncated = True

        self.total_reward += reward
        return self._obs(), float(reward), terminated, truncated, {"event": self.last_event}

    def render(self):
        pass

    def close(self):
        pass


# ---------------- 自检 ----------------
if __name__ == "__main__":
    env = GateNavEnv(num_gates=4, mode="random", seed=0)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs), "obs 不在空间内"
    print("obs shape:", obs.shape)
    print("gates:")
    for i, g in enumerate(env.gates):
        print(f"  G{i+1}: pos=({g[0]:.2f},{g[1]:.2f}) heading={g[2]:.2f} normal={g[3]:.2f}")

    # 随机动作跑一回合
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        done = term or trunc
    print(f"random policy: reward={total:.2f} steps={env.steps} event={env.last_event} "
          f"passed={env.passed_in_order}/{env.num_gates}")
