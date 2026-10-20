"""
PPO 训练 + 分析图：小车按顺序过四道门
=====================================
课程：阶段A（基准布局小扰动，60% episodes）→ 阶段B（全随机布局，40%）
输出：
    gate_learning_curve.png   学习曲线（回合奖励 / 过门数 / 成功率）
    gate_paths.png            训练前 vs 训练后 轨迹对比
    gate_eval_stats.png       20 个新布局评估统计
    ppo_gate_model.zip        训练好的策略
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from gate_env import GateNavEnv, FIELD

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))

# 中文可显示的门颜色
GATE_COLORS = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad"]


# ============================================================
# 课程回调：按训练进度切换布局模式
# ============================================================
class CurriculumCallback(BaseCallback):
    def __init__(self, switch_frac=0.6, verbose=0):
        super().__init__(verbose)
        self.switch_frac = switch_frac

    def _on_step(self):
        frac = self.num_timesteps / self.model._total_timesteps
        # 用环境集合的公共属性切换模式
        for e in self.training_env.envs:
            inner = getattr(e, "env", e)  # Monitor 包装
            inner = getattr(inner, "unwrapped", inner)
            if frac < self.switch_frac:
                inner.mode = "fixed"
            else:
                inner.mode = "random"
        return True


# ============================================================
# 记录回调：每回合记录奖励与过门数
# ============================================================
class EpisodeLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.passed = []
        self.successes = []
        self._ep_ret = 0.0

    def _on_step(self):
        for i in range(self.training_env.num_envs):
            info = self.locals["infos"][i]
            if "episode" in info:                      # Monitor 包装输出
                self.rewards.append(info["episode"]["r"])
            ep_done = self.locals["dones"][i]
            if ep_done:
                inner = self.training_env.envs[i].unwrapped
                self.passed.append(inner.passed_in_order)
                self.successes.append(1 if inner.last_event == "all-passed" else 0)
        return True


# ============================================================
# 轨迹可视化
# ============================================================
def plot_traj(ax, env, traj, gates, title, target_seq=None):
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.3, FIELD + 0.3)
    ax.set_ylim(-0.3, FIELD + 0.3)
    ax.set_xticks(range(0, 11, 2))
    ax.set_yticks(range(0, 11, 2))
    ax.set_aspect("equal")
    # 场地边框
    ax.add_patch(plt.Rectangle((0, 0), FIELD, FIELD, fill=False, ec="k", lw=1.5))
    # 门
    for i, g in enumerate(gates):
        gx, gy, heading, _ = g
        ex = math.cos(heading) * 0.6
        ey = math.sin(heading) * 0.6
        ax.plot([gx - ex, gx + ex], [gy - ey, gy + ey], color=GATE_COLORS[i], lw=4,
                label=f"门{i+1}", alpha=0.85, solid_capstyle="round")
    # 轨迹
    t = np.array(traj)
    ax.plot(t[:, 0], t[:, 1], "-", color="#34495e", lw=1.4, alpha=0.8)
    ax.plot(t[0, 0], t[0, 1], "o", color="#2c3e50", ms=7, label="起点")
    ax.plot(t[-1, 0], t[-1, 1], "X", color="#c0392b", ms=9, label="终点")
    # 车过门顺序标记
    if target_seq:
        for k, (px, py) in enumerate(target_seq):
            ax.annotate(f"{k+1}", (px, py), textcoords="offset points", xytext=(6, 6),
                        color=GATE_COLORS[k], fontsize=10, fontweight="bold")


def rollout(env, model=None, deterministic=True, seed=None):
    """跑一回合，返回 (轨迹, 总奖励, 过门数, 事件)。"""
    obs, _ = env.reset(seed=seed)
    traj = [env.pos.copy()]
    total, done = 0.0, False
    while not done:
        if model is None:
            a = env.action_space.sample()
        else:
            a, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(a)
        total += r
        traj.append(env.pos.copy())
        done = term or trunc
    return traj, total, env.passed_in_order, env.last_event


def draw_gate_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    # 去重
    seen, h, l = set(), [], []
    for hd, lb in zip(handles, labels):
        if lb not in seen:
            seen.add(lb)
            h.append(hd)
            l.append(lb)
    ax.legend(h, l, loc="upper right", fontsize=7, ncol=2)


if __name__ == "__main__":
    N_ENVS = 8
    TOTAL = 1_500_000
    EVAL_ONLY = "--eval-only" in os.sys.argv

    # ---------------- 阶段A环境（fixed）----------------
    def make_env(mode):
        return Monitor(GateNavEnv(num_gates=4, mode=mode))

    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([lambda: make_env("fixed")] * N_ENVS)

    model = PPO(
        "MlpPolicy", venv,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0,
        seed=42,
    )

    logger = EpisodeLogger()
    if EVAL_ONLY:
        model = PPO.load(os.path.join(HERE, "ppo_gate_model"))
        print("eval-only 模式：加载已训练模型")
        data = np.load(os.path.join(HERE, "train_log.npz"))
        logger.rewards = list(data["rewards"])
        logger.passed = list(data["passed"])
        logger.successes = list(data["successes"])
    else:
        print(f"开始 PPO 训练：{TOTAL:,} 步（8 环境并行）...")
        model.learn(total_timesteps=TOTAL, callback=[CurriculumCallback(switch_frac=0.6), logger])
        model.save(os.path.join(HERE, "ppo_gate_model"))
        np.savez(os.path.join(HERE, "train_log.npz"),
                 rewards=np.array(logger.rewards),
                 passed=np.array(logger.passed),
                 successes=np.array(logger.successes))
        print("训练完成，模型已保存。")

    # ---------------- 学习曲线 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    ep_r = np.array(logger.rewards, dtype=float)
    ep_p = np.array(logger.passed, dtype=float)
    ep_s = np.array(logger.successes, dtype=float)

    def smooth(x, k=25):
        if len(x) < k:
            return x
        return np.convolve(x, np.ones(k) / k, mode="valid")

    def smooth_x(x, k=25):
        """与 smooth(y) 等长的 x 轴（valid 模式从第 k-1 个点开始）。"""
        return np.arange(k - 1, len(x))

    ax = axes[0]
    ax.plot(ep_r, color="#bdc3c7", lw=0.6, label="回合奖励")
    if len(ep_r) >= 25:
        ax.plot(smooth_x(ep_r), smooth(ep_r),
                color="#c0392b", lw=2, label="滑动平均(25)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Episode"); ax.set_ylabel("回报")
    ax.set_title("回合回报（课程：前60%小扰动→后40%全随机）")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ep_p, color="#bdc3c7", lw=0.6)
    if len(ep_p) >= 25:
        ax.plot(smooth(ep_p), color="#2980b9", lw=2)
    ax.axhline(4, color="#27ae60", ls="--", lw=1, label="满分 4 门")
    ax.set_yticks(range(0, 5))
    ax.set_xlabel("Episode"); ax.set_ylabel("过门数")
    ax.set_title("按顺序过门数")
    ax.legend(fontsize=8)

    ax = axes[2]
    if len(ep_s) >= 25:
        ax.plot(smooth(ep_s, 25), color="#8e44ad", lw=2)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Episode"); ax.set_ylabel("成功率")
    ax.set_title("四门全过成功率（滑动平均）")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "gate_learning_curve.png"), dpi=130)
    print("已保存 gate_learning_curve.png")
    plt.close(fig)

    # ---------------- 训练前 vs 后 轨迹 ----------------
    rng_seed = 7
    env_eval = GateNavEnv(num_gates=4, mode="random", seed=rng_seed)
    env_eval.reset(seed=rng_seed)

    # 未训练模型（同结构）
    untrained = PPO("MlpPolicy", GateNavEnv(num_gates=4),
                    policy_kwargs=dict(net_arch=[128, 128]), verbose=0, seed=1)
    traj_before, r_b, p_b, ev_b = rollout(env_eval, model=untrained, seed=rng_seed)
    # 注意：untrained 在同一布局上 rollout，需重建同 seed 布局
    env_eval.reset(seed=rng_seed)
    traj_after, r_a, p_a, ev_a = rollout(env_eval, model=model, seed=rng_seed)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    plot_traj(axes[0], env_eval, traj_before, env_eval.gates,
              f"训练前（随机策略）：回报 {r_b:.1f}｜过门 {p_b}/4｜{ev_b}")
    draw_gate_legend(axes[0])
    plot_traj(axes[1], env_eval, traj_after, env_eval.gates,
              f"训练后（PPO 策略）：回报 {r_a:.1f}｜过门 {p_a}/4｜{ev_a}")
    draw_gate_legend(axes[1])
    fig.suptitle("训练前 vs 训练后 · 同一随机门布局（seed=7）", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "gate_paths.png"), dpi=130)
    print("已保存 gate_paths.png")
    plt.close(fig)

    # ---------------- 50 个新布局评估 ----------------
    n_eval, results = 50, []
    for i in range(n_eval):
        env_ev = GateNavEnv(num_gates=4, mode="random", seed=1000 + i)
        traj, r, p, ev = rollout(env_ev, model=model, seed=1000 + i)
        results.append((r, p, ev, traj))

    rs = [x[0] for x in results]
    ps = [x[1] for x in results]
    succ = sum(1 for x in results if x[2] == "all-passed")
    print(f"\n新布局评估（{n_eval} 个）：成功率 {succ}/{n_eval}，"
          f"平均过门 {np.mean(ps):.2f}/4，平均回报 {np.mean(rs):.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 2.2]})
    ax = axes[0]
    bars = ax.bar(["0门", "1门", "2门", "3门", "4门(成功)"],
                  [ps.count(k) for k in range(5)], color=["#95a5a6", "#c0392b", "#e67e22", "#2980b9", "#27ae60"])
    ax.set_title(f"新布局过门分布（n={n_eval}）", fontsize=11)
    ax.set_ylabel("回合数")
    for b in bars:
        ax.annotate(f"{int(b.get_height())}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=10)

    ax = axes[1]
    ax.scatter(range(n_eval), rs, c=["#27ae60" if x[2] == "all-passed" else "#e74c3c" for x in results],
               s=40, zorder=3)
    ax.axhline(np.mean(rs), color="#2980b9", ls="--", lw=1.2,
               label=f"平均回报 {np.mean(rs):.1f}")
    ax.set_xlabel("评估布局编号（seed=1000..1049）")
    ax.set_ylabel("回合回报")
    ax.set_title(f"新布局回报 · 成功率 {succ}/{n_eval}（绿=四门全过）", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "gate_eval_stats.png"), dpi=130)
    print("已保存 gate_eval_stats.png")
    plt.close(fig)

    print("\n全部完成。")
