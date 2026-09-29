"""Record a bank of real walking states for the PoliteBow walk→bow hand-off.

Runs a walking ONNX policy (e.g. Pollen's ``alpha_walking.onnx`` from
``pollen-robotics/microduck-policies``) inside the PoliteBow training env, so
the MJCF, BAM actuators and qpos layout are exactly the ones the bow trains
on, with random velocity commands. Snapshots (qpos, qvel, last action) are
taken at random moments of the gait from envs that are walking upright, and
saved for ``bow_mdp.reset_from_gait_bank``.

    uv run python scripts/make_gait_bank.py --onnx policies/alpha_walking.onnx
    uv run python scripts/make_gait_bank.py --onnx policies/alpha_walking.onnx \\
        --num-envs 1024 --samples 20000 --out data/polite_bow_gait_bank.pt

It prints a sanity report (falls, speed tracking, time with a foot in the air):
if the walker is not actually walking in this env, the bank is worthless, so
read it before training.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-PoliteBow-Flat-MicroDuck"
# The official walking task's command envelope (microduck_velocity_env_cfg).
WALK_RANGES = dict(lin_vel_x=(-0.4, 0.4), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0))


class BatchedOnnxPolicy:
    """Run a [1, 61] -> [1, 14] exported policy on a whole batch at once.

    Exported policies declare a fixed batch of 1. Their graphs are plain MLPs
    (normaliser + Gemm/MatMul + activations), so relabelling the batch
    dimension as symbolic is enough; if the graph disagrees, onnxruntime
    raises and we fall back to one row at a time.
    """

    def __init__(self, path: str):
        model = onnx.load(path)
        for vi in list(model.graph.input) + list(model.graph.output):
            dims = vi.type.tensor_type.shape.dim
            if len(dims) >= 1:
                dims[0].ClearField("dim_value")
                dims[0].dim_param = "batch"
        self.sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.obs_dim = self.sess.get_inputs()[0].shape[-1]
        self.batched = True

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32)
        if self.batched:
            try:
                return self.sess.run(None, {self.in_name: obs})[0]
            except Exception as exc:  # noqa: BLE001
                print(f"[bank] batched ONNX inference failed ({exc}); falling back to per-row")
                self.batched = False
        return np.concatenate([self.sess.run(None, {self.in_name: obs[i : i + 1]})[0] for i in range(len(obs))])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True, help="walking policy ONNX (61-D obs, 14 actions)")
    p.add_argument("--out", default="data/polite_bow_gait_bank.pt")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--samples", type=int, default=20000, help="states to keep")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--warmup-s", type=float, default=1.0, help="ignore the first seconds after each (re)spawn")
    p.add_argument("--stop-frac", type=float, default=0.2,
                   help="share of commands that are zero (walker stopping / marking time)")
    p.add_argument("--snapshot-prob", type=float, default=0.02,
                   help="per-env, per-step chance of taking a snapshot (spreads samples over the gait)")
    p.add_argument("--max-seconds", type=float, default=120.0, help="simulated-time safety cap")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_env_cfg(TASK, play=False)  # training DR: the bank covers what training sees
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.episode_length_s = 20.0  # long walks; falls still terminate and respawn
    cfg.events["gait_bank_spawn"].params["prob"] = 0.0  # no bank yet, obviously
    twist = cfg.commands["twist"]
    for k, v in WALK_RANGES.items():
        setattr(twist.ranges, k, v)
    twist.resampling_time_range = (1.5, 4.0)
    if hasattr(twist, "rel_standing_envs"):
        twist.rel_standing_envs = args.stop_frac
    if hasattr(twist, "heading_command"):
        twist.heading_command = False

    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    policy = BatchedOnnxPolicy(args.onnx)
    n_obs = env.observation_manager.group_obs_dim["actor"][0]
    if policy.obs_dim != n_obs:
        raise SystemExit(f"{args.onnx} expects {policy.obs_dim}-D obs, the env gives {n_obs}-D")

    robot = env.scene["robot"]
    feet_sensor = env.scene.sensors["feet_ground_contact"]
    warmup = int(round(args.warmup_s / env.step_dt))
    max_steps = int(round(args.max_seconds / env.step_dt))

    qpos_l, qvel_l, act_l, cmd_l = [], [], [], []
    kept = 0
    steps = falls = 0
    upright_steps = one_foot_steps = 0
    speed_err_sum = cmd_speed_sum = 0.0

    obs, _ = env.reset()
    t0 = time.time()
    while kept < args.samples and steps < max_steps:
        act = torch.as_tensor(policy(obs["actor"].detach().cpu().numpy()), device=env.device)
        obs, _, terminated, truncated, _ = env.step(act)
        steps += 1
        falls += int((terminated & ~truncated).sum())

        # Candidates: past warm-up, upright, trunk at walking height.
        grav = robot.data.projected_gravity_b
        upright = grav[:, 2] < -math.cos(math.radians(20.0))
        high = (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]) > 0.08
        ok = (env.episode_length_buf >= warmup) & upright & high & ~(terminated | truncated)

        feet = (feet_sensor.data.found > 0).float().sum(-1)
        cmd = env.command_manager.get_command("twist")[:, :3]
        vel = robot.data.root_link_lin_vel_b[:, :2]
        upright_steps += int(ok.sum())
        one_foot_steps += int((ok & (feet < 2)).sum())
        speed_err_sum += float(((vel - cmd[:, :2]).norm(dim=-1) * ok).sum())
        cmd_speed_sum += float((cmd[:, :2].norm(dim=-1) * ok).sum())

        take = ok & (torch.rand(env.num_envs, device=env.device) < args.snapshot_prob)
        ids = take.nonzero().squeeze(-1)
        if len(ids):
            qpos = env.sim.data.qpos[ids].clone()
            qpos[:, 0:3] -= env.scene.env_origins[ids]  # store relative to the env origin
            qpos_l.append(qpos.cpu())
            qvel_l.append(env.sim.data.qvel[ids].clone().cpu())
            act_l.append(env.action_manager.action[ids].clone().cpu())
            cmd_l.append(cmd[ids].clone().cpu())
            kept += len(ids)
        if steps % 250 == 0:
            print(f"[bank] {steps * env.step_dt:6.1f} s simulated, {kept}/{args.samples} states, "
                  f"{time.time() - t0:5.1f} s wall")

    env.close()
    if kept == 0:
        raise SystemExit("no usable states recorded: the walker never walked upright. Check the policy.")

    bank = {
        "qpos": torch.cat(qpos_l)[: args.samples],
        "qvel": torch.cat(qvel_l)[: args.samples],
        "action": torch.cat(act_l)[: args.samples],
        "command": torch.cat(cmd_l)[: args.samples],
        "meta": {
            "task": TASK,
            "walking_onnx": os.path.abspath(args.onnx),
            "walk_ranges": WALK_RANGES,
            "stop_frac": args.stop_frac,
            "warmup_s": args.warmup_s,
            "seed": args.seed,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(bank, args.out)

    n_env_s = steps * args.num_envs * env.step_dt
    upr = max(upright_steps, 1)
    print(f"\n── Gait bank: {args.out} ──")
    print(f"states kept         : {bank['qpos'].shape[0]}  (nq={bank['qpos'].shape[1]}, nv={bank['qvel'].shape[1]})")
    print(f"falls               : {falls} in {n_env_s / 60:.0f} env-minutes"
          f"  ({falls / max(n_env_s / 60, 1e-9):.2f} per env-minute)")
    print(f"one foot in the air : {100 * one_foot_steps / upr:4.1f}% of upright steps  (0% would mean it is not stepping)")
    print(f"speed tracking error: {speed_err_sum / upr:.3f} m/s mean  (mean commanded speed {cmd_speed_sum / upr:.3f} m/s)")
    trunk_speed = bank["qvel"][:, :2].norm(dim=-1)
    print(f"trunk speed in bank : median {trunk_speed.median():.3f} m/s, 90th pct {trunk_speed.quantile(0.9):.3f} m/s")


if __name__ == "__main__":
    main()
