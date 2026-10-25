"""Headless evaluation of a Sleepy Duckling ONNX policy, the deployment artefact.

Runs the exported ONNX exactly as the robot would (raw 61-D actor obs in, 14
actions out, normaliser baked in, no clock) in the training env and reports
what the rollouts actually show (AGENTS.md: measure before theorising).

    CUDA_VISIBLE_DEVICES= uv run python scripts/sleepy_duck_eval.py --onnx output.onnx
    uv run python scripts/sleepy_duck_eval.py --onnx output.onnx --num-envs 256 --device cuda:0
    MUJOCO_GL=egl uv run python scripts/sleepy_duck_eval.py --onnx output.onnx --video sleepy.mp4 --repeats 2
    uv run python scripts/sleepy_duck_eval.py --onnx output.onnx --num-envs 256 --device cuda:0 --from-walk

``--from-walk`` starts every episode from a recorded mid-walk state (the gait
bank) with the walker's last action handed over. Foot lifts are then only
counted after the first 0.5 s (the stride finishing).

The 13-D command block is all zeros, as the runtime feeds an episodic trick.
The training env does not: the roulade family keeps mjlab's velocity default
``rel_forward_envs = 0.2``, so at every command resample a fifth of the ducks
are told to walk forward at >= 0.3 m/s (the ±0.01 ranges notwithstanding).
``--train-commands`` evaluates with the training env's commands instead, which
is what this script did before 25 Oct 2026.

Reports: episodes that toppled, the head touching down too early, the
fastest head impact, how many nods the duck actually did and how sharp the
jerks were, trunk pitch and head tracking against the reference, and the
final pose (asleep on its beak and bum? snuggled onto its cheek?). Watch the
video too: sim metrics can pass while the motion looks wrong to a human eye.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import onnxruntime as ort
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from mjlab_microduck.tasks import bow_mdp, play_dead_mdp
from mjlab_microduck.tasks.microduck_sleepy_duck_env_cfg import (
    EPISODE_LENGTH_S,
    GAIT_BANK_PATH,
    HEAD_DOWN_ALLOWED,
    HEAD_YAW_SNUGGLE,
    KEYFRAMES,
    SLUMP_START,
    SNUGGLE_PITCH,
)

TASK = "Mjlab-SleepyDuck-Flat-MicroDuck"
_NECK, _HEAD_PITCH, _HEAD_YAW = 5, 6, 7
WALK_SETTLE_S = 0.5  # --from-walk: foot lifts before this are the stride finishing
VIDEO_AZIMUTH = 215.0  # with the duck facing +x: front-left, the side its face turns to (checked)
NODS_FROM = 2.3  # s: seated, the nodding starts
NOD_PROMINENCE = 0.12  # rad: a droop-and-catch smaller than this is not a nod


def nose_down(joint_pos: torch.Tensor) -> torch.Tensor:
    """Head angle relative to the trunk, nose-down positive (HOME = 0)."""
    return -(joint_pos[..., _NECK] - 0.3491) + (joint_pos[..., _HEAD_PITCH] - 0.3491)


def count_nods(x: np.ndarray, prominence: float = NOD_PROMINENCE) -> tuple[int, float]:
    """Droop-then-catch events in a nose-down trace, and the fastest catch.

    A nod is a local peak followed by a drop of at least ``prominence`` before
    the trace rises again. Returns (count, fastest catch in rad per sample).
    """
    nods, fastest = 0, 0.0
    peak, peak_i, falling = x[0], 0, False
    for i in range(1, len(x)):
        if not falling:
            if x[i] > peak:
                peak, peak_i = x[i], i
            elif peak - x[i] >= prominence:
                nods += 1
                falling = True
                fastest = max(fastest, float(np.max(-np.diff(x[peak_i : i + 1]))))
                trough = x[i]
        else:
            if x[i] < trough:
                trough = x[i]
            elif x[i] - trough >= prominence / 2:
                falling, peak, peak_i = False, x[i], i
    return nods, fastest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--train-dr", action="store_true", help="use the training (DR) cfg, not play")
    p.add_argument("--video", default=None, help="also write an MP4 of env 0 (needs MUJOCO_GL=egl headless)")
    p.add_argument("--repeats", type=int, default=1, help="episodes to record back to back in the video")
    p.add_argument("--from-walk", nargs="?", const=GAIT_BANK_PATH, default=None, metavar="BANK",
                   help=f"start every episode mid-walk from a gait bank (default {GAIT_BANK_PATH})")
    p.add_argument("--train-commands", action="store_true",
                   help="feed the training env's sampled commands, not the runtime's all-zero block")
    args = p.parse_args()
    if args.video:
        _record(args)
        if args.num_envs <= 1:
            return

    cfg = load_env_cfg(TASK, play=not args.train_dr)
    cfg.scene.num_envs = args.num_envs
    _apply_eval_cfg(cfg, args)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    sess = ort.InferenceSession(args.onnx)
    in_name = sess.get_inputs()[0].name
    assert sess.get_inputs()[0].shape[-1] == 61, "expected a 61-D actor policy"

    robot = env.scene["robot"]
    servo_ids, _ = robot.find_joints(r"^(?!passive_).*")
    head_id = robot.find_bodies("jaw_soft")[0][0]
    home = robot.data.default_joint_pos[:, servo_ids].cpu()

    obs, _ = env.reset()
    n_steps = int(round(EPISODE_LENGTH_S / env.step_dt))
    B = env.num_envs
    alive = torch.ones(B, dtype=torch.bool)
    toppled = torch.zeros(B, dtype=torch.bool)
    early_head = torch.zeros(B, dtype=torch.bool)
    head_impact = torch.zeros(B)
    lifted_steps = torch.zeros(B)
    pitch_log, ref_log, nose_log, nose_ref_log, yaw_log = [], [], [], [], []
    start_xy = None

    for k in range(n_steps):
        actor = obs["actor"].detach().cpu().numpy().astype(np.float32)
        act = np.concatenate([sess.run(None, {in_name: actor[i : i + 1]})[0] for i in range(B)])
        # Everything is read BEFORE stepping, while episode_length_buf = k.
        t = k * env.step_dt
        tt = torch.full((B,), t)
        pitch = bow_mdp.trunk_pitch(env).cpu()
        ref = play_dead_mdp.reference_pitch_at(tt, KEYFRAMES)
        q = robot.data.joint_pos[:, servo_ids].cpu()
        q_ref = play_dead_mdp.reference_pose_at(tt, KEYFRAMES, home)
        xy = (robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).cpu()
        if start_xy is None:
            start_xy = xy.clone()
        obs, _, terminated, truncated, _ = env.step(torch.as_tensor(act, device=env.device))
        term = terminated.cpu()
        if k < n_steps - 1:
            toppled |= term & alive
        alive &= ~(term | truncated.cpu())
        head_down = (env.scene.sensors["head_ground_contact"].data.found > 0).any(-1).cpu()
        head_speed = robot.data.body_link_lin_vel_w[:, head_id].norm(dim=-1).cpu()
        head_impact = torch.maximum(head_impact, torch.where(head_down & alive, head_speed, 0.0))
        if t < HEAD_DOWN_ALLOWED:
            early_head |= head_down & alive
        feet = (env.scene.sensors["feet_ground_contact"].data.found > 0).float().sum(-1).cpu()
        if not args.from_walk or t >= WALK_SETTLE_S:
            lifted_steps += (feet < 2).float() * alive
        nan = torch.tensor(torch.nan)
        pitch_log.append(torch.where(alive, pitch, nan))
        ref_log.append(torch.where(alive, ref, nan))
        nose_log.append(torch.where(alive, nose_down(q), nan))
        nose_ref_log.append(nose_down(q_ref))
        yaw_log.append(torch.where(alive, q[:, _HEAD_YAW], nan))
        if k == n_steps - 2:
            g = robot.data.projected_gravity_b.cpu()
            final_pitch = pitch.clone()
            final_roll = torch.asin(g[:, 1].clamp(-1, 1))
            final_yaw = q[:, _HEAD_YAW].clone()
            final_head_down = head_down.clone()
            final_feet = feet.clone()
            final_alive = alive.clone()
            drift = (xy - start_xy).norm(dim=-1) * 1000

    P, R = torch.stack(pitch_log), torch.stack(ref_log)  # (T, B)
    Y = torch.stack(yaw_log)
    N, NR = torch.stack(nose_log), torch.stack(nose_ref_log)
    deg = math.degrees
    ok = final_alive
    asleep = (
        ok & ((final_pitch - SNUGGLE_PITCH).abs() < math.radians(15)) & (final_roll.abs() < math.radians(15))
        & final_head_down & (final_feet == 2)
    )
    snuggled = ok & ((final_yaw - HEAD_YAW_SNUGGLE).abs() < math.radians(20))

    # Nods: in the seated window, on each surviving duck's own head trace.
    k0, k1 = int(NODS_FROM / env.step_dt), int((SLUMP_START + 0.3) / env.step_dt)
    counts, catches = [], []
    for b in torch.nonzero(ok).flatten().tolist():
        n, fastest = count_nods(N[k0:k1, b].numpy())
        counts.append(n)
        catches.append(fastest / env.step_dt)
    ref_nods, ref_catch = count_nods(NR[k0:k1, 0].numpy())
    # The head-turn counter at each nod's droop (run 2): how far the head has
    # turned towards its sleeping side, against the keyframe.
    droops = [k for k in KEYFRAMES if NODS_FROM < k["t"] <= SLUMP_START][0::2]
    turn_at = [(k["pose"][_HEAD_YAW], torch.nanmedian(Y[int(round(k["t"] / env.step_dt)) - 1]).item()) for k in droops]
    counts_arr = np.array(counts) if counts else np.array([0])

    def med(x: torch.Tensor) -> float:
        return x[ok].median().item() if ok.any() else float("nan")

    start = f"from mid-walk ({args.from_walk})" if args.from_walk else "from a still stand"
    commands = "training commands" if args.train_commands else "runtime commands (all zero)"
    print(f"\n── SleepyDuck eval: {args.onnx}  ({B} envs, {'train-DR' if args.train_dr else 'play'} cfg, {start}, "
          f"{commands}) ──")
    print(f"toppled             : {toppled.float().mean() * 100:5.1f}% of episodes (target 0)")
    print(f"asleep at the end   : {asleep.float().mean() * 100:5.1f}% (trunk within 15° of {deg(SNUGGLE_PITCH):.0f}°, "
          "head on the floor, both feet down)")
    print(f"snuggled            : {snuggled.float().mean() * 100:5.1f}% head yaw within 20° of {deg(HEAD_YAW_SNUGGLE):.0f}°")
    print(f"nods                : {np.median(counts_arr):.0f} median (target {ref_nods}); exactly {ref_nods} in "
          f"{(counts_arr == ref_nods).mean() * 100:.0f}%, histogram {np.bincount(counts_arr).tolist()}")
    print("head turn at nods   : " + ", ".join(f"{deg(got):.0f}°" for _, got in turn_at)
          + "   (target " + ", ".join(f"{deg(want):.0f}°" for want, _ in turn_at) + ")")
    print(f"jerk (catch) speed  : {np.median(catches) if catches else float('nan'):5.2f} rad/s median "
          f"(reference {ref_catch / env.step_dt:.2f})")
    print(f"head down too early : {early_head.float().mean() * 100:5.1f}% (before {HEAD_DOWN_ALLOWED:.1f} s)")
    print(f"a foot lifted       : {lifted_steps.mean() * env.step_dt:5.2f} s per episode (target 0)")
    landed = head_impact[head_impact > 0]
    p99 = torch.quantile(landed, 0.99).item() if len(landed) else float("nan")
    print(f"head impact speed   : {head_impact.median():5.2f} m/s median, {p99:.2f} 99th percentile, "
          f"{head_impact.max():.2f} max   (lower is kinder to the camera)")
    print(f"final trunk pitch   : {deg(med(final_pitch)):6.1f}° median (target {deg(SNUGGLE_PITCH):.0f}°)")
    print(f"final trunk roll    : {deg(med(final_roll.abs())):6.1f}° median |roll| (target 0°)")
    print(f"final head yaw      : {deg(med(final_yaw)):6.1f}° median (target {deg(HEAD_YAW_SNUGGLE):.0f}°)")
    print(f"drift               : {med(drift):6.1f} mm median")
    print("\n  t(s)   pitch ref°  mean°    head ref°  mean°   (nose-down vs the trunk)")
    for k in range(0, n_steps, max(1, n_steps // 24)):
        print(f"  {k * env.step_dt:5.2f}  {deg(torch.nanmean(R[k]).item()):9.1f}  {deg(torch.nanmean(P[k]).item()):6.1f}"
              f"  {deg(NR[k, 0].item()):9.1f}  {deg(torch.nanmean(N[k]).item()):6.1f}")
    rmse = torch.sqrt(torch.nanmean((P - R) ** 2)).item()
    head_rmse = torch.sqrt(torch.nanmean((N - NR) ** 2)).item()
    print(f"\npitch tracking RMSE: {deg(rmse):.1f}°   head (nose-down) tracking RMSE: {deg(head_rmse):.1f}°")
    env.close()


def _apply_eval_cfg(cfg, args) -> None:
    """Spawns and commands for an evaluation.

    Every episode starts at t = 0 from a still stand (or mid-walk with
    --from-walk). The training cfg (--train-dr) would otherwise also start
    some episodes mid-walk and some mid-routine; those end early and their
    clocks don't match this script's, which understated every training-DR
    result before 25 Oct 2026 (runs 2 to 5).

    The twist command is pinned to exactly zero, as on the robot, unless
    --train-commands (the head and body slots are zero padding already)."""
    cfg.events["mid_routine_spawn"].params["prob"] = 0.0
    if args.from_walk:
        cfg.events["gait_bank_spawn"].params.update(bank_path=args.from_walk, prob=1.0)
    else:
        cfg.events["gait_bank_spawn"].params["prob"] = 0.0
    if not getattr(args, "train_commands", False):
        twist = cfg.commands["twist"]
        twist.rel_forward_envs = 0.0
        twist.rel_turn_in_place_envs = 0.0
        twist.ranges.lin_vel_x = twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0.0, 0.0)


def _record(args) -> None:
    """Roll out env 0 for ``--repeats`` episodes and write an MP4."""
    import mediapy

    cfg = load_env_cfg(TASK, play=not args.train_dr)
    cfg.scene.num_envs = 1
    _apply_eval_cfg(cfg, args)
    # Front-left three-quarter view: the snuggle turns the face to the duck's
    # left, so this is the side where you see it asleep. Resets give the duck a
    # random heading, so each episode is turned to face the camera (_face_x).
    cfg.viewer.distance = 0.6
    cfg.viewer.elevation = -15.0
    cfg.viewer.azimuth = VIDEO_AZIMUTH
    cfg.viewer.width, cfg.viewer.height = 640, 480
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")
    sess = ort.InferenceSession(args.onnx)
    in_name = sess.get_inputs()[0].name
    frames = []
    obs, _ = env.reset()
    n_steps = int(round(EPISODE_LENGTH_S / env.step_dt))
    for _ in range(args.repeats):
        _face_x(env)
        for _ in range(n_steps):
            actor = obs["actor"].detach().cpu().numpy().astype(np.float32)
            act = sess.run(None, {in_name: actor})[0]
            obs, *_ = env.step(torch.as_tensor(act, device=env.device))
            frame = env.render()
            if frame is not None:
                frames.append(frame[0] if frame.ndim == 4 else frame)
    env.close()
    mediapy.write_video(args.video, frames, fps=round(1.0 / env.step_dt))
    print(f"wrote {args.video} ({len(frames)} frames)")


def _face_x(env: ManagerBasedRlEnv) -> None:
    """Turn env 0's duck to face +x, keeping its tilt: the camera's front."""
    q = env.sim.data.qpos[0, 3:7]
    w, x, y, z = q.tolist()
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = math.cos(-yaw / 2), math.sin(-yaw / 2)  # q_unyaw = (c, 0, 0, s)
    env.sim.data.qpos[0, 3:7] = torch.tensor(
        [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w], device=env.device
    )
    env.sim.forward()


if __name__ == "__main__":
    main()
