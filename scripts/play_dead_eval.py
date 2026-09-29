"""Headless evaluation of a PlayDead ONNX policy, the deployment artefact.

Runs the exported ONNX exactly as the robot would (raw 61-D actor obs in, 14
actions out, normaliser baked in, no clock) in the training env and reports
what the rollouts actually show (AGENTS.md: measure before theorising).

    CUDA_VISIBLE_DEVICES= uv run python scripts/play_dead_eval.py --onnx output.onnx
    uv run python scripts/play_dead_eval.py --onnx output.onnx --num-envs 256 --device cuda:0
    MUJOCO_GL=egl uv run python scripts/play_dead_eval.py --onnx output.onnx --video dead.mp4 --repeats 3
    uv run python scripts/play_dead_eval.py --onnx output.onnx --num-envs 256 --device cuda:0 --from-walk

``--from-walk`` starts every episode from a recorded mid-walk state (the gait
bank) with the walker's last action handed over: "bang!" while walking. Foot
lifts are then only counted after the first 0.5 s (the stride finishing).

Reports: episodes that ended on a side or face, head touching down too early,
the fastest head impact, trunk pitch vs the reference over time, the final
pose (on its back? head turned?), drift. Watch the video too: sim metrics can
pass while the motion looks wrong to a human eye.
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
from mjlab_microduck.tasks.microduck_play_dead_env_cfg import (
    EPISODE_LENGTH_S,
    GAIT_BANK_PATH,
    HEAD_YAW_DEAD,
    KEYFRAMES,
    LIE_DOWN_START,
    LYING_PITCH,
)

TASK = "Mjlab-PlayDead-Flat-MicroDuck"
_HEAD_YAW = 7
WALK_SETTLE_S = 0.5  # --from-walk: foot lifts before this are the stride finishing


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
    args = p.parse_args()
    if args.video:
        _record(args)
        if args.num_envs <= 1:
            return

    cfg = load_env_cfg(TASK, play=not args.train_dr)
    cfg.scene.num_envs = args.num_envs
    _apply_from_walk(cfg, args)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    sess = ort.InferenceSession(args.onnx)
    in_name = sess.get_inputs()[0].name
    assert sess.get_inputs()[0].shape[-1] == 61, "expected a 61-D actor policy"

    robot = env.scene["robot"]
    servo_ids, _ = robot.find_joints(r"^(?!passive_).*")
    head_id = robot.find_bodies("jaw_soft")[0][0]

    obs, _ = env.reset()
    n_steps = int(round(EPISODE_LENGTH_S / env.step_dt))
    B = env.num_envs
    alive = torch.ones(B, dtype=torch.bool)
    wrong_side = torch.zeros(B, dtype=torch.bool)
    early_head = torch.zeros(B, dtype=torch.bool)
    head_impact = torch.zeros(B)
    lifted_steps = torch.zeros(B)
    pitch_log, ref_log = [], []
    start_xy = None

    for k in range(n_steps):
        actor = obs["actor"].detach().cpu().numpy().astype(np.float32)
        act = np.concatenate([sess.run(None, {in_name: actor[i : i + 1]})[0] for i in range(B)])
        # Pitch/reference are read BEFORE stepping, while episode_length_buf = k.
        t = k * env.step_dt
        pitch = bow_mdp.trunk_pitch(env).cpu()
        ref = play_dead_mdp.reference_pitch_at(torch.full((B,), t), KEYFRAMES)
        xy = (robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).cpu()
        if start_xy is None:
            start_xy = xy.clone()
        obs, _, terminated, truncated, _ = env.step(torch.as_tensor(act, device=env.device))
        term = terminated.cpu()
        if k < n_steps - 1:
            wrong_side |= term & alive
        alive &= ~(term | truncated.cpu())
        head_down = (env.scene.sensors["head_ground_contact"].data.found > 0).any(-1).cpu()
        head_speed = robot.data.body_link_lin_vel_w[:, head_id].norm(dim=-1).cpu()
        head_impact = torch.maximum(head_impact, torch.where(head_down & alive, head_speed, 0.0))
        if t < LIE_DOWN_START:
            early_head |= head_down & alive
            feet = (env.scene.sensors["feet_ground_contact"].data.found > 0).float().sum(-1).cpu()
            if not args.from_walk or t >= WALK_SETTLE_S:
                lifted_steps += (feet < 2).float() * alive
        pitch_log.append(torch.where(alive, pitch, torch.nan))
        ref_log.append(torch.where(alive, ref, torch.nan))
        if k == n_steps - 2:
            g = robot.data.projected_gravity_b.cpu()
            final_pitch = pitch.clone()
            final_roll = torch.asin(g[:, 1].clamp(-1, 1))
            final_yaw = robot.data.joint_pos[:, servo_ids[_HEAD_YAW]].cpu()
            final_alive = alive.clone()
            drift = (xy - start_xy).norm(dim=-1) * 1000

    P = torch.stack(pitch_log)  # (T, B)
    R = torch.stack(ref_log)
    deg = math.degrees
    ok = final_alive
    on_back = ok & ((final_pitch - LYING_PITCH).abs() < math.radians(20)) & (final_roll.abs() < math.radians(20))
    head_turned = ok & ((final_yaw - HEAD_YAW_DEAD).abs() < 0.35)

    def med(x: torch.Tensor) -> float:
        return x[ok].median().item() if ok.any() else float("nan")

    start = f"from mid-walk ({args.from_walk})" if args.from_walk else "from a still stand"
    print(f"\n── PlayDead eval: {args.onnx}  ({B} envs, {'train-DR' if args.train_dr else 'play'} cfg, {start}) ──")
    print(f"ended on side/face  : {wrong_side.float().mean() * 100:5.1f}% of episodes (target 0)")
    print(f"dead at the end     : {on_back.float().mean() * 100:5.1f}% on its back within 20°")
    print(f"head turned         : {head_turned.float().mean() * 100:5.1f}% within 20° of {deg(HEAD_YAW_DEAD):.0f}°")
    print(f"head down too early : {early_head.float().mean() * 100:5.1f}% (before {LIE_DOWN_START:.1f} s)")
    print(f"a foot lifted early : {lifted_steps.mean() * env.step_dt:5.2f} s per episode (target 0)")
    print(f"head impact speed   : {head_impact.median():5.2f} m/s median, {head_impact.max():.2f} max"
          "   (lower is kinder to the camera)")
    print(f"final trunk pitch   : {deg(med(final_pitch)):6.1f}° median (target {deg(LYING_PITCH):.0f}°)")
    print(f"final trunk roll    : {deg(med(final_roll.abs())):6.1f}° median |roll| (target 0°)")
    print(f"final head yaw      : {deg(med(final_yaw)):6.1f}° median (target {deg(HEAD_YAW_DEAD):.0f}°)")
    print(f"drift               : {med(drift):6.1f} mm median")
    print("\n  t(s)   ref°   mean°   (trunk pitch vs reference)")
    for k in range(0, n_steps, max(1, n_steps // 16)):
        m = torch.nanmean(P[k]).item()
        print(f"  {k * env.step_dt:4.2f}  {deg(torch.nanmean(R[k]).item()):6.1f}  {deg(m):6.1f}")
    rmse = torch.sqrt(torch.nanmean((P - R) ** 2)).item()
    print(f"\npitch tracking RMSE: {deg(rmse):.1f}°")
    env.close()


def _apply_from_walk(cfg, args) -> None:
    if args.from_walk:
        cfg.events["gait_bank_spawn"].params.update(bank_path=args.from_walk, prob=1.0)


def _record(args) -> None:
    """Roll out env 0 for ``--repeats`` episodes and write an MP4."""
    import mediapy

    cfg = load_env_cfg(TASK, play=not args.train_dr)
    cfg.scene.num_envs = 1
    _apply_from_walk(cfg, args)
    # Three-quarter view from slightly above: the lie-down is sagittal but the
    # punchline (the head turn) is only visible from above the belly.
    cfg.viewer.distance = 0.75
    cfg.viewer.elevation = -25.0
    cfg.viewer.azimuth = 135.0
    cfg.viewer.width, cfg.viewer.height = 640, 480
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")
    sess = ort.InferenceSession(args.onnx)
    in_name = sess.get_inputs()[0].name
    frames = []
    obs, _ = env.reset()
    n_steps = int(round(EPISODE_LENGTH_S / env.step_dt))
    for _ in range(args.repeats):
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


if __name__ == "__main__":
    main()
