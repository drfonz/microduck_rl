"""Headless evaluation of a PoliteBow ONNX policy, the deployment artefact.

"Measure before theorising" (AGENTS.md): run the exported ONNX exactly as the
robot would (raw 61-D actor obs in, 14 actions out, normaliser baked in, no
clock) in the training env, and report what the rollouts actually show.

    CUDA_VISIBLE_DEVICES= uv run python scripts/bow_eval.py --onnx output.onnx
    uv run python scripts/bow_eval.py --onnx output.onnx --num-envs 256 --device cuda:0
    MUJOCO_GL=egl uv run python scripts/bow_eval.py --onnx output.onnx --video bow.mp4 --repeats 3

Reports: falls, head strikes, feet lifted, trunk pitch vs the reference over
time, peak pitch, the final stand, drift. Watch the video too: sim metrics can
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

from mjlab_microduck.tasks import bow_mdp
from mjlab_microduck.tasks.microduck_polite_bow_env_cfg import (
    BOW_PITCH,
    BOW_PROFILE,
    EPISODE_LENGTH_S,
)

TASK = "Mjlab-PoliteBow-Flat-MicroDuck"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--train-dr", action="store_true", help="use the training (DR) cfg, not play")
    p.add_argument("--video", default=None, help="also write an MP4 of env 0 (needs MUJOCO_GL=egl headless)")
    p.add_argument("--repeats", type=int, default=1, help="episodes to record back to back in the video")
    args = p.parse_args()
    if args.video:
        _record(args)
        if args.num_envs <= 1:
            return

    cfg = load_env_cfg(TASK, play=not args.train_dr)
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    sess = ort.InferenceSession(args.onnx)
    in_name = sess.get_inputs()[0].name
    assert sess.get_inputs()[0].shape[-1] == 61, "expected a 61-D actor policy"

    obs, _ = env.reset()
    n_steps = int(round(EPISODE_LENGTH_S / env.step_dt))
    B = env.num_envs
    alive = torch.ones(B, dtype=torch.bool)
    fell = torch.zeros(B, dtype=torch.bool)
    head_hit = torch.zeros(B, dtype=torch.bool)
    lifted_steps = torch.zeros(B)
    pitch_log, ref_log = [], []
    start_xy = None

    for k in range(n_steps):
        actor = obs["actor"].detach().cpu().numpy().astype(np.float32)
        act = np.concatenate([sess.run(None, {in_name: actor[i : i + 1]})[0] for i in range(B)])
        # Pitch/reference are read BEFORE stepping, while episode_length_buf = k.
        pitch = bow_mdp.trunk_pitch(env).cpu()
        ref = bow_mdp.bow_blend(env, BOW_PROFILE).cpu() * BOW_PITCH
        robot = env.scene["robot"]
        xy = (robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).cpu()
        if start_xy is None:
            start_xy = xy.clone()
        obs, _, terminated, truncated, _ = env.step(torch.as_tensor(act, device=env.device))
        term = terminated.cpu()
        if k < n_steps - 1:
            fell |= term & alive
        alive &= ~(term | truncated.cpu())
        feet = (env.scene.sensors["feet_ground_contact"].data.found > 0).float().sum(-1).cpu()
        lifted_steps += (feet < 2).float() * alive
        head_hit |= (env.scene.sensors["head_ground_contact"].data.found > 0).any(-1).cpu() & alive
        pitch_log.append(torch.where(alive, pitch, torch.nan))
        ref_log.append(torch.where(alive, ref, torch.nan))
        if k == n_steps - 2:
            final_pitch = pitch.clone()
            drift = (xy - start_xy).norm(dim=-1) * 1000

    P = torch.stack(pitch_log)  # (T, B)
    R = torch.stack(ref_log)
    deg = math.degrees
    print(f"\n── PoliteBow eval: {args.onnx}  ({B} envs, {'train-DR' if args.train_dr else 'play'} cfg) ──")
    print(f"fell over          : {fell.float().mean() * 100:5.1f}% of episodes")
    print(f"head hit the floor : {head_hit.float().mean() * 100:5.1f}%")
    print(f"a foot lifted      : {lifted_steps.mean() * env.step_dt:5.2f} s per episode (target 0)")
    print(f"peak trunk pitch   : {deg(torch.nanquantile(P.nan_to_num(-9).amax(0), 0.5).item()):5.1f}° median"
          f"   (target {deg(BOW_PITCH):.0f}°)")
    print(f"final trunk pitch  : {deg(final_pitch.abs().median().item()):5.1f}° median (target 0°)")
    print(f"drift              : {drift.median():5.1f} mm median")
    print("\n  t(s)   ref°   mean°   (trunk pitch vs reference)")
    for k in range(0, n_steps, max(1, n_steps // 14)):
        m = torch.nanmean(P[k]).item()
        print(f"  {k * env.step_dt:4.2f}  {deg(torch.nanmean(R[k]).item()):5.1f}  {deg(m):6.1f}")
    rmse = torch.sqrt(torch.nanmean((P - R) ** 2)).item()
    print(f"\npitch tracking RMSE: {deg(rmse):.1f}°")
    env.close()


def _record(args) -> None:
    """Roll out env 0 for ``--repeats`` episodes and write an MP4."""
    import mediapy

    cfg = load_env_cfg(TASK, play=not args.train_dr)
    cfg.scene.num_envs = 1
    # Close-up side view: the bow is a sagittal motion of a 25 cm robot.
    cfg.viewer.distance = 0.7
    cfg.viewer.elevation = -10.0
    cfg.viewer.azimuth = 90.0  # side-on
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
