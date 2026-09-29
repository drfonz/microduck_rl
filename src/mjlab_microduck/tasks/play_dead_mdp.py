"""MDP terms for the PlayDead trick (Mjlab-PlayDead-Flat-MicroDuck).

Stand → sit → lie belly-up with the legs in the air → turn the head to one
side, then stay "dead". Same teaching method as PoliteBow (see bow_mdp.py):

* the runtime feeds a publishable episodic policy a CONSTANT all-zero command
  and no clock, so the actor cannot see time;
* the environment keeps a hidden reference clock and turns it into a smooth
  keyframed target (joint pose + trunk pitch), which the rewards track;
* the critic gets the clock as a privileged observation (not exported);
* no jackpots: being ahead of the moving target pays nothing, so a slow,
  controlled descent is the argmax. That matters here more than for the bow:
  a free backward fall from the sit would slam the head (38% of the mass,
  camera inside) into the floor.

The difference from the bow is that the reference has several stages, so the
profile is a list of keyframes, each blended to the next with a smoothstep.

Sign convention (same as bow_mdp): every cost returns >= 0 and takes a
NEGATIVE weight; every reward returns [0, 1] and takes a POSITIVE weight. In
the training log every ``Episode_Reward/*_cost`` must read <= 0.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks.bow_mdp import _episode_time, _servo_ids, trunk_pitch

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")

# A keyframe profile is a list of dicts, in time order:
#   {"t": seconds, "pose": {servo_idx: rad, ...}, "pitch": rad}
# Servos missing from "pose" sit at HOME (the asset's default joint pos).


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def keyframe_segment(t: torch.Tensor, times: Sequence[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Index of the active segment and the smoothstep blend within it.

    Before the first keyframe the blend is 0 (first pose); after the last it
    is 1 on the final segment (last pose). Smoothstep starts and ends each
    segment at zero velocity, so the reference never asks for a jerk.
    """
    tt = torch.as_tensor(times, device=t.device, dtype=t.dtype)
    idx = (torch.searchsorted(tt, t.contiguous(), right=True) - 1).clamp(0, len(times) - 2)
    t0, t1 = tt[idx], tt[idx + 1]
    alpha = _smoothstep((t - t0) / (t1 - t0).clamp_min(1e-6))
    return idx, alpha


def reference_pitch_at(t: torch.Tensor, keyframes: Sequence[dict]) -> torch.Tensor:
    """(B,) reference trunk pitch (rad) at episode time ``t``."""
    idx, a = keyframe_segment(t, [k["t"] for k in keyframes])
    p = torch.as_tensor([k["pitch"] for k in keyframes], device=t.device, dtype=t.dtype)
    return p[idx] + a * (p[idx + 1] - p[idx])


def reference_pose_at(
    t: torch.Tensor, keyframes: Sequence[dict], home: torch.Tensor
) -> torch.Tensor:
    """(B, 14) reference servo targets at episode time ``t``.

    ``home`` is the (14,) or (B, 14) HOME pose that fills unlisted servos.
    """
    idx, a = keyframe_segment(t, [k["t"] for k in keyframes])
    base = home if home.dim() == 2 else home.unsqueeze(0).expand(t.shape[0], -1)
    poses = []
    for k in keyframes:
        p = base.clone()
        for j, v in k["pose"].items():
            p[:, j] = v
        poses.append(p)
    P = torch.stack(poses, dim=1)  # (B, K, 14)
    rows = torch.arange(t.shape[0], device=t.device)
    p0, p1 = P[rows, idx], P[rows, idx + 1]
    return p0 + a.unsqueeze(-1) * (p1 - p0)


def _reference_pose(env: ManagerBasedRlEnv, keyframes: Sequence[dict], asset: Entity) -> torch.Tensor:
    ids = _servo_ids(env, asset)
    home = asset.data.default_joint_pos[:, ids]
    return reference_pose_at(_episode_time(env), keyframes, home)


def _pose_error(
    env: ManagerBasedRlEnv,
    keyframes: Sequence[dict],
    joint_indices: Sequence[int],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    ids = _servo_ids(env, asset)
    target = _reference_pose(env, keyframes, asset)[:, list(joint_indices)]
    return asset.data.joint_pos[:, [ids[i] for i in joint_indices]] - target


def _before(env: ManagerBasedRlEnv, t_end: float) -> torch.Tensor:
    """(B,) 1.0 while the episode clock is before ``t_end``, else 0.0."""
    return (_episode_time(env) < t_end).float()


# ── Rewards (positive weight) ────────────────────────────────────────────────


def pose_track(
    env: ManagerBasedRlEnv,
    keyframes: Sequence[dict],
    joint_indices: Sequence[int],
    std: float = 0.3,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Gaussian tracking of the keyframed joint target, in [0, 1]."""
    err = _pose_error(env, keyframes, joint_indices, asset_cfg)
    return torch.nan_to_num(torch.exp(-((err / std) ** 2)).mean(dim=-1), nan=0.0)


def pitch_track(
    env: ManagerBasedRlEnv,
    keyframes: Sequence[dict],
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Gaussian tracking of the reference trunk pitch, in [0, 1].

    This is the main term: it asks the trunk to go from upright (0) to on its
    back (−90°) at the pace of the reference, which is what makes the lie-down
    a controlled lowering instead of a fall.
    """
    target = reference_pitch_at(_episode_time(env), keyframes)
    err = trunk_pitch(env, asset_cfg) - target
    return torch.nan_to_num(torch.exp(-((err / std) ** 2)), nan=0.0)


def feet_planted_until(env: ManagerBasedRlEnv, sensor_name: str, t_end: float) -> torch.Tensor:
    """Fraction of feet on the floor, in [0, 1], only before ``t_end``.

    Standing and sitting keep both feet down; once the lie-down starts the
    feet are meant to leave the floor, so the term switches off by the clock
    (never by state, so it cannot be farmed from a bad pose).
    """
    found = env.scene.sensors[sensor_name].data.found > 0
    frac = found.float().mean(dim=-1) if found.dim() > 1 else found.float()
    return frac * _before(env, t_end)


# ── Costs (negative weight) ──────────────────────────────────────────────────


def pose_l1_cost(
    env: ManagerBasedRlEnv,
    keyframes: Sequence[dict],
    joint_indices: Sequence[int],
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Mean |joint error| vs the keyframed target (>= 0): constant gradient
    far from the target, where the Gaussian has flattened out."""
    err = _pose_error(env, keyframes, joint_indices, asset_cfg)
    return torch.nan_to_num(err.abs().mean(dim=-1), nan=0.0)


def contact_cost_until(env: ManagerBasedRlEnv, sensor_name: str, t_end: float) -> torch.Tensor:
    """1 when the watched body touches the floor before ``t_end`` (>= 0).

    Used for the head: it must not touch down while standing or sitting. Once
    lying down the head rests on the floor, which is the point.
    """
    found = env.scene.sensors[sensor_name].data.found > 0
    if found.dim() > 1:
        found = found.any(dim=-1)
    return found.float() * _before(env, t_end)


def head_impact_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    head_body: str = "jaw_soft",
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Head speed squared while the head touches the floor (>= 0).

    Resting on the floor is free; arriving fast, or scraping, is not. This is
    what protects the camera and the neck servos on the real robot when the
    duck lies back.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cache = env.__dict__.setdefault("_play_dead_head_id", {})
    if head_body not in cache:
        ids, _ = asset.find_bodies(head_body)
        cache[head_body] = ids[0]
    v = asset.data.body_link_lin_vel_w[:, cache[head_body]]
    found = env.scene.sensors[sensor_name].data.found > 0
    if found.dim() > 1:
        found = found.any(dim=-1)
    return torch.nan_to_num(torch.sum(v * v, dim=-1), nan=0.0) * found.float()


# ── Termination ──────────────────────────────────────────────────────────────


def wrong_side_down(
    env: ManagerBasedRlEnv,
    max_roll: float = math.radians(60.0),
    max_forward_pitch: float = math.radians(50.0),
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """True when the duck is on its side or tipping onto its face.

    Lying on the BACK is the task, so the usual "fell over" test cannot be
    used. Rolling sideways past ``max_roll`` or pitching forward past
    ``max_forward_pitch`` ends the episode instead.
    """
    asset: Entity = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    roll = torch.asin(g[:, 1].clamp(-1.0, 1.0)).abs()
    return (roll > max_roll) | (trunk_pitch(env, asset_cfg) > max_forward_pitch)


# ── Privileged critic observation ─────────────────────────────────────────────


def play_dead_clock_obs(
    env: ManagerBasedRlEnv, keyframes: Sequence[dict], period: float
) -> torch.Tensor:
    """(B, 3) = [reference pitch, sin, cos] of the hidden clock. CRITIC ONLY.

    Never add this to the actor group: the exported policy must run on the
    runtime's constant command, with no clock.
    """
    t = _episode_time(env)
    w = 2.0 * math.pi * t / period
    return torch.stack((reference_pitch_at(t, keyframes), torch.sin(w), torch.cos(w)), dim=-1)
