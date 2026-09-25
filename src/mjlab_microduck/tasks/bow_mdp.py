"""MDP terms for the PoliteBow trick (Mjlab-PoliteBow-Flat-MicroDuck).

Design in one paragraph
-----------------------
The deployed policy must be publishable with ``uv run publish --kind episodic``:
the runtime feeds it a CONSTANT (all-zero) 13-D command block and no clock, so
the actor cannot see time. The bow is therefore taught through a *hidden
reference clock*: the environment knows the episode time and derives a smooth
bow profile ``blend(t)`` (0 = HOME stand, 1 = full bow). Rewards track the
joint pose and trunk pitch interpolated by ``blend``, and the CRITIC is given
the clock as a privileged observation (only the actor is exported to ONNX, so
this does not break the 61-D deployment contract). Every episode starts from a
still stand, so the actor learns one deterministic down-hold-up trajectory
driven by its own state (velocities, last action, trunk tilt), exactly like the
roulade learns its roll without a phase clock.

Why a moving target instead of "reach the bow pose" bonuses: AGENTS.md's
no-jackpot rule. Being ahead of the ramp pays nothing, so going slow and smooth
is the argmax; there is no terminal bonus to rush towards.

Sign convention used in THIS module (keep it consistent):
every cost returns a value >= 0 and is paired with a NEGATIVE weight; every
reward returns a value in [0, 1] and is paired with a POSITIVE weight. So in the
training log every ``Episode_Reward/bow_*_cost`` must read <= 0.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


# ── Reference profile ────────────────────────────────────────────────────────


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def bow_blend_at(
    t: torch.Tensor,
    down_start: float,
    down_end: float,
    hold_end: float,
    up_end: float,
) -> torch.Tensor:
    """Bow amount in [0, 1] at episode time ``t`` (seconds).

    0 before ``down_start``, smooth rise to 1 at ``down_end``, hold until
    ``hold_end``, smooth return to 0 at ``up_end``, then 0 (standing).
    Smoothstep ramps start and end with zero velocity, so the reference never
    asks for a jerk.
    """
    down = _smoothstep((t - down_start) / max(down_end - down_start, 1e-6))
    up = _smoothstep((t - hold_end) / max(up_end - hold_end, 1e-6))
    return down * (1.0 - up)


def _episode_time(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.episode_length_buf.to(torch.float32) * env.step_dt


def bow_blend(env: ManagerBasedRlEnv, profile: dict) -> torch.Tensor:
    """(B,) current reference bow amount for every env."""
    return bow_blend_at(_episode_time(env), **profile)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _servo_ids(env: ManagerBasedRlEnv, asset: Entity) -> list:
    cache = env.__dict__.setdefault("_bow_servo_ids_cache", {})
    ids = cache.get(id(asset))
    if ids is None:
        ids, _ = asset.find_joints(r"^(?!passive_).*")
        cache[id(asset)] = ids
    return ids


def _pose_error(
    env: ManagerBasedRlEnv,
    profile: dict,
    bow_delta: dict[int, float],
    joint_indices: Sequence[int],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """(B, k) joint error vs the blended HOME→BOW target, servo-index keyed."""
    asset: Entity = env.scene[asset_cfg.name]
    ids = _servo_ids(env, asset)
    sel = [ids[i] for i in joint_indices]
    default = asset.data.default_joint_pos[:, sel]
    delta = torch.tensor(
        [bow_delta.get(i, 0.0) for i in joint_indices],
        device=env.device,
        dtype=default.dtype,
    ).unsqueeze(0)
    target = default + bow_blend(env, profile).unsqueeze(-1) * delta
    return asset.data.joint_pos[:, sel] - target


def trunk_pitch(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """(B,) trunk pitch in radians from projected gravity. Positive = leaning
    forward, the direction of the bow (verified in sim: folding the hips
    forward makes this grow)."""
    asset: Entity = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    return torch.atan2(g[:, 0], -g[:, 2])


# ── Rewards (positive weight) ────────────────────────────────────────────────


def bow_pose_track(
    env: ManagerBasedRlEnv,
    profile: dict,
    bow_delta: dict[int, float],
    joint_indices: Sequence[int],
    std: float = 0.3,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Gaussian tracking of the moving HOME→BOW→HOME joint target, in [0, 1]."""
    err = _pose_error(env, profile, bow_delta, joint_indices, asset_cfg)
    out = torch.exp(-((err / std) ** 2)).mean(dim=-1)
    return torch.nan_to_num(out, nan=0.0)


def bow_pitch_track(
    env: ManagerBasedRlEnv,
    profile: dict,
    bow_pitch: float,
    stand_pitch: float = 0.0,
    std: float = 0.15,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Gaussian tracking of the reference trunk pitch, in [0, 1].

    Joint tracking alone would let the robot hit the joint targets while tipping
    as a rigid block; this term pins down what the trunk itself should do.
    """
    target = stand_pitch + bow_blend(env, profile) * (bow_pitch - stand_pitch)
    err = trunk_pitch(env, asset_cfg) - target
    return torch.nan_to_num(torch.exp(-((err / std) ** 2)), nan=0.0)


def feet_planted(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Fraction of feet in ground contact, in [0, 1]. A bow never steps."""
    found = env.scene.sensors[sensor_name].data.found > 0  # counts → bool
    if found.dim() > 1:
        return found.float().mean(dim=-1)
    return found.float()


# ── Costs (negative weight) ──────────────────────────────────────────────────


def bow_pose_l1_cost(
    env: ManagerBasedRlEnv,
    profile: dict,
    bow_delta: dict[int, float],
    joint_indices: Sequence[int],
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Mean |joint error| vs the moving target (>= 0). Constant gradient far
    from the target, where the Gaussian above has flattened out."""
    err = _pose_error(env, profile, bow_delta, joint_indices, asset_cfg)
    return torch.nan_to_num(err.abs().mean(dim=-1), nan=0.0)


def base_drift_cost(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Horizontal trunk speed squared (>= 0): bow on the spot, don't shuffle."""
    asset: Entity = env.scene[asset_cfg.name]
    v = asset.data.root_link_lin_vel_w[:, :2]
    return torch.nan_to_num(torch.sum(v * v, dim=-1), nan=0.0)


def trunk_roll_cost(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Sideways lean squared (>= 0): a bow is purely sagittal."""
    asset: Entity = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    return torch.nan_to_num(g[:, 1] ** 2, nan=0.0)


def contact_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """1 when the watched body touches the floor (>= 0). Used for the head:
    a bow that nods into the carpet is a face-plant."""
    found = env.scene.sensors[sensor_name].data.found > 0  # counts → bool
    if found.dim() > 1:
        found = found.any(dim=-1)
    return found.float()


# ── Privileged critic observation ─────────────────────────────────────────────


def bow_clock_obs(env: ManagerBasedRlEnv, profile: dict, period: float) -> torch.Tensor:
    """(B, 3) = [blend, sin, cos] of the hidden clock. CRITIC ONLY.

    Never add this to the actor group: the exported policy must run on the
    runtime's constant command, with no clock.
    """
    t = _episode_time(env)
    w = 2.0 * math.pi * t / period
    return torch.stack((bow_blend_at(t, **profile), torch.sin(w), torch.cos(w)), dim=-1)
