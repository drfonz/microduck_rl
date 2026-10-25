"""MDP terms for the Sleepy Duckling trick (Mjlab-SleepyDuck-Flat-MicroDuck).

Yawn → sit → nod off (each nod deeper, a small jerk as it catches itself) →
asleep with the beak on the floor and the bum in the air. Same teaching method
as PoliteBow and PlayDead, and most of the terms are PlayDead's (keyframed
reference, Gaussian pose/pitch tracking, contact and impact costs, critic-only
clock: see play_dead_mdp.py). This module only adds what PlayDead lacks:

* a termination that works both ways: this duck must not topple onto its back
  (the yawn leans the heavy head back) nor roll on over its face;
* a reward for actually resting the beak on the floor once asleep, so the
  policy lays its head down instead of hovering it a few millimetres up;
* a reverse curriculum: some episodes start partway through the routine
  (seated, between nods, at the start of the slump, slumped on the beak), so
  the slump and the snuggle get practice from the first iteration. Run 1
  learnt the yawn and the sit by iteration 500 but never slumped. The episode
  clock gets a per-env offset (see bow_mdp._episode_time) and a clock-based
  time-out ends those episodes at the routine's end.

Sign convention (same as bow_mdp): every cost returns >= 0 and takes a
NEGATIVE weight; every reward returns [0, 1] and takes a POSITIVE weight.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import play_dead_mdp
from mjlab_microduck.tasks.bow_mdp import _episode_time, _handoff_buffers, _servo_ids, trunk_pitch

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def contact_reward_after(env: ManagerBasedRlEnv, sensor_name: str, t_start: float) -> torch.Tensor:
    """1 while the watched body touches the floor from ``t_start`` on, in [0, 1].

    Gated by the clock, never by state: before ``t_start`` touching pays
    nothing (and the early-contact cost applies), so the pose cannot be
    reached early and farmed.
    """
    found = env.scene.sensors[sensor_name].data.found > 0
    if found.dim() > 1:
        found = found.any(dim=-1)
    return found.float() * (_episode_time(env) >= t_start).float()


def pose_track_window(
    env: ManagerBasedRlEnv,
    keyframes: Sequence[dict],
    joint_indices: Sequence[int],
    std: float = 0.3,
    t_start: float = 0.0,
    t_end: float = math.inf,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """play_dead_mdp.pose_track, paid only while ``t_start <= clock < t_end``.

    The head turn has two jobs that want different stds: the small nod
    counter (tight, so a 0.15 rad step is worth taking) and the snuggle,
    which starts ~1 rad from its target, where a tight Gaussian is flat.
    """
    t = _episode_time(env)
    gate = ((t >= t_start) & (t < t_end)).float()
    return play_dead_mdp.pose_track(env, keyframes, joint_indices, std, asset_cfg) * gate


def tipped_over(
    env: ManagerBasedRlEnv,
    max_roll: float = math.radians(45.0),
    max_back_pitch: float = math.radians(-40.0),
    max_forward_pitch: float = math.radians(85.0),
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """True when the duck has fallen: on a side, onto its back, or over its face.

    The trick ends leaning forward about 55° with the beak on the floor, so the
    usual "fell over" test would end the episode at its goal. These limits
    leave room for that pose and nothing past it.
    """
    asset: Entity = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    roll = torch.asin(g[:, 1].clamp(-1.0, 1.0)).abs()
    pitch = trunk_pitch(env, asset_cfg)
    return (roll > max_roll) | (pitch < max_back_pitch) | (pitch > max_forward_pitch)


# ── Reverse curriculum: start partway through the routine ────────────────────


def clock_offsets(env: ManagerBasedRlEnv) -> torch.Tensor:
    """(B,) per-env episode clock offset in seconds, created on first use."""
    offsets = env.__dict__.get("_clock_offset_s")
    if offsets is None:
        offsets = torch.zeros(env.num_envs, device=env.device)
        env._clock_offset_s = offsets
    return offsets


def reset_mid_routine(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    prob: float = 0.0,
    states: Sequence[dict] = (),
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
    """Reset event, LAST: start a ``prob`` share of the episodes mid-routine.

    Each state is {"t", "pose", "pitch", "z", "weight"}: a keyframe time, its
    servo pose (servos not listed sit at HOME), and the trunk pitch and height
    it settles at (measured with scripts/sleepy_pose_study.py). The duck is
    placed there at rest, keeping the reset's random heading and position,
    its episode clock starts at ``t``, and the pose is handed over as the
    previous action (the gait bank's mechanism), so the first action-rate
    step is measured against it. Every other episode gets a zero offset.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    offsets = clock_offsets(env)
    offsets[env_ids] = 0.0
    if prob <= 0.0 or not states:
        return
    pick = env_ids[torch.rand(len(env_ids), device=env.device) < prob]
    if len(pick) == 0:
        return
    weights = torch.tensor([float(st.get("weight", 1.0)) for st in states], device=env.device)
    which = torch.multinomial(weights, len(pick), replacement=True)

    asset: Entity = env.scene[asset_cfg.name]
    servo = _servo_ids(env, asset)
    cols = torch.tensor([7 + j for j in servo], device=env.device)
    home = asset.data.default_joint_pos[0, servo]
    pending, handoff_action = _handoff_buffers(env)
    qpos, qvel = env.sim.data.qpos, env.sim.data.qvel

    # The reset's heading, from its quaternion (w, x, y, z).
    w, x, y, z = qpos[pick, 3:7].unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    for k, st in enumerate(states):
        mask = which == k
        sel = pick[mask]
        if len(sel) == 0:
            continue
        target = home.clone()
        for j, v in st["pose"].items():
            target[j] = v
        cy, sy = torch.cos(yaw[mask] / 2), torch.sin(yaw[mask] / 2)
        cp, sp = math.cos(st["pitch"] / 2), math.sin(st["pitch"] / 2)
        # q_yaw ⊗ q_pitch: heading first, then the lean in the duck's own frame.
        qpos[sel, 3:7] = torch.stack((cy * cp, -sy * sp, cy * sp, sy * cp), dim=-1)
        qpos[sel, 2] = env.scene.env_origins[sel, 2] + st["z"] + 0.003
        qpos[sel.unsqueeze(1), cols.unsqueeze(0)] = target
        qvel[sel] = 0.0
        handoff_action[sel] = target - home  # action = target − HOME (scale 1)
        pending[sel] = True
        offsets[sel] = st["t"]


def clock_out(env: ManagerBasedRlEnv, t_end: float) -> torch.Tensor:
    """True once the (offset) episode clock reaches ``t_end``: the time-out
    for episodes that started mid-routine. For the others it fires on the same
    step as the stock time-out."""
    return _episode_time(env) >= t_end - 0.5 * env.step_dt
