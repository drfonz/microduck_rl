"""MDP terms for the Sleepy Duckling trick (Mjlab-SleepyDuck-Flat-MicroDuck).

Yawn → sit → nod off (each nod deeper, a small jerk as it catches itself) →
asleep with the beak on the floor and the bum in the air. Same teaching method
as PoliteBow and PlayDead, and most of the terms are PlayDead's (keyframed
reference, Gaussian pose/pitch tracking, contact and impact costs, critic-only
clock: see play_dead_mdp.py). This module only adds what PlayDead lacks:

* a termination that works both ways: this duck must not topple onto its back
  (the yawn leans the heavy head back) nor roll on over its face;
* a reward for actually resting the beak on the floor once asleep, so the
  policy lays its head down instead of hovering it a few millimetres up.

Sign convention (same as bow_mdp): every cost returns >= 0 and takes a
NEGATIVE weight; every reward returns [0, 1] and takes a POSITIVE weight.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks.bow_mdp import _episode_time, trunk_pitch

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
