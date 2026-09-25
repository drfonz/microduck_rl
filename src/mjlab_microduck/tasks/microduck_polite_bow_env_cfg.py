"""Microduck polite bow: Mjlab-PoliteBow-Flat-MicroDuck.

Episodic, expressive trick: from a still stand the duck bows (trunk pitches
forward, neck and head dip), holds for a beat, and comes back to a still stand,
feet planted the whole time. Publishable as a community policy:

    uv run publish --task Mjlab-PoliteBow-Flat-MicroDuck ... \
        --kind episodic --duration-s 3.5

How it is taught (see bow_mdp.py for the full rationale):
  • hidden reference clock → smooth HOME→BOW→HOME joint + trunk-pitch targets;
  • the actor never sees the clock (constant-command deployment contract);
    the critic does (privileged ``bow_clock`` obs, not exported);
  • no terminal bonus anywhere (no-jackpot rule): tracking a moving target
    makes slow and smooth the argmax.

Everything else (BAM actuators, DR, obs noise/delays, 61-D obs layout,
symmetry loss, NaN guard) is inherited from the roulade env, which itself
mirrors standup/velocity for sim2real parity. Building on it keeps those stacks
in sync for free (AGENTS.md, "pick the closest template").

Tuning knobs you will most likely touch are the CONSTANTS block below.
How BOW_PITCH and BOW_DELTA were chosen is documented next to them; after
training, judge the result with scripts/bow_eval.py AND the video, not the
reward curve alone (AGENTS.md: measure before theorising).
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.envs.mdp import terminations as mjlab_terminations
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import bow_mdp
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import make_microduck_roulade_env_cfg
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

# ── CONSTANTS ────────────────────────────────────────────────────────────────

ENABLE_SYMMETRY = True  # the bow is left/right symmetric

# Reference timeline (seconds from episode start). Episode = UP_END + tail.
BOW_PROFILE = dict(
    down_start=0.0,  # start bowing straight away: the actor cannot see a clock,
    down_end=1.2,  #   so an initial "wait" would be unlearnable anyway
    hold_end=1.6,  # short hold: a still bottom pose is ambiguous without a clock
    up_end=2.9,
)
TAIL_S = 0.6  # still stand at the end, scored like the start
EPISODE_LENGTH_S = BOW_PROFILE["up_end"] + TAIL_S  # 3.5 s → --duration-s 3.5

# What the bow IS, in two parts:
#  1. the trunk pitches forward by BOW_PITCH (task space: the policy discovers
#     the leg configuration that keeps the CoM over the feet (hips back,
#     ankles in), which is exactly the part that is hard to hand-design);
#  2. the neck and head dip by BOW_DELTA (joint space: the expressive bit).
# Measured in sim before choosing these numbers (see docs/POLITE_BOW.md):
#  * open-loop holding of ANY pose, even HOME, falls within ~2 s under BAM +
#    actuator delay, so equilibrium must be judged closed-loop;
#  * the official StandUp policy is trained to track body-pitch commands of
#    ±15°, so 18° is a modest step beyond a proven envelope;
#  * the head is ~38% of the body mass: dipping it moves the CoM forward a lot,
#    so keep the neck/head dip moderate or the task becomes "don't faceplant".
BOW_PITCH = math.radians(18.0)  # trunk pitch at the bottom of the bow
STAND_PITCH = 0.0  # trunk pitch at HOME (measured: −0.3°)
# Servo-index keyed offsets from HOME at the bottom of the bow.
# Layout: 0-4 left leg, 5-8 neck/head (neck_pitch, head_pitch, head_yaw,
# head_roll), 9-13 right leg. Legs are deliberately absent (see 1. above).
BOW_DELTA: dict[int, float] = {
    5: 0.35,  # neck_pitch  (dip the neck)
    6: 0.30,  # head_pitch  (look down)
}

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]


def make_microduck_polite_bow_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the PoliteBow environment configuration."""

    cfg = make_microduck_roulade_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Drop the roulade task: rewards, curricula ───────────────────────────
    for name in [n for n in list(cfg.rewards.keys()) if n.startswith("roulade_")]:
        del cfg.rewards[name]
    for name in ("roulade_spawn_mix", "arrival_damping_weight", "gentle_landing_weight"):
        cfg.curriculum.pop(name, None)
    cfg.rewards.pop("arrival_damping", None)

    # Spawn: always a still stand (reuse the roulade reset with the mid-roll
    # reverse curriculum switched off).
    spawn = cfg.events["set_roulade_state"].params
    spawn["standing_prob"] = 1.0
    spawn["midroll_prob"] = 0.0
    spawn["forward_vel_range"] = (0.0, 0.0)
    spawn["standing_tilt_max"] = math.radians(3.0)
    spawn["joint_noise_std"] = 0.03

    # ── Task rewards ────────────────────────────────────────────────────────
    common = {"profile": BOW_PROFILE, "bow_delta": BOW_DELTA}
    # Main term: the trunk follows the reference pitch.
    cfg.rewards["bow_pitch"] = RewardTermCfg(
        func=bow_mdp.bow_pitch_track,
        weight=4.0,
        params={
            "profile": BOW_PROFILE,
            "bow_pitch": BOW_PITCH,
            "stand_pitch": STAND_PITCH,
            "std": 0.15,
        },
    )
    # Expressive term: neck/head follow the reference dip.
    cfg.rewards["bow_neck_pose"] = RewardTermCfg(
        func=bow_mdp.bow_pose_track,
        weight=3.0,
        params={**common, "joint_indices": _NECK_JOINTS, "std": 0.2},
    )
    cfg.rewards["bow_neck_l1_cost"] = RewardTermCfg(
        func=bow_mdp.bow_pose_l1_cost,
        weight=-1.0,
        params={**common, "joint_indices": _NECK_JOINTS},
    )
    # Legs: a WEAK, generous pull towards HOME (BOW_DELTA has no leg entries),
    # a regulariser that returns the legs to the stand at the end without
    # dictating how they fold during the bow.
    cfg.rewards["bow_leg_home"] = RewardTermCfg(
        func=bow_mdp.bow_pose_track,
        weight=1.0,
        params={**common, "joint_indices": _LEG_JOINTS, "std": 0.5},
    )
    cfg.rewards["feet_planted"] = RewardTermCfg(
        func=bow_mdp.feet_planted,
        weight=1.0,
        params={"sensor_name": "feet_ground_contact"},
    )

    # ── Style and safety costs ─────────────────────────────────────────────
    cfg.rewards["bow_drift_cost"] = RewardTermCfg(func=bow_mdp.base_drift_cost, weight=-2.0)
    cfg.rewards["bow_roll_cost"] = RewardTermCfg(func=bow_mdp.trunk_roll_cost, weight=-2.0)
    cfg.rewards["bow_head_contact_cost"] = RewardTermCfg(
        func=bow_mdp.contact_cost,
        weight=-2.0,
        params={"sensor_name": "head_ground_contact"},
    )
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,  # returns >= 0 (cost) → negative weight
        weight=-1.0,
    )
    # Smoothness from step 0 (slow careful tasks want heavier smoothness than
    # walking, AGENTS.md), tightened by curriculum once the bow exists.
    cfg.rewards["action_rate_l2"].weight = -0.2
    cfg.rewards["self_collisions"].weight = -1.0
    cfg.rewards["body_ang_vel"].weight = -0.02
    cfg.rewards["angular_momentum"].weight = -0.01
    # gentle_landing (self-negating |a_z|, POSITIVE weight) stays from roulade.

    # ── Terminations: falling over ends the episode (loses future reward) ──
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mjlab_terminations.bad_orientation,
        params={"limit_angle": math.radians(60.0)},
    )

    # ── Critic-only privileged clock ─────────────────────────────────────────
    cfg.observations["critic"].terms["bow_clock"] = ObservationTermCfg(
        func=bow_mdp.bow_clock_obs,
        params={"profile": BOW_PROFILE, "period": EPISODE_LENGTH_S},
    )

    # ── Curriculum: polish after discovery ───────────────────────────────────
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.2},
                {"step": 800 * 24, "weight": -0.4},
                {"step": 1600 * 24, "weight": -0.6},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 800 * 24, "weight": -5e-4},
                {"step": 1600 * 24, "weight": -1e-3},
            ],
        },
    )

    return cfg


MicroduckPoliteBowRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # baked into the ONNX by scripts/export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.6,  # smaller than roulade's 1.0: a bow is a small motion
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_polite_bow",
    run_name="microduck_polite_bow",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=3_000,  # simple episodic tricks ≈ 1000-3000 iters at 4096 envs
)
