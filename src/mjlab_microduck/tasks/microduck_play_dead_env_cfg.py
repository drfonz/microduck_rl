"""Microduck play dead: Mjlab-PlayDead-Flat-MicroDuck.

Episodic, comic trick for "finger gun ... bang!": from a still stand the duck
sits, lowers itself onto its back with its legs in the air, turns its head to
one side, and stays there, dead. Publishable as a community policy:

    uv run publish --task Mjlab-PlayDead-Flat-MicroDuck ... \
        --kind episodic --duration-s 5.0

The episode ENDS LYING DOWN. When the runtime hands control back after
``duration_s`` the duck is on its back, so the next policy must be one that
can get up from there (StandUp trains face-up recovery). Rehearse the
hand-off in scripts/infer_policy.py before trying it on the robot.

How it is taught (see play_dead_mdp.py for the rationale): the same method as
PoliteBow. A hidden reference clock drives a keyframed target (joints + trunk
pitch); the critic sees the clock, the actor does not; no terminal bonuses.

The beak is NOT part of this policy. The jaw has no servo in the 14-joint
action space (in the sim it is fixed to the head), so opening it has to be
done by the runtime, outside the policy, if the hardware allows it.

Everything else (BAM actuators, DR, obs noise/delays, 61-D obs layout, NaN
guard) is inherited from the roulade env, like PoliteBow.

Tuning knobs you will most likely touch are the CONSTANTS block below.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import bow_mdp, play_dead_mdp
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import make_microduck_roulade_env_cfg
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# ── CONSTANTS ────────────────────────────────────────────────────────────────

# The head turns to ONE side, so the trick is not left/right symmetric and the
# mirror loss would fight it (AGENTS.md: never for asymmetric tasks).
ENABLE_SYMMETRY = False

# Servo layout: 0-4 left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle),
# 5-8 neck/head (neck_pitch, head_pitch, head_yaw, head_roll), 9-13 right leg.
# Left and right legs have mirrored signs.

# Sit: the SitStand env's stability-verified keyframe (settles at 3-5° tilt).
# Keep in sync with microduck_sitstand_env_cfg.SITTING_TARGET_OVERRIDES.
SIT_POSE: dict[int, float] = {
    1: 0.0, 2: -0.4079, 3: 1.35, 4: 0.0,
    10: 0.0, 11: 0.4079, 12: -1.35, 13: 0.0,
}
# On the back with the legs in the air ("dead bug"). Checked in sim on the
# groundcontact model (29 Sep 2026): dropped supine at this pose it settles
# flat on its back (pitch −91°, roll 0°) and stays put for 3 s. Folding the
# hips the other way (+1.0 left) rolls it up onto its head instead.
SUPINE_POSE: dict[int, float] = {
    1: 0.0, 2: -1.2, 3: 0.5, 4: 0.0,
    10: 0.0, 11: 1.2, 12: -0.5, 13: 0.0,
}
# Head turned to the side, resting on its cheek. +1.4 rad of head_yaw (−1.4
# for the other side) lays the head on its side when the trunk is supine; the
# pose stays flat (roll 0°) in sim. Neck and head pitch stay at HOME.
HEAD_YAW_DEAD = 1.4
DEAD_POSE: dict[int, float] = {**SUPINE_POSE, 7: HEAD_YAW_DEAD}

LYING_PITCH = math.radians(-90.0)  # trunk_pitch() on the back (measured −90.2°)

# Reference timeline (seconds from episode start). Each keyframe blends into
# the next with a smoothstep. The pause at the sit is kept short: a still pose
# is ambiguous for an actor that cannot see a clock (PoliteBow lesson).
KEYFRAMES: list[dict] = [
    {"t": 0.0, "pose": {}, "pitch": 0.0},                # stand (HOME)
    {"t": 1.2, "pose": SIT_POSE, "pitch": 0.0},          # sat down
    {"t": 1.5, "pose": SIT_POSE, "pitch": 0.0},          # a beat
    {"t": 2.9, "pose": SUPINE_POSE, "pitch": LYING_PITCH},  # lowered onto its back
    {"t": 3.5, "pose": DEAD_POSE, "pitch": LYING_PITCH},    # head flops to the side
    {"t": 5.0, "pose": DEAD_POSE, "pitch": LYING_PITCH},    # stay dead
]
EPISODE_LENGTH_S = KEYFRAMES[-1]["t"]  # 5.0 s → --duration-s 5.0
LIE_DOWN_START = KEYFRAMES[2]["t"]  # feet may leave, head may touch, from here

# "Bang!" usually comes mid-walk. PoliteBow v1 fell whenever it was triggered
# while not perfectly still, so, like PoliteBow v2, half of the training
# episodes start from a real mid-walk state (with the walker's last action
# handed over). Same bank as the bow: same base env and robot model. It is
# recorded on the GPU box by scripts/make_gait_bank.py; data/ is gitignored.
GAIT_BANK_PATH = "data/polite_bow_gait_bank.pt"
GAIT_BANK_PROB = 0.5
STAND_JOINT_NOISE_STD = 0.03  # rad, servo noise on still-stand spawns

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 8]  # neck_pitch, head_pitch, head_roll
_HEAD_YAW = [7]


def make_microduck_play_dead_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the PlayDead environment configuration."""

    cfg = make_microduck_roulade_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Drop the roulade task: rewards, curricula ───────────────────────────
    for name in [n for n in list(cfg.rewards.keys()) if n.startswith("roulade_")]:
        del cfg.rewards[name]
    for name in ("roulade_spawn_mix", "arrival_damping_weight", "gentle_landing_weight"):
        cfg.curriculum.pop(name, None)
    cfg.rewards.pop("arrival_damping", None)

    # Spawn: always a still stand (roulade reset, mid-roll spawns off).
    spawn = cfg.events["set_roulade_state"].params
    spawn["standing_prob"] = 1.0
    spawn["midroll_prob"] = 0.0
    spawn["forward_vel_range"] = (0.0, 0.0)
    spawn["standing_tilt_max"] = math.radians(3.0)
    spawn["joint_noise_std"] = 0.03
    # Mid-walk spawns + servo noise on still stands. Must run AFTER
    # set_roulade_state (event dicts apply in insertion order).
    cfg.events["gait_bank_spawn"] = EventTermCfg(
        func=bow_mdp.reset_from_gait_bank,
        mode="reset",
        params={
            "bank_path": GAIT_BANK_PATH,
            "prob": 0.0 if play else GAIT_BANK_PROB,
            "stand_joint_noise_std": STAND_JOINT_NOISE_STD,
        },
    )
    # The runtime keeps the previous policy's last action in the observation
    # at a hand-off; mirror that for the walk spawns (actor AND critic).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["actions"].func = bow_mdp.last_action_with_handoff

    kf = {"keyframes": KEYFRAMES}

    # ── Task rewards ────────────────────────────────────────────────────────
    # Main term: the trunk goes from upright to on its back at the pace of the
    # reference. This is what turns "fall backwards" into "lie down".
    cfg.rewards["play_dead_pitch"] = RewardTermCfg(
        func=play_dead_mdp.pitch_track, weight=4.0, params={**kf, "std": 0.25}
    )
    # Legs: sit, then legs in the air. They ARE the shape of the trick, so
    # unlike the bow they are tracked, with a generous std.
    cfg.rewards["play_dead_legs"] = RewardTermCfg(
        func=play_dead_mdp.pose_track,
        weight=2.0,
        params={**kf, "joint_indices": _LEG_JOINTS, "std": 0.4},
    )
    cfg.rewards["play_dead_legs_l1_cost"] = RewardTermCfg(
        func=play_dead_mdp.pose_l1_cost,
        weight=-0.5,
        params={**kf, "joint_indices": _LEG_JOINTS},
    )
    # Head turn: the punchline, so it gets its own term.
    cfg.rewards["play_dead_head_yaw"] = RewardTermCfg(
        func=play_dead_mdp.pose_track,
        weight=2.0,
        params={**kf, "joint_indices": _HEAD_YAW, "std": 0.3},
    )
    # Neck: loose. During the lie-down the policy may need to curl the neck
    # forward to keep the heavy head from swinging back into the floor.
    cfg.rewards["play_dead_neck"] = RewardTermCfg(
        func=play_dead_mdp.pose_track,
        weight=1.0,
        params={**kf, "joint_indices": _NECK_JOINTS, "std": 0.5},
    )
    cfg.rewards["feet_planted"] = RewardTermCfg(
        func=play_dead_mdp.feet_planted_until,
        weight=1.0,
        params={"sensor_name": "feet_ground_contact", "t_end": LIE_DOWN_START},
    )

    # ── Style and safety costs ─────────────────────────────────────────────
    cfg.rewards["play_dead_drift_cost"] = RewardTermCfg(
        func=bow_mdp.base_drift_cost, weight=-1.0
    )
    # Straight back, belly up: no rolling onto a side (g_y is 0 both standing
    # and lying on the back, so one term covers the whole trick).
    cfg.rewards["play_dead_roll_cost"] = RewardTermCfg(
        func=bow_mdp.trunk_roll_cost, weight=-2.0
    )
    cfg.rewards["play_dead_early_head_contact_cost"] = RewardTermCfg(
        func=play_dead_mdp.contact_cost_until,
        weight=-2.0,
        params={"sensor_name": "head_ground_contact", "t_end": LIE_DOWN_START},
    )
    cfg.rewards["play_dead_head_impact_cost"] = RewardTermCfg(
        func=play_dead_mdp.head_impact_cost,
        weight=-10.0,  # a 1 m/s slam ≈ 10 per step; the head turn itself (≈0.15 m/s) ≈ 0.2
        params={"sensor_name": "head_ground_contact"},
    )
    # gentle_landing (self-negating |a_z| on the trunk, POSITIVE weight) is
    # inherited from the roulade: it prices the bump onto the bum and back.
    cfg.rewards["gentle_landing"].weight = 0.005
    cfg.rewards["action_rate_l2"].weight = -0.2
    cfg.rewards["self_collisions"].weight = -1.0
    # Lying down is a 90° rotation in ~1.4 s: keep the motion-blockers low.
    cfg.rewards["body_ang_vel"].weight = -0.005
    cfg.rewards["angular_momentum"].weight = -0.005

    # ── Terminations ─────────────────────────────────────────────────────────
    # Lying on the back IS the task, so no generic "fell over". Ending up on a
    # side or on the face ends the episode instead.
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["wrong_side_down"] = TerminationTermCfg(
        func=play_dead_mdp.wrong_side_down,
        params={"max_roll": math.radians(60.0), "max_forward_pitch": math.radians(50.0)},
    )

    # ── Critic-only privileged clock ─────────────────────────────────────────
    cfg.observations["critic"].terms["play_dead_clock"] = ObservationTermCfg(
        func=play_dead_mdp.play_dead_clock_obs,
        params={**kf, "period": EPISODE_LENGTH_S},
    )

    # ── Curriculum: polish after discovery ───────────────────────────────────
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.2},
                {"step": 1000 * 24, "weight": -0.4},
                {"step": 2000 * 24, "weight": -0.6},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 1000 * 24, "weight": -5e-4},
                {"step": 2000 * 24, "weight": -1e-3},
            ],
        },
    )

    return cfg


MicroduckPlayDeadRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # baked into the ONNX by scripts/export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.8,  # between the bow (0.6) and the roulade (1.0)
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
        symmetry_cfg=None,  # asymmetric trick, see ENABLE_SYMMETRY
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_play_dead",
    run_name="microduck_play_dead",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=3_000,
)
