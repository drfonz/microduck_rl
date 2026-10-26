"""Microduck Sleepy Duckling: Mjlab-SleepyDuck-Flat-MicroDuck.

Episodic, cute trick: from a stand (or mid-walk) the duck yawns, settles into
a sit and starts to nod off. Each nod goes a little deeper, with a small jerk
as it catches itself, until it falls asleep with its beak on the floor and its
bum in the air; a beat later it snuggles, rolling its head onto its cheek.
Publishable as a community policy:

    uv run publish --task Mjlab-SleepyDuck-Flat-MicroDuck ... \\
        --kind episodic --duration-s 10.5

The episode ENDS ASLEEP and the pose is static, so the runtime can keep this
policy running for as long as the duck should sleep. Whatever wakes it (the
stand policy) starts from a crouch with the head on the floor.

How it is taught: the PlayDead method (see play_dead_mdp.py and
docs/SLEEPY_DUCK.md). A hidden reference clock drives a keyframed target
(joints + trunk pitch); the critic sees the clock, the actor does not; no
terminal bonuses. One run in three stages (see STAGE_CATCHES): the slump, then
the catches, then the snuggle.

The beak is NOT part of this policy: the jaw has no servo in the 14-joint
action space (in the sim it is fixed to the head). The yawn's open beak and
the snoring (the beak gently opening and closing once asleep) are the
runtime's job, like PlayDead's beak.

Everything else (BAM actuators, DR, obs noise/delays, 61-D obs layout, NaN
guard) is inherited from the roulade env, like PoliteBow and PlayDead.

Tuning knobs you will most likely touch are the CONSTANTS block below. The
poses come from scripts/sleepy_pose_study.py, which drops the duck into each
one and checks where it settles.
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

from mjlab_microduck.tasks import bow_mdp, play_dead_mdp, sleepy_duck_mdp
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import make_microduck_roulade_env_cfg
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# ── CONSTANTS ────────────────────────────────────────────────────────────────

# The head snuggles onto ONE cheek, so the trick is not left/right symmetric
# and the mirror loss would fight it (AGENTS.md: never for asymmetric tasks).
ENABLE_SYMMETRY = False

# Servo layout: 0-4 left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle),
# 5-8 neck/head (neck_pitch, head_pitch, head_yaw, head_roll), 9-13 right leg.
# Left and right legs have mirrored signs. HOME neck/head pitch = 0.3491.
#
# Signs, measured with scripts/sleepy_pose_study.py (24 Oct 2026):
#   neck_pitch −  head forward and down;  +  head back (tips the duck backwards)
#   head_pitch +  nose down (the nod);    −  nose up (the yawn)
# The head's nose-down angle is about −(neck − 0.349) + (head − 0.349) + trunk
# pitch. Every sitting pose below settles in sim with both feet down.

# Sit: the SitStand env's stability-verified keyframe (a crouch, really).
# Keep in sync with microduck_sitstand_env_cfg.SITTING_TARGET_OVERRIDES.
# In it each ankle bracket rests against its hip (self-contact): that contact
# is part of why it is so stable, and it is harmless (a resting touch, like
# sitting on your heels). A search for a contact-free sit (25 Oct 2026) found
# none that stays put through the nods: with the head forward the servos give
# a little under load and the ankle meets the hip anyway. So the sit stays and
# the self-collision cost is turned down instead (see below).
SIT_POSE: dict[int, float] = {
    1: 0.0, 2: -0.4079, 3: 1.35, 4: 0.0,
    10: 0.0, 11: 0.4079, 12: -1.35, 13: 0.0,
}


def _head(neck: float, head: float, yaw: float = 0.0, roll: float = 0.0) -> dict[int, float]:
    return {5: neck, 6: head, 7: yaw, 8: roll}


# Yawn, standing: nose up about 37°, while the runtime opens the beak. Not
# more: neck +0.55 / head −0.45 tips even the sitting duck onto its back.
YAWN_HEAD = _head(0.45, -0.2)
# Three nods. Each droops deeper (nose down 29°, 43°, 63°) and each catch
# recovers less (4°, 21°, 40°); the droops are slow, the catches are the jerks
# (0.3 s: run 2's 0.2 s jerks were never attempted; a clockless actor has to
# time them from its own pose, and a wider feature forgives small timing
# errors). The actor has no clock, and run 1 showed that depth alone does not
# tell the nods apart: it averaged them into one slow droop. So each nod also
# turns the head a little further towards its sleeping side (yaw 9°, 17°,
# 26°, roll 7°, 14°, 21°) and the catches keep that turn: a posture "counter"
# the actor can read. The snuggle then finishes the turn.
# Nod 1 is the biggest jerk of the three (29° → 4°): run 3 learnt nods 2 and 3
# but skipped the first, then the smallest (23° → 9°).
NOD_1 = _head(0.3491, 0.85, 0.15, 0.12)
CATCH_1 = _head(0.3491, 0.42, 0.15, 0.12)
NOD_2 = _head(0.25, 1.0, 0.30, 0.24)
CATCH_2 = _head(0.28, 0.65, 0.30, 0.24)
NOD_3 = _head(0.1, 1.2, 0.45, 0.36)
CATCH_3 = _head(0.15, 0.85, 0.45, 0.36)
# Asleep on its beak: the body slumps forward (trunk ≈ 63°, bum highest) and
# the beak (the orange jaw under the head) lands on the floor, the head still
# turned as after nod 3 (pose study "beak_turn_a": 64°, beak and feet down).
BEAK_DOWN_HEAD = _head(-0.9, -0.9, 0.45, 0.36)
BEAK_DOWN_PITCH = math.radians(63.0)
# Snuggled: head_yaw rolls the head onto its cheek, face turned to the left,
# the way an animal settles to sleep. Settles flat on the cheek in sim (pose
# study "side_5").
HEAD_YAW_SNUGGLE = 1.4
SNUGGLE_HEAD = _head(-0.9, -0.4, HEAD_YAW_SNUGGLE)
SNUGGLE_PITCH = math.radians(60.0)  # settled at 60–61° in the pose study

ASLEEP_BEAK_POSE: dict[int, float] = {**SIT_POSE, **BEAK_DOWN_HEAD}
ASLEEP_POSE: dict[int, float] = {**SIT_POSE, **SNUGGLE_HEAD}

# Trunk height above the floor (m), measured in sim: standing at HOME 115 mm (evaluation, t = 0,
# 26 Oct 2026); the settled sit 61 mm, asleep on the beak 66 mm and snuggled 68 mm (pose study,
# as SPAWN_STATES below). The sit is what tells sitting from standing, which the leg pose term
# barely does (runs 5 and 8 nodded on their feet).
STAND_Z, SIT_Z, BEAK_Z, SNUGGLE_Z = 0.115, 0.061, 0.066, 0.068

# Reference timeline (seconds from episode start). Each keyframe blends into
# the next with a smoothstep. Still holds are kept short (≤ 0.3 s): a still
# pose is ambiguous for an actor that cannot see a clock (PoliteBow lesson).
KEYFRAMES: list[dict] = [
    {"t": 0.0, "pose": {}, "pitch": 0.0, "z": STAND_Z},                         # standing (HOME)
    {"t": 0.7, "pose": YAWN_HEAD, "pitch": 0.0, "z": STAND_Z},                  # yawn: nose up
    {"t": 1.0, "pose": YAWN_HEAD, "pitch": 0.0, "z": STAND_Z},                  # ...hold the yawn
    {"t": 2.3, "pose": SIT_POSE, "pitch": 0.0, "z": SIT_Z},                     # sat down, head level
    {"t": 3.1, "pose": {**SIT_POSE, **NOD_1}, "pitch": 0.0, "z": SIT_Z},        # nod 1: droop
    {"t": 3.4, "pose": {**SIT_POSE, **CATCH_1}, "pitch": 0.0, "z": SIT_Z},      # ...jerk!
    {"t": 4.3, "pose": {**SIT_POSE, **NOD_2}, "pitch": 0.04, "z": SIT_Z},       # nod 2: deeper
    {"t": 4.6, "pose": {**SIT_POSE, **CATCH_2}, "pitch": 0.02, "z": SIT_Z},     # ...jerk
    {"t": 5.6, "pose": {**SIT_POSE, **NOD_3}, "pitch": 0.08, "z": SIT_Z},       # nod 3: deeper still
    {"t": 5.9, "pose": {**SIT_POSE, **CATCH_3}, "pitch": 0.04, "z": SIT_Z},     # ...a weak jerk
    {"t": 7.5, "pose": ASLEEP_BEAK_POSE, "pitch": BEAK_DOWN_PITCH, "z": BEAK_Z},  # slumps onto its beak
    {"t": 7.8, "pose": ASLEEP_BEAK_POSE, "pitch": BEAK_DOWN_PITCH, "z": BEAK_Z},  # a beat
    {"t": 8.7, "pose": ASLEEP_POSE, "pitch": SNUGGLE_PITCH, "z": SNUGGLE_Z},    # snuggles onto its cheek
    {"t": 10.5, "pose": ASLEEP_POSE, "pitch": SNUGGLE_PITCH, "z": SNUGGLE_Z},   # fast asleep
]
EPISODE_LENGTH_S = KEYFRAMES[-1]["t"]  # 10.5 s → --duration-s 10.5
SAT_DOWN = KEYFRAMES[3]["t"]  # 2.3 s: sat, before the first nod
SLUMP_START = KEYFRAMES[9]["t"]  # 5.9 s: the final droop begins
# The head may touch the floor from here (the beak lands at ~7.5 s); before
# this, touching down is a face-plant.
HEAD_DOWN_ALLOWED = 6.5  # run 3 landed ~0.3 s early (harmless); the nods end at 5.9 s
BEAK_LANDED = KEYFRAMES[10]["t"]  # 7.5 s: resting the head on the floor pays from here
SNUGGLE_START = KEYFRAMES[11]["t"]  # 7.8 s: the head turn's second job begins


def _kf(t: float) -> dict:
    return next(k for k in KEYFRAMES if k["t"] == t)


# Mid-routine starts (see MID_ROUTINE_PROB below): keyframe, settled trunk
# pitch and height (pose study, 25 Oct 2026), and how often to pick it. Run 3
# adds each nod's lowest point: run 2 averaged the nods and so never reached a
# droop's bottom, where the catch has to happen. And the snuggled pose, so the
# critic learns early how good it is to be there.
SPAWN_STATES: list[dict] = [
    {"t": 2.3, "pose": _kf(2.3)["pose"], "pitch": math.radians(-0.5), "z": 0.0609, "weight": 1.0},  # just sat
    {"t": 3.1, "pose": _kf(3.1)["pose"], "pitch": math.radians(1.2), "z": 0.0612, "weight": 1.0},   # bottom of nod 1
    {"t": 4.3, "pose": _kf(4.3)["pose"], "pitch": math.radians(2.8), "z": 0.0611, "weight": 1.0},   # bottom of nod 2
    {"t": 5.6, "pose": _kf(5.6)["pose"], "pitch": math.radians(4.5), "z": 0.0611, "weight": 1.0},   # bottom of nod 3
    {"t": 5.9, "pose": _kf(5.9)["pose"], "pitch": math.radians(5.2), "z": 0.0608, "weight": 1.5},   # slump starts
    {"t": 7.5, "pose": _kf(7.5)["pose"], "pitch": math.radians(63.3), "z": 0.0658, "weight": 1.5},  # on its beak
    {"t": 8.7, "pose": _kf(8.7)["pose"], "pitch": math.radians(60.0), "z": 0.0678, "weight": 0.5},  # snuggled
]

# "Sleepy time" can come mid-walk. As for PoliteBow v2 and PlayDead, a share
# of the training episodes start from a real mid-walk state (with the
# walker's last action handed over). Same bank: same base env and robot model.
# It is recorded on the GPU box by scripts/make_gait_bank.py; data/ is
# gitignored.
GAIT_BANK_PATH = "data/polite_bow_gait_bank.pt"
GAIT_BANK_PROB = 0.3
STAND_JOINT_NOISE_STD = 0.03  # rad, servo noise on still-stand spawns

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_HEAD_PITCH_JOINTS = [5, 6]  # neck_pitch, head_pitch: the yawn and the nods
_HEAD_TURN_JOINTS = [7, 8]  # head_yaw, head_roll: the nod counter, then the snuggle

# Reverse curriculum (run 2): a share of the episodes start mid-routine, at
# rest in a keyframe pose, with the episode clock set to that keyframe. Run 1
# never slumped by iteration 500: the slump and the snuggle got no practice.
# Trunk pitch and height are where each pose settles (scripts/sleepy_pose_study.py,
# candidates kf_*s). Weighted towards the slump, the hardest part.
MID_ROUTINE_PROB = 0.4

# One run, three stages (run 8). Runs 6 and 7 trained the finished config from
# scratch and never learnt the nods: the duck slumps and snuggles but averages
# the three nods into one slow droop. The staged runs that did learn them
# (2 → 3 → 4 → 5, each fine-tuning the last) taught one thing at a time: the
# slump, then the catches, then the snuggle. Each fine-tune also restarted the
# torque-rate penalty at zero and the centre-of-mass randomisation small, so
# the catches were found with no tax on quick torque changes. These stages
# replay that inside one run (in iterations).
STAGE_CATCHES = 1500  # tight head tracking, rounds starting at the bottom of a nod
STAGE_SNUGGLE = 3500  # the snuggle terms, and rounds starting snuggled (runs 3 + 4: 2,000)
MAX_ITERATIONS = 5000  # the snuggle stage as long as run 5 (1,500)

_SPAWN_AT = {st["t"]: st for st in SPAWN_STATES}
# The slump: run 2's mix (no nod bottoms yet; they come with the catches).
SPAWN_SLUMP: list[dict] = [_SPAWN_AT[2.3], {**_SPAWN_AT[5.9], "weight": 2.0}, {**_SPAWN_AT[7.5], "weight": 2.0}]
# The catches: run 3's mix, without the snuggled pose.
SPAWN_CATCHES: list[dict] = [st for st in SPAWN_STATES if st["t"] < SNUGGLE_START]


def make_microduck_sleepy_duck_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the SleepyDuck environment configuration."""

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
    # Mid-routine starts (reverse curriculum). LAST reset event: it overrides
    # whatever the stand/walk spawns chose for the envs it picks.
    cfg.events["mid_routine_spawn"] = EventTermCfg(
        func=sleepy_duck_mdp.reset_mid_routine,
        mode="reset",
        params={"prob": 0.0 if play else MID_ROUTINE_PROB, "states": SPAWN_SLUMP},  # staged below
    )
    # The runtime keeps the previous policy's last action in the observation
    # at a hand-off; mirror that for the walk spawns (actor AND critic).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["actions"].func = bow_mdp.last_action_with_handoff

    kf = {"keyframes": KEYFRAMES}

    # ── Task rewards ────────────────────────────────────────────────────────
    # The head is the performance: the yawn and the three nods. Tracked
    # tighter than the legs, so a 0.3 rad nod is clearly worth doing. Run 2
    # (std 0.25, weight 3) settled for averaging the nods; tighter and heavier
    # makes the average cost more.
    # The nods only pay while sitting (see seated_gate): runs 5 and 8 nodded on their feet.
    seated = {"seated_start": SAT_DOWN, "seated_end": SLUMP_START, "height_std": 0.02}
    cfg.rewards["sleepy_head"] = RewardTermCfg(
        func=sleepy_duck_mdp.pose_track_seated,
        weight=4.0,
        params={**kf, "joint_indices": _HEAD_PITCH_JOINTS, "std": 0.2, **seated},
    )
    # Run 2's looser head tracking, for the first stage: follow the head's
    # overall path while the slump is learnt, before the nods are asked for.
    cfg.rewards["sleepy_head_coarse"] = RewardTermCfg(
        func=sleepy_duck_mdp.pose_track_seated,
        weight=3.0,
        params={**kf, "joint_indices": _HEAD_PITCH_JOINTS, "std": 0.25, **seated},
    )
    cfg.rewards["sleepy_head_l1_cost"] = RewardTermCfg(
        func=play_dead_mdp.pose_l1_cost,
        weight=-0.5,
        params={**kf, "joint_indices": _HEAD_PITCH_JOINTS},
    )
    # Trunk pitch: upright through the nods, then the slump that lifts the
    # bum. This is what makes the slump a lowering instead of a fall.
    cfg.rewards["sleepy_pitch"] = RewardTermCfg(
        func=play_dead_mdp.pitch_track, weight=3.0, params={**kf, "std": 0.25}
    )
    # Trunk height: stand for the yawn, sit down (5 cm lower) before the nods, then the slump.
    # Run 8 sat only as it fell asleep, at 6.2 s; run 5 halfway through the nods.
    cfg.rewards["sleepy_height"] = RewardTermCfg(
        func=sleepy_duck_mdp.height_track, weight=3.0, params={**kf, "std": 0.015}
    )
    # Legs: stand, then sit, then stay sat. Generous std, as in PlayDead.
    cfg.rewards["sleepy_legs"] = RewardTermCfg(
        func=play_dead_mdp.pose_track,
        weight=2.0,
        params={**kf, "joint_indices": _LEG_JOINTS, "std": 0.4},
    )
    cfg.rewards["sleepy_legs_l1_cost"] = RewardTermCfg(
        func=play_dead_mdp.pose_l1_cost,
        weight=-0.5,
        params={**kf, "joint_indices": _LEG_JOINTS},
    )
    # The head turn, job 1: a little more with each nod (the counter that
    # tells the nods apart). std 0.25 so a 0.15 rad step between nods is
    # clearly worth taking. Until the snuggle starts.
    cfg.rewards["sleepy_head_turn"] = RewardTermCfg(
        func=sleepy_duck_mdp.pose_track_seated,
        weight=2.0,
        params={**kf, "joint_indices": _HEAD_TURN_JOINTS, "std": 0.25, "t_end": SNUGGLE_START, **seated},
    )
    # Job 2, the snuggle onto the cheek, the punchline. Runs 2 and 3 never
    # learnt it with the tight std above: it starts ~1 rad from its target,
    # where that Gaussian is flat (an open-loop test showed the servos do it
    # easily). Run 4 tried one wide term over yaw AND roll: the policy took
    # the roll half (roll 0, the easy part) and never turned. So yaw gets its
    # own wide term plus a linear pull (every degree of turning pays), and
    # roll its own small one, from the snuggle's start.
    cfg.rewards["sleepy_snuggle"] = RewardTermCfg(
        func=sleepy_duck_mdp.pose_track_window,
        weight=3.0,
        params={**kf, "joint_indices": [_HEAD_TURN_JOINTS[0]], "std": 1.0, "t_start": SNUGGLE_START},
    )
    cfg.rewards["sleepy_snuggle_l1_cost"] = RewardTermCfg(
        func=sleepy_duck_mdp.pose_l1_cost_window,
        weight=-1.0,
        params={**kf, "joint_indices": [_HEAD_TURN_JOINTS[0]], "t_start": SNUGGLE_START},
    )
    cfg.rewards["sleepy_snuggle_roll"] = RewardTermCfg(
        func=sleepy_duck_mdp.pose_track_window,
        weight=1.0,
        params={**kf, "joint_indices": [_HEAD_TURN_JOINTS[1]], "std": 0.25, "t_start": SNUGGLE_START},
    )
    # The L1 pull on the head turn (both jobs) gives a gradient from far away.
    cfg.rewards["sleepy_head_turn_l1_cost"] = RewardTermCfg(
        func=play_dead_mdp.pose_l1_cost,
        weight=-0.5,
        params={**kf, "joint_indices": _HEAD_TURN_JOINTS},
    )
    # Both feet stay down the whole time: it sits and sleeps where it stands.
    cfg.rewards["feet_planted"] = RewardTermCfg(
        func=play_dead_mdp.feet_planted_until,
        weight=1.0,
        params={"sensor_name": "feet_ground_contact", "t_end": EPISODE_LENGTH_S + 1.0},
    )
    # Once asleep, actually rest the head on the floor (by the clock).
    cfg.rewards["sleepy_head_resting"] = RewardTermCfg(
        func=sleepy_duck_mdp.contact_reward_after,
        weight=1.0,
        params={"sensor_name": "head_ground_contact", "t_start": BEAK_LANDED},
    )

    # ── Style and safety costs ─────────────────────────────────────────────
    cfg.rewards["sleepy_drift_cost"] = RewardTermCfg(func=bow_mdp.base_drift_cost, weight=-1.0)
    # No leaning sideways: the whole trick is sagittal except the head.
    cfg.rewards["sleepy_roll_cost"] = RewardTermCfg(func=bow_mdp.trunk_roll_cost, weight=-2.0)
    cfg.rewards["sleepy_early_head_contact_cost"] = RewardTermCfg(
        func=play_dead_mdp.contact_cost_until,
        weight=-2.0,
        params={"sensor_name": "head_ground_contact", "t_end": HEAD_DOWN_ALLOWED},
    )
    # The head (38% of the mass, camera inside) must land softly, and the
    # snuggle must roll it gently rather than scrub it across the floor.
    cfg.rewards["sleepy_head_impact_cost"] = RewardTermCfg(
        func=play_dead_mdp.head_impact_cost,
        weight=-10.0,  # a 1 m/s slam ≈ 10 per step; a 0.1 m/s snuggle roll ≈ 0.1
        params={"sensor_name": "head_ground_contact"},
    )
    # gentle_landing (self-negating |a_z| on the trunk, POSITIVE weight) is
    # inherited from the roulade: it prices the bump into the sit.
    cfg.rewards["gentle_landing"].weight = 0.005
    cfg.rewards["action_rate_l2"].weight = -0.2
    # Counts ANY touch between robot parts (no force threshold: the sensor has
    # no force history), so the sit's resting ankle-on-hip contacts (6) cost as
    # much as a crash. At PlayDead's −1.0 that was ~−300/s while sitting or
    # asleep, more than the whole pitch term: runs 1 and 2 sat shallow and
    # dodged the asleep poses. −0.05 keeps a nudge against real collisions.
    cfg.rewards["self_collisions"].weight = -0.05
    # Slow, small motions, but keep the motion-blockers low anyway: the
    # catches are quick, and the slump is a 55° rotation in 1.7 s.
    cfg.rewards["body_ang_vel"].weight = -0.005
    cfg.rewards["angular_momentum"].weight = -0.005

    # ── Terminations ─────────────────────────────────────────────────────────
    # The trick ends leaning forward ~55°, so no generic "fell over". Toppling
    # onto the back, onto a side, or on over the face ends the episode.
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["tipped_over"] = TerminationTermCfg(func=sleepy_duck_mdp.tipped_over)
    # Episodes that start mid-routine end when the routine does.
    cfg.terminations["clock_out"] = TerminationTermCfg(
        func=sleepy_duck_mdp.clock_out, params={"t_end": EPISODE_LENGTH_S}, time_out=True
    )

    # ── Critic-only privileged clock ─────────────────────────────────────────
    cfg.observations["critic"].terms["sleepy_clock"] = ObservationTermCfg(
        func=play_dead_mdp.play_dead_clock_obs,
        params={**kf, "period": EPISODE_LENGTH_S},
    )

    # ── Smoothing: held low ──────────────────────────────────────────────────
    # The action-rate penalty stays at its starting weight, and the torque-rate
    # penalty is capped at −5e-4, the most the staged runs (2 → 3 → 4 → 5) ever
    # trained with. Trained from scratch with the ramps to −0.4 and −1e-3
    # (run 6), the duck slumped and snuggled but never nodded: heavy smoothing
    # rounds off the jerks, and the jerks are the joke. (action_rate_l2 is set
    # to −0.2 with the costs above.) The torque-rate penalty restarts at zero
    # for the catches, as each fine-tune restarted it.
    cfg.curriculum.pop("action_rate_weight", None)
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 1000 * 24, "weight": -5e-4},
                {"step": STAGE_CATCHES * 24, "weight": 0.0},
                {"step": (STAGE_SNUGGLE + 500) * 24, "weight": -5e-4},
            ],
        },
    )

    # ── One run, three stages (see STAGE_CATCHES) ────────────────────────────
    # Weights per stage (the slump, the catches, the snuggle). The terms keep
    # their stage-1 weight until the curriculum's first update.
    starts = (0, STAGE_CATCHES, STAGE_SNUGGLE)
    staged = {
        "sleepy_head_coarse": (3.0, 0.0, 0.0),
        "sleepy_head": (0.0, 4.0, 4.0),
        "sleepy_snuggle": (0.0, 0.0, 3.0),
        "sleepy_snuggle_l1_cost": (0.0, 0.0, -1.0),
        "sleepy_snuggle_roll": (0.0, 0.0, 1.0),
    }
    for name, weights in staged.items():
        cfg.rewards[name].weight = weights[0]
        cfg.curriculum[f"{name}_stages"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": name,
                "weight_stages": [{"step": it * 24, "weight": w} for it, w in zip(starts, weights)],
            },
        )
    # Where the mid-routine rounds start (never their share: that stays
    # MID_ROUTINE_PROB, and 0 in play/eval).
    cfg.curriculum["mid_routine_states"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "mid_routine_spawn",
            "param_stages": [
                {"step": it * 24, "params": {"states": states}}
                for it, states in zip(starts, (SPAWN_SLUMP, SPAWN_CATCHES, SPAWN_STATES))
            ],
        },
    )
    # The centre-of-mass randomisation ramps up again in each stage, as it
    # did in each fine-tune (the roulade's ramp, cut short by the next stage).
    ramp = [(0, 0.003), (500, 0.005), (1000, 0.01), (1500, 0.015)]
    bounds = (*starts, MAX_ITERATIONS)
    for term, top in (("com_range", 0.015), ("head_com_range", 0.01)):
        if term in cfg.curriculum:
            cfg.curriculum[term].params["range_stages"] = [
                {"step": (s0 + off) * 24, "range": min(rng, top)}
                for s0, s1 in zip(bounds, bounds[1:])
                for off, rng in ramp
                if s0 + off < s1
            ]

    return cfg


MicroduckSleepyDuckRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # baked into the ONNX by scripts/export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.8,  # as PlayDead
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
    experiment_name="microduck_sleepy_duck",
    run_name="microduck_sleepy_duck",
    # Every 50 iterations (not 250): dense checkpoints show how the trick
    # develops over training.
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=MAX_ITERATIONS,
)
