"""Config invariants for Mjlab-SleepyDuck-Flat-MicroDuck (CPU, no GPU needed)."""

import math

import torch

from mjlab_microduck.tasks import play_dead_mdp
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import SITTING_TARGET_OVERRIDES
from mjlab_microduck.tasks.microduck_sleepy_duck_env_cfg import (
    ASLEEP_POSE,
    BEAK_DOWN_PITCH,
    BEAK_LANDED,
    EPISODE_LENGTH_S,
    HEAD_DOWN_ALLOWED,
    HEAD_YAW_SNUGGLE,
    KEYFRAMES,
    SIT_POSE,
    SLUMP_START,
    SNUGGLE_PITCH,
    YAWN_HEAD,
    MicroduckSleepyDuckRlCfg,
    make_microduck_sleepy_duck_env_cfg,
)

# Every cost returns >= 0, every reward returns [0, 1].
_COSTS = {
    "sleepy_head_l1_cost",
    "sleepy_legs_l1_cost",
    "sleepy_head_turn_l1_cost",
    "sleepy_snuggle_l1_cost",
    "sleepy_drift_cost",
    "sleepy_roll_cost",
    "sleepy_early_head_contact_cost",
    "sleepy_head_impact_cost",
}
_REWARDS = {
    "sleepy_head",
    "sleepy_pitch",
    "sleepy_legs",
    "sleepy_head_turn",
    "sleepy_snuggle",
    "sleepy_snuggle_roll",
    "feet_planted",
    "sleepy_head_resting",
}
_HOME = torch.tensor(
    [0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0, 0,
     0, 0.0873, 0.4579, 0.0049, -0.4530]
)
_NECK, _HEAD_PITCH, _HEAD_YAW = 5, 6, 7


def _nose_down(neck: float, head: float) -> float:
    """Head angle vs the trunk, nose-down positive (measured in the pose study)."""
    return -(neck - 0.3491) + (head - 0.3491)


def test_reward_signs():
    """The sign-convention footgun from AGENTS.md: costs negative, rewards positive."""
    r = make_microduck_sleepy_duck_env_cfg().rewards
    for name in _COSTS:
        assert r[name].weight < 0, name
    for name in _REWARDS:
        assert r[name].weight > 0, name
    # gentle_landing is self-negating (returns -|a_z|): POSITIVE weight.
    assert r["gentle_landing"].weight > 0


def test_inherited_tasks_fully_removed():
    cfg = make_microduck_sleepy_duck_env_cfg()
    assert not [n for n in cfg.rewards if n.startswith(("roulade_", "bow_", "play_dead_"))]
    assert "roulade_spawn_mix" not in cfg.curriculum
    spawn = cfg.events["set_roulade_state"].params
    assert spawn["standing_prob"] == 1.0 and spawn["midroll_prob"] == 0.0


def test_asleep_pose_does_not_end_the_episode():
    """Generic fell_over would terminate the trick at its goal (leaning ~55°)."""
    cfg = make_microduck_sleepy_duck_env_cfg()
    assert "fell_over" not in cfg.terminations
    term = cfg.terminations["tipped_over"]
    from mjlab_microduck.tasks import sleepy_duck_mdp
    import inspect

    defaults = inspect.signature(sleepy_duck_mdp.tipped_over).parameters
    max_fwd = term.params.get("max_forward_pitch", defaults["max_forward_pitch"].default)
    assert max_fwd > max(BEAK_DOWN_PITCH, SNUGGLE_PITCH) + math.radians(15.0)


def test_clock_is_critic_only():
    """The deployed actor must not see the clock (constant-command contract)."""
    cfg = make_microduck_sleepy_duck_env_cfg()
    assert "sleepy_clock" in cfg.observations["critic"].terms
    assert "sleepy_clock" not in cfg.observations["actor"].terms


def test_sit_matches_the_verified_sitstand_keyframe():
    assert SIT_POSE == SITTING_TARGET_OVERRIDES


def test_resting_self_contact_is_cheap():
    """The sit rests its ankles on its hips (6 contacts): at −1.0 that cost
    more than the whole pitch term and the duck would not sit or sleep."""
    r = make_microduck_sleepy_duck_env_cfg().rewards
    assert -0.1 <= r["self_collisions"].weight < 0.0
    assert 6 * abs(r["self_collisions"].weight) <= 0.1 * r["sleepy_pitch"].weight


def test_keyframes_in_order_and_end_asleep():
    times = [k["t"] for k in KEYFRAMES]
    assert times == sorted(times) and times[0] == 0.0
    assert times[-1] == EPISODE_LENGTH_S
    assert KEYFRAMES[0]["pose"] == {} and KEYFRAMES[0]["pitch"] == 0.0
    assert KEYFRAMES[-1]["pose"] == ASLEEP_POSE
    assert ASLEEP_POSE[_HEAD_YAW] == HEAD_YAW_SNUGGLE
    # The head only ever turns further towards its sleeping side (the nod
    # counter, then the snuggle), and the big turn comes after the landing.
    yaws = [k["pose"].get(_HEAD_YAW, 0.0) for k in KEYFRAMES]
    assert yaws == sorted(yaws)
    for k in KEYFRAMES:
        if k["t"] <= BEAK_LANDED:
            assert k["pose"].get(_HEAD_YAW, 0.0) <= 0.5, k
    # Still holds stay short: the actor cannot see a clock.
    for a, b in zip(KEYFRAMES, KEYFRAMES[1:]):
        if a["pose"] == b["pose"] and a["pitch"] == b["pitch"] and b is not KEYFRAMES[-1]:
            assert b["t"] - a["t"] <= 0.3 + 1e-9, (a["t"], b["t"])


def test_nods_get_deeper_and_the_catches_weaker():
    """Monotonic nods: the actor tells them apart by how deep they are."""
    nods = [k["pose"] for k in KEYFRAMES if 2.3 < k["t"] <= SLUMP_START]
    droops, catches = nods[0::2], nods[1::2]
    assert len(droops) == 3 and len(catches) == 3
    d = [_nose_down(p[_NECK], p[_HEAD_PITCH]) for p in droops]
    c = [_nose_down(p[_NECK], p[_HEAD_PITCH]) for p in catches]
    assert d == sorted(d) and len(set(d)) == 3
    assert c == sorted(c) and len(set(c)) == 3
    assert all(ci < di for ci, di in zip(c, d))  # each catch lifts the head
    necks = [p[_NECK] for p in nods]
    assert all(b <= a + 0.05 for a, b in zip(necks, necks[1:]))  # neck only creeps forward


def test_each_nod_has_its_own_head_turn():
    """Run 1 averaged the nods: depth alone did not tell them apart. Each nod
    now turns the head further (a posture counter); a catch keeps its turn."""
    nods = [k["pose"] for k in KEYFRAMES if 2.3 < k["t"] <= SLUMP_START]
    turns = [p[_HEAD_YAW] for p in nods[0::2]]
    assert turns == sorted(turns) and min(b - a for a, b in zip([0.0] + turns, turns)) >= 0.1
    for droop, catch in zip(nods[0::2], nods[1::2]):
        assert catch[_HEAD_YAW] == droop[_HEAD_YAW] and catch[8] == droop[8]


def test_yawn_is_nose_up_but_not_too_far_back():
    up = -_nose_down(YAWN_HEAD[_NECK], YAWN_HEAD[_HEAD_PITCH])
    assert math.radians(20.0) < up < math.radians(50.0)  # 57° tipped the sitting duck over


def test_reference_profile_shape():
    t = torch.linspace(0.0, EPISODE_LENGTH_S, 2000)
    pitch = play_dead_mdp.reference_pitch_at(t, KEYFRAMES)
    pose = play_dead_mdp.reference_pose_at(t, KEYFRAMES, _HOME)
    assert pitch[0].item() == 0.0
    assert math.isclose(pitch[-1].item(), SNUGGLE_PITCH, abs_tol=1e-6)
    assert (pitch.diff().abs() < 0.01).all()  # smooth at this resolution
    assert (pose.diff(dim=0).abs() < 0.02).all()
    assert torch.allclose(pose[0], _HOME)
    end = _HOME.clone()
    for j, v in ASLEEP_POSE.items():
        end[j] = v
    assert torch.allclose(pose[-1], end, atol=1e-6)
    # Upright until the slump.
    assert (pitch[t < SLUMP_START] < math.radians(6.0)).all()
    # Seated (legs) from the sit keyframe on.
    seated = play_dead_mdp.reference_pose_at(torch.tensor([2.3, 6.0, 9.0]), KEYFRAMES, _HOME)
    for j, v in SIT_POSE.items():
        assert torch.allclose(seated[:, j], torch.tensor(v), atol=1e-6)


def test_phase_gates():
    r = make_microduck_sleepy_duck_env_cfg().rewards
    assert r["sleepy_early_head_contact_cost"].params["t_end"] == HEAD_DOWN_ALLOWED
    assert SLUMP_START < HEAD_DOWN_ALLOWED < BEAK_LANDED
    assert r["sleepy_head_resting"].params["t_start"] == BEAK_LANDED
    assert r["feet_planted"].params["t_end"] > EPISODE_LENGTH_S  # feet down throughout


def test_symmetry_off_for_one_sided_snuggle():
    assert HEAD_YAW_SNUGGLE != 0.0
    assert MicroduckSleepyDuckRlCfg.algorithm.symmetry_cfg is None


def test_mid_routine_spawns_match_the_keyframes():
    from mjlab_microduck.tasks import sleepy_duck_mdp
    from mjlab_microduck.tasks.microduck_sleepy_duck_env_cfg import MID_ROUTINE_PROB, SPAWN_STATES

    by_t = {k["t"]: k for k in KEYFRAMES}
    assert SPAWN_STATES and 0.0 < MID_ROUTINE_PROB < 0.5
    for st in SPAWN_STATES:
        assert st["t"] in by_t and st["pose"] == by_t[st["t"]]["pose"], st["t"]
        assert abs(st["pitch"] - by_t[st["t"]]["pitch"]) < math.radians(8.0)  # settles near the reference
        assert 0.05 < st["z"] < 0.08 and st["weight"] > 0.0
    # The slump gets practice: some spawns start at or after it.
    assert any(st["t"] >= SLUMP_START for st in SPAWN_STATES)
    train = make_microduck_sleepy_duck_env_cfg()
    names = list(train.events.keys())
    assert names[-1] == "mid_routine_spawn"  # last reset event: overrides stand/walk spawns
    ev = train.events["mid_routine_spawn"]
    assert ev.func is sleepy_duck_mdp.reset_mid_routine and ev.params["prob"] == MID_ROUTINE_PROB
    assert make_microduck_sleepy_duck_env_cfg(play=True).events["mid_routine_spawn"].params["prob"] == 0.0


def test_mid_routine_episodes_end_with_the_routine():
    term = make_microduck_sleepy_duck_env_cfg().terminations["clock_out"]
    assert term.time_out and term.params["t_end"] == EPISODE_LENGTH_S


def test_clock_offset_shifts_the_episode_clock():
    """bow_mdp._episode_time adds the per-env offset when one exists, and is
    unchanged for tasks that never set it (PoliteBow, PlayDead)."""
    from types import SimpleNamespace

    from mjlab_microduck.tasks import bow_mdp

    env = SimpleNamespace(episode_length_buf=torch.tensor([10, 10]), step_dt=0.02)
    assert torch.allclose(bow_mdp._episode_time(env), torch.tensor([0.2, 0.2]))
    env._clock_offset_s = torch.tensor([0.0, 7.6])
    assert torch.allclose(bow_mdp._episode_time(env), torch.tensor([0.2, 7.8]))


def test_head_turn_jobs_hand_over_at_the_snuggle():
    """Tight counter during the nods, wide snuggle afterwards: no gap, no overlap."""
    from mjlab_microduck.tasks import sleepy_duck_mdp
    from mjlab_microduck.tasks.microduck_sleepy_duck_env_cfg import SNUGGLE_START

    r = make_microduck_sleepy_duck_env_cfg().rewards
    counter, snuggle = r["sleepy_head_turn"], r["sleepy_snuggle"]
    assert counter.func is snuggle.func is sleepy_duck_mdp.pose_track_window
    assert counter.params["t_end"] == snuggle.params["t_start"] == SNUGGLE_START
    assert snuggle.params["std"] > 2 * counter.params["std"]
    # Yaw alone (run 4's yaw+roll term was satisfied by roll alone), with a
    # linear pull over the same window.
    assert snuggle.params["joint_indices"] == [_HEAD_YAW]
    pull = r["sleepy_snuggle_l1_cost"]
    assert pull.params["joint_indices"] == [_HEAD_YAW] and pull.params["t_start"] == SNUGGLE_START
    assert r["sleepy_snuggle_roll"].params["joint_indices"] == [8]
    assert BEAK_LANDED < SNUGGLE_START < EPISODE_LENGTH_S
    yaw_at = {k["t"]: k["pose"].get(_HEAD_YAW, 0.0) for k in KEYFRAMES}
    assert yaw_at[SNUGGLE_START] < 0.5 < HEAD_YAW_SNUGGLE  # the big turn is the snuggle's


def test_dense_checkpoints():
    assert MicroduckSleepyDuckRlCfg.save_interval <= 50


def test_play_variant_builds_and_registered():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    cfg = make_microduck_sleepy_duck_env_cfg(play=True)
    assert cfg.episode_length_s == EPISODE_LENGTH_S
    assert load_env_cfg("Mjlab-SleepyDuck-Flat-MicroDuck", play=True).episode_length_s == EPISODE_LENGTH_S


def test_gait_bank_spawn_wiring():
    """Mid-walk starts, as for PoliteBow v2 and PlayDead."""
    from mjlab_microduck.tasks import bow_mdp
    from mjlab_microduck.tasks.microduck_sleepy_duck_env_cfg import GAIT_BANK_PATH, GAIT_BANK_PROB

    train = make_microduck_sleepy_duck_env_cfg()
    names = list(train.events.keys())
    assert names.index("gait_bank_spawn") > names.index("set_roulade_state")
    ev = train.events["gait_bank_spawn"]
    assert ev.func is bow_mdp.reset_from_gait_bank and ev.mode == "reset"
    assert ev.params["bank_path"] == GAIT_BANK_PATH
    assert 0.0 < ev.params["prob"] == GAIT_BANK_PROB < 1.0
    play = make_microduck_sleepy_duck_env_cfg(play=True)
    assert play.events["gait_bank_spawn"].params["prob"] == 0.0
    for cfg in (train, play):
        for group in ("actor", "critic"):
            assert cfg.observations[group].terms["actions"].func is bow_mdp.last_action_with_handoff
