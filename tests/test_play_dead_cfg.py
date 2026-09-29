"""Config invariants for Mjlab-PlayDead-Flat-MicroDuck (CPU, no GPU needed)."""

import math

import torch

from mjlab_microduck.tasks import play_dead_mdp
from mjlab_microduck.tasks.microduck_play_dead_env_cfg import (
    DEAD_POSE,
    EPISODE_LENGTH_S,
    KEYFRAMES,
    LIE_DOWN_START,
    LYING_PITCH,
    SIT_POSE,
    make_microduck_play_dead_env_cfg,
)
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import SITTING_TARGET_OVERRIDES

# Every play_dead_mdp / bow_mdp cost returns >= 0, every reward returns [0, 1].
_COSTS = {
    "play_dead_legs_l1_cost",
    "play_dead_drift_cost",
    "play_dead_roll_cost",
    "play_dead_early_head_contact_cost",
    "play_dead_head_impact_cost",
}
_REWARDS = {
    "play_dead_pitch",
    "play_dead_legs",
    "play_dead_head_yaw",
    "play_dead_neck",
    "feet_planted",
}
_HOME = torch.tensor(
    [0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0, 0,
     0, 0.0873, 0.4579, 0.0049, -0.4530]
)


def test_reward_signs():
    """The sign-convention footgun from AGENTS.md: costs negative, rewards positive."""
    r = make_microduck_play_dead_env_cfg().rewards
    for name in _COSTS:
        assert r[name].weight < 0, name
    for name in _REWARDS:
        assert r[name].weight > 0, name
    # gentle_landing is self-negating (returns -|a_z|): POSITIVE weight.
    assert r["gentle_landing"].weight > 0


def test_inherited_tasks_fully_removed():
    cfg = make_microduck_play_dead_env_cfg()
    assert not [n for n in cfg.rewards if n.startswith(("roulade_", "bow_"))]
    assert "roulade_spawn_mix" not in cfg.curriculum
    spawn = cfg.events["set_roulade_state"].params
    assert spawn["standing_prob"] == 1.0 and spawn["midroll_prob"] == 0.0


def test_lying_on_the_back_does_not_end_the_episode():
    """Generic fell_over would terminate the trick at its goal."""
    terms = make_microduck_play_dead_env_cfg().terminations
    assert "fell_over" not in terms
    assert "wrong_side_down" in terms


def test_clock_is_critic_only():
    """The deployed actor must not see the clock (constant-command contract)."""
    cfg = make_microduck_play_dead_env_cfg()
    assert "play_dead_clock" in cfg.observations["critic"].terms
    assert "play_dead_clock" not in cfg.observations["actor"].terms


def test_sit_matches_the_verified_sitstand_keyframe():
    assert SIT_POSE == SITTING_TARGET_OVERRIDES


def test_keyframes_in_order_and_end_dead():
    times = [k["t"] for k in KEYFRAMES]
    assert times == sorted(times) and times[0] == 0.0
    assert times[-1] == EPISODE_LENGTH_S
    assert KEYFRAMES[0]["pose"] == {} and KEYFRAMES[0]["pitch"] == 0.0
    assert KEYFRAMES[-1]["pose"] == DEAD_POSE
    assert KEYFRAMES[-1]["pitch"] == LYING_PITCH
    # The head only turns once the duck is lying down.
    for k in KEYFRAMES:
        if k["pitch"] != LYING_PITCH:
            assert 7 not in k["pose"]


def test_reference_profile_shape():
    t = torch.linspace(0.0, EPISODE_LENGTH_S, 1000)
    pitch = play_dead_mdp.reference_pitch_at(t, KEYFRAMES)
    pose = play_dead_mdp.reference_pose_at(t, KEYFRAMES, _HOME)
    assert pitch[0].item() == 0.0 and math.isclose(pitch[-1].item(), LYING_PITCH, abs_tol=1e-6)
    assert (pitch <= 1e-6).all() and (pitch >= LYING_PITCH - 1e-6).all()
    assert (pitch.diff() <= 1e-6).all()  # only ever leans further back
    assert (pitch.diff().abs() < 0.01).all()  # smooth at this resolution
    assert (pose.diff(dim=0).abs() < 0.02).all()
    assert torch.allclose(pose[0], _HOME)
    end = _HOME.clone()
    for j, v in DEAD_POSE.items():
        end[j] = v
    assert torch.allclose(pose[-1], end, atol=1e-6)
    # Seated (legs) at the sit keyframe, still upright.
    mid = play_dead_mdp.reference_pose_at(torch.tensor([KEYFRAMES[1]["t"]]), KEYFRAMES, _HOME)[0]
    for j, v in SIT_POSE.items():
        assert math.isclose(mid[j].item(), v, abs_tol=1e-6)
    # Before and after the last keyframe the reference is clamped.
    assert torch.allclose(
        play_dead_mdp.reference_pitch_at(torch.tensor([-1.0, 99.0]), KEYFRAMES),
        torch.tensor([0.0, LYING_PITCH]),
    )


def test_phase_gates_close_at_lie_down():
    for name in ("feet_planted", "play_dead_early_head_contact_cost"):
        term = make_microduck_play_dead_env_cfg().rewards[name]
        assert term.params["t_end"] == LIE_DOWN_START


def test_symmetry_off_for_one_sided_head_turn():
    from mjlab_microduck.tasks.microduck_play_dead_env_cfg import MicroduckPlayDeadRlCfg

    assert DEAD_POSE[7] != 0.0
    assert MicroduckPlayDeadRlCfg.algorithm.symmetry_cfg is None


def test_play_variant_builds_and_registered():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    cfg = make_microduck_play_dead_env_cfg(play=True)
    assert cfg.episode_length_s == EPISODE_LENGTH_S
    assert load_env_cfg("Mjlab-PlayDead-Flat-MicroDuck", play=True).episode_length_s == EPISODE_LENGTH_S


def test_gait_bank_spawn_wiring():
    """Mid-walk starts, as for PoliteBow v2: "bang!" usually comes mid-walk."""
    from mjlab_microduck.tasks import bow_mdp
    from mjlab_microduck.tasks.microduck_play_dead_env_cfg import GAIT_BANK_PATH, GAIT_BANK_PROB

    train = make_microduck_play_dead_env_cfg()
    names = list(train.events.keys())
    assert names.index("gait_bank_spawn") > names.index("set_roulade_state")
    ev = train.events["gait_bank_spawn"]
    assert ev.func is bow_mdp.reset_from_gait_bank and ev.mode == "reset"
    assert ev.params["bank_path"] == GAIT_BANK_PATH
    assert 0.0 < ev.params["prob"] == GAIT_BANK_PROB < 1.0
    play = make_microduck_play_dead_env_cfg(play=True)
    assert play.events["gait_bank_spawn"].params["prob"] == 0.0
    for cfg in (train, play):
        for group in ("actor", "critic"):
            assert cfg.observations[group].terms["actions"].func is bow_mdp.last_action_with_handoff
