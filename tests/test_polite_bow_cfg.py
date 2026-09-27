"""Config invariants for Mjlab-PoliteBow-Flat-MicroDuck (CPU, no GPU needed)."""

import math

import torch

from mjlab_microduck.tasks import bow_mdp
from mjlab_microduck.tasks.microduck_polite_bow_env_cfg import (
    BOW_DELTA,
    BOW_PROFILE,
    EPISODE_LENGTH_S,
    make_microduck_polite_bow_env_cfg,
)

# Every bow_mdp cost returns >= 0, every reward returns [0, 1] (module docstring).
_COSTS = {
    "bow_neck_l1_cost",
    "bow_drift_cost",
    "bow_roll_cost",
    "bow_head_contact_cost",
    "feet_flat",
}
_REWARDS = {"bow_pitch", "bow_neck_pose", "bow_leg_home", "feet_planted"}


def test_reward_signs():
    """The sign-convention footgun from AGENTS.md: costs negative, rewards positive."""
    r = make_microduck_polite_bow_env_cfg().rewards
    for name in _COSTS:
        assert r[name].weight < 0, name
    for name in _REWARDS:
        assert r[name].weight > 0, name
    # gentle_landing is self-negating (returns -|a_z|): POSITIVE weight.
    assert r["gentle_landing"].weight > 0


def test_roulade_task_fully_removed():
    cfg = make_microduck_polite_bow_env_cfg()
    assert not [n for n in cfg.rewards if n.startswith("roulade_")]
    assert "roulade_spawn_mix" not in cfg.curriculum
    spawn = cfg.events["set_roulade_state"].params
    assert spawn["standing_prob"] == 1.0 and spawn["midroll_prob"] == 0.0


def test_clock_is_critic_only():
    """The deployed actor must not see the clock (constant-command contract)."""
    cfg = make_microduck_polite_bow_env_cfg()
    assert "bow_clock" in cfg.observations["critic"].terms
    assert "bow_clock" not in cfg.observations["actor"].terms


def test_legs_are_not_hand_posed():
    """Legs are discovered by RL; only neck/head carry a joint-space bow target."""
    assert set(BOW_DELTA) <= {5, 6, 7, 8}


def test_profile_shape():
    t = torch.linspace(0.0, EPISODE_LENGTH_S, 500)
    b = bow_mdp.bow_blend_at(t, **BOW_PROFILE)
    assert b[0].item() == 0.0  # starts at the stand
    assert math.isclose(b.max().item(), 1.0, abs_tol=1e-6)  # reaches the full bow
    assert b[-1].item() == 0.0  # ends at the stand
    assert ((b >= 0) & (b <= 1)).all()
    # Smooth: no jump larger than a smoothstep allows at this resolution.
    assert (b.diff().abs() < 0.02).all()
    # The hold is really held.
    mid = bow_mdp.bow_blend_at(
        torch.tensor([BOW_PROFILE["down_end"], BOW_PROFILE["hold_end"]]), **BOW_PROFILE
    )
    assert torch.allclose(mid, torch.ones(2))


def test_play_variant_builds():
    cfg = make_microduck_polite_bow_env_cfg(play=True)
    assert "bow_pitch" in cfg.rewards
    assert cfg.episode_length_s == EPISODE_LENGTH_S


def test_feet_flat_is_scoped_to_feet():
    """Regression: without an explicit asset_cfg the default SceneEntityCfg is
    never resolved and feet_flat sums over every robot site (head included),
    which penalised the bow itself (first 3080 Ti run, 26 Sep 2026)."""
    term = make_microduck_polite_bow_env_cfg().rewards["feet_flat"]
    asset_cfg = term.params.get("asset_cfg")
    assert asset_cfg is not None
    assert list(asset_cfg.site_names) == ["left_foot", "right_foot"]


def test_viser_joystick_skipped_for_near_zero_command():
    """Regression: `play --viewer viser` asserted in viser's slider for tasks whose
    command ranges are below viser's 0.1 slider minimum (PoliteBow, roulade)."""
    from types import SimpleNamespace

    from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnly

    class _NoServer:
        def __getattr__(self, _):
            raise AssertionError("GUI must not be built for a near-zero command")

    cmd_cfg = make_microduck_polite_bow_env_cfg(play=True).commands["twist"]
    fake = SimpleNamespace(cfg=cmd_cfg, _JOYSTICK_MIN_RANGE=0.1)
    VelocityCommandCommandOnly.create_gui(fake, "twist", _NoServer(), lambda: 0)


def test_gait_bank_spawn_wiring():
    """Walk→bow hand-off (rehearsal 27 Sep 2026: any mid-stride trigger fell).
    Training mixes in mid-walk spawns AFTER the still-stand reset; play/eval
    default to still stands; the actor and critic see the handed-over action."""
    from mjlab_microduck.tasks.microduck_polite_bow_env_cfg import GAIT_BANK_PATH, GAIT_BANK_PROB

    train = make_microduck_polite_bow_env_cfg()
    names = list(train.events.keys())
    assert names.index("gait_bank_spawn") > names.index("set_roulade_state")
    ev = train.events["gait_bank_spawn"]
    assert ev.func is bow_mdp.reset_from_gait_bank and ev.mode == "reset"
    assert ev.params["bank_path"] == GAIT_BANK_PATH
    assert 0.0 < ev.params["prob"] == GAIT_BANK_PROB < 1.0  # still stands stay in the mix
    assert ev.params["stand_joint_noise_std"] > 0.0
    play = make_microduck_polite_bow_env_cfg(play=True)
    assert play.events["gait_bank_spawn"].params["prob"] == 0.0
    for cfg in (train, play):
        for group in ("actor", "critic"):
            assert cfg.observations[group].terms["actions"].func is bow_mdp.last_action_with_handoff


def test_gait_bank_loader_errors(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="make_gait_bank"):
        bow_mdp.load_gait_bank(str(tmp_path / "missing.pt"), "cpu")
    bad = tmp_path / "bad.pt"
    torch.save({"qpos": torch.zeros(3, 21), "qvel": torch.zeros(3, 20)}, bad)
    with pytest.raises(ValueError, match="action"):
        bow_mdp.load_gait_bank(str(bad), "cpu")
    ragged = tmp_path / "ragged.pt"
    torch.save({"qpos": torch.zeros(3, 21), "qvel": torch.zeros(2, 20), "action": torch.zeros(3, 14)}, ragged)
    with pytest.raises(ValueError, match="ragged"):
        bow_mdp.load_gait_bank(str(ragged), "cpu")
    good = tmp_path / "good.pt"
    torch.save({"qpos": torch.ones(4, 21), "qvel": torch.ones(4, 20), "action": torch.ones(4, 14)}, good)
    bank = bow_mdp.load_gait_bank(str(good), "cpu")
    assert bank["qpos"].shape == (4, 21) and bank["action"].dtype == torch.float32
