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
