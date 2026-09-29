"""The optional rehearsal jaw (robot/jaw.py) must not disturb the servo layout."""

import mujoco
import numpy as np
import pytest

from mjlab_microduck.robot import jaw

SCENE = "src/mjlab_microduck/robot/microduck/scene.xml"


def _models():
    plain = mujoco.MjModel.from_xml_path(SCENE)
    spec = mujoco.MjSpec.from_file(SCENE)
    jaw.add_jaw_hinge(spec)
    return plain, spec.compile()


def _servo_names(m):
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) for i in range(m.nu)]


def test_servos_unchanged_and_jaw_unactuated():
    plain, hinged = _models()
    assert hinged.nu == plain.nu == 14
    assert _servo_names(hinged) == _servo_names(plain)
    assert jaw.JOINT_NAME not in _servo_names(hinged)
    assert jaw.JOINT_NAME.startswith("passive_")  # repo-wide servo selectors skip it
    assert hinged.nq == plain.nq + 1


def test_jaw_follows_its_spring_target():
    _, m = _models()
    m.opt.gravity[:] = 0.0  # isolate the jaw from the robot falling over
    d = mujoco.MjData(m)
    adr = jaw.jaw_qpos_adr(m)
    for target in (jaw.OPEN_ANGLE, 0.0):
        jaw.set_jaw_target(m, adr, target)
        for _ in range(int(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        assert d.qpos[adr] == pytest.approx(target, abs=0.03)


def test_positive_angle_opens_downwards():
    """The lower-jaw tip must move down (towards the head's -x) when opening."""
    _, m = _models()
    d = mujoco.MjData(m)
    adr = jaw.jaw_qpos_adr(m)
    head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, jaw.HEAD_BODY)
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "jaw")
    tip_local = np.array([-0.01, 0.0, -0.06]) - jaw.PIVOT  # a point near the beak tip

    def tip_in_head_frame():
        mujoco.mj_forward(m, d)
        world = d.xpos[body] + d.xmat[body].reshape(3, 3) @ tip_local
        return d.xmat[head].reshape(3, 3).T @ (world - d.xpos[head])

    shut = tip_in_head_frame()
    d.qpos[adr] = jaw.OPEN_ANGLE
    opened = tip_in_head_frame()
    assert opened[0] < shut[0] - 0.01
