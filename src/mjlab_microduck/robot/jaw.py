"""Optional hinged jaw (beak) for rehearsal and videos. NOT used in training.

The exported robot models fix the jaw rigidly to the head (the ``jaw`` mesh
inside the ``jaw_soft`` body) and the policies drive only the 14 servos. On
the real robot the jaw has its own servo, which the runtime drives, not the
policy. This module adds a matching hinge at load time so a rehearsal or a
video can open and close the beak the same way the runtime would.

The joint is named ``passive_jaw`` and has NO actuator, so everything that
selects servos with ``^(?!passive_).*`` or through the actuators (the BAM
loader and ``infer_policy.py`` both do) ignores it. It is held by a joint
spring: set its target angle with :func:`set_jaw_target`.

Measured from the ``jaw`` mesh on the 2026-09 groundcontact model: the side
arms end in a round boss (radius ~6 mm) around the pivot pin at PIVOT in the
``jaw_soft`` frame; +y is the pin axis; a positive angle opens the beak
(lower jaw tip goes down).
"""

from __future__ import annotations

import mujoco
import numpy as np

HEAD_BODY = "jaw_soft"
JAW_MESH = "jaw"
JOINT_NAME = "passive_jaw"
PIVOT = np.array([0.0053, 0.0, -0.0179])  # m, in the jaw_soft frame
MAX_OPEN = 0.7  # rad; the soft mouth shows well before this
OPEN_ANGLE = 0.45  # rad, a good "dead duck" gape

_GEOM_ATTRS = (
    "type", "meshname", "material", "quat", "contype", "conaffinity", "group",
    "rgba", "friction", "condim", "margin", "gap", "priority", "solref",
    "solimp", "solmix",
)


def add_jaw_hinge(spec: mujoco.MjSpec) -> None:
    """Move the jaw meshes onto a spring-held hinge (edits ``spec`` in place)."""
    head = spec.body(HEAD_BODY)
    jaw = head.add_body(name="jaw", pos=PIVOT)
    jaw.add_joint(
        name=JOINT_NAME, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0],
        range=[0.0, MAX_OPEN], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        stiffness=0.2, damping=0.005, springref=0.0,
        # The body inherits the servo defaults of the "microduck" class
        # (frictionloss 0.1 N·m, armature 0.005), which would out-pull the
        # spring. A jaw on a pin has neither.
        frictionloss=0.0, armature=1e-6,
    )
    # A few grams: the head's mass/inertia stay as exported (explicit inertial).
    jaw.mass = 0.004
    jaw.inertia = [2e-7, 2e-7, 2e-7]
    jaw.explicitinertial = True
    moved = 0
    for g in list(head.geoms):
        if g.meshname != JAW_MESH:
            continue
        ng = jaw.add_geom(default=g.classname)
        for attr in _GEOM_ATTRS:
            setattr(ng, attr, getattr(g, attr))
        ng.pos = np.array(g.pos) - PIVOT
        spec.delete(g)
        moved += 1
    if moved == 0:
        raise ValueError(f"no '{JAW_MESH}' mesh geom found in body '{HEAD_BODY}'")


def jaw_qpos_adr(model: mujoco.MjModel) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)
    if jid < 0:
        raise KeyError(f"model has no '{JOINT_NAME}' joint (call add_jaw_hinge first)")
    return int(model.jnt_qposadr[jid])


def set_jaw_target(model: mujoco.MjModel, adr: int, angle: float) -> None:
    """Point the jaw spring at ``angle`` rad (0 = shut)."""
    model.qpos_spring[adr] = float(np.clip(angle, 0.0, MAX_OPEN))
