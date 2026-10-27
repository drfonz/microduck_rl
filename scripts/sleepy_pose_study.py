"""Sleepy Duckling pose study: where do candidate poses settle?

Drops the duck into each candidate pose in the training environment (the
groundcontact model with Pollen's BAM servo model, no domain randomisation
beyond the play config's), holds the pose as a constant servo target for a few
seconds under gravity, and reports where it ends up: trunk pitch and roll,
whether the feet and the beak touch the floor, beak and bum heights, and
whether it is still moving. It also renders a still of each settled pose, for
choosing the style.

It only checks that a pose is a place the duck can rest. Whether the policy
can get there smoothly, and how, is what training finds out.

The plain MuJoCo servos in the XML are no good for this: they are too soft to
hold even the standing pose (the duck tips onto its face within a second).

Usage:
    CUDA_VISIBLE_DEVICES= uv run python scripts/sleepy_pose_study.py              # all candidates
    CUDA_VISIBLE_DEVICES= uv run python scripts/sleepy_pose_study.py sit asleep    # some of them
    CUDA_VISIBLE_DEVICES= uv run python scripts/sleepy_pose_study.py --out <dir>   # with stills
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-PlayDead-Flat-MicroDuck"  # same base env and robot model as the sleepy trick
SCENE_XML = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/robot/microduck/scene.xml"

# Servo order = actuator order in the XML (and the policy's 14-D action).
SERVOS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
# HOME (STAND2), as in microduck_constants.HOME_FRAME.
HOME = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
])


def legs(hip_pitch: float, knee: float, ankle: float, hip_roll: float = 0.0) -> dict[int, float]:
    """Both legs from LEFT-leg values (the right leg mirrors the signs)."""
    return {1: hip_roll, 2: hip_pitch, 3: knee, 4: ankle,
            10: -hip_roll, 11: -hip_pitch, 12: -knee, 13: -ankle}


def head(neck_pitch: float, head_pitch: float, head_yaw: float = 0.0, head_roll: float = 0.0) -> dict[int, float]:
    return {5: neck_pitch, 6: head_pitch, 7: head_yaw, 8: head_roll}


def pose(*parts: dict[int, float]) -> np.ndarray:
    q = HOME.copy()
    for part in parts:
        for i, v in part.items():
            q[i] = v
    return q


# The SitStand keyframe (a crouch, really), as used by PlayDead.
SIT_LEGS = legs(-0.4079, 1.35, 0.0)

# Signs, from the first run of this study (24 Oct 2026), seen from the side:
#   neck_pitch −  head forward and down (−0.6 from the sit: beak on the floor)
#   neck_pitch +  head back (+0.9 from the sit tips the duck onto its back)
#   head_pitch +  nose down (the nod);  head_pitch −  nose up (the yawn)
#   hip_pitch more negative (left)  trunk folds forward

# name -> (servo targets, initial trunk pitch in rad, forward positive)
CANDIDATES: dict[str, tuple[np.ndarray, float]] = {
    "home": (pose(), 0.0),
    "sit": (pose(SIT_LEGS), 0.0),
    # Sign checks for the head, one joint at a time from the sit.
    "sit_neck+": (pose(SIT_LEGS, head(0.9, 0.3491)), 0.0),
    "sit_neck-": (pose(SIT_LEGS, head(-0.6, 0.3491)), 0.0),
    "sit_headp+": (pose(SIT_LEGS, head(0.3491, 1.1)), 0.0),
    "sit_headp-": (pose(SIT_LEGS, head(0.3491, -0.6)), 0.0),
    # Sign check for the legs: fold the hips forward from the sit.
    "sit_hips_fwd": (pose(legs(-0.9, 1.35, 0.0)), 0.3),
    # The yawn, sitting (standing poses need a policy to balance them).
    "yawn_sit": (pose(SIT_LEGS, head(0.55, -0.45)), 0.0),
    # The nods: each droop deeper, each catch recovering less.
    "nod1": (pose(SIT_LEGS, head(0.3491, 0.85)), 0.0),
    "nod1_catch": (pose(SIT_LEGS, head(0.3491, 0.55)), 0.0),
    "nod2": (pose(SIT_LEGS, head(0.1, 1.2)), 0.0),
    "nod2_catch": (pose(SIT_LEGS, head(0.2, 0.85)), 0.0),
    # Asleep, beak on the floor. A: face-plant with the head upright on its beak.
    "asleep_a": (pose(SIT_LEGS, head(-0.6, 0.3491)), 0.4),
    # B: chin tucked, nose down, beak resting near the feet.
    "asleep_b": (pose(SIT_LEGS, head(-0.35, 1.2)), 0.3),
    # C: bum high: straighter legs fold the trunk forward over the feet.
    "asleep_c": (pose(legs(-0.95, 1.0, -0.25), head(-0.45, 0.8)), 0.6),
    # D: B with the head rolled onto one cheek.
    "asleep_d": (pose(SIT_LEGS, head(-0.35, 1.2, 0.0, 0.38)), 0.3),
    # E: C with the head rolled onto one cheek.
    "asleep_e": (pose(legs(-0.95, 1.0, -0.25), head(-0.45, 0.8, 0.0, 0.38)), 0.6),
    # A to E rest on the FACE (camera) or the head's front edge. Chin rests
    # instead: neck far forward, head pitched back up so the beak (the orange
    # jaw under the head) lies on the floor.
    "chin_f": (pose(SIT_LEGS, head(-0.9, -0.3)), 0.3),
    "chin_g": (pose(SIT_LEGS, head(-1.1, -0.1)), 0.3),
    "chin_h": (pose(SIT_LEGS, head(-0.8, 0.15)), 0.3),
    "chin_g_bum": (pose(legs(-0.95, 1.0, -0.25), head(-1.1, -0.1)), 0.5),
    "chin_g_tilt": (pose(SIT_LEGS, head(-1.1, -0.1, 0.0, 0.38)), 0.3),
    # F to H were still face-down. Measured: the head's nose-down angle is about
    # −(neck − 0.349) + (head − 0.349) + trunk pitch, so a beak rest needs the
    # head pitched well back up as the neck goes forward.
    "beak_1": (pose(SIT_LEGS, head(-0.9, -0.9)), 0.3),
    "beak_2": (pose(SIT_LEGS, head(-1.0, -1.3)), 0.3),
    "beak_3": (pose(SIT_LEGS, head(-0.7, -0.6)), 0.3),
    "beak_1_bum": (pose(legs(-0.95, 1.0, -0.25), head(-0.9, -0.9)), 0.4),
    "beak_1_tilt": (pose(SIT_LEGS, head(-0.9, -0.9, 0.0, 0.38)), 0.3),
    "beak_2_tilt": (pose(SIT_LEGS, head(-1.0, -1.3, 0.0, 0.38)), 0.3),
    "beak_2_turn": (pose(SIT_LEGS, head(-1.0, -1.3, 0.45, 0.3)), 0.3),
    # Asleep with the head on its side, the way an animal settles to sleep.
    # Head yaw (range ±2.97 rad) rolls the head onto its cheek, as PlayDead's
    # +1.4 does.
    "side_1": (pose(SIT_LEGS, head(-0.9, -0.9, 1.2)), 0.3),
    "side_2": (pose(SIT_LEGS, head(-0.9, -0.9, 1.5)), 0.3),
    "side_3": (pose(SIT_LEGS, head(-1.0, -1.3, 1.3)), 0.3),
    "side_4": (pose(SIT_LEGS, head(-0.6, 0.0, 1.4)), 0.3),
    "side_5": (pose(SIT_LEGS, head(-0.9, -0.4, 1.4)), 0.3),
    "side_6": (pose(SIT_LEGS, head(-0.7, -0.9, 1.4, 0.3)), 0.3),
    # Run 1 averaged the three nods (the actor can't tell them apart without a
    # clock). A posture "counter": the head lolls further to one side with each
    # nod. Both signs, to pick the one that leads into the snuggle.
    "tilt+_nod1": (pose(SIT_LEGS, head(0.3491, 0.75, 0.0, 0.12)), 0.0),
    "tilt+_nod2": (pose(SIT_LEGS, head(0.25, 1.0, 0.0, 0.24)), 0.0),
    "tilt+_nod3": (pose(SIT_LEGS, head(0.1, 1.2, 0.0, 0.36)), 0.0),
    "tilt-_nod1": (pose(SIT_LEGS, head(0.3491, 0.75, 0.0, -0.12)), 0.0),
    "tilt-_nod2": (pose(SIT_LEGS, head(0.25, 1.0, 0.0, -0.24)), 0.0),
    "tilt-_nod3": (pose(SIT_LEGS, head(0.1, 1.2, 0.0, -0.36)), 0.0),
    # Stronger counter: the head also turns towards its sleeping side (the
    # snuggle's +yaw) a little more with each nod; the snuggle finishes it.
    "turn+_nod1": (pose(SIT_LEGS, head(0.3491, 0.75, 0.15, 0.12)), 0.0),
    "turn+_nod2": (pose(SIT_LEGS, head(0.25, 1.0, 0.30, 0.24)), 0.0),
    "turn+_nod3": (pose(SIT_LEGS, head(0.1, 1.2, 0.45, 0.36)), 0.0),
    "turn-_nod1": (pose(SIT_LEGS, head(0.3491, 0.75, 0.15, -0.12)), 0.0),
    "turn-_nod2": (pose(SIT_LEGS, head(0.25, 1.0, 0.30, -0.24)), 0.0),
    "turn-_nod3": (pose(SIT_LEGS, head(0.1, 1.2, 0.45, -0.36)), 0.0),
    # Beak-down with the head already partly turned (it arrives from nod 3 at
    # yaw 0.45, roll 0.36): does the slumped pose still rest on the beak?
    "beak_turn_a": (pose(SIT_LEGS, head(-0.9, -0.9, 0.45, 0.36)), 1.0),
    "beak_turn_b": (pose(SIT_LEGS, head(-0.9, -0.9, 0.45, 0.2)), 1.0),
    "beak_turn_c": (pose(SIT_LEGS, head(-0.9, -0.9, 0.45, 0.0)), 1.0),
    # The standing pose held open-loop: it falls without a policy balancing it.
    "home_unbalanced": (pose(), 0.0),
    # Standing yawn: head back, nose up. Only rendered (a stand needs a policy).
    "yawn": (pose(head(0.5, -0.25)), 0.0),
}
# The task's own sitting keyframes (microduck_sleepy_duck_env_cfg.KEYFRAMES),
# so the study checks exactly what training will ask for.
from mjlab_microduck.tasks.microduck_sleepy_duck_env_cfg import KEYFRAMES as _TASK_KEYFRAMES  # noqa: E402

for _k in _TASK_KEYFRAMES:
    if _k["pose"].get(3, 0.0) > 1.0:  # sitting (left knee bent)
        CANDIDATES[f"kf_{_k['t']:.1f}s"] = (pose(_k["pose"]), _k["pitch"])

# Standing poses fall over without a policy balancing them: they are rendered
# as posed, not settled.
KINEMATIC = {"home", "yawn"}
# The storyboard: one still per beat of the routine, with its caption.
STORYBOARD = [
    ("home", "1  Standing"),
    ("yawn", "2  Yawn (the beak opens: runtime)"),
    ("sit", "3  Settles into a sit"),
    ("nod1", "4  Nod 1: the head droops"),
    ("nod1_catch", "5  ...jerk! Awake"),
    ("nod2", "6  Nod 2: deeper"),
    ("nod2_catch", "7  ...jerk! (less awake)"),
    ("side_5", "8  Asleep, head on its side (snores: runtime)"),
]


def place_height(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, pitch0: float) -> float:
    """Highest trunk height at which the robot, posed at ``q``, touches the floor."""
    set_pose(model, data, q, pitch0, z=0.3)
    floor = model.geom("floor").id
    lo, hi = 0.0, 0.3
    for _ in range(30):
        data.qpos[2] = 0.5 * (lo + hi)
        mujoco.mj_forward(model, data)
        touching = any(floor in (c.geom1, c.geom2) and c.dist < 0 for c in data.contact[: data.ncon])
        lo, hi = (data.qpos[2], hi) if touching else (lo, data.qpos[2])
    return hi + 0.002


def pitch_quat(pitch: float) -> list[float]:
    """Trunk quaternion (w, x, y, z) leaning forward by ``pitch`` rad."""
    return [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]


def set_pose(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, pitch0: float, z: float) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[2] = z
    data.qpos[3:7] = pitch_quat(pitch0)
    for i, name in enumerate(SERVOS):
        data.qpos[model.joint(name).qposadr[0]] = q[i]


def floor_parts(model: mujoco.MjModel, data: mujoco.MjData, margin: float = 0.003) -> list[str]:
    """Mesh names of the robot parts within ``margin`` of the floor."""
    floor = model.geom("floor").id
    parts = set()
    for c in data.contact[: data.ncon]:
        if floor in (c.geom1, c.geom2) and c.dist < margin:
            g = c.geom2 if c.geom1 == floor else c.geom1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if not name and model.geom_dataid[g] >= 0:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])
            parts.add(name or f"geom{g}")
    return sorted(parts)


def soften_scene(model: mujoco.MjModel) -> None:
    """A plain warm floor instead of the blue checkerboard, for the stills."""
    mat = model.material("groundplane").id
    model.mat_texid[mat] = -1
    model.mat_rgba[mat] = [0.86, 0.82, 0.76, 1.0]
    model.mat_reflectance[mat] = 0.05
    model.vis.global_.offwidth, model.vis.global_.offheight = 1920, 1440


def render(model: mujoco.MjModel, data: mujoco.MjData, azimuth: float, elevation: float = -10.0,
           size: tuple[int, int] = (960, 720), focus: str | None = None, distance: float | None = None):
    """A still framed on the whole duck, or close up on one body (``focus``)."""
    from PIL import Image

    renderer = mujoco.Renderer(model, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if focus:
        cam.lookat[:] = data.xpos[model.body(focus).id]
        cam.distance = distance or 0.14
    else:
        bodies = data.xpos[1:]  # every robot body (0 is the world)
        lo, hi = bodies.min(axis=0), bodies.max(axis=0)
        cam.lookat[:] = 0.5 * (lo + hi)
        cam.distance = distance or (0.22 + 1.1 * float(np.max(hi - lo)))
    cam.azimuth = azimuth
    cam.elevation = elevation
    renderer.update_scene(data, camera=cam)
    image = Image.fromarray(renderer.render())
    renderer.close()
    return image


def contact_sheet(stills: list[tuple[str, object]], path: Path, cols: int = 4, width: int = 480) -> None:
    """Captioned grid of stills."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = width, width * 3 // 4
    rows = (len(stills) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), "white")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = ImageFont.load_default()
    for i, (caption, image) in enumerate(stills):
        tile = image.resize((w, h))
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, h - 34, w, h], fill=(255, 255, 255))
        draw.text((10, h - 30), caption, fill=(40, 40, 40), font=font)
        sheet.paste(tile, ((i % cols) * w, (i // cols) * h))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="candidates to run (default: all)")
    parser.add_argument("--out", type=Path, default=None, help="folder for the stills (default: none)")
    parser.add_argument("--storyboard", action="store_true", help="run the STORYBOARD beats, write storyboard.png")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--azimuth", type=float, default=140.0, help="camera azimuth of the front stills (180 = face on)")
    parser.add_argument("--elevation", type=float, default=-10.0, help="camera elevation of the front stills")
    parser.add_argument("--look", default="sim", choices=("sim", "warm"), help="sim: the blue shiny floor; warm: studio")
    parser.add_argument("--closeup", default=None, help="body to frame in a close-up still (e.g. ankle_left)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    captions = dict(STORYBOARD) if args.storyboard else {}
    names = [n for n, _ in STORYBOARD] if args.storyboard else (args.names or list(CANDIDATES))

    # Raw MuJoCo copy of the same robot: for placing it on the floor and for the stills.
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    if args.look == "warm":
        soften_scene(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1920, 1440
    data = mujoco.MjData(model)

    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = len(names)
    cfg.episode_length_s = args.seconds + 10.0
    # Keep every pose on screen: no episode ends early, whatever it does.
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env.reset()
    robot = env.scene["robot"]
    servo_ids, servo_names = robot.find_joints(r"^(?!passive_).*")
    assert list(servo_names) == SERVOS, servo_names
    qcols = torch.tensor([7 + j for j in servo_ids], device=env.device)

    targets = torch.zeros(len(names), 14, device=env.device)
    placed = []
    for b, name in enumerate(names):
        q, pitch0 = CANDIDATES[name]
        z = place_height(model, data, q, pitch0)
        placed.append(data.qpos.copy())
        root = torch.tensor([0.0, 0.0, z, *pitch_quat(pitch0)], device=env.device)
        root[:3] += env.scene.env_origins[b]
        env.sim.data.qpos[b, :7] = root
        env.sim.data.qpos[b, qcols] = torch.tensor(q, device=env.device, dtype=torch.float32)
        targets[b] = torch.tensor(q, device=env.device)
    env.sim.data.qvel[:] = 0.0
    env.sim.forward()
    actions = targets - robot.data.default_joint_pos[:, servo_ids]
    xy0 = robot.data.root_link_pos_w[:, :2].clone()

    for _ in range(int(round(args.seconds / env.step_dt))):
        env.step(actions)

    g = robot.data.projected_gravity_b
    pitch = torch.atan2(g[:, 0], -g[:, 2])
    roll = torch.asin(g[:, 1].clamp(-1.0, 1.0))
    feet = env.scene.sensors["feet_ground_contact"].data.found > 0
    beak = env.scene.sensors["head_ground_contact"].data.found > 0
    feet_n = feet.float().sum(dim=-1) if feet.dim() > 1 else feet.float()
    beak = beak.any(dim=-1) if beak.dim() > 1 else beak
    jaw_z = robot.data.body_link_pos_w[:, robot.find_bodies("jaw_soft")[0][0], 2]
    trunk_quat = robot.data.root_link_quat_w
    trunk_pos = robot.data.root_link_pos_w
    drift = (trunk_pos[:, :2] - xy0).norm(dim=-1)
    speed = robot.data.root_link_lin_vel_w.norm(dim=-1)
    joint_err = (robot.data.joint_pos[:, servo_ids] - targets).abs().max(dim=-1).values

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    print(f"{'pose':<16}{'pitch°':>8}{'roll°':>7}{'feet':>5}{'head':>5}{'head z':>8}{'bum z':>7}{'trunk z':>8}"
          f"{'drift':>7}{'speed':>7}{'max err':>8}  touching the floor")
    front, side = [], []
    for b, name in enumerate(names):
        if name in KINEMATIC:
            print(f"{name:<16}  (standing: rendered as posed; it needs a policy to balance)")
            data.qpos[:] = placed[b]
        else:
            # Copy the settled state into the raw model: which PART touches (the
            # env's head sensor is per body, and the whole head is one body).
            mujoco.mj_resetData(model, data)
            data.qpos[:7] = env.sim.data.qpos[b, :7].cpu().numpy()
            data.qpos[:3] -= env.scene.env_origins[b].cpu().numpy()
            for i, sname in enumerate(SERVOS):
                data.qpos[model.joint(sname).qposadr[0]] = env.sim.data.qpos[b, qcols[i]].item()
        mujoco.mj_forward(model, data)
        if name not in KINEMATIC:
            # Bum: 4 cm behind the trunk origin along the trunk's own x axis.
            w, x, y, z = trunk_quat[b].tolist()
            bum_z = trunk_pos[b, 2].item() - 0.04 * (2 * (x * z - w * y))
            print(f"{name:<16}{math.degrees(pitch[b]):8.1f}{math.degrees(roll[b]):7.1f}{int(feet_n[b]):5d}"
                  f"{'yes' if beak[b] else '-':>5}{jaw_z[b] * 100:7.1f}c{bum_z * 100:6.1f}c"
                  f"{(trunk_pos[b, 2] - env.scene.env_origins[b, 2]).item() * 100:7.2f}c"
                  f"{drift[b] * 100:6.1f}c{speed[b]:7.3f}{joint_err[b]:8.2f}  {', '.join(floor_parts(model, data))}")
        if args.out:
            caption = captions.get(name, name)
            front.append((caption, render(model, data, azimuth=args.azimuth, elevation=args.elevation)))
            side.append((caption, render(model, data, azimuth=90.0)))
            front[-1][1].save(args.out / f"{name}-front34.png")
            side[-1][1].save(args.out / f"{name}-side.png")
            if args.closeup:
                render(model, data, azimuth=args.azimuth, elevation=args.elevation, size=(1600, 1200),
                       focus=args.closeup).save(args.out / f"{name}-closeup-{args.closeup}.png")
    if args.out:
        stem = "storyboard" if args.storyboard else "sheet"
        contact_sheet(front, args.out / f"{stem}-front34.png")
        contact_sheet(side, args.out / f"{stem}-side.png")


if __name__ == "__main__":
    main()
