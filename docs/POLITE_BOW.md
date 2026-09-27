# PoliteBow: design notes

Task id: `Mjlab-PoliteBow-Flat-MicroDuck` (branch `polite-bow` of `microduck_rl`).

The files:

- `src/mjlab_microduck/tasks/microduck_polite_bow_env_cfg.py`: the task. Every knob you are likely to tune is in the CONSTANTS block at the top.
- `src/mjlab_microduck/tasks/bow_mdp.py`: the reward, cost and observation functions. The upstream convention is to put these in `mdp.py`. I kept them in their own small module so they are easy to read.
- `tests/test_polite_bow_cfg.py`: config invariants. These are the tests that catch the classic mistakes.
- `scripts/bow_eval.py`: a headless evaluation of the exported ONNX, with an optional MP4 video.

## The constraint that shapes everything

A community policy is published with `uv run publish --kind episodic`. On the robot the runtime then feeds it a **constant, all-zero 13-D command and no clock**, runs it for `duration_s`, and hands control back to the walk/stand policy. The actor therefore has to perform the bow from its own state alone, as the Roulade does.

## How it is taught

1. **A hidden reference clock.** The environment knows the episode time and turns it into a smooth bow profile, `blend(t)`:
   - it goes from 0 to 1 between 0 and 1.2 s;
   - it holds at 1 until 1.6 s;
   - it returns to 0 by 2.9 s;
   - the robot then stands still until 3.5 s.
2. **Rewards track the moving target:**
   - `bow_pitch` (main term): trunk pitch follows `blend × 18°`.
   - `bow_neck_pose`: neck and head follow `blend × (neck +0.35, head +0.30 rad)`.
   - `bow_leg_home`: a weak, generous pull back to the HOME legs.
   - `feet_planted`, and costs for drift, roll, head strikes, tilted feet, action rate and self-collision.
3. **The critic sees the clock and the actor does not.** `bow_clock` is a critic-only observation. Only the actor is exported, so the 61-D deployment contract holds, and a test enforces this.
4. **There are no jackpots.** Nothing pays for arriving early. Tracking a moving target makes slow and smooth the best strategy (AGENTS.md).
5. **Legs are discovered, not hand-posed.** Only the trunk pitch and the neck/head dip are specified. The policy finds the hip/ankle configuration that keeps the centre of mass over the feet.

## What was measured before choosing the numbers

- **Open-loop holding does not work, even at HOME.** Holding any pose open-loop, HOME included, falls within about 2 s under BAM actuators and actuator delay (I swept 40 variants). So equilibrium has to be judged closed-loop, and hand-picked leg angles are unreliable. That is why the legs are left to RL.
- **Folding the hips forward raises the measured trunk pitch.** This confirms the sign of `trunk_pitch()`.
- **18° is a modest step beyond something already proven.** The official StandUp policy is trained to track body-pitch commands of ±15°.
- **The head is about 38% of the robot's mass.** Dipping it moves the centre of mass forward a lot, so the neck and head dip are kept moderate.

## The known risk, and what to do if it bites

The actor cannot see the clock, so the bottom "hold" is ambiguous. The policy may linger at the bottom or rise early. If the evaluation shows the pitch curve lagging the reference badly:

1. Shorten the hold (`hold_end`), or remove it.
2. Lengthen the ramps: slower is easier to time from state.
3. As a last resort, put a phase clock in the twist slot, as GroundPick does. That learns easily, but the policy is then **not publishable** as a community policy: your runtime would have to drive the phase.

## How to judge a run

- **In the log** (`status.sh` does the sign check):
  - every `*_cost` must read ≤ 0;
  - `Episode_Reward/bow_pitch` should climb;
  - `Episode_Termination/fell_over` should fall towards 0;
  - mean episode length should reach 175 steps (3.5 s at 50 Hz).
- **With `export.sh --video`:**
  - fall rate close to 0% in both the play and the training-DR evaluations;
  - peak pitch around 18°;
  - final pitch under 3°;
  - no foot lifts;
  - no head strikes.

  Then watch the MP4. Sim metrics can pass while the motion looks wrong.
- **Budget:** 1,000 to 3,000 iterations at 4,096 envs, roughly 1 to 2 hours on a 3080 Ti (an estimate, not yet measured).

## Bowing from a walk (v2)

The first run (2026-09-26_20-56-52) passed every still-stand check, but in the
Mac rehearsal it fell over every time it was triggered while the duck was not
perfectly still: it had only ever practised from a dead-still stand.

v2 starts half of the training episodes from **real mid-walk states**:

- `scripts/make_gait_bank.py` runs the official walking policy
  (`alpha_walking.onnx` from `pollen-robotics/microduck-policies`) inside this
  same env with random velocity commands. It saves qpos, qvel and last action
  from random moments of the gait to `data/polite_bow_gait_bank.pt`
  (gitignored).
- `bow_mdp.reset_from_gait_bank` spawns `GAIT_BANK_PROB` (0.5) of the
  episodes from that bank, after the still-stand reset. The walker's last
  action is handed over into the `actions` observation, as the runtime does
  (`bow_mdp.last_action_with_handoff`).
- Still stands now really get `STAND_JOINT_NOISE_STD` servo noise. The
  roulade reset's `joint_noise_std` only ever applied to its mid-roll bucket.
- There is no "settle, then bow" window. The actor has no clock, so from a
  still stand it could not know when the wait ends. The smoothstep ramp is
  gentle enough to absorb the stride. The published duration stays 3.5 s.

```bash
# on the GPU box, once per walking policy
uv run python scripts/make_gait_bank.py --onnx policies/alpha_walking.onnx
uv run train Mjlab-PoliteBow-Flat-MicroDuck --env.scene.num-envs 4096
# judge it both ways
uv run python scripts/bow_eval.py --onnx exports/politebow-v2.onnx --num-envs 256 --device cuda:0
uv run python scripts/bow_eval.py --onnx exports/politebow-v2.onnx --num-envs 256 --device cuda:0 --from-walk
```

`--from-walk` must show falls close to 0%, the same as a still-stand start.

## Deploying it

```bash
# rehearse on the Mac, hand-off exactly like a trick (press R to trigger):
uv run scripts/infer_policy.py --walking walk.onnx --standing stand.onnx \
    --roulade exports/politebow-flat-N.onnx --roulade-duration 3.5 --new-cmd-obs
# publish for the robot:
uv run publish --onnx exports/politebow-flat-N.onnx --repo <you>/microduck-polite-bow \
    --kind episodic --duration-s 3.5 --description "Bows politely and stands back up."
```
