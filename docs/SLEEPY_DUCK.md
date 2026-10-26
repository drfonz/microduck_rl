# Sleepy Duckling: design notes

Task id: `Mjlab-SleepyDuck-Flat-MicroDuck` (branch `sleepy-duck` of `microduck_rl`, built on `main`, which carries PoliteBow and PlayDead).

The trick: the duck yawns, settles into a sit and starts to nod off. Each nod goes a little deeper, with a small jerk as it catches itself, until it falls asleep with its beak on the floor and its bum in the air. A beat later it snuggles, rolling its head onto its cheek. Then it sleeps (and, on the robot, snores: see "The beak").

![Storyboard](sleepy_duck_storyboard.png)

The files:

- `src/mjlab_microduck/tasks/microduck_sleepy_duck_env_cfg.py`: the task. The poses and the timeline are in the CONSTANTS block at the top.
- `src/mjlab_microduck/tasks/sleepy_duck_mdp.py`: the two terms PlayDead lacks (a two-way "tipped over" termination and a clock-gated "head resting" reward). Everything else is reused from `play_dead_mdp.py`.
- `tests/test_sleepy_duck_cfg.py`: config invariants, including "each nod deeper than the last".
- `scripts/sleepy_pose_study.py`: drops the duck into each pose in the training env (BAM servos) and checks where it settles; renders stills and the storyboard.
- `scripts/sleepy_duck_eval.py`: headless evaluation of the exported ONNX, with an optional MP4. It counts the nods and measures how sharp the jerks are.

## The timeline

| Time (s) | Beat | Head (nose-down vs trunk) | Head turn (yaw / roll) | Trunk pitch |
|---|---|---|---|---|
| 0.0 | standing (HOME) | 0° | 0° / 0° | 0° |
| 0.7–1.0 | yawn: nose up | −37° | 0° / 0° | 0° |
| 2.3 | sat down (the SitStand keyframe) | 0° | 0° / 0° | 0° |
| 3.1 / 3.4 | nod 1: droop / jerk | 29° / 4° | 9° / 7° | 0° |
| 4.3 / 4.6 | nod 2: droop / jerk | 43° / 21° | 17° / 14° | ~2° |
| 5.6 / 5.9 | nod 3: droop / jerk | 63° / 40° | 26° / 21° | ~4° |
| 7.5 | slumped asleep: beak on the floor, bum highest | — | 26° / 21° | 63° |
| 7.8 | a beat | — | 26° / 21° | 63° |
| 8.7 | snuggled: lying on its cheek | — | 80° / 0° | 60° |
| 10.5 | fast asleep: the episode ends | — | 80° / 0° | 60° |

Each keyframe blends into the next with a smoothstep. The droops are slow (0.8 to 1.0 s) and the jerks fast (0.3 s).

## Design choices

- **Three nods**, not two: more comic build-up, at some cost in learnability (below).
- **Head on its side when asleep**, the way an animal settles to sleep, rather than resting on the beak with the head straight (the first candidates).
- **Each nod turns the head a little further towards its sleeping side** (added for run 2, see below); the snuggle finishes the turn.
- **The snuggle comes after landing** (lands on its beak, then rolls its head onto its cheek), rather than turning on the way down. It is more expressive; the cost is that the head rolls along the floor, so `head_impact_cost` prices it (it charges head speed only while touching).
- **Snoring** once asleep: the runtime opens and closes the beak with a snore sound.

## Run 1 → run 2

**Run 1** (the design above without the head turn during the nods), checked at iteration 500 with the headless eval (64 ducks, from a still stand):

- learnt: no falls at all; the yawn almost exactly (−27°, −36°, −33° against −24°, −37°, −33°); the sit;
- **no nods:** the head sank slowly and steadily from 7° to 34° nose-down: the average of the three nods. Nod depth alone did not tell the nods apart: at the same head angle, nod 1 and nod 2 differ by only ~0.1 rad of neck;
- **no slump:** trunk upright at the end (2° against 58°), head never on the floor.

**Run 2** changes:

1. **A posture counter for the nods:** each nod turns the head further towards its sleeping side (yaw 9°, 17°, 26°; roll 7°, 14°, 21°), the catches keep the turn, and the snuggle finishes it. The head-turn term is tighter (std 0.25) with a stronger L1 pull (−0.5).
2. **Mid-routine starts (reverse curriculum):** 30% of training episodes start at rest in a keyframe pose (just sat, after nod 2, at the start of the slump, slumped on the beak; weighted 1:1:2:2), with the episode clock set to that keyframe. `bow_mdp._episode_time` accepts a per-env clock offset (no effect on tasks that don't set one) and a clock-based time-out ends those episodes with the routine. Mid-walk starts drop from 40% to 30%.
3. **Eval video** turns the duck to face the camera (resets give it a random heading) and reports the head turn at each nod.

## Run 2 → run 3

**Run 2**, headless eval from a still stand (play config):

| | Iteration 500 | Iteration 1500 |
|---|---|---|
| Asleep at the end (trunk ≈ 60°, head on the floor, feet down) | 64% | 91% (66% with training DR) |
| Slump (trunk pitch rising) | from ~8 s, late | ~7 to 9.5 s (reference 5.9 to 7.6 s) |
| Nods counted | 0 | 0 |
| Head turn at the nods (target 9°, 17°, 26°) | 3°, 9°, 15° | 6°, 12°, 19° |
| Final head yaw (target 80°) | 26° | 14° |
| Head landing speed (median) | 0.22 m/s | 0.20 m/s |

The training reward was flat from iteration 500 to 1500.

**Is the snuggle physically possible?** Open loop, with no policy: from the settled beak-down pose, sweeping the servo targets to the snuggled pose over 0.9 s (or 2 s) rolls the head onto its cheek (yaw 89° against 80°, head speed at most 0.16 m/s). So the snuggle is a learning problem, not friction: while the slump lands late, a full episode leaves no time for it.

**Run 2 at iteration 2999:** asleep at the end in 100% of episodes from a still stand (70% with training DR), slump nearly on time (pitch RMSE 9.6°), head turn at the nods 5°, 14°, 24°; one nod in 40% of ducks, never three; no snuggle (it turns the head *back* to straight at the end).

**The hidden cost.** Reading every reward term in the settled asleep poses: `self_collisions` was −300 per second on the beak (−150 snuggled), against +142 per second for the whole pitch term. The term counts *any* touch between robot parts (the sensor has no force history, so no threshold), and the SitStand sit keyframe rests each ankle bracket against its hip: 6 contacts, all the time the duck sits or sleeps. At PlayDead's weight (−1.0) the reference poses were simply too expensive to reach; the policy sat shallow and dodged them. A search for a contact-free sit found none that stays put through the nods (under load the ankle meets the hip anyway), and that resting contact is what makes the sit so stable. So the sit stays and the weight drops to −0.05.

**Run 3** (fine-tuned from run 2's last checkpoint):

1. **Starts at each nod's lowest point** (and in the snuggled pose): run 2 averaged the nods and so never reached the bottom of a droop, where the catch has to happen. Mid-routine share 30% → 40%.
2. **Jerks 0.3 s instead of 0.2 s:** the actor has to time each catch from its own pose; a wider feature forgives small timing errors.
3. **Head tracking tighter and heavier** (std 0.25 → 0.2, weight 3 → 4): the average of the nods costs more.
4. **Self-collision weight −1.0 → −0.05**, see above.

## Run 3 → run 4

**Run 3** (fine-tuned from run 2), headless eval from a still stand:

| | Iteration 500 | Iteration 1499 |
|---|---|---|
| Asleep at the end | 100% | 100% (53% with training DR) |
| Nods counted (histogram 0/1/2/3) | 10 / 38 / 208 / 0 | 0 / 0 / 256 / 0 |
| Jerk speed (reference 1.85 rad/s) | 1.54 rad/s | 1.14 rad/s |
| Head turn at the nods (target 9°, 17°, 26°) | 7°, 17°, 25° | 9°, 18°, 25° |
| Pitch tracking RMSE | 4.9° | 4.8° |
| Head down before 6.9 s | 22% | 73% (the slump now lands ~0.3 s early) |
| Snuggled | 0% | 0% |

The nods arrived (the catch practice and the self-collision fix), and the slump is on time. Two gaps remain: the duck skips **nod 1** (the smallest, 23° → 9°), and it never **snuggles**. The snuggle shares the tight head-turn term (std 0.25) with the nod counter, and it starts ~1 rad from its target, where that Gaussian is flat.

**Run 4** (fine-tuned from run 3's last checkpoint):

1. **A separate, wide snuggle term:** `sleepy_head_turn` (std 0.25) until the snuggle starts (7.8 s), then `sleepy_snuggle` (std 0.6, weight 3); the L1 pull stays on throughout.
2. **Nod 1 is now the biggest jerk** (29° → 4°).
3. **Head contact allowed from 6.5 s** (was 6.9 s): run 3's early landing is harmless, and the nods end at 5.9 s.

## Run 4 → run 5: the whole trick

**Run 4** (fine-tuned from run 3), iteration 500: **three nods in 100% of episodes**, jerks at 1.89 rad/s (reference 1.99), head turn at the nods 9°, 18°, 24°, asleep 100%. Still no snuggle: the wide snuggle term averaged yaw and roll, and the policy took the roll half (roll held at 0) without turning (final yaw 5°).

**Run 5** (fine-tuned from run 4's iteration 500): the snuggle term is yaw only (std 1.0, weight 3), with a linear pull on yaw (−1 per rad) and a separate roll term (std 0.25), all from the snuggle's start.

Headless evals, 256 ducks each (training DR = the training config's randomisation, every episode from t = 0):

| | Iteration 500 | Iteration 1000 | Iteration 1499 |
|---|---|---|---|
| Three nods: stand / training DR / mid-walk | 99% / 100% / 99% | 100% / 100% / 98% | 100% / 100% / 100% |
| Jerk speed (reference 1.99 rad/s) | 1.72 | 2.06 | 2.03 |
| Head turn at the nods (target 9°, 17°, 26°) | 9°, 17°, 26° | 9°, 20°, 28° | 9°, 17°, 27° |
| Asleep / snuggled, stand (and training DR) | 100% / 100% | 100% / 100% | 100% / 100% |
| Asleep / snuggled, mid-walk + training DR | 98.4% / 98.4% | 98.8% / 98.0% | 98.8% / 98.8% |
| Toppled, mid-walk + training DR | 1.2% | 1.2% | 1.2% |
| Head landing speed: median, worst (mid-walk) | 0.20, 0.83 m/s | 0.19, 1.03 m/s | 0.17, 0.45 m/s |
| Pitch tracking RMSE | 6.3° | 6.0° | 6.3° |

**Candidate policy: run 5, iteration 1499.** It is reached by a chain of fine-tunes (run 2 → run 3 → run 4 @500 → run 5), so the config at the tip of this branch is being retrained from scratch to check that it reproduces the trick on its own.

**Note on the evaluator:** until commit `56775a66`, `sleepy_duck_eval.py --train-dr` used the training config with its mid-walk and mid-routine starts; those episodes end early and their clocks don't match the evaluator's, so the training-DR figures for runs 2 to 4 above are understated.

**Note on the command block:** the table above was measured with the training env's commands. The roulade family keeps mjlab's velocity default `rel_forward_envs = 0.2`, so at every command resample a fifth of the ducks get a forward command of at least 0.3 m/s, whatever the ±0.01 ranges say (training included). The robot's runtime feeds an episodic trick exact zeros, which the evaluator now does by default (commit `e13b60b0`; `--train-commands` for the old behaviour). Run 5 at iteration 1499 re-evaluated with zeros, 1,024 ducks each:

| | Toppled | Asleep / snuggled | Head landing: median, 99th percentile, worst |
|---|---|---|---|
| Stand (play) | 0% | 100% / 100% | 0.17, 0.45, 0.47 m/s |
| Stand, training DR | 0% | 100% / 100% | 0.18, 0.45, 0.47 m/s |
| Mid-walk + training DR, zero commands (two runs) | 2.0%, 1.2% | 97.9% / 97.5%, 98.7% / 98.2% | 0.16, –, 1.90; 0.17, 0.46, 1.16 m/s |
| Mid-walk + training DR, training commands | 1.4% | 98.6% / 98.1% | 0.17, –, 1.86 m/s |

Three nods in 100% from a stand, head turn 9°, 17°, 27°, jerks 2.03 rad/s: the zero commands change nothing measurable. The worst head landing grows with the number of ducks (0.45 m/s over 256 mid-walk ducks, up to 1.9 m/s over 1,024), so the evaluator now also reports the 99th percentile.

## Run 6 → run 7: does it work from scratch?

**Run 6** trained the config at the tip of the branch (commit `29a96065`) from scratch, 3,000 iterations. Headless evals, 256 ducks:

| | Iteration 2000 | Iteration 2500 | Iteration 2999 |
|---|---|---|---|
| Asleep / snuggled, stand | 91% / 91% | 98% / 97% | 98% / 99% |
| Asleep / snuggled, mid-walk + training DR | 91% / 84% | 95% / 95% | 93% / 94% |
| Nods counted (0 / 1 / 2 / 3) | 255 / 0 / 0 / 0 | 230 / 26 / 0 / 0 | 228 / 28 / 0 / 0 |
| Pitch tracking RMSE | 19.6° | 15.1° | 14.1° |

It falls asleep and snuggles, but **never learns the nods**: the staged runs were needed. One difference stands out. Run 6's smoothing penalties ramp up (action rate −0.2 → −0.4, torque rate 0 → −1e-3 by iteration 2,000), while each fine-tune in the staged chain restarted those ramps, so the chain never trained with more than −0.3 / −5e-4. Heavy smoothing rounds off the jerks.

**Run 7** trains from scratch again with the action-rate penalty held at −0.2 and the torque-rate penalty capped at −5e-4.

Headless evals, 256 ducks each, zero commands:

| | Iteration 1500 | Iteration 2000 | Iteration 2500 | Iteration 2999 |
|---|---|---|---|---|
| Asleep / snuggled, stand | 100% / 100% | 100% / 99.6% | 99.2% / 100% | 100% / 100% |
| Asleep / snuggled, mid-walk + training DR | 97.7% / 97.3% | 96.5% / 95.3% | 96.9% / 97.3% | 96.1% / 94.1% |
| Three nods | 0% | 0% | 0% | 0% |
| Head turn at the nods (target 9°, 17°, 26°) | 8°, 13°, 20° | 7°, 12°, 19° | 9°, 14°, 19° | 8°, 14°, 20° |
| Pitch tracking RMSE | 13.2° | 15.2° | 16.7° | 14.2° |

**No nods, again.** Holding the smoothing penalties low did not help: from a fresh start, the final config learns to fall asleep and snuggle but never the nods. The video shows why. The duck barely sits (it stays up on bent knees), lowers its head in one slow droop through the nod window (17° → 32° nose down while the reference swings 26° → 5° → 42° → 21° → 61°), which is the three nods averaged, as in run 1, and then tips onto its beak. The staged chain (run 2 → run 3 → run 4 @500 → run 5) taught the slump first, then the catches from droop-peak spawns, then the snuggle, each stage starting from the previous policy; that order is what this task needs. Until a single-run curriculum reproduces it, the trick's recipe is the staged chain.

## How it is taught

It works the same way as PlayDead: a hidden reference clock, a moving target, the clock given to the critic only, and no jackpots.

- **Head** (neck pitch + head pitch): the performance, so the tightest tracking (std 0.25 rad, weight 3) plus an L1 pull.
- **Trunk pitch:** upright through the nods, then the slump. This makes the slump a lowering, not a fall.
- **Legs:** stand, then sit, then stay sat. Generous std, as in PlayDead.
- **Head turn** (head yaw and roll): the nod counter, then the snuggle. Its own term, std 0.25 plus an L1 pull. PlayDead's head turn, at std 0.3 with no L1, only learnt after ~1,600 iterations.
- **Feet planted** for the whole episode, and **head resting** paid by the clock once the beak should have landed (never by state, so it cannot be farmed early).
- **Safety costs:** head speed while touching the floor (a slam or a scrub is expensive, resting is free), the head touching down before 7 s, roll, drift.
- **Termination:** the asleep pose leans forward ~60°, so the usual `fell_over` is removed. Tipping onto the back (< −40°, the yawn's risk), onto a side (> 45°) or on over the face (> 85°) ends the episode.
- **Smoothing** is held low: action rate at −0.2 throughout, torque rate capped at −5e-4 (see run 6 → run 7). Heavy smoothing rounds off the jerks, and the jerks are the joke.
- **Symmetry loss is off:** the head snuggles onto one side only.
- **Mid-walk starts:** 40% of training episodes start from a real mid-walk state (PoliteBow v2's gait bank).

## What was checked in sim before choosing the numbers

With `scripts/sleepy_pose_study.py`: each pose dropped into the training env (groundcontact model, BAM servos) and held as a constant target for 3 s.

- **The plain MuJoCo servos are useless for this:** too soft to hold even the standing pose. Everything below uses the BAM model.
- **Standing needs a policy:** held open-loop, HOME tips onto the face within a second. The sit holds.
- **Signs:** neck_pitch − brings the head forward and down; head_pitch + is nose down. The head's nose-down angle is about −(neck − 0.349) + (head − 0.349) + trunk pitch.
- **Every sitting keyframe settles still** with both feet down (the `kf_*` candidates are built from the task's own keyframes).
- **The yawn:** neck +0.55 / head −0.45 (57° nose up) tips the *sitting* duck onto its back, so the yawn is done standing, at 37°.
- **Asleep:** the first candidates rested on the face (the camera!). Pitching the head back up as the neck goes forward puts the beak (the orange jaw) on the floor instead: trunk 56–62°, bum the highest point. With head yaw +1.4 the head lies flat on its cheek, face turned to the duck's left.

Not checked: the transitions. Whether the duck can slump softly, and roll its head without scrubbing, is what training finds out.

## Run 8: the staged chain in one run

The staged chain differed from the from-scratch runs in **when** things arrived, not in its final rewards: each fine-tune added one lesson to a policy that already did the previous one, and each restarted the torque-rate penalty at zero (so run 3 found the catches, and run 4 sharpened them, with no tax on quick torque changes) and the centre-of-mass randomisation small. Run 8 (commit `f37eee9a`) replays that inside one run of 5,000 iterations:

| Stage | Iterations | Head tracking | Mid-routine starts | Snuggle terms | Torque-rate penalty | CoM randomisation |
|---|---|---|---|---|---|---|
| The slump | 0–1,500 | run 2's: std 0.25, weight 3 | just sat, slump starts, on the beak | off | 0, then −5e-4 from 1,000 | ramps from 3 mm |
| The catches | 1,500–3,500 | std 0.2, weight 4 | + the bottom of each nod | off | 0 | restarts at 3 mm |
| The snuggle | 3,500–5,000 | std 0.2, weight 4 | + snuggled | on | 0, then −5e-4 from 4,000 | restarts at 3 mm, 10 mm by the end |

The mid-routine share stays 0.4 throughout (and 0 in play and evaluation). `tests/test_sleepy_duck_cfg.py::test_one_run_three_stages` drives every curriculum term through the stages; a live check on the GPU box did the same on the real env.

**It works: the whole trick in one run.** Headless evals, 256 ducks each, zero commands:

| | Iteration 1500 | 2500 | 3500 | 4000 | 4500 | 4999 |
|---|---|---|---|---|---|---|
| Nods counted (0 / 1 / 2 / 3 / more), stand | 238 / 17 / 1 / 0 | 160 / 87 / 9 / 0 | 2 / 82 / 152 / 20 | 0 / 0 / 7 / 248 / 1 | 0 / 0 / 0 / 234 / 22 | 0 / 0 / 2 / 254 |
| Asleep / snuggled, stand | 99% / 0% | 100% / 0% | 100% / 0% | 0% / 100% | 100% / 100% | 100% / 100% |
| Jerk speed (reference 1.99 rad/s) | – | – | 0.90 | 1.91 | 2.00 | 1.69 |
| Pitch / head tracking RMSE | 11.4° / 16.7° | 8.3° / 10.8° | 8.0° / 9.9° | 5.9° / 9.3° | 7.8° / 10.2° | 5.9° / 7.6° |

(The snuggle is off until iteration 3,500 by design; at 4,000 it snuggles before settling fully into the asleep pose.) Iteration 4999 with 1,024 ducks each, zero commands, against run 5's iteration 1499:

| | Run 8, iteration 4999 | Run 5, iteration 1499 |
|---|---|---|
| Three nods, stand / training DR | 99% / 99% | 100% / 100% |
| Asleep / snuggled, stand (and training DR) | 100% / 100% | 100% / 100% |
| Mid-walk + training DR: asleep / snuggled / toppled | 97.9% / 96.3% / 2.1% | 98.3% / 97.9% / 1.6% (two runs) |
| Three nods, mid-walk + training DR (of the ducks still up) | 97% | – |
| Head turn at the nods (target 9°, 17°, 26°) | 9°, 18°, 25° | 9°, 17°, 27° |
| Jerk speed | 1.67 rad/s | 2.04 rad/s |
| Head landing: median, 99th percentile (mid-walk) | 0.15, 0.69 m/s | 0.17, 0.46 m/s |
| Pitch / head tracking RMSE, stand | 5.6° / 7.2° | 6.5° / 10.7° |

So the recipe is one command:

```bash
uv run train Mjlab-SleepyDuck-Flat-MicroDuck --env.scene.num-envs 4096   # 5,000 iterations, about 4 h 50 on an RTX 3080 Ti
```

Run 8 tracks the routine more closely; run 5's catches are a little snappier and its worst landings mid-walk a little gentler.

**But it nods standing up.** Watching the video, it does the nods on its feet and only sits as it falls asleep, the opposite of the routine (sit from 2.3 s, then nod), and a duck nodding off on its feet lands from standing height. Measured over the routine (play config, 256 ducks): run 8 first gets down to sitting height at **6.24 s** and spends **0%** of the nod window (2.3–5.9 s) seated; run 5 sits at **4.26 s**, seated for 45% of it. The leg pose term is a mean of per-joint Gaussians over ten joints, and only the two knees tell standing from sitting, so standing cost under a point a step against the head, pitch and head-turn terms. None of the evaluator's metrics measured the sit.

## Run 9: sit down before the nods

Commit `85c2490a`, run 8's three stages plus:

- **`sleepy_height`** (weight 3, std 15 mm): Gaussian tracking of the trunk height, from a new `"z"` per keyframe: standing 115 mm, sat 61 mm (from 2.3 s), on the beak 66 mm, snuggled 68 mm, all measured in sim.
- **A seated gate on the nods**: from 2.3 s to the slump, the head, coarse head and head-turn terms are multiplied by a Gaussian on the trunk height (std 20 mm), so a nod only pays while sitting.
- **The evaluator reports the sit**: when the trunk first gets below 73 mm, and the share of the nod window spent below it.

**It works: the whole routine, in order, from one run.** Headless evals, zero commands:

| | Iteration 1500 | 3500 | 4000 | 4500 | 4999 (1,024 ducks) |
|---|---|---|---|---|---|
| Sat down at (reference 2.3 s) / nod window seated | 1.80 s / 100% | 1.76 s / 100% | 1.84 s / 100% | 1.82 s / 100% | 1.78 s / 100% |
| Three nods, stand / training DR | 0% / 0% | 98% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| Asleep / snuggled, stand | 75% / 0% | 100% / 0% | 97% / 39% | 100% / 100% | 100% / 100% |
| Jerk speed (reference 1.99 rad/s) | – | 2.29 | 2.12 | 1.81 | 2.21 |
| Head turn at the nods (target 9°, 17°, 26°) | 6°, 10°, 14° | 10°, 21°, 29° | 8°, 17°, 26° | 10°, 17°, 27° | 9°, 17°, 26° |

Iteration 4999, mid-walk with training DR (1,024 ducks): **0% toppled**, asleep 100%, snuggled 99.9%, three nods 99%, sat down at 1.80 s; head landing median 0.20 m/s, 99th percentile 0.45 m/s. From a stand: head landing median 0.20 m/s, 99th percentile 0.45, worst 0.48. Training took 5 h 14 m on an RTX 3080 Ti (114 days of simulated practice).

**Candidate policy: run 9, iteration 4999**, the config at commit `85c2490a`, trained from scratch with one command:

```bash
uv run train Mjlab-SleepyDuck-Flat-MicroDuck --env.scene.num-envs 4096
```

## The known risks

1. **Telling the nods apart.** The actor has no clock. Depth alone was not enough in run 1; run 2 adds the head-turn counter. If the eval still shows fewer than three nods, make the turn steps bigger (or go back to two nods).
2. **The last mile.** Run 1 never slumped by iteration 500; run 2 starts 30% of episodes mid-routine. If it still never slumps from a full episode, raise the share of slump-start spawns.
3. **The jerks get smoothed away** by the action-rate curriculum. Lower the final stage.
4. **The head slam or scrub.** If `head impact speed` stays above about 0.5 m/s, raise `sleepy_head_impact_cost` or lengthen the slump.
5. **The beak.** The jaw has no servo in the 14-joint action space, so the yawn's open beak and the snoring are the runtime's job, as for PlayDead.

## How to judge a run

- **In the log** (`status.sh`): every `*_cost` ≤ 0; `sleepy_head` and `sleepy_pitch` climbing; `sleepy_snuggle` climbing late; `tipped_over` falling towards 0; mean episode length reaching 525 steps.
- **With `fetch.sh`** (eval + video): toppled ≈ 0%, nods = 3, asleep and snuggled ≈ 100%, head impact low. Then watch the MP4.
- **Budget:** 3,000 iterations at 4,096 envs (about 2 to 3 hours). Checkpoints every 50 iterations, to see how the behaviour develops.

## Deploying it

```bash
uv run publish --onnx exports/sleepyduck-flat-N.onnx --repo drfonz-indigo/microduck-sleepy-duckling \
    --kind episodic --duration-s 10.5 --description "Yawns, sits, nods off three times and falls asleep on its cheek."
```

The final pose is static, so the runtime can keep the policy running while the duck sleeps, then hand over to the stand policy to wake it.
