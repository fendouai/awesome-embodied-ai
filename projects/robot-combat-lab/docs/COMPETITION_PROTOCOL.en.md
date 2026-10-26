# Competition timing, evaluation and commentary record (v2)

Date: 2026-10-26. Round-04 records a new declared protocol; previous 30-second visual demonstrations remain in round-03.

The local preset uses a 180-second upper limit for MuJoCo boxing and four independent 180-second K1 visual soccer matches. Soccer has 20 seconds of initialization, then the native READY → SET → PLAYING flow. Native goal, out-of-bounds and finish decisions are retained; the kickoff side alternates. The upstream 600-second period is restored as the source default, with local duration selected per batch.

Official references are documented separately: RoboStriker's pinned source has a 50-second episode; NCP's public configuration has 500 control steps at 30Hz (about16.67 seconds); HSL specifies two ten-minute periods. Local MuJoCo ports and local contest scores are not presented as a reproduction of native source benchmarks. Source references: [RoboStriker](https://yinkangning0124.github.io/RoboStriker/), [NCP configuration](https://github.com/Tencent-RoboticsX/NCP/blob/master/ncp/data/cfg/env/humanoid_boxing.yaml), [HSL rules](https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules/blob/main/rules/game.tex).

Boxing records valid upper-body hand contacts, short-window normal impulse, falls, recovery and upright duration. Ten-second recovery initiation and one-second stable-standing confirmation can produce technical stoppages or mutual-recovery technical draws. Otherwise the 180-second local score decides. All controls and physics integration steps are archived and independently replayed. These are contest outcomes, not hardware-damage predictions. NCP characters and different competition tracks stay separate.

Visual recordings include native game states, remaining time, goals, ball position and six player positions. Scores reset between matches. A monotonic physics-step clock is required; resets reject publication instead of stitching scenes. Ball motion during PLAYING provides a separate activity-quality check.

Each completed match receives the original MP4, a Mandarin narrated MP4 with burned captions, SRT/VTT, 48kHz narration WAV, a transcript and evidence-linked commentary metadata. Local macOS Tingting synthesis is used. Speech describes only logged facts; timing derives from actual audio lengths. Opening and closing freeze frames are recorded explicitly, while the match retains its original order and speed.

The machine-readable protocol is `configs/competition/protocol-v2.json`. Execution scripts are `run_simulation_series_v2.py`, `finish_visual_series_v2.py`, `render_native_soccer_v2.py`, and `produce_match_commentary.py`. Artifacts and progress indexes are under `results/tournaments/round-04/`.

Optional display-only mesh LOD uses the project's `[render]` dependencies. Source collision models and integration trajectories remain exact. Rendered body positions and rotations are compared with the archived model at initial, middle and final states before publishing; the display-only changes are recorded separately. Both K1 soccer sides use the installed B-Human program with distinct team settings, not two independently installed vision-model families.

The180-second limit is the scoring window. A recovery count already pending at the buzzer may extend by at most11seconds without additional scoring. Edited source footage begins at READY; initialization telemetry and raw camera captures remain available. Native soccer logs contain the documented referee, ball and player-position fields, not a full joint-state/control physics replay archive.
