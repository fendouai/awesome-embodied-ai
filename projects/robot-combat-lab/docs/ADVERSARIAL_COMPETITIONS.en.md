# Three Robot-Adversarial Tracks: Distinct Rules, Separate Results

The local platform has completed three forms of opponent interaction: learned-policy boxing on Unitree G1, NCP digital-human boxing, and camera-based soccer with B-Human K1 robots. Their bodies, inputs, and judging rules differ, so their results stay in separate groups rather than becoming one cross-task power ranking.

| Track | Opponents and control | Completed results | Full videos and records |
| --- | --- | --- | --- |
| RoboStriker G1 | Two Unitree G1 humanoids; Stage3 ego / opponent learned motion policies; rigid-body contact in MuJoCo | ego vs opponent: 16:17, opponent wins; seat swap: 22:16, opponent wins. Ego self-play: 20:23. Opponent self-play: 2:2 technical draw. Self-play shows contact behavior and is not a policy ranking. | [Round 02 videos](../results/tournaments/round-02/videos/) · [Per-match results and replay verification](../results/tournaments/round-02/index.json) |
| NCP boxing | Two NCP digital-human characters, public checkpoints 106 / 100; MuJoCo body dynamics | 106 vs 100: 18:22, boxer 100 wins; seat swap: 1:2, mutual recovery failure and technical draw. 106 self-play 0:0 and 100 self-play 10:12; both were technical draws after mutual recovery failure. | [Round 02 videos](../results/tournaments/round-02/videos/) · [Per-match results and replay verification](../results/tournaments/round-02/index.json) |
| B-Human K1 visual soccer | Six Booster K1 robots, B-Human vs B-Team, 3v3; SimRobot RGB cameras and B-Human's normal visual control pipeline | Four local 30-second exhibition matches, all completed by the native referee; all four ended 0:0. | [Four videos](../results/tournaments/round-03/videos/) · [Tournament index](../results/tournaments/round-03/index.json) · [Score CSV](../results/tournaments/round-03/vision-matches.csv) · [Per-match referee telemetry](../results/tournaments/round-03/matches/) |

## Visual soccer: camera input and the local 30-second format

Each match uses the normal RGB-perception path: SimRobot `CameraColor` virtual cameras → B-Human `CameraImage` / `ECImage` / `RobotBallPerceptor` → B-Human K1 controllers. Six K1 robots play, with B-Human Red and B-Team White. Videos come from the native SimRobot framebuffer; scores and `FINISHED` states come from native B-Human `GameController` telemetry. No scripted goals or manual score edits were used.

To let matches finish on the available Mac, all four use a local 30-second exhibition half. The existing B-Human automatic timeout rule ends each bout. **This is a reproducible local short-period format, not an official ten-minute half, and it does not measure official competition strength.** All four scores were 0:0, so the records preserve the draws without assigning a winner. The normal camera-perception mode was enabled; `GameFast` truth-state mode was not used.

The four wall-clock recording intervals, from the first in-match camera frame through the referee finish, total about 9.6 minutes (113–197 seconds per match, excluding scene reload waits). The playable video files run longer than the 30-second match clock because they preserve native camera samples at the Mac's perception-mode runtime speed.

| Match | Red | White | Final score | Referee state | Video |
| --- | --- | --- | --- | --- | --- |
| 01 | B-Human | B-Team | 0:0, draw | FINISHED | [Watch](../results/tournaments/round-03/videos/vision-bhuman-k1-3v3-match-01.mp4) · [Telemetry](../results/tournaments/round-03/matches/vision-k1-match-01/match.csv) |
| 02 | B-Human | B-Team | 0:0, draw | FINISHED | [Watch](../results/tournaments/round-03/videos/vision-bhuman-k1-3v3-match-02.mp4) · [Telemetry](../results/tournaments/round-03/matches/vision-k1-match-02/match.csv) |
| 03 | B-Human | B-Team | 0:0, draw | FINISHED | [Watch](../results/tournaments/round-03/videos/vision-bhuman-k1-3v3-match-03.mp4) · [Telemetry](../results/tournaments/round-03/matches/vision-k1-match-03/match.csv) |
| 04 | B-Human | B-Team | 0:0, draw | FINISHED | [Watch](../results/tournaments/round-03/videos/vision-bhuman-k1-3v3-match-04.mp4) · [Telemetry](../results/tournaments/round-03/matches/vision-k1-match-04/match.csv) |

Reproducible local settings and acceptance metadata are documented in [`integrations/README.md`](../integrations/README.md), [`visual-exhibition-acceptance.json`](../integrations/bhuman/visual-exhibition-acceptance.json), and [`integrations/bhuman/`](../integrations/bhuman/). Local framebuffer and referee telemetry hooks are read-only; they do not alter perception, robot control, or physics parameters.

## MuJoCo boxing coverage and judging

Round 02 covers the two installed compatible policy families with autonomous attacks: ego / opponent on G1 and checkpoints 106 / 100 on NCP. Cross-policy bouts swap seats, while each policy also plays itself, for eight matches total. Ordinary locomotion, balance, disturbance-rejection, and recovery controllers do not include compatible attack policies and are not presented as boxing algorithms. NCP is a digital human, not G1 hardware.

The G1 / NCP bouts have a 90-second limit. A fall is confirmed when tilt exceeds 65 degrees or root height drops below 55% of its initial value for 0.25 seconds. Recovery must begin within 10 seconds and hold tilt below 25 degrees and height above 75% of the initial value for one second. One failed recovery while the other side is upright causes a technical stoppage; mutual failure is a technical draw. Only hand-to-upper-body contacts that meet the published speed, location, and force-impulse thresholds score.

These scores are results under specified simulation rules; **they do not predict hardware damage**. Calibrated G1 / NCP motor thermal-failure, gearbox-life, and structural-failure data are unavailable. Each pairing uses one fixed seed, suitable for initial checks but insufficient for a statistically grounded ranking.

## Complete indexes

- The bilingual page source is `web/competitions.html` (kept local for this round; not uploaded)
- [Round 02: eight simulation boxing results](../results/tournaments/round-02/index.json)
- [Round 03: four completed visual matches](../results/tournaments/round-03/index.json)
- [Round 03: video SHA-256 manifest](../results/tournaments/round-03/video-archive.json)
