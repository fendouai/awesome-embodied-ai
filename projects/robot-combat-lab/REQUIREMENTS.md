# BotsTest Robot Combat Lab — acceptance requirements

User objective: independent project; MuJoCo simulation; a robot-body ranking and an existing-motion-model/VLA ranking; complete encounter archives, intuitive 3D combat videos, and a report. Physical authenticity is the first principle; game comparison concerns visualization only.

- Actual MuJoCo rigid-body dynamics, free floating roots, real contacts, bounded actuators; no animated displacement, hidden stabilizing external forces, health, invented injury, or scripted winners.
- Downloaded robot descriptions and existing neural policy checkpoints pinned to upstream commits with SHA-256 hashes and retained licenses.
- Robot-body protocol must hold control algorithm, stimulus and outcome rules fixed; disclose morphology-specific calibration. Do not describe a body+different-controller competition as pure hardware ranking.
- AI protocol fixes body, match driver and physics while comparing actual runnable downloaded neural motion policies. Do not rename heuristics as commercial AI or VLA models. Locomotion-under-contact results are not general intelligence or combat-trained capability.
- Entire encounters: initial config, compiled model, integration state/control trajectory at physics cadence, telemetry, inter-robot contacts, events, result and hashes. Replay from recorded states.
- Every published encounter has a full-duration 3D video with synchronized physical telemetry and state labels; browser viewer offers scrub/play, cameras and full replay.
- Two separate rankings with matches, uncertainty and protocol; mirrored repeated trials; failures, draws, model limitations preserved.
- Report describes methods, results, validity, limitations and reproducible commands.
- Run physics/control sanity checks, replay checks, metric consistency, report/video/browser validation. No claim of real hardware combat equivalence.

## Revised objective: continuous contact until hardware loss (2026-10-25)

- Falling, ring exit and short time limits must not declare a winner in continuous mode.
- Preserve physical state, neural observation history/LSTM memory and hardware-model states through chunking and resume; never reset to create a recovery.
- Resource limits cause resumable technical pauses, not losses.
- Physical hardware damage requires version-specific calibrated failure data. Peak torque/current limits, work, protection trips and solver contact force are not fracture/burnout thresholds.
- Search primary hardware/manual/research sources proactively. Record facts, applicability and gaps; do not invent missing parameters or borrow ratings from different actuators.
- The current default has UNKNOWN temperatures/damage because calibration data remain incomplete. Parameterized thermal model is explicitly unvalidated; structural/gearbox breakage is not implemented. Do not claim the full requested damage-to-inoperability simulator is complete.
