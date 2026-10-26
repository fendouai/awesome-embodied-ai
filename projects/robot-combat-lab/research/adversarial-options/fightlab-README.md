# FightLab

Autonomous humanoid combat league. AI policies fight as Unitree G1 humanoids in MuJoCo simulation. King of the hill, always evolving.

## The Concept

Miners submit a policy — a neural network that controls a Unitree G1 humanoid in combat. The league runs deterministic evaluation to rank them. The best policy becomes king. Every king's weights are open so new miners can build on them. You must beat the king to take the crown.

## Combat Rules

- **Striking:** punches (wrists to torso/head)
- **Damage:** Head = 2x, body = 1x
- **Scoring:** 10-point must per round (3 rounds × 20s)
- **Win:** KO (HP ≤ 0), fall (pelvis < 0.4m), or decision
- **Draw:** damage gap < 2 HP = draw round; scorecards tied + damage < 1 = draw
- **King stays on draw:** you must beat the champion to take the crown

## How to Enter

1. Fork this repo (includes arena, scoring, eval pipeline, and all past kings' weights)
2. Train your policy — any method, any architecture, any observation format
   - Train from scratch, fine-tune a king, or train against a king
   - A walking controller is provided as infrastructure, or build your own
3. Test locally with the deterministic eval
4. Submit a PR adding your `models/*.zip` with a fighter name
5. CI auto-runs trustless eval (deterministic, multi-seed, auditable)
6. If you pass the gate (win on 2+ seeds, deal damage) → title bout vs king

## Open Weights Policy

All kings' weights are published in `models/kings/`. Anyone can:
- Download and study them
- Fine-tune them as a starting point
- Use them as training opponents
- The gate prevents copying: you must BEAT the king to take the crown, not match it.

## What's Open Source

- The G1 model + MuJoCo scene (physics)
- The scoring engine (damage detection, 10-point must)
- The eval pipeline (deterministic, multi-seed, hashed, auditable)
- The league system (ELO, king of the hill, evolution loop)
- All past kings' weights
- A walking controller (use it or build your own)

## What Miners Provide

Their trained policy (`models/*.zip`). One file. The policy controls the G1 in combat — how it stands, moves, and fights is entirely up to the miner. Any training method, any architecture, any observation format. The only requirement: it must work in the arena under the rules.

## Trustless Evaluation

- Deterministic physics (fixed seeds 42, 123, 777)
- Model SHA256 hash verified
- Per-step bout logs (HP, damage events, decisions) published as artifacts
- Runs in GitHub Actions — not our infrastructure
- Anyone can clone the repo and reproduce identical results

## Tech Stack

- **Physics:** MuJoCo (GPU-accelerated EGL rendering)
- **Training:** PPO via Stable Baselines3 (or any RL framework)
- **Platform:** Unitree G1 (29-DoF humanoid)
- **Rendering:** EGL 720p, broadcast tracking camera

## Files

```
scene_2bot.xml           2-G1 combat arena (MuJoCo)
fight_env.py             Combat environment (scoring, damage, rules)
train_fight.py           Training script (PPO on FightEnv)
eval.py                  Deterministic eval + CI gate + bout rendering
league.py                Round-robin + ELO + page generation
evolve.py                Evolution loop (auto-train challengers)
build_2bot_scene.py      Arena builder (duplicates G1 with r2_ prefix)
walker_arena.py          Walker controller wrapper
g1.xml                   Unitree G1 model (from Lucky Robots)
model_config.json        G1 joint config + walker normalization
docs/                    GitHub Pages site (standings + bouts)
models/kings/            Archived king weights (open)
.github/workflows/       CI: trustless league gate
```
