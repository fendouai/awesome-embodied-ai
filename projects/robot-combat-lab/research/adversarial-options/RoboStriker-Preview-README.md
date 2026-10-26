# RoboStriker Code

This is a preview release of RoboStriker for early users who would like to try it out, including motions and the Stage-3 checkpoint. The current codebase is still being refined, and a more organized, ready-to-use version will be released soon.

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Installation

- Install Isaac Lab v2.1.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.
- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
# Option 1: SSH
git clone git@github.com:yinkangning0124/RoboStriker-Preview.git
# Option 2: HTTPS
git clone https://github.com/yinkangning0124/RoboStriker-Preview.git
```

- Download motion from  [https://drive.google.com/drive/folders/1v2oOB4LaF-yvxH-V9_-2psPvKUg5gAEO?usp=sharing ]() and put them under ''RoboStriker-Preview/source/whole_body_tracking/whole_body_tracking''
- Download assets from  [https://drive.google.com/drive/folders/1UwfpoKiL2i8e-KQ1-YK67zlR7eZj-oDC?usp=sharing ]() and put them under "RoboStriker-Preview/source/whole_body_tracking/whole_body_tracking"
- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/whole_body_tracking
```

## Command

```bash
# stage1 
## train
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 --motion_file source/whole_body_tracking/whole_body_tracking/motions/boxing_20251226 --headless --logger wandb --log_project_name xxx --run_name xxx --num_envs 4096

## play
python scripts/rsl_rl/play --task=Tracking-Flat-G1-v0 --motion_file source/whole_body_tracking/whole_body_tracking/motions/boxing_20251226/kangning-044.npz --load_run xxx --num_envs 1

# stage2
## train
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0-stage2-latent --motion_file source/whole_body_tracking/whole_body_tracking/motions/boxing_20251226 --headless --logger wandb --log_project_name xxx --run_name xxx --num_envs 4096

## play
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0-stage2-latent --motion_file=source/whole_body_tracking/whole_body_tracking/motions/boxing_20251226 --is_stage2 --load_run xxx --num_envs=1

# stage3 sandbag warmup
## train
python scripts/rsl_rl/run_marl_env.py --task=CombatSandbag-Stage3-SMPLOLYMPICRewards --num_envs 4096 --headless --log_project_name xxx --run_name xxx --logger wandb 

## play
python scripts/rsl_rl/play_selfplay.py --task=CombatSandbag-Stage3-BoxingRing --num_envs 1 --load_run xxx
```
