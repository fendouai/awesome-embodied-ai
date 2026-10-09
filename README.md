# Awesome Embodied AI [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[简体中文](README.zh-CN.md)

A curated guide to embodied AI and robotics: components, hardware, software, models, datasets, benchmarks, companies, products and research.

Maintained by [BotsTest](https://botstest.com/). Includes directly relevant resources and explicitly labeled supporting tools. Resource links point to their source.

Edition: 2026-10-09. **958 curated URLs**, 9 categories and 55 available subcategories. The full candidate archive preserves 1,000 historically checked URLs: 16 excluded and 26 awaiting review.

Editorial inclusion and endpoint checks are separate. Check dates are historical; this edition does not imply fresh network checks or installation tests. Scores help discovery; they do not measure performance or suitability for a specific deployment.

## Contents

- [Core components (28)](lists/components.md)
- [Hardware (38)](lists/hardware.md)
- [Software (579)](lists/software.md)
- [Models (76)](lists/models.md)
- [Datasets (33)](lists/datasets.md)
- [Benchmarks (31)](lists/benchmarks.md)
- [Companies (44)](lists/companies.md)
- [Products (27)](lists/products.md)
- [Research & learning (102)](lists/research.md)

## How to use this list

Start with the category and subcategory for your task. Check versions, interfaces, training/inference requirements and license scope at the source. English and Chinese editions use the same data and order.

The sections below show up to eight top-scoring entries per category; category files contain every included resource. Entries sort by descending score within their category, then name and canonical URL. See [ranking and curation rules](RANKING.md).

## Core components

[All 28 resources](lists/components.md)

- [ATI Industrial Automation](https://www.ati-ia.com/) — ATI Industrial Automation supplies force-torque sensors and robot end-of-arm interface equipment. [Details](https://botstest.com/resources/ati-industrial-automation/)
- [Bosch Sensortec](https://www.bosch-sensortec.com/) **[Supporting resource]** — Bosch Sensortec develops MEMS sensors for inertial measurement and motion sensing. [Details](https://botstest.com/resources/bosch-sensortec/)
- [Bota Systems](https://www.botasys.com/) — Bota Systems develops force-torque sensing devices for robot interaction and manipulation. [Details](https://botstest.com/resources/bota-systems/)
- [Hailo](https://hailo.ai/) **[Supporting resource]** — Hailo develops edge AI accelerators for local perception and inference workloads. [Details](https://botstest.com/resources/hailo/)
- [Harmonic Drive](https://www.harmonicdrive.net/) — Harmonic Drive supplies strain-wave gearing and precision motion components for robot joints. [Details](https://botstest.com/resources/harmonic-drive/)
- [Infineon](https://www.infineon.com/) **[Supporting resource]** — Infineon supplies power and control semiconductors for drive electronics and embedded systems. [Details](https://botstest.com/resources/infineon/)
- [maxon](https://www.maxongroup.com/) — maxon supplies electric motors, gearheads and drive systems for compact robotic mechanisms. [Details](https://botstest.com/resources/maxon/)
- [Moog](https://www.moog.com/) **[Supporting resource]** — Moog develops motion-control components and actuation systems for demanding applications. [Details](https://botstest.com/resources/moog/)

## Hardware

[All 38 resources](lists/hardware.md)

- [OpenArm](https://github.com/enactic/openarm) — OpenArm humanoid-arm project for physical AI research. [Details](https://botstest.com/resources/enactic-openarm/)
- [BetaBots Robot Arm Project](https://github.com/4ndreas/BetaBots-Robot-Arm-Project) — BetaBots Robot Arm Project provides a 3D-printable robot arm design. [Details](https://botstest.com/resources/4ndreas-betabots-robot-arm-project/)
- [open quadruped](https://github.com/adham-elarabawy/open-quadruped) — open-quadruped is a 3D-printed quadrupedal robot project with gait generation and body-pose control. [Details](https://botstest.com/resources/adham-elarabawy-open-quadruped/)
- [SO ARM100 101 Parallel Gripper](https://github.com/roboninecom/SO-ARM100-101-Parallel-Gripper) — SO-ARM100/101 Parallel Gripper provides a 3D-printed gripper design for SO-ARM projects. [Details](https://botstest.com/resources/roboninecom-so-arm100-101-parallel-gripper/)
- [microban](https://github.com/Rhoban/microban) — Microban is a compact 3D-printable humanoid robot project with CAD and setup resources. [Details](https://botstest.com/resources/rhoban-microban/)
- [Thor Robot Arm](https://github.com/AngelLM/Thor) — Thor is a 3D-printable robot arm project with mechanical design and assembly materials. [Details](https://botstest.com/resources/thor-robot-arm/)
- [makerspet loki](https://github.com/makerspet/makerspet_loki) — Maker's Pet Loki is a 3D-printed DIY home robot with assembly and setup resources. [Details](https://botstest.com/resources/makerspet-makerspet_loki/)
- [BCN3D Moveo](https://github.com/BCN3D/BCN3D-Moveo) — BCN3D Moveo is a 3D-printed robotic arm design for educational projects. [Details](https://botstest.com/resources/bcn3d-moveo/)

## Software

[All 579 resources](lists/software.md)

- [LeRobot](https://github.com/huggingface/lerobot) — LeRobot tools for robot control, datasets and end-to-end learning. [Details](https://botstest.com/resources/huggingface-lerobot/)
- [MuJoCo](https://github.com/google-deepmind/mujoco) **[Supporting resource]** — MuJoCo physics simulator for multijoint dynamics and contact. [Details](https://botstest.com/resources/google-deepmind-mujoco/)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) **[Supporting resource]** — Standard environment API and reference environments for reinforcement learning. [Details](https://botstest.com/resources/farama-foundation-gymnasium/)
- [cleanrl](https://github.com/vwxyzjn/cleanrl) **[Supporting resource]** — Single-file implementations of deep reinforcement-learning algorithms. [Details](https://botstest.com/resources/vwxyzjn-cleanrl/)
- [CARLA](https://github.com/carla-simulator/carla) — An open-source simulator for autonomous-driving research. [Details](https://botstest.com/resources/carla/)
- [Bullet Physics](https://github.com/bulletphysics/bullet3) **[Supporting resource]** — A physics SDK for collision detection and multi-physics simulation. [Details](https://botstest.com/resources/bulletphysics-bullet3/)
- [scenario runner](https://github.com/carla-simulator/scenario_runner) — A traffic-scenario definition and execution engine for CARLA. [Details](https://botstest.com/resources/carla-simulator-scenario_runner/)
- [onshape to robot](https://github.com/Rhoban/onshape-to-robot) — Converts Onshape assemblies into URDF, SDF, and MuJoCo robot descriptions. [Details](https://botstest.com/resources/rhoban-onshape-to-robot/)

## Models

[All 76 resources](lists/models.md)

- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) — NVIDIA Isaac GR00T provides foundation-model resources for generalist robots. [Details](https://botstest.com/resources/nvidia-isaac-gr00t/)
- [iros20 6d pose tracking](https://github.com/wenbowen123/iros20-6d-pose-tracking) — se(3)-TrackNet provides 6D object-pose tracking research and visual-feedback-control examples. [Details](https://botstest.com/resources/wenbowen123-iros20-6d-pose-tracking/)
- [Octo](https://github.com/octo-models/octo) — Octo provides a transformer-based robot policy and adaptation resources. [Details](https://botstest.com/resources/octo/)
- [openpi](https://github.com/Physical-Intelligence/openpi) — openpi provides code and model resources for Physical Intelligence robot policies. [Details](https://botstest.com/resources/openpi/)
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) — Diffusion Policy learns visuomotor robot actions using action diffusion. [Details](https://botstest.com/resources/diffusion-policy/)
- [Segment Anything](https://github.com/facebookresearch/segment-anything) **[Supporting resource]** — Segment Anything provides image-segmentation models as perception building blocks. [Details](https://botstest.com/resources/segment-anything/)
- [DeeR VLA](https://github.com/yueyang130/DeeR-VLA) — DeeR-VLA studies dynamic multimodal-model inference for robot execution. [Details](https://botstest.com/resources/yueyang130-deer-vla/)
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft) — OpenVLA-OFT provides fine-tuning tools for OpenVLA action prediction. [Details](https://botstest.com/resources/openvla-oft/)

## Datasets

[All 33 resources](lists/datasets.md)

- [example robot data](https://github.com/Gepetto/example-robot-data) — Example Robot Data collects robot URDF models for benchmarks and examples. [Details](https://botstest.com/resources/gepetto-example-robot-data/)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Collection of robot and scene models for MuJoCo. [Details](https://botstest.com/resources/mujoco-menagerie/)
- [nasa urdf robots](https://github.com/gkjohnson/nasa-urdf-robots) — URDF robot models derived from NASA Robonaut 2 and Valkyrie projects. [Details](https://botstest.com/resources/gkjohnson-nasa-urdf-robots/)
- [eDO description](https://github.com/ianathompson/eDO_description) — URDF description package for the eDO robot. [Details](https://botstest.com/resources/ianathompson-edo_description/)
- [AgiBot World](https://agibot-world.com/) — AgiBot World provides robot demonstration data and embodied-manipulation resources. [Details](https://botstest.com/resources/agibot-world/)
- [AgiBot World](https://github.com/OpenDriveLab/AgiBot-World) — AgiBot World provides manipulation-platform resources for embodied-system data and learning. [Details](https://botstest.com/resources/opendrivelab-agibot-world/)
- [AMASS](https://amass.is.tue.mpg.de/) **[Supporting resource]** — AMASS unifies human motion-capture datasets into a shared representation. [Details](https://botstest.com/resources/amass/)
- [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) — BridgeData V2 provides multi-task robot-manipulation demonstrations. [Details](https://botstest.com/resources/bridgedata-v2/)

## Benchmarks

[All 31 resources](lists/benchmarks.md)

- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — LIBERO benchmarks knowledge transfer in lifelong robot learning. [Details](https://botstest.com/resources/libero/)
- [actionshift](https://github.com/Archerkattri/actionshift) — ActionShift evaluates adaptation to hidden robot action-interface contracts on ManiSkill tasks. [Details](https://botstest.com/resources/archerkattri-actionshift/)
- [CALVIN](https://github.com/mees/calvin) — CALVIN benchmarks language-conditioned policy learning for long-horizon robot manipulation. [Details](https://botstest.com/resources/calvin/)
- [RLBench](https://github.com/stepjam/RLBench) — RLBench provides a task suite for visual robot-learning evaluation. [Details](https://botstest.com/resources/rlbench/)
- [SimplerEnv](https://github.com/simpler-env/SimplerEnv) — SimplerEnv provides simulated environments for evaluating robot-manipulation policies under shared setups. [Details](https://botstest.com/resources/simplerenv/)
- [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) — RoboTwin provides bimanual manipulation benchmarks and data-generation tools. [Details](https://botstest.com/resources/robotwin/)
- [Meta-World](https://github.com/Farama-Foundation/Metaworld) — Meta-World provides robot-manipulation tasks for multi-task and meta-learning evaluation. [Details](https://botstest.com/resources/meta-world/)
- [softgym](https://github.com/Xingyu-Lin/softgym) — SoftGym provides benchmark environments for deformable-object manipulation. [Details](https://botstest.com/resources/xingyu-lin-softgym/)

## Companies

[All 44 resources](lists/companies.md)

- [1X Technologies](https://www.1x.tech/) — 1X Technologies develops humanoid robots and embodied learning systems. [Details](https://botstest.com/resources/1x-technologies/)
- [AgiBot](https://www.agibot.com/) — AgiBot develops embodied robotic platforms and robot learning tools. [Details](https://botstest.com/resources/agibot/)
- [Apptronik](https://apptronik.com/) — Apptronik develops humanoid robots for real-world work. [Details](https://botstest.com/resources/apptronik/)
- [Boston Dynamics](https://bostondynamics.com/) — Boston Dynamics develops legged robots and mobile manipulation systems. [Details](https://botstest.com/resources/boston-dynamics/)
- [Clearpath Robotics](https://clearpathrobotics.com/) — Clearpath Robotics develops mobile robot platforms for research and automation. [Details](https://botstest.com/resources/clearpath-robotics/)
- [Covariant](https://covariant.ai/) — Covariant develops AI-driven robotic manipulation systems for industrial automation. [Details](https://botstest.com/resources/covariant/)
- [DEEP Robotics](https://www.deeprobotics.cn/) — DEEP Robotics develops quadruped robots for industrial applications. [Details](https://botstest.com/resources/deep-robotics/)
- [Dobot](https://www.dobot-robots.com/) — DOBOT develops collaborative and industrial robotic arms for automation. [Details](https://botstest.com/resources/dobot/)

## Products

[All 27 resources](lists/products.md)

- [Linorobot](https://github.com/linorobot/linorobot2) — Linorobot2 provides ROS 2 resources for building and simulating autonomous mobile robots. [Details](https://botstest.com/resources/linorobot/)
- [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) — Open Duck Mini is a compact bipedal robot hardware project for builders. [Details](https://botstest.com/resources/open-duck-mini/)
- [Stanford Doggo](https://github.com/Nate711/StanfordDoggoProject) — Stanford Doggo is a quadruped hardware project for legged locomotion experiments. [Details](https://botstest.com/resources/stanford-doggo/)
- [NASA JPL Open Source Rover](https://github.com/nasa-jpl/open-source-rover) — NASA JPL Open Source Rover is a build-it-yourself six-wheel rover project inspired by Mars rovers. [Details](https://botstest.com/resources/nasa-jpl-open-source-rover/)
- [1X NEO](https://www.1x.tech/neo) — 1X NEO is a humanoid robot platform developed by 1X Technologies. [Details](https://botstest.com/resources/1x-neo/)
- [Bitcraze Crazyflie](https://www.bitcraze.io/) — Bitcraze Crazyflie is a small aerial robotics platform with development hardware and software. [Details](https://botstest.com/resources/bitcraze-crazyflie/)
- [Boston Dynamics Atlas](https://bostondynamics.com/atlas/) — Atlas is a humanoid robot platform developed by Boston Dynamics. [Details](https://botstest.com/resources/boston-dynamics-atlas/)
- [Boston Dynamics Stretch](https://bostondynamics.com/products/stretch/) — Boston Dynamics Stretch is a mobile manipulation robot for warehouse material handling. [Details](https://botstest.com/resources/boston-dynamics-stretch/)

## Research & learning

[All 102 resources](lists/research.md)

- [awesome multimodal data](https://github.com/backblaze-labs/awesome-multimodal-data) **[Supporting resource]** — Awesome Multimodal Data lists multimodal datasets and infrastructure for video, image, audio, 3D and robotics data. [Details](https://botstest.com/resources/backblaze-labs-awesome-multimodal-data/)
- [Embodied AI Lab](https://github.com/Benjamindaoson/Embodied-AI-Lab) **[Supporting resource]** — Embodied AI Lab is a research hub covering VLA, data infrastructure and robotics evaluation. [Details](https://botstest.com/resources/benjamindaoson-embodied-ai-lab/)
- [awesome robotics](https://github.com/kiloreux/awesome-robotics) — Awesome Robotics by kiloreux collects robotics courses, books, software, papers and publications. [Details](https://botstest.com/resources/kiloreux-awesome-robotics/)
- [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python) — Robotics Toolbox for Python provides tools and examples for robotics education and algorithm exploration. [Details](https://botstest.com/resources/robotics-toolbox-for-python/)
- [awesome computer vision](https://github.com/jbhuang0604/awesome-computer-vision) **[Supporting resource]** — Awesome Computer Vision collects computer vision resources, books and courses. [Details](https://botstest.com/resources/jbhuang0604-awesome-computer-vision/)
- [Awesome Implicit NeRF Robotics](https://github.com/zubair-irshad/Awesome-Implicit-NeRF-Robotics) — Awesome Implicit NeRF Robotics collects implicit-representation and NeRF research for robotics and reinforcement learning. [Details](https://botstest.com/resources/zubair-irshad-awesome-implicit-nerf-robotics/)
- [awesome robotics](https://github.com/ahundt/awesome-robotics) — Awesome Robotics by ahundt collects robot-related links and software libraries. [Details](https://botstest.com/resources/ahundt-awesome-robotics/)
- [Unity Robotics Hub](https://github.com/Unity-Technologies/Unity-Robotics-Hub) — Unity Robotics Hub provides tools, tutorials and documentation for robotics simulation in Unity. [Details](https://botstest.com/resources/unity-technologies-unity-robotics-hub/)

## Data and maintenance

- [Structured resources](data/resources.json) · [CSV](data/resources.csv) · [JSON Schema](schema/resource.schema.json)
- [Candidate archive](data/url-registry.json) · [Editorial decisions](data/decisions.json) · [Provenance](PROVENANCE.md)
- [Ranking rules](RANKING.md) · [Contributing](CONTRIBUTING.md) · [Metadata contract](METADATA.md)

```sh
# Python 3.10+; install validation dependencies once
python3 -m pip install -r requirements-dev.lock
python3 scripts/build.py
python3 scripts/validate.py
python3 -m unittest discover -s tests
```

After dependencies are installed, generation and validation run offline and require no credentials. Update the relevant source data, review decision and both languages together; generated files should not be edited manually.

## License

Original list text and compilation: [CC0 1.0](LICENSE). Python tooling: [MIT](LICENSE-CODE). External projects, third-party source excerpts, names and trademarks retain their own rights and licenses.
