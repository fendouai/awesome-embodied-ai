# 控制器调研与适配范围（2026-10-25）

目标是改善当前3种机器人在接触扰动下的控制能力。没有公开统一榜证明某个控制器在所有机器人、所有扰动下“全球最强”；这里选择有实际代码/权重/部署定义的候选，并在当前模型上比较。

| 候选 | 一手来源 | 当前处理 |
| --- | --- | --- |
| NVIDIA GEAR-SONIC / WBC | https://github.com/NVlabs/GR00T-WholeBodyControl | SONIC是全身动作跟踪基础模型，不能把跟踪演示等同抗推最优。本轮选择仓库中可直接按速度/站姿指令部署的Decoupled WBC Balance/Walk实际权重参加适配与测试；没有把它冒称为SONIC。 |
| Amazon FAR Holosoma | https://github.com/amazon-far/holosoma | 实际FastSAC与PPO G1 ONNX；采用模型元数据增益、关节名映射，按原生C++定义对观测项排序并处理站立步态相位。 |
| AMP locomotion + recovery | https://github.com/ccrpRepo/AMP_mjlab 和 https://github.com/ccrpRepo/wbc_fsm | 取部署仓库的真实model_30000权重，复现4帧历史、力矩/Kp动作尺度、增益与关节顺序。训练目标包含行走和恢复，但本项目不先验保证恢复性能。 |
| ASAP | https://github.com/LeCAR-Lab/ASAP | 有公开动态技能与权重；主要是参考动作跟踪与迁移，不能直接称为无参考抗推最优。当前保留为后续候选。 |
| XHugWBC | https://xhugwbc.github.io/ | 跨本体研究方向相关，但研究时项目页代码入口仍写Coming Soon，未编造可运行权重。 |
| Unitree RL Gym | https://github.com/unitreerobotics/unitree_rl_gym | 对H1/H1-2采用原公开腿部TorchScript，按关节名映射到完整本体，其余关节保持伺服。原训练身体是降自由度模型，此项属于明确的迁移适配，需实测，不冒充原生完整本体训练。 |

本轮G1适配：AMP、Holosoma FastSAC/PPO、NVIDIA WBC Balance/Walk；对照原RL Lab/MjLab及中性PD。H1与H1-2分别按本体权重适配，不将G1权重直接复用。

所有来源固定到研究时提交；代码与权重逐文件SHA见candidate-assets.json。实际测试结果在results/control-upgrade。筛选阶段不用于重写旧榜单；正式抗推验证需保存完整积分状态与显式扰动力，避免把外部试验推力与隐藏扶正混淆。
