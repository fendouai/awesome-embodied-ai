## 10种机器人与10类控制方案

当前入口：http://127.0.0.1:8767/web/comparison.html 。新增Go1、B2、Go2-W、B2-W，以及WTW隐变量自适应和BeyondMimic参考跟踪。详见reports/ten-platform.html。

## 历史阶段：6种机器人与8类控制方案

当前本机对比入口：http://127.0.0.1:8767/web/comparison.html 。支持G1、H1、H1-2、Go2、A1、K1，提供兼容矩阵、同本体配对实验和视频。详细说明见reports/robot-control-expansion.html。

# BotsTest Robot Combat Lab · 仿真战力榜

独立 MuJoCo 项目。真实刚体动力学与实际预训练运动模型，记录接触、失衡、倒地、越界与恢复。所有数值来自仿真；没有血量、虚构损伤、根部动画或预设赢家。

## 当前默认：持续对抗（硬件损伤未标定）

按用户的新目标，默认模式不再以倒地、越界或12/20秒限时判输。已经实测连续60秒，其中机器人间接触累计54.2秒；按30秒分段保存完整500Hz状态、控制、神经网络历史/LSTM记忆和硬件模型状态。支持从原始状态续跑，不把技术暂停当作胜负。

- 平台“持续对抗”页可顺序播放完整分段；交互页可暂停/续跑，默认120秒为计算批次。
- `results/continuous/summary.json`：连续记录索引；`verification.json`：轨迹、原始接触冲量、精确分段边界与续跑检查。
- `reports/hardware-evidence.html` / `.pdf`：官方手册、规格、SDK、论文和一手问题报告的查证结果。
- `research/hardware-failure/`：官方31页手册图像与OCR、源文件SHA、证据清单及未填数值的参数模板。
- `src/combatlab/endurance.py`：持续动力学与检查点；`hardware.py`：可配置集总热模型，区别保护停机与不可逆故障。

**尚未完成真实硬件损坏预测。** 当前未找到可匹配全部3本体/各关节版本的热容量/散热/烧毁参数、齿轮冲击疲劳曲线和结构材料/破坏试验。默认均显示未知，不会计算虚假损伤或强制产生赢家。现有控制器是运动策略，持续接触推挤不等于已经训练了搏斗策略。

还发现当前G1公开MJCF膝关节限幅139N·m与官网G1/EDU峰值90/120N·m存在版本/口径待核对，不能混用作破坏载荷。

```sh
PYTHONPATH=src .venv/bin/python scripts/run_continuous.py --seconds 60
PYTHONPATH=src .venv/bin/python scripts/run_continuous.py --resume results/continuous/某个part目录 --seconds 60
PYTHONPATH=src .venv/bin/python scripts/verify_continuous.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_endurance.py
```

分段视频使用该段相对时长，HUD显示原始绝对仿真时钟。参数化热模型仅在显式提供来源与全部系数时启用，始终标为未经实机验证；纯单元测试中的合成热参数不会进入机器人配置。只有该模型中全部执行器不可逆失效才触发模型运行能力丧失判据。结构和减速器破坏仍未建模。

## 2026-10-25 控制器升级（当前默认）

原中性PD本体榜已降级为历史探索，不能解释本体固有实力。平台默认进入“控制器升级验证”，交互模式可选择3种机器人并自动匹配控制器。

- G1：NVIDIA Decoupled WBC，原生Balance/Walk组合；静止时Balance，速度指令范数>0.05时Walk。不是GEAR-SONIC。
- H1/H1-2：各自官方Unitree RL Gym真实LSTM腿部策略，按关节名迁移到完整本体，上肢保持伺服；不冒称完整全身原生训练。
- 调研AMP、Holosoma FastSAC/PPO、NVIDIA WBC等，完成182项筛选/细化；G1实际优选在30/40N·s四方向、两个初始种子的16项完整验证均通过；H1/H1-2各通过30N·s的8项。旧PD在这32项条件全部失败。
- G1 60N·s失败案例保留。4场新控制器接触对局均发生接触并到12秒限时，未触发倒地/越界；每场视频完整保留后续1.5秒。这不是对抗训练或普遍不倒的证明。
- 新报告：`reports/control-upgrade.pdf` / `.html`；新结果：`results/control-upgrade/summary.json`；来源与适配说明：`research/control-upgrade/selection.md`。
- 33项验证均保存500Hz完整状态、控制、显式外力时序和源码快照。182项筛选保存50Hz测量，明确不冒称完整积分档案。8个完整示例视频（4抗推对照/失败、4接触对局）。
- `verification.json`核验33项外力、源码、一步积分和神经网络输出；`delivery-audit.json`覆盖新接触、视频及3本体交互。所有数据位于`results/control-upgrade/`。

复现新增测试（先保留交付结果；命令会写同名输出）：

```sh
PYTHONPATH=src .venv/bin/python scripts/screen_controllers.py
PYTHONPATH=src .venv/bin/python scripts/refine_controllers.py
PYTHONPATH=src .venv/bin/python scripts/validate_controllers.py
PYTHONPATH=src .venv/bin/python scripts/validate_contact_upgrade.py
PYTHONPATH=src .venv/bin/python scripts/verify_control_upgrade.py
PYTHONPATH=src MUJOCO_GL=cgl .venv/bin/python scripts/render_control_upgrade.py
PYTHONPATH=src .venv/bin/python scripts/report_control_upgrade.py
```

完整抗推档案的`disturbance.npz`保存每步实际躯干外力；验证器确保除声明推力外没有任何外部扶正力。H1/H1-2 LSTM验证复现两次初始化前向，再重执行保存的399步观测；各类初始化与观测顺序均需遵循适配器。早期源码审计前记录保存在`archives-source-audit-v0/`，不计入当前33项验证。

## 直接查看

启动后打开 **http://127.0.0.1:8767**：完整视频、同步物理数据、自由视角3D状态回放、两个排行榜、人机/人人/模型交互仿真。

- `reports/report.pdf`：11页实验报告；`reports/report.html`：网页报告。
- `results/leaderboards.json`：两个榜单、全部26场结果、积分和区间。
- `results/matches/<对局ID>/battle.mp4`：每场1280×720、24fps完整3D视频。
- `results/verification.json`：逐场原始轨迹、规则、功/冲量、模型输入输出及视频核验。
- `evidence/`：实际浏览器测试截图及检查输出。

## 历史探索结果（旧控制器，已降级）

| 榜单 | 参赛项 | 胜 / 平 / 负 | 解释 |
| --- | --- | --- | --- |
| 本体 1 | H1-2，无灵巧手，27 DoF | 6 / 6 / 0 | 对H1获胜，与G1平局 |
| 本体 2 | G1 rev1.0，29 DoF | 0 / 12 / 0 | 与两种本体均平局 |
| 本体 3 | H1，19 DoF | 0 / 6 / 6 | 在本协议下对H1-2失利 |
| 运动模型 1 | Unitree RL MjLab G1 velocity/v0 | 2 / 6 / 0 | 高速条件的边界结果构成积分差 |
| 运动模型 2 | Unitree RL Lab G1 velocity/v0 | 0 / 6 / 2 | 同一G1本体，保留原生部署参数 |

26场正式对局全部发生机器人间接触。Elo对200个确定性洗牌顺序取平均；胜1、平0.5、负0。区间为本次记录重抽样范围，存在小样本和条件依赖，不能外推为实机强弱或总体置信度。本体榜不能把质量、控制校准、任务定义完全解耦；运动模型榜比较部署包，不是网络架构单独排名。

## 启动现有环境

在本目录运行：

```sh
PYTHONPATH=src MUJOCO_GL=cgl .venv/bin/python -m combatlab.cli serve --port 8767
```

本机已验证：macOS ARM64、16GB内存；Python3.13、MuJoCo3.17.0，CPU ONNX Runtime。当前`.venv`使用Anaconda系统包提供torch、ONNX Runtime、SciPy等，不能直接复制虚拟环境到另一台电脑。

## 新环境安装

Python 3.11+，安装ffmpeg/ffprobe（视频编码与检查），建议至少数GB可用磁盘。源码与资产都在本文件夹，不需要外部服务或API密钥。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
# macOS原生离屏渲染
export MUJOCO_GL=cgl
# Linux可尝试egl或osmesa；需自行安装对应图形运行库
```

依赖版本范围用于安装；本次实际版本见`results/preflight.json`。物理引擎固定3.17.0；跨平台图形后端、数值库与硬件差异不保证位级一致。

## 复核与重新实验

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m combatlab.cli verify
# 以下命令会重新生成相应结果，应先复制本目录保留已交付实验包
PYTHONPATH=src .venv/bin/python -m combatlab.cli preflight
PYTHONPATH=src .venv/bin/python -m combatlab.cli tournament
PYTHONPATH=src .venv/bin/python -m combatlab.cli render
PYTHONPATH=src .venv/bin/python scripts/sensitivity.py
PYTHONPATH=src .venv/bin/python -m combatlab.cli verify
PYTHONPATH=src .venv/bin/python -m combatlab.cli report
```

报告PDF使用Playwright和Chrome打印实际HTML：先启动服务器，`npm install`，再`npm run report:pdf`。macOS默认使用已安装Google Chrome；其他系统可设置`CHROME_PATH`，或安装Playwright Chromium。`npm run check:browser`会实际启动并保存两个人工交互会话，检查视频/数值同步、3D视角、双榜、协议及报告排版。

`verify`逐场检查资产/档案SHA、完整轨迹判据、接触冲量、机械功、测量一致性，选5个状态重新积分一步，并对每个模型4个观测重新执行真实ONNX推理。它不声称已从头重新积分全部长期轨迹。

## 档案结构与恢复

正式对局位于`results/matches/`：

| 文件 | 内容 |
| --- | --- |
| `config.json` / `scene.xml` | 初始条件、规则、组合后的MJCF |
| `model.ref.json` / `model.patch.json.gz` | 精确编译模型的共享基底引用及无损字节补丁 |
| `trajectory.npz` | 每2ms完整MuJoCo积分状态、实际电机控制 |
| `policy_io.npz` | 每20ms模型观测和动作 |
| `contacts.npz` | 每个机器人间接触的时间、几何、位置、6维局部力/力矩与法向 |
| `telemetry.json` / `events.json` | 每20ms测量、状态及判定事件 |
| `result.json` / `manifest.json` | 结果、源权重SHA、软件指纹、档案哈希 |
| `battle.mp4` / `poster.jpg` | 完整3D录像及封面 |

搬移时必须保留`results/model-cache/`；加载器按SHA检查并字节级恢复编译模型，不丢失网格精度或物理帧。`scene.xml`中的资产绝对路径反映生成机器；跨机器回放优先使用随附编译模型和`combatlab.storage.load_model`。重新构建场景时源码会从新目录生成正确路径。

`results/software/<SHA>/`保留对应的engine/models/policies源码。`results/experiments-contact-v0/`与`diagnostics/`为旧探索，`baselines/`为无接触基线，`validation/`为时间步减半检查，均不进入正式榜单。

人工交互通过速度指令操纵实际运动策略，支持人机、人人、模型对抗。WASD/QE操作左侧，方向键/JL操作右侧；仿真可能慢于墙钟。停止后完整档案和自动生成的视频位于`results/live-sessions/`，不计入正式榜单。

## 真实性边界

- 本体采用Unitree官方G1、H1、H1-2公开模型；两个ONNX来自官方RL Lab和RL MjLab。固定提交、逐文件SHA及许可/版权材料见`assets/provenance.json`与vendor目录。无厂商合作或背书。
- 本体测试使用共同重力补偿PD、统一几何校准及规定伸臂前倾。静态基线与无对手执行同动作均通过。没有给自由根部施加扶正外力。
- 源运动模型缺少手碰撞，依据原网格增加无质量凸包代理，支持跨机器人及地面接触。H1空占位关节固定为零并移除无用电机，原质量惯量保留。这些均是明确的模型处理，不能称为实机标定。
- dt=2ms；两个代表对局减半至1ms，结局一致、冲量差约0.20%及0.42%，不构成全面收敛证明。
- 真实接触力是求解器结果；机械功不是电池消耗。没有材料损伤、热、电流、通信时延、视觉感知误差的实测模型。
- 两个参赛AI是**运动策略**，没有把它们称为VLA或格斗训练模型。当前输出是接触抗扰与规定推挤实验，不能外推为真实拳击、结构抗损伤或通用智能榜。

作者：**BotsTest** · https://botstest.com/  
A field guide to embodied intelligence.

### 有终局的自主对抗

打开 http://127.0.0.1:8767/web/duels.html 查看G1与NCP的90秒完整比赛、实时有效上身接触计分、6秒终局停留和逐次交锋回看。页面可重新运行两项比赛，顺序执行策略对抗、逐物理步复核和录像生成；与原对比任务共享CPU任务锁。

规则与硬件损坏分开：倒地确认后给10秒恢复机会，站立稳定1秒才确认恢复；恢复失败且对手站立则技术终止。仍可运行则90秒限时判定，先比有效上身接触次数，再按公开冲量规则破平；没有差异则平局。异常中断不授予胜者。本规则用于本地仿真比赛，不代表官方拳击规则或实机损伤阈值。

可复现脚本（项目虚拟环境，`PYTHONPATH=src MUJOCO_GL=cgl`）：`scripts/run_duel.py`、`scripts/run_ncp_duel.py`；随后对新归档执行 `scripts/verify_duel.py <folder>`、`scripts/render_duel.py <folder>`，最后执行 `scripts/publish_duels.py`。发布要求完整结果与轨迹校验相符，并有终局录像。原20秒演示与其他对抗实验保留。

### 首轮兼容格斗策略循环赛

`/web/tournament.html` 汇总G1（ego/opponent）与NCP（106/100）两组的双向比赛，共4场。首回合复用已有核验的90秒比赛，次回合交换策略所在位置；NCP随机流随策略身份交换。上限90秒，技术终止可提前结束，双方恢复失败可判技术平局。分组展示胜平负，不视作多种子统计排名，也不把行走控制器当作攻击策略。

全部完整视频另有具名归档：`results/tournaments/round-01/videos/`；使用APFS写时复制生成独立副本并逐个校验SHA256，原始轨迹目录同时保留。`matches.csv`为结果表，`video-archive.json`为录像清单。`scripts/run_round_robin.py`支持从已完成场次继续，`scripts/export_tournament.py`生成录像归档。后台批量赛运行期间，网页CPU实验接口拒绝并发启动。

### 全部兼容配对录像

`/web/all-combinations.html` 汇总所有现成兼容配对：63个固定种子0的同本体能力对比任务（跟踪、标准抗扰、参考动作跟踪），4场真实自主格斗，另附1份B-Human原生足球实录。能力对比枚举每个本体上接受同一任务输入的所有不同控制器无序配对；两边在隔离测试位运行，不计格斗胜负。63个任务中复用29份完整记录，补跑34个遗漏配对，原有种子1与其他历史记录保留。

执行入口：`scripts/run_all_combinations.py`；增量校验：`scripts/audit_all_combinations.py`；统一录像副本与CSV：`scripts/export_all_combinations.py`。每条能力记录检查文件SHA256、视频时长/编码，并抽样恢复三个原生物理步重放；格斗4场保留原有完整轨迹重放复核。界面按本体、任务和控制器筛选，视频为完整物理记录渲染，并非伪造攻击动作。

统一归档：`results/all-combinations/round-01/videos/`，使用APFS写时复制生成独立副本，原路径视频和完整轨迹都保留。未支持的本体与控制器组合见 `compatibility-audit.json`；不同关节/观测接口和没有攻击策略的组合不能声称已完成真实格斗。


首轮之后，`scripts/run_all_combinations.py`继续跑完round-02的全部unordered策略配对，包含同一策略自对抗。每种控制架构分别记录配对结果；异策略席位双向都打，单个种子只是兼容性和组合筛查。页面战绩只汇总异策略对局，同策略自对抗单独展示。

视觉赛另外跑了 B-Human K1 3v3 正常相机感知路径：SimRobot `CameraColor` RGB 虚拟相机、B-Human `CameraImage → ECImage / RobotBallPerceptor`。本次录制覆盖 19.96 秒仿真物理时钟，比分从 2:0 变为红队 3:0。官方半场仍在进行，所以这是阶段记分和视觉演示，不宣称完整常规赛终局。原生 OpenGL 帧、B-Human 裁判遥测、H.264 视频和比分来源说明保存在 `results/bhuman-native/vision-native-capture-20261025/`；发布用平面视频位于 `results/tournaments/round-02/videos/`。

### 最终视觉展示赛

`/web/tournament.html` 现汇总8场兼容攻击策略的仿真配对录像，并附一段 B-Human K1 3v3 正常 RGB 相机感知录像。该视觉段来自 SimRobot CameraColor 虚拟相机进入 B-Human 正常感知链路，按 MuJoCo 物理时钟选取原生 framebuffer 帧，最终阶段比分红队3:0；录制段结束时官方10分钟半场仍在进行，因此不声称它是完整官方比赛的终场胜负。公开包带9段MP4、poster、裁判/轨迹复核摘要和录像校验清单；原始高分辨率捕获帧只留本机，以控制仓库体积。
