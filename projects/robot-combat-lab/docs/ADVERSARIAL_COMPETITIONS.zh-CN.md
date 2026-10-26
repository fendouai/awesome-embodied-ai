# 三条机器人对抗路线：赛制不同，结果分开看

本地平台已经完成三类有对手互动的比赛：Unitree G1 学习策略格斗、NCP 数字人拳击、B-Human K1 视觉足球。它们的本体、输入和裁判规则不同，分组报告，不合成一个“战力榜”。

| 赛道 | 对抗对象与控制 | 已完成结果 | 完整录像与原始数据 |
| --- | --- | --- | --- |
| RoboStriker G1 | 两台 Unitree G1，Stage3 ego / opponent 学习运动策略；MuJoCo 刚体接触 | ego 对 opponent：16:17、opponent 胜；交换站位：22:16、opponent 胜。ego 自对抗 20:23；opponent 自对抗 2:2 技术平局。自对抗用于观察接触，不计算策略排名。 | [Round 02 录像](../results/tournaments/round-02/videos/) · [逐场结果与重放核验](../results/tournaments/round-02/index.json) |
| NCP 拳击 | 两个 NCP 数字人角色，公开 checkpoint 106 / 100；MuJoCo 人体动力学 | 106 对 100：18:22，100 胜；交换站位：1:2，双方恢复失败，技术平局。106 自对抗 0:0、100 自对抗 10:12，均因双方恢复失败判技术平局。 | [Round 02 录像](../results/tournaments/round-02/videos/) · [逐场结果与重放核验](../results/tournaments/round-02/index.json) |
| B-Human K1 视觉足球 | 6 台 Booster K1，B-Human 对 B-Team，3v3；SimRobot RGB 相机和 B-Human 正常视觉控制链路 | 4 场本地 30 秒展示赛，均由原生裁判正常结束；四场都是 0:0 平局。 | [4 场录像](../results/tournaments/round-03/videos/) · [赛程索引](../results/tournaments/round-03/index.json) · [比分 CSV](../results/tournaments/round-03/vision-matches.csv) · [逐场裁判遥测](../results/tournaments/round-03/matches/) |

## 视觉足球：相机输入与 30 秒本地赛制

每场采用同一套正常 RGB 视觉链路：SimRobot `CameraColor` 虚拟相机 → B-Human `CameraImage` / `ECImage` / `RobotBallPerceptor` → B-Human K1 控制器。场上是六台 K1，红队 B-Human、白队 B-Team。录像来自原生 SimRobot 帧缓冲，比分和 `FINISHED` 状态来自 B-Human `GameController` 遥测，没有脚本进球或人工改分。

为了让 Mac 上的比赛完整结束，四场使用本地 30 秒展示赛半场，由 B-Human 原有的自动裁判超时规则结束。**这是可重复的本地短时赛制，不是官方 10 分钟半场，也不据此评估正式赛事实力。**四场比分均为 0:0；如实保留平局，不人为指定胜者。常规相机感知模式已开启，没有使用 `GameFast` 真值赛道。

四场从首个比赛画面到裁判终局的墙钟录像采集时间合计约 9.6 分钟（每场约 113–197 秒；不含重新载入场景的等待）。因此网页视频长于 30 秒的比赛计时：它保留了本机相机感知运行速度下的原生采样画面。

| 场次 | 红队 | 白队 | 最终比分 | 裁判状态 | 视频 |
| --- | --- | --- | --- | --- | --- |
| 01 | B-Human | B-Team | 0:0，平局 | FINISHED | [播放](../results/tournaments/round-03/videos/vision-bhuman-k1-3v3-match-01.mp4) · [遥测](../results/tournaments/round-03/matches/vision-k1-match-01/match.csv) |
| 02 | B-Human | B-Team | 0:0，平局 | FINISHED | [播放](../results/tournaments/round-03/videos/vision-bhuman-k1-3v3-match-02.mp4) · [遥测](../results/tournaments/round-03/matches/vision-k1-match-02/match.csv) |
| 03 | B-Human | B-Team | 0:0，平局 | FINISHED | [播放](../results/tournaments/round-03/videos/vision-bhuman-k1-3v3-match-03.mp4) · [遥测](../results/tournaments/round-03/matches/vision-k1-match-03/match.csv) |
| 04 | B-Human | B-Team | 0:0，平局 | FINISHED | [播放](../results/tournaments/round-03/videos/vision-bhuman-k1-3v3-match-04.mp4) · [遥测](../results/tournaments/round-03/matches/vision-k1-match-04/match.csv) |

赛制参数、验收元数据和可复现配置见 [`integrations/README.md`](../integrations/README.md)、[`visual-exhibition-acceptance.json`](../integrations/bhuman/visual-exhibition-acceptance.json) 与 [`integrations/bhuman/`](../integrations/bhuman/)。本机新增的 framebuffer / 裁判遥测代码只读采集，不改视觉模块、机器人控制或物理参数。

## MuJoCo 格斗赛覆盖与判定

Round 02 覆盖当前已安装且真正具备自主攻击策略的两组兼容部署：G1 的 ego / opponent 与 NCP 的 106 / 100；异策略对抗交换双方站位，同策略各赛一场，共 8 场。普通行走、平衡、抗扰和恢复控制器没有兼容攻击策略，因此不伪装成拳击算法。NCP 是数字人，不是 G1 硬件。

G1 / NCP 使用 90 秒上限。倾斜超过 65° 或根部高度低于初始值 55% 且持续 0.25 秒，确认倒地；倒地后 10 秒内开始恢复，并保持倾斜低于 25°、高度高于初始值 75% 达 1 秒，才确认站稳。一方恢复失败、另一方站稳时技术终止；双方恢复失败判技术平局。只把满足速度、接触位置和力冲量阈值的手部上身接触计分。

这些分数是指定仿真规则下的对抗结果，**不预测硬件损坏**。目前没有经过标定的 G1 / NCP 电机热失效、减速器寿命或结构破坏数据。每个配对使用一个固定种子，适合初步检查，不足以得出统计排名。

## 全量索引

- 本地双语网页源文件为 `web/competitions.html`（本轮暂留本机，未上传）
- [Round 02：8 场仿真格斗结果](../results/tournaments/round-02/index.json)
- [Round 03：4 场完整视觉赛结果](../results/tournaments/round-03/index.json)
- [Round 03：视频 SHA-256 清单](../results/tournaments/round-03/video-archive.json)
