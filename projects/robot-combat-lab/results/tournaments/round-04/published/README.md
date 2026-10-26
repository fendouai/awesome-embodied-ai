# Round 04：完整比赛、中文解说和字幕

8场MuJoCo格斗与4场B-Human相机视觉足球。均使用明确公布的180秒本地计分/比赛窗口；格斗可因恢复条件提前结束。每段解说MP4包含完整原比赛过程与明确记录的首尾静帧，中文配音、烧录字幕以及独立SRT/VTT与解说稿。

官方参考赛时分别记录于protocol.json：RoboStriker源配置50秒，NCP约16.67秒，HSL两个10分钟半场。本轮本地成绩不冒充官方基准复现。G1、NCP数字人和足球分别呈现，不跨任务排名。

| 比赛 | 赛道 | 接触统计 / 进球 | 裁判结果 | 视频 | 字幕与解说稿 |
| --- | --- | --- | --- | --- | --- |
| g1-robostriker_ego-robostriker_ego | simulation | 41:40 | 蓝方胜 · 180.00s | [中文解说](videos/g1-robostriker_ego-robostriker_ego.zh-CN.mp4) | [SRT](matches/g1-robostriker_ego-robostriker_ego/captions.zh-CN.srt) · [VTT](matches/g1-robostriker_ego-robostriker_ego/captions.zh-CN.vtt) · [解说稿](matches/g1-robostriker_ego-robostriker_ego/script.zh-CN.txt) |
| g1-robostriker_ego-robostriker_opponent | simulation | 34:40 | 橙方胜 · 180.00s | [中文解说](videos/g1-robostriker_ego-robostriker_opponent.zh-CN.mp4) | [SRT](matches/g1-robostriker_ego-robostriker_opponent/captions.zh-CN.srt) · [VTT](matches/g1-robostriker_ego-robostriker_opponent/captions.zh-CN.vtt) · [解说稿](matches/g1-robostriker_ego-robostriker_opponent/script.zh-CN.txt) |
| g1-robostriker_opponent-robostriker_ego | simulation | 28:21 | 技术平局 · 140.66s | [中文解说](videos/g1-robostriker_opponent-robostriker_ego.zh-CN.mp4) | [SRT](matches/g1-robostriker_opponent-robostriker_ego/captions.zh-CN.srt) · [VTT](matches/g1-robostriker_opponent-robostriker_ego/captions.zh-CN.vtt) · [解说稿](matches/g1-robostriker_opponent-robostriker_ego/script.zh-CN.txt) |
| g1-robostriker_opponent-robostriker_opponent | simulation | 2:2 | 技术平局 · 21.79s | [中文解说](videos/g1-robostriker_opponent-robostriker_opponent.zh-CN.mp4) | [SRT](matches/g1-robostriker_opponent-robostriker_opponent/captions.zh-CN.srt) · [VTT](matches/g1-robostriker_opponent-robostriker_opponent/captions.zh-CN.vtt) · [解说稿](matches/g1-robostriker_opponent-robostriker_opponent/script.zh-CN.txt) |
| ncp-106-106 | simulation | 0:0 | 技术平局 · 12.64s | [中文解说](videos/ncp-106-106.zh-CN.mp4) | [SRT](matches/ncp-106-106/captions.zh-CN.srt) · [VTT](matches/ncp-106-106/captions.zh-CN.vtt) · [解说稿](matches/ncp-106-106/script.zh-CN.txt) |
| ncp-106-100 | simulation | 23:27 | 技术平局 · 149.49s | [中文解说](videos/ncp-106-100.zh-CN.mp4) | [SRT](matches/ncp-106-100/captions.zh-CN.srt) · [VTT](matches/ncp-106-100/captions.zh-CN.vtt) · [解说稿](matches/ncp-106-100/script.zh-CN.txt) |
| ncp-100-106 | simulation | 1:2 | 技术平局 · 23.81s | [中文解说](videos/ncp-100-106.zh-CN.mp4) | [SRT](matches/ncp-100-106/captions.zh-CN.srt) · [VTT](matches/ncp-100-106/captions.zh-CN.vtt) · [解说稿](matches/ncp-100-106/script.zh-CN.txt) |
| ncp-100-100 | simulation | 10:12 | 技术平局 · 57.41s | [中文解说](videos/ncp-100-100.zh-CN.mp4) | [SRT](matches/ncp-100-100/captions.zh-CN.srt) · [VTT](matches/ncp-100-100/captions.zh-CN.vtt) · [解说稿](matches/ncp-100-100/script.zh-CN.txt) |
| vision-01 | vision | 3:2 | 红队胜 | [中文解说](videos/vision-01.zh-CN.mp4) | [SRT](matches/vision-01/captions.zh-CN.srt) · [VTT](matches/vision-01/captions.zh-CN.vtt) · [解说稿](matches/vision-01/script.zh-CN.txt) |
| vision-02 | vision | 0:4 | 白队胜 | [中文解说](videos/vision-02.zh-CN.mp4) | [SRT](matches/vision-02/captions.zh-CN.srt) · [VTT](matches/vision-02/captions.zh-CN.vtt) · [解说稿](matches/vision-02/script.zh-CN.txt) |
| vision-03 | vision | 0:2 | 白队胜 | [中文解说](videos/vision-03.zh-CN.mp4) | [SRT](matches/vision-03/captions.zh-CN.srt) · [VTT](matches/vision-03/captions.zh-CN.vtt) · [解说稿](matches/vision-03/script.zh-CN.txt) |
| vision-04 | vision | 1:3 | 白队胜 | [中文解说](videos/vision-04.zh-CN.mp4) | [SRT](matches/vision-04/captions.zh-CN.srt) · [VTT](matches/vision-04/captions.zh-CN.vtt) · [解说稿](matches/vision-04/script.zh-CN.txt) |

完整原片、配音成片母版、48kHz配音WAV和详细物理归档保留在本机；公开包的大文件另行压缩，保持完整时序、720p画面与音频字幕，哈希单独记录。网站按要求暂留本机。audit.json记录视频、配音字幕和原始证据检查结果。

## G1

[![G1](videos/g1-robostriker_ego-robostriker_opponent.jpg)](videos/g1-robostriker_ego-robostriker_opponent.zh-CN.mp4)

## NCP

[![NCP](videos/ncp-106-100.jpg)](videos/ncp-106-100.zh-CN.mp4)

## B-Human

[![B-Human](videos/vision-01.jpg)](videos/vision-01.zh-CN.mp4)
