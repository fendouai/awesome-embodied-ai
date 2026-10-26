from pathlib import Path
from collections import defaultdict
import json,html,hashlib
import numpy as np
from combatlab.models import ROOT
from combatlab.engine import write_json
from combatlab.control_upgrade import LABELS

P=ROOT/'results/control-upgrade';reports=ROOT/'reports';reports.mkdir(exist_ok=True)
screen=json.loads((P/'screen-results.json').read_text())+json.loads((P/'refine-results.json').read_text());validation=json.loads((P/'validation-results.json').read_text());contacts=json.loads((P/'contact-results.json').read_text());verified=json.loads((P/'verification.json').read_text());assert verified['passed']
selected={'g1_full':'nvidia_wbc','h1_full':'unitree_full_gym','h1_2_full':'unitree_full_gym'};group=defaultdict(list)
for r in screen:
 if r['impulse_ns']>0:group[r['robot'],r['policy']].append(r)
rows=[]
for (robot,policy),rs in group.items():
 levels=sorted({r['impulse_ns'] for r in rs});allmax=max([0]+[j for j in levels if len([r for r in rs if r['impulse_ns']==j])==4 and all(r['actors'][0]['survived'] for r in rs if r['impulse_ns']==j)])
 walks=[r for r in screen if r['robot']==robot and r['policy']==policy and r['walk_command_m_s']>0];walk=walks[0]['actors'][0] if walks else None
 rows.append(dict(robot=robot,policy=policy,passed=sum(r['actors'][0]['survived'] for r in rs),trials=len(rs),baseline_passed=sum(r['actors'][1]['survived'] for r in rs),all_direction_impulse_ns=allmax,walk_displacement_m=walk['displacement_m'] if walk else None,walk_command_passed=bool(walk and walk['survived'] and walk['displacement_m']>.5)))
rows.sort(key=lambda r:(r['robot'],-r['all_direction_impulse_ns'],-r['passed'],r['policy']))
videos=[]
for robot,j in [('g1_full',40),('h1_full',30),('h1_2_full',30)]:
 r=next(x for x in validation if x['robot']==robot and x['direction']=='forward' and x['impulse_ns']==j and x['seed']==1);videos.append(dict(title=f'{robot} · {j:g} N·s 抗推对照',path=f'results/control-upgrade/archives/{r["id"]}/battle.mp4',note='左侧新策略，右侧旧中性PD；相同本体、相同外部冲量。红箭头仅示方向，推力数值来自记录。此录像完整8秒。'))
r=next(x for x in validation if x['impulse_ns']==60);videos.append(dict(title='失败案例 · G1 60 N·s 正向推力',path=f'results/control-upgrade/archives/{r["id"]}/battle.mp4',note='保留超过当前恢复范围的失败，避免只展示成功。600N持续0.1秒；不代表实机极限。'))
for r in contacts:videos.append(dict(title=f'新控制器接触对局 · {r["robots"][0]} · {r["policies"][0]} vs {r["policies"][1]}',path=f'results/control-upgrade/contact/{r["match_id"]}/battle.mp4',note='真实控制器驱动的完整接触对局；12秒内未触发倒地或越界判据，另保留1.5秒后续。非格斗训练策略，不用于跨本体排名。'))
summary=dict(status='complete',date='2026-10-25',screen_trials=len(screen),validation_trials=len(validation),rows=rows,selected=selected,videos=videos,holdout=[dict(robot=robot,passed=sum(r['actors'][0]['survived'] for r in validation if r['robot']==robot and r['impulse_ns']<60),trials=sum(r['robot']==robot and r['impulse_ns']<60 for r in validation),tested_all_direction_impulse_ns=40 if robot=='g1_full' else 30) for robot in selected],contacts=contacts,verification='results/control-upgrade/verification.json',selection_scope='Best among actually tested public deployment candidates on the current bodies; not a global strongest claim. NVIDIA native Balance/Walk switch; H1/H1-2 leg-policy transfer.',historical_rankings_deprecated=True)
write_json(P/'summary.json',summary)
# Actual scientific plot, not an illustrative recovery trajectory.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
folder=P/'archives'/next(r['id'] for r in validation if r['robot']=='g1_full' and r['impulse_ns']==40 and r['seed']==1 and r['direction']=='forward');tele=json.loads((folder/'telemetry.json').read_text());ts=[f['time'] for f in tele]
fig,axes=plt.subplots(2,1,figsize=(10,4.4),sharex=True,layout='constrained',dpi=160)
for side,col,label in [(0,'#1687b0','NVIDIA WBC'),(1,'#bf732e','Old neutral PD')]:
 axes[0].plot(ts,[f['actors'][side]['tilt_deg'] for f in tele],color=col,label=label);axes[1].plot(ts,[f['actors'][side]['height_m'] for f in tele],color=col)
for ax in axes:ax.axvspan(3,3.1,color='#bd4646',alpha=.2);ax.grid(alpha=.2)
axes[0].set_ylabel('Root tilt / deg');axes[0].legend();axes[1].set_ylabel('Root height / m');axes[1].set_xlabel('Simulation time / s');fig.savefig(reports/'control-upgrade-response.png');plt.close(fig)
def e(v):return html.escape(str(v))
def table(head,rows):return '<table><thead><tr>'+''.join('<th>'+e(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
names={'g1_full':'G1 rev1.0','h1_full':'H1','h1_2_full':'H1-2无灵巧手'};old={'unitree_rl_lab':'原Unitree RL Lab','unitree_rl_mjlab':'原Unitree RL MjLab'};label=lambda k:LABELS.get(k,old.get(k,k))
pages=[]
def page(title,body):pages.append('<section class="page"><header>BOTSTEST / CONTROL UPGRADE <span>2026-10-25 · MuJoCo</span></header><h1>'+title+'</h1>'+body+'<footer>作者：BotsTest · https://botstest.com/ <span>'+str(len(pages)+1)+'</span></footer></section>')
page('从“碰一下就倒”<br>到可量化的抗推验证','<div class="box">182项筛选与细化 · 33项完整轨迹验证<br>3种本体完成适配 · 4场新接触对局</div><p>当前G1优选：NVIDIA Decoupled WBC，按原生规则切换Balance与Walk。H1/H1-2：各自Unitree RL Gym腿部权重，按关节名映射到完整本体。候选在同条件中实测选择，未声称全球最强。</p>'+table(['机器人','验证条件','新控制器','同场旧PD'],[(names[x['robot']],('30 / 40' if x['robot']=='g1_full' else str(x['tested_all_direction_impulse_ns']))+' N·s，四方向，2个初始种子',str(x['passed'])+'/'+str(x['trials']),str(sum(r['actors'][1]['survived'] for r in validation if r['robot']==x['robot'] and r['impulse_ns']<60))+'/'+str(x['trials'])) for x in summary['holdout']])+'<h2>哪些结论已经有证据？</h2><p>G1在30和40N·s条件的16项完整验证全部通过；H1和H1-2分别通过30N·s的8项验证。G1在60N·s正向推力下失败，该录像与数据一并保留。4场新接触对局均发生实际机器人接触并持续到12秒限时。</p><p>这证明当前适配在这些条件下明显优于旧中性PD；不证明机器人可以承受任意持续推挤，也不代表真实硬件或格斗能力。</p>')
page('候选比较：强弱来自实测',table(['本体 / 控制器','扰动通过','旧PD','四方向均通过的最大已测冲量'],[(names[r['robot']]+' / '+e(label(r['policy'])),f'{r["passed"]}/{r["trials"]}',f'{r["baseline_passed"]}/{r["trials"]}',str(r['all_direction_impulse_ns'])+' N·s') for r in rows])+'<p>G1的7个候选各有20个扰动条件：四方向×10/30/40/50/60N·s。H1/H1-2各12个条件：四方向×10/30/60N·s。此外每个候选有无推力站立、0.35m/s行走检查，共182项。计数不是总体成功率，不按不同质量的机器人跨本体排名。</p><p>Balance单模型虽能抗推，但不能执行本次行走指令（仅移动约0.01m）；Walk完成约1.84m。因此采用源部署的组合：指令范数≤0.05用Balance，否则用Walk，保留同一历史和上次动作。组合的静止抗推验证单独保存。</p>')
page('相同推力，真实的恢复差异','<img class="plot" src="control-upgrade-response.png"><p>G1、400N持续0.1秒，即40N·s。两条曲线直接取自同一对照实验：新控制器与旧PD使用同一本体模型，分别承受与各自初始朝向对应的相同水平冲量。</p><h2>试验和通过判据</h2><p>t=3秒向躯干施加0.1秒水平推力，方向固定于初始朝向；试验总长8秒。前3秒必须站立就绪；推力后倾角>65°或根部高度<中性高度55%持续0.3秒算跌倒；最后0.8秒需保持倾角<20°、根部高度>70%、平面速度<0.35m/s，才算通过。</p><p>筛选用2ms物理步长、20ms测量；完整验证保存每2ms状态、控制和显式外力。所有恢复力矩必须由电机产生，没有额外扶正或位置动画。支持脚移动由物理接触产生。</p><h2>验证范围</h2><p>33项档案的哈希、实际加载适配源码、全部外力时序、零未声明外力与8个关键状态的一步重新积分均通过。ONNX逐项抽4个观测重新推理；H1/H1-2的LSTM在复现两次初始化调用后，完整399个已记录观测序列重执行一致。没有把这些检查描述为全程长期重积分。</p>')
assets=json.loads((ROOT/'research/control-upgrade/candidate-assets.json').read_text());repos={r['repository']:r['commit'] for r in assets}
page('适配边界、来源与复现','<h2>具体改了什么</h2><p>Holosoma按原生C++排序观测项，读取ONNX关节名和增益，保持站立/行走相位。AMP保持四帧历史和力矩/Kp动作缩放。NVIDIA保留86维单步观测、6帧历史、15维下肢/腰动作和上肢原生伺服。H1/H1-2按关节名映射各自10/12维腿部策略，其余关节伺服保持。</p><p>H1/H1-2源策略在降自由度身体上训练，本次是到完整本体的迁移。它改善了平衡，却不能称为完整全身策略；不应强行套用G1权重。当前均未做对抗训练、结构损伤、电流温升或电池仿真。</p>'+table(['代码与权重源','固定提交'],[(f'<a href="https://github.com/{repo}/tree/{sha}">{e(repo)}</a>',sha[:16]) for repo,sha in repos.items()])+'<p>补充调研：ASAP侧重动态动作跟踪与迁移；XHugWBC项目页研究时仍标注Code Coming Soon，未编造可运行权重。本轮NVIDIA权重来自Decoupled WBC，未将其冒称GEAR-SONIC。</p><pre>PYTHONPATH=src .venv/bin/python scripts/screen_controllers.py\nPYTHONPATH=src .venv/bin/python scripts/refine_controllers.py\nPYTHONPATH=src .venv/bin/python scripts/validate_controllers.py\nPYTHONPATH=src .venv/bin/python scripts/verify_control_upgrade.py\nPYTHONPATH=src .venv/bin/python -m combatlab.cli serve</pre><p>原始数据：results/control-upgrade；候选来源：research/control-upgrade；当前平台默认进入升级验证页。旧榜单保留并降级为历史探索，不再将其解释为机器人本体实力。</p>')
css='''*{box-sizing:border-box}body{margin:0;background:#e1e6ee;color:#172b43;font-family:"PingFang SC","Helvetica Neue",sans-serif}.page{width:1120px;height:1584px;padding:60px 76px;position:relative;overflow:hidden;background:#f6f8fc;break-after:page}header{border-top:3px solid #24728d;border-bottom:1px solid #b8cadc;padding:16px 0;font-size:15px;display:flex;justify-content:space-between}h1{font-size:47px;line-height:1.23;margin:35px 0 25px}h2{font-size:27px;margin:24px 0 10px}p{font-size:21px;line-height:1.8;margin:15px 0}table{width:100%;border-collapse:collapse;font-size:17px;line-height:1.6;margin:20px 0}th,td{padding:12px 10px;text-align:left;border-bottom:1px solid #b9cbdc;vertical-align:top}th{background:#dfebf5}a{color:#246f99}.box{background:#e1edf6;padding:24px;font-size:29px;line-height:1.6}.plot{width:100%}pre{font:15px/1.8 monospace;white-space:pre-wrap;background:#e6edf5;padding:18px}footer{position:absolute;bottom:38px;left:76px;right:76px;border-top:2px solid #397e9a;padding-top:16px;font-size:17px;display:flex;justify-content:space-between}'''
(reports/'control-upgrade.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>控制器升级与抗推验证报告</title><style>'+css+'</style><body>'+''.join(pages)+'</body></html>');print('SUMMARY',len(screen),'screen',len(validation),'validation',len(videos),'videos; REPORT',len(pages),'pages')
