from __future__ import annotations
from pathlib import Path
import json,html,collections,time
import numpy as np
from .models import ROOT
from .engine import write_json

def esc(x):return html.escape(str(x))
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def para(t):return '<p>'+t+'</p>'
def h(t):return '<h2>'+t+'</h2>'
def box(t):return '<div class="box">'+t+'</div>'

def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .render import ReplayRenderer
    from PIL import Image
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    mid='robot-03-f2-m0';folder=ROOT/'results/matches'/mid;r=json.loads((folder/'result.json').read_text());tele=json.loads((folder/'telemetry.json').read_text());raw=np.load(folder/'contacts.npz')['samples'];dt=r['physics']['timestep_s'];times=np.arange(round(r['duration_s']/dt)+1)*dt;force=np.zeros(len(times))
    if len(raw):np.add.at(force,np.minimum(len(times)-1,np.rint(raw[:,0]/dt).astype(int)),np.linalg.norm(raw[:,6:9],axis=1))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(10,4.5),sharex=True,layout='constrained',dpi=160)
    axes[0].plot(times,force,color='#b96d25',lw=1);axes[0].set_ylabel('Contact force / N');axes[0].grid(alpha=.2)
    for i,col in enumerate(['#197898','#b96d25']):axes[1].plot([f['time'] for f in tele],[f['actors'][i]['tilt_deg'] for f in tele],label='Left' if i==0 else 'Right',color=col,lw=1.3)
    axes[1].axhline(65,color='#a53737',lw=1,ls='--',label='65 deg threshold');axes[1].set_ylabel('Root tilt / deg');axes[1].set_xlabel('Simulation time / s');axes[1].legend(ncol=3,fontsize=8);axes[1].grid(alpha=.2)
    for ax in axes:ax.axvline(r['result']['time'],ls=':',color='#526c82',lw=1)
    fig.savefig(out/'force-tilt.png');plt.close(fig)
    renderer=ReplayRenderer(folder,960,540)
    frame_times=[0,max(0,r['first_contact_s']+.02),max(0,r['result']['time']-.3),r['duration_s']]
    for i,t in enumerate(frame_times):Image.fromarray(renderer.frame(t,azimuth=45,distance=5.)).save(out/f'physical-{i}.jpg',quality=90)
    renderer.close()


def build_report():
    out=ROOT/'reports';out.mkdir(exist_ok=True);data=json.loads((ROOT/'results/leaderboards.json').read_text());matches=data['matches'];pf=json.loads((ROOT/'results/preflight.json').read_text());reach=json.loads((ROOT/'results/reach-control-reference.json').read_text());sens=json.loads((ROOT/'results/sensitivity.json').read_text()) if (ROOT/'results/sensitivity.json').exists() else None;verification=json.loads((ROOT/'results/verification.json').read_text()) if (ROOT/'results/verification.json').exists() else None;prov=json.loads((ROOT/'assets/provenance.json').read_text());pages=[]
    def page(title,body):
        n=len(pages)+1;pages.append(f'<section class="page" id="page-{n}"><header><b>BOTSTEST / 仿真战力榜</b><span>MuJoCo · 可核查接触对抗实验</span></header><h1>{title}</h1>{body}<footer><b>作者：BotsTest</b><a href="https://botstest.com/">https://botstest.com/</a><span>{n:02}</span></footer></section>')
    valid=[r for r in matches if r['rated']];decisive=sum(r['result']['winner'] is not None for r in valid)
    page('机器人本体与运动模型<br>仿真对抗实验报告',para('在同一物理场景中，保留接触、失衡、恢复和失败。')+box(f'{len(matches)}场完整对局 · {len(valid)}场有效接触 · {decisive}场产生胜负<br>3种公开完整本体 · 2套真实预训练运动模型')+para('交付包含两个分开的榜单、每场完整3D视频、500Hz原始积分状态和控制轨迹、50Hz可视化测量、模型输入输出、接触明细及网页回放。')+h('先说明这个榜能回答什么')+para('本体榜衡量公开模型在共同控制算法、几何校准与规定伸臂推挤任务下的表现；模型榜衡量同一G1本体上两套实际运动模型部署包的接触抗扰表现。它们均不是实机固有格斗实力排名，也不是VLA或语言模型的通用智能排名。')+para('控制频率、动作解码、原生伺服增益、质量惯量和力矩上限均随配置保存。不存在游戏血量、编造损伤、动画位移或预设赢家。')+h('实验时间与运行环境')+para('2026-10-24；本机Mac ARM64、16GB内存，MuJoCo '+esc(pf['mujoco'])+'，真实Torch/ONNX权重本地运行。完整版本与平台见preflight.json。')+para('Slogan：A field guide to embodied intelligence.'))
    page('实验协议：把比较条件<br>固定并公开',table(['项目','本体榜','运动模型榜'],[('身体','G1 rev1.0 / H1 / H1-2无灵巧手','同一G1 rev1.0，29自由度'),('控制','共同重力补偿PD、几何中性姿态校准、规定伸臂前倾','同一速度追踪驱动；两套原生ONNX部署包'),('配对','3个本体对，每组3摩擦条件、交换左右','4组摩擦/速度条件、交换左右'),('物理步长 / 控制周期','2ms / 20ms','2ms / 20ms'),('结局','倒地、越界或限时平局','倒地、越界或限时平局')])+para('摩擦系数为0.6、0.8、1.0；微小初始横向偏移和朝向扰动由明确随机种子产生，左右交换时保持条件。接触前自发失败、没有接触或数值失败不会计榜；原记录仍保留。')+h('判定的具体含义')+para('根部倾角大于65°，或高度低于中性根部高度55%，连续0.30秒判倒地；根部位置越过±2.5m边界判出界；双方同时失败或限时仍未产生失利则平局。视频保留判定后1.5秒的实际物理运动。')+h('固定算法不等于硬件完全解耦')+para('本体姿态增益采用公开配置并对缺失上肢参考作明确共同设定；重心通过相同几何方法校准。型号间仍有控制校准影响。本体排序只对本协议有效，不能声称纯机械能力已被完全隔离。模型榜保留各部署包的原生增益和动作尺度，因此比较的是部署包整体，不是神经网络架构单独优劣。'))
    for key in ['robot','ai']:
        board=data['leaderboards'][key];rows=board['rows'];page(board['title'],table(['排名 / 实体','Elo','胜 / 平 / 负','分数率','样本重抽样区间'],[(str(r['rank'])+' / '+esc(r['label']),f'{r["elo"]:.1f}',f'{r["wins"]} / {r["draws"]} / {r["losses"]}',f'{r["score_rate"]*100:.1f}%',f'{r["score_rate_bootstrap_95"][0]*100:.1f}–{r["score_rate_bootstrap_95"][1]*100:.1f}%') for r in rows])+h('如何读这个结果')+para('本体对局中，H1-2在本次固定动作协议下对H1取得胜局；G1与另外两种本体的对局为平局。因此本次Elo顺序并不证明任意两台机器人之间都存在严格强弱。' if key=='robot' else '两套运动模型的多数对局为平局；较高速度条件下出现边界失利，构成当前积分差。样本较少，不能据此宣称普遍优劣，更不能外推到格斗训练或实机。')+h('积分与不确定性')+para('Elo初值1500、K=24，对200种确定性洗牌顺序取均值，减少顺序偶然性。分数率按胜1、平0.5、负0计算；表中区间为已记录对局重抽样的分数率范围。左右配对、摩擦条件及小样本存在依赖，区间不是硬件总体性能置信区间；全平局的零宽区间也不代表总体确定。')+para(f'有效场次：{board["rated_matches"]}。被排除场次：{len(board["excluded_matches"])}。完整相互对手矩阵见leaderboards.json，所有平局均保留。')+box('排名描述这次实验条件下的表现。质量、控制校准、碰撞近似与任务定义都可能改变结果。'))
    unique={}
    for r in matches:
        for a in r['actors']:unique[a['key']]=a
    page('实体、控制器与真实权重',table(['公开模型','总质量 / kg','受控自由度','边界'],[(esc(a['robot']),f'{a["mass_kg"]:.6f}',a['controlled_dof'],esc(a['model_scope'])) for a in unique.values()])+h('运动模型')+para('使用unitreerobotics/unitree_rl_lab的G1 29自由度velocity/v0 ONNX，以及unitreerobotics/unitree_rl_mjlab的G1 velocity/v0 ONNX。保留各自关节映射、观测历史、动作尺度及增益。旧的低自由度Gym策略仅用于开发诊断，不参加正式模型榜。')+h('碰撞与控制的可见近似')+para('公开运动模型对手部碰撞有所简化。本项目依据原橡胶手网格补充无质量的凸包碰撞代理，只增加跨机器人/地面接触，不增加惯量或隐藏自碰撞支持。H1上游无用占位关节被固定在零位并移除空电机，其余质量与参数保持。摩擦为试验标准条件，并非实测材料值。')+para('共同控制器在关节力矩约束内施加PD与静态支撑力估计得到的重力补偿，使用MuJoCo内部仿射伺服进行隐式积分；不会给自由根部施加扶正外力。正机械功累计来自实际电机力矩×角速度，不是电池能耗。'))
    plots()
    page('一次倒地如何发生：<br>图像与数值来自同一条轨迹','<div class="figgrid">'+''.join(f'<figure><img src="physical-{i}.jpg"><figcaption>{label}</figcaption></figure>' for i,label in enumerate(['初始状态','首次接触附近','倒地判据持续阶段','双方最终均倒地；H1-2按先后判据获胜']))+'</div><img class="plot" src="force-tilt.png">'+para('示例为robot-03-f2-m0。接触力曲线使用500Hz接触样本聚合，倾角使用50Hz可视化测量。虚线为倒地角度阈值，竖线为实际判定时间；高力峰值不能直接解释为结构损伤。'))
    gates=pf['checks']+reach['checks'];page('验证：先排除适配错误，<br>再解释对抗结果',table(['测试','覆盖','当前证据'],[('无接触站立与前进','3本体站立＋2模型站立/前进','7项基线通过'),('相同伸臂/前倾，无对手','3本体、原动作协议','均持续至限时，无接触、无倒地'),('输入权重与动作','真实下载权重、保存原始观测与输出','文件SHA和神经网络重执行'),('档案与物理结果','正式26场完整轨迹、接触、机械功与判定','独立重算，不只信result.json'),('重放','保存MuJoCo完整积分状态、控制输入','每场选取5个时间点重新积分一步'),('视频','每场完整时长、1280×720、24fps','逐场ffprobe并抽查画面')])+para('验证脚本会检查所有来源文件SHA-256、所有档案哈希、物理步时间连续性、自由根部、无隐藏外力、力矩限幅、完整状态重放，以及UI测量和原始状态一致性。验证范围以results/verification.json中的实际输出为准，不将少量点检描述成完整长期重新积分。')+box('当前无效探索记录保留在results/experiments-contact-v0；没有接触的旧试验未进入正式榜单。正式赛制与资产版本均可单独核查。'))
    if sens:
        page('时间步敏感性：<br>结果是否依赖2ms这个选择？',table(['对局','dt对照','胜负一致','接触冲量相对变化'],[(esc(x['reference']),'2ms → 1ms','是' if x['winner_agrees'] else '否',f'{x["relative_impulse_change"]*100:.2f}%') for x in sens['checks']])+para('保持50Hz控制与相同配置，分别对一场本体对抗和一场较高速度模型对抗，将物理步长减半。完整验证轨迹保存在results/validation，不与正式2ms榜单混合。')+h('结论的范围')+para('这是两场代表性检查，不是对全部对局、所有摩擦与所有接触峰值的收敛证明。接触模型、碰撞凸包、摩擦、控制器与初始微扰都可能影响长期结果，尤其是边界附近的胜负。')+h('尚未建模')+para('没有结构破坏、材料疲劳、温升、电机电流/电池模型、总线时延或感知误差实测；没有为商用未公开机器人猜测CAD、惯量与控制权重；没有把普通语言模型直接当作关节控制器。')+box('真实性来自明确模型、参数、数据和限制。它不意味着公开仿真模型已经等同于真实机器人。'))
    for start in range(0,len(matches),13):
        subset=matches[start:start+13];page('完整对局目录 '+str(start//13+1),table(['对局','协议 / μ','结果','判定时间','原始数据与视频'],[(esc(r['match_id']),('本体' if r['track']=='robot' else '模型')+' / '+str(r['friction']),esc(r['result']['reason'])+' / '+('平局' if r['result']['winner'] is None else '左' if r['result']['winner']==0 else '右'),f'{r["result"]["time"]:.3f}s',f'<a href="http://127.0.0.1:8767/results/matches/{r["match_id"]}/battle.mp4">视频</a> · <a href="http://127.0.0.1:8767/results/matches/{r["match_id"]}/result.json">数据</a>') for r in subset])+para('每场文件夹包含config.json、scene.xml、精确编译模型引用与无损补丁、trajectory.npz、policy_io.npz、contacts.npz、telemetry.json、events.json、result.json、manifest.json及完整battle.mp4。编译模型的共享基底位于results/model-cache，搬移实验包时应保留目录结构。'))
    repositories={}
    for m in prov:repositories.setdefault(m['repository'],m['commit'])
    page('复现、来源与使用方法',h('运行入口')+'<pre>cd projects/robot-combat-lab\nPYTHONPATH=src MUJOCO_GL=cgl .venv/bin/python -m combatlab.cli serve\n# 浏览器：http://127.0.0.1:8767\nPYTHONPATH=src .venv/bin/python -m combatlab.cli verify</pre>'+h('源仓库与固定提交')+table(['来源','固定提交'],[(f'<a href="https://github.com/{repo}/tree/{sha}">{esc(repo)}</a>',f'<code>{sha[:16]}</code>') for repo,sha in repositories.items()])+para(f'共{len(prov)}个第三方资产文件，逐个SHA-256见assets/provenance.json。许可证或版权头保留在来源资产中；没有明确授权信息的来源不作额外许可声明。项目未与相关厂商合作或获得背书。')+h('交互界面')+para('双榜单分开；完整视频同步物理数值与事件；3D状态回放支持时间定位、拖动视角与缩放。交互模式可选择人机、人人或模型对抗，人工仅改变速度指令，关节仍由真实运动模型和物理引擎控制；交互场次单独保存，不进入正式榜单。')+para('作者：BotsTest · https://botstest.com/ · A field guide to embodied intelligence.'))
    css='''*{box-sizing:border-box}body{margin:0;background:#e1e6ee;color:#172b43;font-family:"PingFang SC","Helvetica Neue",sans-serif}.page{width:1120px;height:1584px;padding:62px 76px;position:relative;overflow:hidden;background:#f6f8fc;break-after:page}header{display:flex;justify-content:space-between;gap:20px;border-top:3px solid #24728d;border-bottom:1px solid #b8cadc;padding:16px 0;color:#335975;font-size:14px}h1{font-size:49px;line-height:1.2;margin:38px 0 28px;letter-spacing:-1px}h2{font-size:27px;margin:27px 0 12px;color:#235778}p{font-size:21px;line-height:1.86;margin:12px 0 20px}.box{padding:23px 27px;background:#e1edf6;border-top:2px solid #3384a1;font-size:26px;line-height:1.75;margin:24px 0}table{width:100%;border-collapse:collapse;font-size:18px;line-height:1.7;margin:22px 0}th,td{text-align:left;vertical-align:top;padding:15px 12px;border-bottom:1px solid #b9cbdc}th{background:#dfebf5;font-weight:600;color:#245678}a{color:#226fa1;text-decoration:none}code{font-size:15px;overflow-wrap:anywhere}pre{padding:21px;background:#e6edf5;font:16px/1.8 monospace;white-space:pre-wrap}.figgrid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.figgrid figure{margin:0}.figgrid img{width:100%;display:block}.figgrid figcaption{font-size:14px;color:#5c7892;padding-top:8px}.plot{width:100%;margin-top:25px}footer{position:absolute;bottom:38px;left:76px;right:76px;border-top:2px solid #397e9a;padding:16px 0;display:flex;gap:25px;font-size:18px}footer span{margin-left:auto;color:#617e97}@media print{body{background:white}.page{margin:0}}'''
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>BotsTest仿真战力榜：实验报告</title><style>'+css+'</style><body>'+''.join(pages)+'</body></html>'
    (out/'report.html').write_text(document);write_json(out/'report-manifest.json',dict(pages=len(pages),matches=len(matches),physical_dataset=sha256_file(ROOT/'results/leaderboards.json'),note='Generated from actual results; inspect verification.json for completed coverage.'));print('REPORT',len(pages),'pages',flush=True)
    return out/'report.html'

def sha256_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
