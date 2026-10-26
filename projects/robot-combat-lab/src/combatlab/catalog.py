"""Explicit robot/control capability matrix. Counts hardware products, not presets."""
from .models import ROOT
from .comparison import COMPAT,LABELS
ROBOTS=[
 dict(id='g1_full',name='Unitree G1',morphology='人形',dof=29,tasks=['tracking','push','motion','combat'],profile_note='基础能力使用rev1.0；格斗使用RoboStriker源URDF，两个配置不混排',video='/web/integrations.html#robostriker'),
 dict(id='h1_full',name='Unitree H1',morphology='人形',dof=19,tasks=['tracking','push'],profile_note='原生腿部策略映射到完整本体，上肢由PD保持',video=None),
 dict(id='h1_2_full',name='Unitree H1-2',morphology='人形',dof=27,tasks=['tracking','push'],profile_note='无灵巧手配置；腿部策略迁移，上肢PD保持',video=None),
 dict(id='go2',name='Unitree Go2',morphology='四足',dof=12,tasks=['tracking','push'],profile_note='Menagerie本体＋rl_sar两套公开权重；CPU/MuJoCo迁移',video=None),
 dict(id='a1',name='Unitree A1',morphology='四足',dof=12,tasks=['tracking','push'],profile_note='Menagerie本体＋rl_sar VQ历史策略；CPU/MuJoCo迁移',video=None),
 dict(id='k1',name='Booster K1',morphology='人形',dof=22,tasks=['soccer'],profile_note='B-Human原生3v3后端，MuJoCo3.3.5；不混入基础能力基准',video='/web/integrations.html#bhuman')]
ROBOTS.extend([
 dict(id='go1',name='Unitree Go1',morphology='四足',dof=12,tasks=['tracking','push'],profile_note='Menagerie本体；Walk These Ways公开Go1权重，历史隐变量适应；PD电机近似不等于原训练actuator_net',video=None),
 dict(id='b2',name='Unitree B2',morphology='重型四足',dof=12,tasks=['tracking','push'],profile_note='源本体＋配套B2策略；本轮0.30m/s跟踪偏慢，保留原始结果；未改限幅以追求成绩',video=None),
 dict(id='go2w',name='Unitree Go2-W',morphology='轮足',dof=16,tasks=['tracking','push'],profile_note='12个腿关节位置控制＋4个轮轴速度控制；独立轮足本体与配套权重',video=None),
 dict(id='b2w',name='Unitree B2-W',morphology='重型轮足',dof=16,tasks=['tracking','push'],profile_note='重型轮足源本体＋独立B2-W权重；轮轴不是锁定的位置关节',video=None)
])
CONTROLS=[
 dict(id='unitree_rl_lab',name='Unitree RL Lab PPO',scheme='ppo',robots=['g1_full'],tasks=['tracking','push']),
 dict(id='unitree_rl_mjlab',name='Unitree RL MjLab PPO',scheme='ppo',robots=['g1_full'],tasks=['tracking','push']),
 dict(id='holosoma_ppo',name='Holosoma PPO',scheme='ppo',robots=['g1_full'],tasks=['tracking','push']),
 dict(id='holosoma_fastsac',name='Holosoma FastSAC',scheme='fastsac',robots=['g1_full'],tasks=['tracking','push']),
 dict(id='amp_recovery',name='AMP 行走策略',scheme='amp',robots=['g1_full'],tasks=['tracking','push']),
 dict(id='nvidia_wbc',name='GR00T WBC Balance/Walk',scheme='wbc',robots=['g1_full'],tasks=['tracking','push']),
 dict(id='unitree_full_gym',name='Unitree RL Gym PPO 腿部迁移',scheme='ppo',robots=['h1_full','h1_2_full'],tasks=['tracking','push']),
 dict(id='go2_robot_lab',name='Go2 Robot Lab PPO',scheme='ppo',robots=['go2'],tasks=['tracking','push']),
 dict(id='go2_him',name='Go2 HIM 历史与状态估计',scheme='him',robots=['go2'],tasks=['tracking','push']),
 dict(id='a1_vq',name='A1 VQ 历史编码策略',scheme='vq',robots=['a1'],tasks=['tracking','push']),
 dict(id='robostriker_stage3',name='RoboStriker Stage3 潜变量格斗',scheme='robostriker',robots=['g1_full'],tasks=['combat'],route='/web/integrations.html#robostriker'),
 dict(id='bhuman',name='B-Human 分层足球控制',scheme='bhuman',robots=['k1'],tasks=['soccer'],route='/web/integrations.html#bhuman'),
 dict(id='go1_wtw',name='Walk These Ways 隐变量自适应',scheme='wtw_adaptation',robots=['go1'],tasks=['tracking','push']),
 dict(id='b2_robot_lab',name='B2 Robot Lab PPO',scheme='ppo',robots=['b2'],tasks=['tracking','push']),
 dict(id='go2w_robot_lab',name='Go2-W Robot Lab PPO',scheme='ppo',robots=['go2w'],tasks=['tracking','push']),
 dict(id='b2w_robot_lab',name='B2-W Robot Lab PPO',scheme='ppo',robots=['b2w'],tasks=['tracking','push']),
 dict(id='g1_beyondmimic',name='BeyondMimic 参考动作跟踪',scheme='beyondmimic',robots=['g1_full'],tasks=['motion']),
 dict(id='pd_stand',name='中性关节PD（参考基线）',scheme='reference',robots=list(COMPAT),tasks=['tracking','push','motion'])]
SCHEMES={'ppo':'PPO部署','fastsac':'FastSAC','amp':'AMP','wbc':'GR00T WBC','him':'HIM状态估计/历史','vq':'VQ历史编码','robostriker':'Stage3潜变量格斗','bhuman':'B-Human分层足球','wtw_adaptation':'Walk These Ways隐变量自适应','beyondmimic':'BeyondMimic参考动作跟踪'}
def catalog():
 return dict(robot_count=len(ROBOTS),control_scheme_count=len(SCHEMES),deployment_package_count=len([c for c in CONTROLS if c['scheme']!='reference']),robots=ROBOTS,controls=CONTROLS,schemes=SCHEMES,compatibility=COMPAT,task_labels={'tracking':'命令跟踪','push':'标准化抗扰','combat':'自主格斗','soccer':'机器人足球','motion':'参考动作跟踪'},counting='10款实体机器人产品的仿真本体；G1不同URDF和不同权重不重复计数。10类控制方案按部署架构分类，不表示10种不同RL训练算法。PPO权重包归为同类，PD另列参考。NCP数字人不计入机器人数量。',fairness='只对同一本体、同一任务、同一随机种子结果做并排比较；原生增益和观测不同，比较的是部署组合，不是隔离训练算法的因果实验。无跨本体综合战力分。')
