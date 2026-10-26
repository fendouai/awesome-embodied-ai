from __future__ import annotations
from itertools import combinations
from pathlib import Path
import json,time,platform
import numpy as np
import mujoco,torch,onnxruntime
from .engine import Simulation,MatchConfig,ROOT,write_json,sha256,SOFTWARE_FINGERPRINT
from .policies import gravity_body

BODY_KEYS=['g1_full','h1_full','h1_2_full']
AI_KEYS=['unitree_rl_lab','unitree_rl_mjlab']
LABELS={'g1_full':'Unitree G1 rev1.0 / 29 DoF','h1_full':'Unitree H1 / 19 DoF','h1_2_full':'Unitree H1-2 handless / 27 DoF','unitree_rl_lab':'Unitree RL Lab · G1 velocity v0','unitree_rl_mjlab':'Unitree RL MjLab · G1 velocity v0'}


def preflight():
    rows=[]
    # Each body must stand without contact or offensive motion before its contest counts.
    for key in BODY_KEYS:
        cfg=MatchConfig('stand-'+key,track='robot',robots=(key,key),policies=('pd_stand','pd_stand'),distance=4.,duration=8.,postroll=0.,rated=False,pose_jitter=0.,position_jitter=0.,ring_half_width=20.)
        sim=Simulation(cfg)
        for p in sim.policies:p.reach_enabled=False
        sim.run();folder,result=sim.save(ROOT/'results/baselines')
        max_tilt=max(max(f['actors'][i]['tilt_deg'] for f in sim.frames) for i in [0,1]);min_height=min(min(f['actors'][i]['height_m']/f['actors'][i]['nominal_height_m'] for f in sim.frames) for i in [0,1]);passed=max_tilt<15 and min_height>.85 and not result['numerical_warnings']
        rows.append(dict(kind='body_stand',participant=key,passed=passed,max_tilt_deg=max_tilt,min_height_ratio=min_height,archive=str(folder.relative_to(ROOT))))
        print('PREFLIGHT',key,'PASS' if passed else 'FAIL',round(max_tilt,2),flush=True)
    for command in [0.,.35]:
        cfg=MatchConfig('policy-baseline-'+str(command),track='ai',robots=('g1_full','g1_full'),policies=tuple(AI_KEYS),distance=8.,duration=8.,postroll=0.,rated=False,pose_jitter=0.,position_jitter=0.,ring_half_width=20.)
        sim=Simulation(cfg)
        sim.desired_command=lambda side:np.array([command,0.,0.])
        sim.run();folder,result=sim.save(ROOT/'results/baselines')
        for i,key in enumerate(AI_KEYS):
            max_tilt=max(f['actors'][i]['tilt_deg'] for f in sim.frames);min_height=min(f['actors'][i]['height_m']/f['actors'][i]['nominal_height_m'] for f in sim.frames);passed=max_tilt<20 and min_height>.8 and not result['numerical_warnings']
            rows.append(dict(kind='policy_walk' if command else 'policy_stand',participant=key,passed=passed,max_tilt_deg=max_tilt,min_height_ratio=min_height,archive=str(folder.relative_to(ROOT))))
            print('PREFLIGHT',key,command,'PASS' if passed else 'FAIL',flush=True)
    report=dict(software_fingerprint=SOFTWARE_FINGERPRINT,passed=all(x['passed'] for x in rows),checks=rows,platform=platform.platform(),mujoco=mujoco.__version__,torch=torch.__version__,onnxruntime=onnxruntime.__version__,compiled_unix=time.time())
    write_json(ROOT/'results/preflight.json',report)
    if not report['passed']:raise RuntimeError('Readiness gate failed; inspect results/preflight.json before publishing ranks')
    return report


def configs():
    configs=[]
    for pidx,pair in enumerate(combinations(BODY_KEYS,2)):
        for condition,friction in enumerate([.6,.8,1.]):
            for mirrored in [0,1]:
                keys=pair if not mirrored else pair[::-1]
                configs.append(MatchConfig(f'robot-{pidx+1:02}-f{condition+1}-m{mirrored}',track='robot',robots=keys,policies=('pd_stand','pd_stand'),seed=20261024+condition*17+pidx,friction=friction,duration=12.,distance=.50,lean=.035,note='Common support-force gravity-compensated joint PD, source neutral stance, prescribed mirrored arm reach/lean; no learned gait in body ranking.'))
    for condition,(friction,speed) in enumerate([(.6,.25),(.8,.35),(1.,.5),(1.,.75)]):
        for mirrored in [0,1]:
            policies=tuple(AI_KEYS if not mirrored else AI_KEYS[::-1])
            configs.append(MatchConfig(f'ai-f{condition+1}-m{mirrored}',track='ai',policies=policies,seed=20261024+condition*31,friction=friction,speed=speed,duration=16.,note='Same G1 rev1.0 body, physics, pursuit driver and timing; native checkpoint observation/action decode and PD gains retained and disclosed.'))
    return configs


def tournament():
    pf=ROOT/'results/preflight.json'
    if not pf.exists() or not json.loads(pf.read_text())['passed'] or json.loads(pf.read_text()).get('software_fingerprint')!=SOFTWARE_FINGERPRINT:preflight()
    cc=configs();write_json(ROOT/'configs/tournament.json',[c.__dict__ for c in cc])
    results=[]
    for i,c in enumerate(cc):
        folder=ROOT/'results/matches'/c.match_id
        if (folder/'result.json').exists() and json.loads((folder/'result.json').read_text()).get('software_fingerprint')==SOFTWARE_FINGERPRINT and (folder/'config.json').exists() and json.loads((folder/'config.json').read_text())==json.loads(json.dumps(c.__dict__)):
            r=json.loads((folder/'result.json').read_text());print('REUSE',c.match_id,flush=True)
        else:
            st=time.time();sim=Simulation(c).run();folder,r=sim.save();print(f'MATCH {i+1}/{len(cc)}',c.match_id,r['result'],'rated',r['rated'],f'{time.time()-st:.1f}s',flush=True)
        results.append(r)
    write_json(ROOT/'results/matches.json',results)
    return rank(results)


def rank(results=None,persist=True):
    if results is None:results=json.loads((ROOT/'results/matches.json').read_text())
    output={}
    for track,keys in [('robot',BODY_KEYS),('ai',AI_KEYS)]:
        rated=[r for r in results if r['track']==track and r['rated']]
        games=[];data={k:dict(key=k,label=LABELS[k],wins=0,draws=0,losses=0,points=[],opponents={},energies=[],tilts=[],durations=[]) for k in keys}
        for r in rated:
            contestants=r['robots'] if track=='robot' else r['policies'];winner=r['result']['winner'];score=.5 if winner is None else 1. if winner==0 else 0.
            games.append((contestants[0],contestants[1],score))
            for side,key in enumerate(contestants):
                d=data[key];s=score if side==0 else 1-score;d['points'].append(s);d['wins']+=s==1;d['draws']+=s==.5;d['losses']+=s==0;d['energies'].append(r['positive_mechanical_work_j'][side]);d['tilts'].append(r['max_tilt_deg'][side]);d['durations'].append(r['duration_s'])
                d['opponents'].setdefault(contestants[1-side],[]).append(s)
        # Order-averaged Elo reduces the arbitrary effect of sequential match ordering.
        rng=np.random.default_rng(731);ratings=[]
        for _ in range(200):
            rating={k:1500. for k in keys}
            for ix in rng.permutation(len(games)):
                a,b,s=games[ix];expected=1/(1+10**((rating[b]-rating[a])/400));delta=24*(s-expected);rating[a]+=delta;rating[b]-=delta
            ratings.append(rating)
        rows=[]
        for k,d in data.items():
            n=len(d['points']);pts=np.array(d['points']);ci=np.quantile(np.mean(rng.choice(pts,size=(5000,n)),axis=1),[.025,.975]).tolist() if n else [None,None]
            row=dict(key=k,label=d['label'],matches=n,wins=d['wins'],draws=d['draws'],losses=d['losses'],score_rate=float(np.mean(pts)) if n else None,win_rate=d['wins']/n if n else None,score_rate_bootstrap_95=ci,elo=float(np.mean([r[k] for r in ratings])) if n else None,mean_positive_mechanical_work_j=float(np.mean(d['energies'])) if n else None,mean_peak_tilt_deg=float(np.mean(d['tilts'])) if n else None,opponent_score_matrix={o:float(np.mean(x)) for o,x in d['opponents'].items()})
            rows.append(row)
        rows.sort(key=lambda x:-(x['elo'] if x['elo'] is not None else -1))
        previous=None;rank_no=0
        for i,row in enumerate(rows):
            if row['elo'] is None:row['rank']=None;continue
            if previous is None or abs(row['elo']-previous)>1:rank_no=i+1
            row['rank']=rank_no;previous=row['elo']
        output[track]=dict(title='机器人本体榜 / 固定控制协议' if track=='robot' else '运动模型榜 / 固定G1本体',protocol='body-push-v2' if track=='robot' else 'motion-policy-contact-v2',rows=rows,rated_matches=len(rated),excluded_matches=[dict(match_id=r['match_id'],reason=r['result']['reason'],contact_before_result=r['result'].get('contact_before_result')) for r in results if r['track']==track and not r['rated']],confidence_note='Score-rate bootstrap interval over recorded paired-condition bouts; small and dependent samples, not hardware population confidence.',interpretation='Task- and calibration-specific simulation ranking, not intrinsic real-world combat strength.' if track=='robot' else 'Downloaded motion-policy deployment packages incl. native decoding/gains; not VLA/LLM general-intelligence ranking.')
    package=dict(schema_version=1,leaderboards=output,matches=results,created_unix=time.time(),project='BotsTest Robot Combat Lab',authenticity=dict(physics='MuJoCo direct dynamics, free roots, motor torques and real contacts',no_fictional_damage=True,actual_pretrained_models=AI_KEYS,mechanical_work_is_not_battery_consumption=True,unforced_or_no_contact_outcomes_excluded_from_rated=True))
    if persist:write_json(ROOT/'results/leaderboards.json',package)
    return package
