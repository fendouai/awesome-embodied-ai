"""Reproducible local CPU suite. Every pair is same robot and same task/seed."""
from pathlib import Path
import argparse,json,time
from combatlab.comparison import Comparison,COMPAT
from combatlab.models import ROOT
from combatlab.engine import write_json
PAIRS=[('g1_full','nvidia_wbc',b) for b in ['amp_recovery','holosoma_fastsac','holosoma_ppo','unitree_rl_lab','unitree_rl_mjlab','pd_stand']]+[('h1_full','unitree_full_gym','pd_stand'),('h1_2_full','unitree_full_gym','pd_stand'),('go2','go2_robot_lab','go2_him'),('a1','a1_vq','pd_stand')]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');args=p.parse_args();jobs=[(*pair,task,seed) for pair in PAIRS for task in ['tracking','push'] for seed in ([0] if args.quick else [0,1])];rows=[];start=time.time()
 for robot,left,right,task,seed in jobs:
  s=Comparison(robot,left,right,task,seed).run();folder,result=s.save();rows.append(dict(folder=str(folder.relative_to(ROOT)),result=result));print(len(rows),'/',len(jobs),robot,left,right,task,seed,'fell',s.fallen.tolist(),'tilt',s.max_tilt.round(1).tolist(),flush=True);write_json(ROOT/'results/comparisons/suite.json',dict(state='running' if len(rows)<len(jobs) else 'complete',trials=len(rows),planned=len(jobs),rows=rows,wall_time_s=time.time()-start,scope='Same-body deployment-package comparisons. Different tasks are never combined into a combat score. Native gains differ and are archived.'))
