from combatlab.comparison import Comparison
from combatlab.comparison_analysis import analyze
from combatlab.models import ROOT
from combatlab.engine import write_json
import json,time
if __name__=='__main__':
 old=json.loads((ROOT/'results/comparisons/suite.json').read_text());rows=list(old['rows']);new=[]
 pairs=[('b2','b2_robot_lab'),('go2w','go2w_robot_lab'),('b2w','b2w_robot_lab'),('go1','go1_wtw')]
 jobs=[(r,p,t,s) for r,p in pairs for t in ['tracking','push'] for s in [0,1]]+[('g1_full','g1_beyondmimic','motion',s) for s in [0,1]]
 for r,p,t,seed in jobs:
  sim=Comparison(r,p,'pd_stand',t,seed).run();folder,result=sim.save();analyze(folder);row=dict(folder=str(folder.relative_to(ROOT)),result=result);rows.append(row);new.append(row);print('TEN',len(new),'/',len(jobs),r,t,seed,'fell',sim.fallen.tolist(),'tilt',sim.max_tilt.round(2).tolist(),'motion',[a.get('reference_joint_rmse_rad') for a in result['actors']],flush=True)
  write_json(ROOT/'results/comparisons/ten-extension.json',dict(state='complete' if len(new)==len(jobs) else 'running',rows=new))
 latest=json.loads((ROOT/'results/comparisons/suite.json').read_text());merged={r['folder']:r for r in latest['rows']};merged.update({r['folder']:r for r in rows});rows=list(merged.values());old.update(rows=rows,trials=len(rows),state='complete',ten_extension_trials=len(new));write_json(ROOT/'results/comparisons/suite.json',old)
