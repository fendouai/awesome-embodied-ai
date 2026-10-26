import json
from combatlab.models import ROOT
from combatlab.engine import write_json
from combatlab.comparison import Comparison
j=json.loads((ROOT/'results/comparisons/suite-initial-placement.json').read_text());rows=[r for r in j['rows'] if r['result']['robots'][0] not in ['go2','a1']]
for robot,left,right in [('go2','go2_robot_lab','go2_him'),('a1','a1_vq','pd_stand')]:
 for task in ['tracking','push']:
  for seed in [0,1]:
   s=Comparison(robot,left,right,task,seed).run();folder,result=s.save();rows.append(dict(folder=str(folder.relative_to(ROOT)),result=result));print(robot,task,seed,'tilt',s.max_tilt.round(2),'fell',s.fallen,flush=True)
j.update(rows=rows,trials=len(rows),state='complete',protocol_revision=2,placement='Source neutral joint pose placed at3mm collision clearance; earlier quadruped placement trials preserved separately, excluded from current suite');write_json(ROOT/'results/comparisons/suite.json',j)
