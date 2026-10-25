import json
from combatlab.catalog import catalog
from combatlab.models import ROOT
from combatlab.engine import write_json
c=catalog();rows=[]
for robot in c['robots']:
 for control in c['controls']:
  supported=robot['id'] in control['robots'];tasks=sorted(set(robot['tasks'])&set(control['tasks'])) if supported else []
  rows.append(dict(robot=robot['id'],robot_name=robot['name'],control=control['id'],control_name=control['name'],supported_tasks=tasks,combat_ready='combat' in tasks,reason='Compatible deployment; only its listed task interfaces may be used' if tasks else 'No installed matching observation/action/joint mapping and checkpoint for this body'))
write_json(ROOT/'results/all-combinations/round-01/compatibility-audit.json',dict(robot_count=c['robot_count'],scheme_count=c['control_scheme_count'],rows=rows,scope='Installed body/deployment capability matrix; unsupported combinations are not executed or counted as fights',ncp='Digital character outside the ten physical robot products; separate boxing model round robin retained'))
print('CAPABILITY MATRIX',len(rows),'body/deployment combinations')
