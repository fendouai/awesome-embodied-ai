import os,sys,json
from combatlab.models import ROOT
from combatlab.research_archive import workspace,run_recorded,atomic_json,event,now
root=workspace();training=json.loads((root/'00_index/training-pass01.json').read_text());index=root/'00_index/evaluation-pass01.json';data=json.loads(index.read_text()) if index.exists() else dict(state='running',started_utc=now(),robots=[])
for row in training['robots']:
 if row['state']!='training_pass_complete':continue
 robot=row['robot']
 if any(r['robot']==robot and r['state']=='evaluated' for r in data['robots']):continue
 data['current']=robot;atomic_json(index,data);folder=row['folder'];checkpoint=row['latest']['checkpoint']
 try:
  run_recorded([sys.executable,str(ROOT/'scripts/evaluate_contact.py'),checkpoint,'--seconds','12'],root/'07_logs'/('evaluate-all-pass01-'+robot+'.log'),cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},timeout=600)
  summary=json.loads(open(folder+'/evaluation/summary.json').read());data['robots'].append(dict(robot=robot,state='evaluated',folder=folder,summary=summary));event('robot_contact_gate_evaluated',robot=robot,all_seeds_continuous_engagement=summary['all_seeds_continuous_engagement'])
 except Exception as error:data['robots'].append(dict(robot=robot,state='error',error=repr(error)));event('robot_contact_gate_failed',robot=robot,error=repr(error))
 atomic_json(index,data)
data.update(state='complete',finished_utc=now());data.pop('current',None);atomic_json(index,data)
