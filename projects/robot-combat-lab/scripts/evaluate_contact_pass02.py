"""Evaluate completed second-pass checkpoints while remaining bodies train."""
import json,os,sys,time
from combatlab.models import ROOT
from combatlab.research_archive import workspace,run_recorded,atomic_json,event,now
root=workspace();path=root/'00_index/evaluation-pass02.json';data=json.loads(path.read_text()) if path.exists() else dict(state='running',started_utc=now(),robots=[])
while True:
 plan=json.loads((root/'00_index/training-pass02.json').read_text())
 for row in plan['robots']:
  if row['state']!='training_pass_complete' or any(r['robot']==row['robot'] for r in data['robots']):continue
  key=row['robot'];folder=row['folder'];data['current']=key;atomic_json(path,data)
  try:
   run_recorded([sys.executable,str(ROOT/'scripts/evaluate_contact.py'),row['latest']['checkpoint'],'--seconds','12'],root/'07_logs'/('evaluate-pass02-'+key+'.log'),cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},timeout=600);summary=json.load(open(folder+'/evaluation/summary.json'));data['robots'].append(dict(robot=key,state='evaluated',folder=folder,summary=summary));event('second_contact_gate_evaluated',robot=key,all_seeds_continuous_engagement=summary['all_seeds_continuous_engagement'])
  except Exception as e:data['robots'].append(dict(robot=key,state='failed',error=repr(e)));event('second_contact_gate_failed',robot=key,error=repr(e))
  atomic_json(path,data)
 if plan['state']!='running':break
 time.sleep(5)
data.update(state='complete',finished_utc=now());data.pop('current',None);atomic_json(path,data)
