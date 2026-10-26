"""Longer randomized engagement curriculum for failed preliminary gates; retain all runs."""
import os,sys,json
from combatlab.models import ROOT
from combatlab.research_archive import workspace,run_recorded,atomic_json,event,now
root=workspace();training=json.loads((root/'00_index/training-pass01.json').read_text());evaluation=json.loads((root/'00_index/evaluation-pass01.json').read_text());evaluations={r['robot']:r for r in evaluation['robots']};path=root/'00_index/training-pass02.json';data=json.loads(path.read_text()) if path.exists() else dict(state='running',robots=[],started_utc=now(),purpose='Retrain only failed continuous-engagement gates, with12s episodes and denser learned-heading/contact-presence reward; corrected native collision masks')
for row in training['robots']:
 robot=row['robot'];gate=evaluations.get(robot,{}).get('summary',{}).get('all_seeds_continuous_engagement',False)
 if gate and robot!='k1':continue
 if any(r['robot']==robot and r['state']=='training_pass_complete' for r in data['robots']):continue
 receipt=root/'00_index'/('training-pass02-receipt-'+robot+'.json');data['current']=robot;atomic_json(path,data);args=[sys.executable,str(ROOT/'scripts/train_contact.py'),robot,'--iterations','100','--episode-seconds','12','--reward-version','2','--receipt',str(receipt)]
 if robot!='k1':args+=['--initial-checkpoint',row['latest']['checkpoint']]
 try:
  run_recorded(args,root/'07_logs'/('train-pass02-'+robot+'.log'),cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},timeout=3600);folder=json.loads(receipt.read_text())['folder'];data['robots'].append(dict(robot=robot,state='training_pass_complete',folder=folder,latest=json.load(open(folder+'/latest.json'))));event('second_contact_training_pass_available',robot=robot,folder=folder)
 except Exception as e:data['robots'].append(dict(robot=robot,state='failed',error=repr(e)));event('second_contact_training_pass_failed',robot=robot,error=repr(e))
 atomic_json(path,data)
data.update(state='training_pass_finished_requires_evaluation',finished_utc=now());data.pop('current',None);atomic_json(path,data)
