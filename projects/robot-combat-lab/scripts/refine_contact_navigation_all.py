"""Refine only failed models, keeping supervised synthetic data separate from physics."""
import os,sys,json
from pathlib import Path
from combatlab.models import ROOT
from combatlab.research_archive import workspace,run_recorded,atomic_json,event,now
root=workspace();training=json.loads((root/'00_index/training-pass02.json').read_text());evals=json.loads((root/'00_index/evaluation-pass02.json').read_text());failed={r['robot'] for r in evals['robots'] if not r.get('summary',{}).get('all_seeds_continuous_engagement',False)};path=root/'00_index/navigation-refinement.json';data=json.loads(path.read_text()) if path.exists() else dict(state='running',started_utc=now(),robots=[],method='PPO physical-contact policy plus synthetic goal-direction supervision; teacher absent from deployment')
for row in training['robots']:
 robot=row['robot']
 if robot not in failed or any(r['robot']==robot for r in data['robots']):continue
 data['current']=robot;atomic_json(path,data);receipt=root/'00_index'/('navigation-'+robot+'-receipt.json')
 try:
  if not receipt.exists():run_recorded([sys.executable,str(ROOT/'scripts/pretrain_contact_navigation.py'),row['latest']['checkpoint'],'--receipt',str(receipt)],root/'07_logs'/('navigation-'+robot+'.log'),cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},timeout=600)
  checkpoint=json.loads(receipt.read_text())['checkpoint'];folder=Path(checkpoint).parent
  if not (folder/'evaluation/summary.json').exists():run_recorded([sys.executable,str(ROOT/'scripts/evaluate_contact.py'),checkpoint],root/'07_logs'/('evaluate-navigation-'+robot+'.log'),cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},timeout=600)
  summary=json.loads((folder/'evaluation/summary.json').read_text());data['robots'].append(dict(robot=robot,state='evaluated',checkpoint=checkpoint,folder=str(folder),summary=summary));event('navigation_refined_contact_gate',robot=robot,all_seeds_continuous_engagement=summary['all_seeds_continuous_engagement'],checkpoint=checkpoint)
 except Exception as error:data['robots'].append(dict(robot=robot,state='failed',error=repr(error)));event('navigation_refinement_failed',robot=robot,error=repr(error))
 atomic_json(path,data)
data.update(state='complete',finished_utc=now());data.pop('current',None);atomic_json(path,data)
