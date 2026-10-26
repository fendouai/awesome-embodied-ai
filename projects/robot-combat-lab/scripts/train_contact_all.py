"""First fully archived randomized CPU training pass over all ten bodies."""
import os,sys,json,time
from combatlab.contact_arena import BODIES
from combatlab.models import ROOT
from combatlab.research_archive import workspace,run_recorded,atomic_json,event,now
root=workspace();index=root/'00_index/training-pass01.json';data=json.loads(index.read_text()) if index.exists() else dict(state='running',started_utc=now(),robots=[dict(robot=r,state='pending') for r in BODIES],purpose='First randomized CPU PPO contact training pass; no claim of validated90-match combat completion')
data['state']='running';atomic_json(index,data)
for row in data['robots']:
 if row['state']=='training_pass_complete':continue
 robot=row['robot'];row['state']='training';data['current']=robot;atomic_json(index,data);receipt=root/'00_index'/('training-receipt-'+robot+'.json');args=[sys.executable,str(ROOT/'scripts/train_contact.py'),robot,'--iterations','24','--receipt',str(receipt)]
 if robot=='go2':args+=['--initial-checkpoint',str(root/'03_training/ppo-go2-20261025T123100-e5e788/checkpoints/iteration-00024.pt')]
 try:
  run_recorded(args,root/'07_logs'/('train-all-pass01-'+robot+'.log'),cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},timeout=3600);folder=json.loads(receipt.read_text())['folder'];row.update(state='training_pass_complete',folder=folder,latest=json.loads(open(folder+'/latest.json').read()));event('robot_training_pass_available',robot=robot,folder=folder)
 except Exception as error:row.update(state='failed',error=repr(error));event('robot_training_pass_failed',robot=robot,error=repr(error))
 atomic_json(index,data)
data.update(state='first_training_pass_finished_requires_evaluation',finished_utc=now());data.pop('current',None);atomic_json(index,data)
