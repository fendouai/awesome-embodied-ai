"""Complete the compatible combat home/away round; retain every full video and replay."""
import json,subprocess,sys,os,time,shutil
from pathlib import Path
from combatlab.models import ROOT
from combatlab.engine import sha256
OUT=ROOT/'results/tournaments/round-01';OUT.mkdir(parents=True,exist_ok=True)
INDEX=OUT/'index.json';ACTIVE=ROOT/'results/tournament-active.json'
def write(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(path)
if INDEX.exists():data=json.loads(INDEX.read_text())
else:
 data=dict(id='round-01',format='Two separate two-policy home/away round robins; one seed, two sides; no cross-body global ranking',state='running',matches=[
 dict(id='g1-leg1',group='g1',leg=1,policies=['robostriker_ego','robostriker_opponent'],folder='results/duels/duel-1792914768741798000-part0001'),
 dict(id='g1-leg2',group='g1',leg=2,policies=['robostriker_opponent','robostriker_ego'],folder='results/duels/duel-1792916992468118000-part0001'),
 dict(id='ncp-leg1',group='ncp',leg=1,policies=['ncp_boxing_106','ncp_boxing_100'],folder='results/ncp-duels/ncp-cpu-1792914871104170000'),
 dict(id='ncp-leg2',group='ncp',leg=2,policies=['ncp_boxing_100','ncp_boxing_106'])],scope=dict(eligible=['G1 / RoboStriker ego vs opponent','NCP / boxing106 vs boxing100'],excluded=[dict(group='Other robot locomotion/balance/recovery deployments',reason='No compatible trained attack policy or common adversarial observation/action contract'),dict(group='B-Human soccer',reason='Only one installed team controller; separate soccer rules, not a second distinct opponent controller'),dict(group='SMPLOlympics multi-sport',reason='Official reference videos are not local competitive policy deployments')]),seed=0,policy_seed_identity_preserved=True,first_legs='Existing verified90s matches retained verbatim',created=time.time())
write(ACTIVE,dict(state='running',tournament=data['id']));env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'MUJOCO_GL':'cgl'}
def execute(args,log):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800,check=True)
try:
 for match in data['matches']:
  if match.get('state')=='complete':continue
  if 'folder' not in match:
   if shutil.disk_usage(ROOT).free<512*1024**2:raise RuntimeError('Storage reserve below512MiB; preserved previous matches')
   data['stage']='simulation';data['current']=match['id'];write(INDEX,data)
   with (OUT/(match['id']+'.log')).open('a') as log:
    execute([ROOT/'scripts/run_ncp_duel.py','--swap'],log)
   match['folder']=json.loads((ROOT/'results/ncp-duels/latest.json').read_text())['folder'];write(INDEX,data)
  folder=ROOT/match['folder'];result=json.loads((folder/'result.json').read_text());assert result['policies']==match['policies']
  with (OUT/(match['id']+'.log')).open('a') as log:
   if not (folder/'duel-verification.json').exists():
    data['stage']='verification';data['current']=match['id'];write(INDEX,data);execute([ROOT/'scripts/verify_duel.py',folder],log)
   if not (folder/'duel-video.json').exists():
    data['stage']='rendering';data['current']=match['id'];write(INDEX,data);execute([ROOT/'scripts/render_duel.py',folder],log)
  verdict=json.loads((folder/'duel-result.json').read_text());verification=json.loads((folder/'duel-verification.json').read_text());video=json.loads((folder/'duel-video.json').read_text())
  assert verdict['complete'] and verification['verified'];assert verification['trajectory_sha256']==sha256(folder/'trajectory.npz');assert video['video_sha256']==sha256(folder/video['video'])
  match.update(state='complete',outcome=verdict['outcome'],verification=verification,video='/'+match['folder']+'/'+video['video'],poster='/'+match['folder']+'/duel-poster.jpg',video_sha256=video['video_sha256']);write(INDEX,data)
  print('MATCH COMPLETE',match['id'],match['outcome']['clean_upper_contacts'],flush=True)
 data.update(state='complete',stage='published',finished=time.time());data.pop('current',None);write(INDEX,data)
 with (OUT/'publish.log').open('a') as log:execute([ROOT/'scripts/publish_duels.py'],log)
except Exception as error:
 data.update(state='error',error=str(error));write(INDEX,data);raise
finally:write(ACTIVE,dict(state=data['state'],tournament=data['id']))
