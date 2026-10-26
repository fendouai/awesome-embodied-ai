"""All pairings among the installed genuine combat policies; resume safely."""
import json,subprocess,sys,os,time,shutil,copy
from pathlib import Path
from combatlab.models import ROOT
from combatlab.engine import sha256
OLD=ROOT/'results/tournaments/round-01/index.json';from combatlab.catalog import ROBOTS,CONTROLS
OUT=ROOT/'results/tournaments/round-02';OUT.mkdir(parents=True,exist_ok=True);INDEX=OUT/'index.json';ACTIVE=ROOT/'results/tournament-active.json'
def write(p,d):t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
if INDEX.exists():data=json.loads(INDEX.read_text())
else:
 old=json.loads(OLD.read_text());matches=copy.deepcopy(old['matches']);
 for group,pairings in [('g1',[('robostriker_ego','robostriker_ego'),('robostriker_opponent','robostriker_opponent')]),('ncp',[('ncp_boxing_106','ncp_boxing_106'),('ncp_boxing_100','ncp_boxing_100')])]:
  for left,right in pairings:matches.append(dict(id=group+'-'+left.rsplit('_',1)[-1]+'-vs-'+right.rsplit('_',1)[-1],group=group,leg=1,policies=[left,right]))
 data=dict(id='round-02',format='Complete unordered policy-pair matrix within each combat-capable backend; cross-policy pairs have both seat assignments, self-pairs run once',state='running',matches=matches,seed=0,coverage=dict(G1=['ego-ego','ego-opponent','opponent-ego','opponent-opponent'],NCP=['106-106','106-100','100-106','100-100'],scope_note='Within the two installed trained attack-policy pairs; locomotion policies are excluded from combat compatibility matrix because they have no trained attack/controller contract.'),created=time.time());write(INDEX,data)
data['coverage']['robot_model_matrix']=[dict(robot=r['name'],robot_id=r['id'],compatible_deployments=[c['id'] for c in CONTROLS if r['id'] in c['robots']],adversarial_deployments=[c['id'] for c in CONTROLS if r['id'] in c['robots'] and 'combat' in c['tasks']],status='included' if 'combat' in r['tasks'] else 'excluded',reason='Installed trained adversarial policy available' if 'combat' in r['tasks'] else 'Available packages are locomotion, balance, motion tracking or soccer; no compatible opponent-targeting combat policy') for r in ROBOTS]
write(INDEX,data)
write(ACTIVE,dict(state='running',tournament=data['id']));env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'MUJOCO_GL':'cgl'}
def execute(args,log):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800,check=True)
try:
 for m in data['matches']:
  if m.get('state')=='complete':continue
  if 'folder' not in m:
   if shutil.disk_usage(ROOT).free<512*1024**2:raise RuntimeError('Disk reserve under512MiB; existing recordings preserved')
   data.update(stage='simulation',current=m['id']);write(INDEX,data)
   script='run_duel.py' if m['group']=='g1' else 'run_ncp_duel.py';args=[ROOT/'scripts'/script,'--limit','90','--left',m['policies'][0].rsplit('_',1)[-1] if m['group']=='ncp' else m['policies'][0],'--right',m['policies'][1].rsplit('_',1)[-1] if m['group']=='ncp' else m['policies'][1]]
   with (OUT/(m['id']+'.log')).open('a') as log:execute(args,log)
   base=ROOT/('results/duels' if m['group']=='g1' else 'results/ncp-duels');folder=base/'latest.json';m['folder']=json.loads(folder.read_text())['folder'];write(INDEX,data)
  f=ROOT/m['folder'];out=json.loads((f/'duel-result.json').read_text())['outcome'];assert json.loads((f/'result.json').read_text())['policies']==m['policies']
  with (OUT/(m['id']+'.log')).open('a') as log:
   if not (f/'duel-verification.json').exists():data.update(stage='verification',current=m['id']);write(INDEX,data);execute([ROOT/'scripts/verify_duel.py',f],log)
   if not (f/'duel-video.json').exists():data.update(stage='rendering',current=m['id']);write(INDEX,data);execute([ROOT/'scripts/render_duel.py',f],log)
  v=json.loads((f/'duel-video.json').read_text());ver=json.loads((f/'duel-verification.json').read_text());assert ver['verified'] and ver['winner']==out['winner'] and ver['score']==out['clean_upper_contacts'] and v['video_sha256']==sha256(f/v['video'])
  m.update(state='complete',outcome=out,verification=ver,video='/'+m['folder']+'/'+v['video'],poster='/'+m['folder']+'/duel-poster.jpg',video_sha256=v['video_sha256']);write(INDEX,data);print('MATCH',m['id'],out['kind'],out['clean_upper_contacts'],flush=True)
 data.update(state='complete',stage='published',finished=time.time());data.pop('current',None);write(INDEX,data)
 with (OUT/'publish.log').open('a') as log:execute([ROOT/'scripts/export_tournament.py','--round','02'],log)
except Exception as error:data.update(state='error',error=str(error));write(INDEX,data);raise
finally:write(ACTIVE,dict(state=data['state'],tournament=data['id']))
