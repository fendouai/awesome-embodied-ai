"""Eight reproducible 180-second matches with two isolated production lanes."""
import json,os,subprocess,sys,time,threading
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from combatlab.models import ROOT
from combatlab.engine import write_json,sha256
base=ROOT/'results/tournaments/round-04';base.mkdir(parents=True,exist_ok=True);index=base/'simulation-index.json';lock=threading.RLock()
if index.exists():data=json.loads(index.read_text())
else:
 matches=[]
 for group,choices in [('g1',['robostriker_ego','robostriker_opponent']),('ncp',['106','100'])]:
  for a in choices:
   for b in choices:matches.append(dict(id=f'{group}-{a}-{b}',group=group,left=a,right=b,state='pending'))
 data=dict(id='round-04-simulation',protocol='competition-v2',state='running',limit_seconds=180,official_rules=False,matches=matches)
data['state']='running';data.pop('error',None)
def publish():
 with lock:
  for m in data['matches']:
   if m['state']=='complete' and m.get('folder'):
    meta=ROOT/m['folder']/'commentary/commentary.json'
    if meta.exists():m['video_sha256']=json.loads(meta.read_text())['video_sha256']
  write_json(index,data)
def update(match,**fields):
 with lock:match.update(fields);publish()
publish();env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'MUJOCO_GL':'cgl','OMP_NUM_THREADS':'2','COMBATLAB_RENDER_LOD':'1'}
write_json(ROOT/'results/tournament-active.json',dict(state='running',tournament='round-04'))
def execute(args,log):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
def run_match(match):
 if match['state']=='complete':return
 with (base/(match['id']+'.log')).open('a') as log:
  if 'folder' not in match:
   update(match,state='simulation');pointer=base/(match['id']+'-pointer.json')
   execute([ROOT/'scripts'/('run_duel.py' if match['group']=='g1' else 'run_ncp_duel.py'),'--limit','180','--seed','0','--left',match['left'],'--right',match['right'],'--output-meta',pointer],log)
   update(match,folder=json.loads(pointer.read_text())['folder'])
  folder=ROOT/match['folder']
  for state,script,target in [('verification','verify_duel.py','duel-verification.json'),('rendering','render_duel.py','duel-video.json')]:
   if not (folder/target).exists():update(match,state=state);execute([ROOT/'scripts'/script,folder],log)
  source=folder/'duel-full.mp4';update(match,state='commentary')
  if not (folder/'commentary/commentary.json').exists():execute([ROOT/'scripts/produce_match_commentary.py',folder,source],log)
 outcome=json.loads((folder/'duel-result.json').read_text())['outcome'];proof=json.loads((folder/'duel-verification.json').read_text());assert proof['verified']
 update(match,state='complete',outcome=outcome,video='/'+str((folder/'commentary/match-commentated.zh-CN.mp4').relative_to(ROOT)),source_video='/'+str(source.relative_to(ROOT)),captions='/'+str((folder/'commentary/captions.zh-CN.vtt').relative_to(ROOT)),script='/'+str((folder/'commentary/script.zh-CN.txt').relative_to(ROOT)),result='/'+str((folder/'duel-result.json').relative_to(ROOT)),video_sha256=sha256(folder/'commentary/match-commentated.zh-CN.mp4'))
 print('COMPLETE',match['id'],outcome['clean_upper_contacts'],flush=True)
try:
 errors=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  jobs={pool.submit(run_match,m):m for m in data['matches'] if m['state']!='complete'}
  for job in as_completed(jobs):
   try:job.result()
   except Exception as e:update(jobs[job],state='error',error=str(e));errors.append(str(e))
 if errors:raise RuntimeError('; '.join(errors))
 data.update(state='complete',finished=time.time());data.pop('current',None);publish()
except Exception as e:data.update(state='error',error=str(e));publish();raise
finally:write_json(ROOT/'results/tournament-active.json',dict(state=data['state'],tournament='round-04'))
