"""Wait for four native finishes, then produce independently archived narrated videos."""
import json,os,subprocess,sys,time
from pathlib import Path
from combatlab.models import ROOT
from combatlab.engine import write_json,sha256
base=ROOT/'results/tournaments/round-04';index=base/'vision-index.json'
data=json.loads(index.read_text()) if index.exists() else dict(id='round-04-vision',state='running',period_seconds=180,official_rules=False,matches=[dict(id=f'vision-{i:02d}',number=i,state='pending') for i in range(1,5)])
data['state']='running';data.pop('error',None);write_json(index,data);env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'2'}
def execute(args,log):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
try:
 for m in data['matches']:
  if m['state']=='complete':continue
  folder=base/'vision-captures'/m['id'];m['folder']=str(folder.relative_to(ROOT));m['state']='waiting_for_native_finish';write_json(index,data)
  deadline=time.monotonic()+14400
  while not (folder/'native-result.json').exists():
   if time.monotonic()>deadline:raise TimeoutError(f'Native match {m["id"]} did not finish within wall budget')
   time.sleep(2)
  with (base/(m['id']+'.log')).open('a') as log:
   if not (folder/'result.json').exists():m['state']='rendering';write_json(index,data);execute([ROOT/'scripts/render_native_soccer_v2.py',folder],log)
   m['state']='commentary';write_json(index,data)
   if not (folder/'commentary/commentary.json').exists():execute([ROOT/'scripts/produce_match_commentary.py',folder,folder/'source.mp4','--kind','vision'],log)
  r=json.loads((folder/'result.json').read_text());analysis=json.loads((folder/'analysis.json').read_text())
  m.update(state='complete',score=[r['red'],r['white']],referee_state=r['state'],quality=analysis,video='/'+str((folder/'commentary/match-commentated.zh-CN.mp4').relative_to(ROOT)),source_video='/'+str((folder/'source.mp4').relative_to(ROOT)),captions='/'+str((folder/'commentary/captions.zh-CN.vtt').relative_to(ROOT)),script='/'+str((folder/'commentary/script.zh-CN.txt').relative_to(ROOT)),result='/'+str((folder/'result.json').relative_to(ROOT)),video_sha256=sha256(folder/'commentary/match-commentated.zh-CN.mp4'));write_json(index,data);print('VISION COMPLETE',m['id'],m['score'],analysis,flush=True)
 data.update(state='complete',finished=time.time());write_json(index,data)
except Exception as e:data.update(state='error',error=str(e));write_json(index,data);raise
