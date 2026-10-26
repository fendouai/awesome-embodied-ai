"""Fetch two pinned public checkpoints; no repo code or unsafe pickle execution."""
from pathlib import Path
import urllib.request,json,hashlib,os,shutil
from combatlab.models import ROOT
from combatlab.engine import write_json

P=ROOT/'assets/adversarial-candidates';P.mkdir(exist_ok=True)
if __name__=='__main__':
 jobs=[('RoboStriker-Preview','yinkangning0124/RoboStriker-Preview','logs/rsl_marl/g1_flat_rsl_rl_ppo_stage3/combat/model_3000.pt'),('NCP','Tencent-RoboticsX/NCP','ncp/data/model/ncp_models/boxing_106.pth')];items=[]
 for name,repo,path in jobs:
  tree=json.loads((ROOT/f'research/adversarial-options/{name}-tree.json').read_text());entry=next(x for x in tree['tree'] if x['path']==path);target=P/name/Path(path).name;target.parent.mkdir(exist_ok=True)
  if shutil.disk_usage(ROOT).free-entry['size']<512*1024**2:raise RuntimeError('Insufficient reserve disk; no checkpoint downloaded')
  if not target.exists():
   url=f'https://raw.githubusercontent.com/{repo}/{tree["sha"]}/{path}'
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'BotsTest-Research'}),timeout=90) as response,open(str(target)+'.partial','wb') as out:
    while chunk:=response.read(1024*1024):out.write(chunk)
   os.replace(str(target)+'.partial',target)
  data=target.read_bytes();assert len(data)==entry['size'];git_blob=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest();assert git_blob==entry['sha']
  items.append(dict(repository=repo,commit=tree['sha'],path=path,local=str(target.relative_to(ROOT)),bytes=len(data),git_blob_sha1=git_blob,sha256=hashlib.sha256(data).hexdigest()));print('CHECKPOINT VERIFIED',name,len(data),flush=True)
 write_json(P/'manifest.json',items)
