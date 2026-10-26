"""Fetch a pinned, documented subset of official Unitree assets and actual policy weights."""
import concurrent.futures, hashlib, json, urllib.request, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPOS={'gym':'unitreerobotics/unitree_rl_gym','lab':'unitreerobotics/unitree_rl_lab'}
def get(u):
 req=urllib.request.Request(u,headers={'User-Agent':'BotsTest-RobotCombatLab/0.1'})
 for attempt in range(3):
  try:return urllib.request.urlopen(req,timeout=30).read()
  except Exception:
   if attempt==2:raise
   time.sleep(attempt+1)
def main():
 manifest=[];jobs=[]
 for key,repo in REPOS.items():
  info=json.loads(get('https://api.github.com/repos/'+repo+'/commits/main'));sha=info['sha']
  tree=json.loads(get(f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1'))['tree']
  for f in tree:
   path=f['path']
   if f['type']!='blob':continue
   if key=='gym':
    pick=(path.startswith(('resources/robots/g1_description/','resources/robots/h1/','resources/robots/h1_2/')) and (path.endswith(('.xml','.STL','.stl','.obj','.png','.jpg')))) or path in ['LICENSE','deploy/pre_train/g1/motion.pt','deploy/pre_train/h1/motion.pt','deploy/pre_train/h1_2/motion.pt','deploy/deploy_mujoco/configs/g1.yaml','deploy/deploy_mujoco/configs/h1.yaml','deploy/deploy_mujoco/configs/h1_2.yaml','deploy/deploy_mujoco/deploy_mujoco.py']
   else:
    pick=path.startswith('deploy/robots/g1_29dof/config/policy/velocity/v0/') or path in ['LICENSE','deploy/include/isaaclab/envs/mdp/observations/observations.h','deploy/include/isaaclab/manager/observation_manager.h','deploy/include/FSM/State_RLBase.h']
   if pick:jobs.append((key,repo,sha,path,f.get('size',0),f['sha']))
 def fetch(j):
  key,repo,sha,path,size,blob=j;target=ROOT/'assets'/'vendor'/key/path;target.parent.mkdir(parents=True,exist_ok=True)
  u=f'https://raw.githubusercontent.com/{repo}/{sha}/{path}'
  b=target.read_bytes() if target.exists() else b''
  if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=blob:b=get(u)
  assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==blob, path
  target.write_bytes(b)
  return dict(repository=repo,commit=sha,path=path,local=str(target.relative_to(ROOT)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),source=u)
 print('Fetching',len(jobs),'files',round(sum(j[-2] for j in jobs)/1e6,1),'MB',flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for i,m in enumerate(pool.map(fetch,jobs)):
   manifest.append(m)
   if i%25==0:print(i+1,'/',len(jobs),flush=True)
 (ROOT/'assets'/'provenance.json').write_text(json.dumps(manifest,indent=2))
 print('Fetched',len(manifest),'files; SHA-256 manifest saved.',flush=True)
if __name__=='__main__':main()
