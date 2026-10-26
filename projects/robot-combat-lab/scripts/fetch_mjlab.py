from pathlib import Path
import urllib.request,json,hashlib
P=Path(__file__).resolve().parents[1];repo='unitreerobotics/unitree_rl_mjlab'
def get(u):return urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'BotsTest-Research/0.1'}),timeout=45).read()
sha=json.loads(get(f'https://api.github.com/repos/{repo}/commits/main'))['sha']
paths=['deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx','deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml','src/assets/robots/unitree_g1/xmls/g1.xml','LICENSE']
m=json.loads((P/'assets/provenance.json').read_text())
for path in paths:
 try:
  u=f'https://raw.githubusercontent.com/{repo}/{sha}/{path}';b=get(u);f=P/'assets/vendor/mjlab'/path;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b);m.append(dict(repository=repo,commit=sha,path=path,local=str(f.relative_to(P)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),source=u));print(path,len(b),flush=True)
 except Exception as e:
  if path=='LICENSE':print('License fetch:',e,flush=True)
  else:raise
(P/'assets/provenance.json').write_text(json.dumps(m,indent=2))
