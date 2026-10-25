from pathlib import Path
import urllib.request,json,hashlib,xml.etree.ElementTree as ET,concurrent.futures,time
P=Path(__file__).resolve().parents[1];repo='unitreerobotics/unitree_mujoco'
def get(u):
 for attempt in range(3):
  try:return urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'BotsTest-Research/0.1'}),timeout=30).read()
  except Exception:
   if attempt==2:raise
   time.sleep(attempt+1)
sha=json.loads(get(f'https://api.github.com/repos/{repo}/commits/main'))['sha'];paths=[]
for x in ['unitree_robots/h1/h1.xml','unitree_robots/h1_2/h1_2_handless.xml']:
 b=get(f'https://raw.githubusercontent.com/{repo}/{sha}/{x}');doc=ET.fromstring(b);meshdir=doc.find('compiler').get('meshdir','');paths.append(x)
 for mesh in doc.findall('asset/mesh'):
  if mesh.get('file'):paths.append(str(Path(x).parent/meshdir/mesh.get('file')))
paths=list(dict.fromkeys(paths));manifest=json.loads((P/'assets/provenance.json').read_text())
def fetch(path):
 u=f'https://raw.githubusercontent.com/{repo}/{sha}/{path}';f=P/'assets/vendor/mujoco'/path;f.parent.mkdir(parents=True,exist_ok=True);b=get(u);f.write_bytes(b);return dict(repository=repo,commit=sha,path=path,local=str(f.relative_to(P)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),source=u)
print('Fetch',len(paths),'official full-body assets',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
 for r in ex.map(fetch,paths):manifest.append(r)
(P/'assets/provenance.json').write_text(json.dumps(manifest,indent=2));print('Full bodies ready',flush=True)
