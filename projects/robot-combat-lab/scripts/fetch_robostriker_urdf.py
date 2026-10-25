"""Stream the pinned asset archive; retain only small URDF/text assets, no unsafe extraction."""
from pathlib import Path
import urllib.request,tarfile,json,hashlib
from combatlab.models import ROOT
if __name__=='__main__':
 repo='yinkangning0124/RoboStriker-Preview';t=json.loads((ROOT/'research/adversarial-options/RoboStriker-Preview-tree.json').read_text());path='source/whole_body_tracking/unitree_description.tar.gz';url=f'https://raw.githubusercontent.com/{repo}/{t["sha"]}/{path}';out=ROOT/'assets/adversarial-candidates/RoboStriker-Preview/description';out.mkdir(exist_ok=True);manifest=[]
 with urllib.request.urlopen(url,timeout=90) as response:
  with tarfile.open(fileobj=response,mode='r|gz') as archive:
   for member in archive:
    if member.isfile() and member.size<500000 and member.name.endswith(('.urdf','.xml','.yaml')):
     data=archive.extractfile(member).read();target=out/Path(member.name).name
     if target.exists():target=out/(Path(member.name).parent.name+'-'+target.name)
     target.write_bytes(data);manifest.append(dict(member=member.name,local=str(target.relative_to(ROOT)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
 (out/'manifest.json').write_text(json.dumps(dict(repository=repo,commit=t['sha'],archive=path,files=manifest),indent=2));print('Streamed asset archive; retained',len(manifest),'small files',flush=True)
