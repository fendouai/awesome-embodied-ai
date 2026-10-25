"""Consolidated independent APFS-COW copies of every full recording; originals retained."""
import json,csv,subprocess
from combatlab.models import ROOT
from combatlab.engine import sha256,write_json
base=ROOT/'results/all-combinations/round-01';d=json.loads((base/'index.json').read_text());assert d['state']=='complete';audit=json.loads((base/'audit.json').read_text());assert audit['state']=='complete';videos=base/'videos';videos.mkdir(exist_ok=True);rows=[]
for j in d['jobs']:rows.append(dict(id=j['id'],kind='comparison',robot=j['robot'],task=j['task'],policies=j['policies'],source_video=j['video'],sha256=j['video_sha256'],source_folder=j['folder']))
combat=json.loads((ROOT/'results/tournaments/round-01/index.json').read_text())
for m in combat['matches']:rows.append(dict(id=m['id'],kind='combat',robot=m['group'],task='combat',policies=m['policies'],source_video=m['video'],sha256=m['video_sha256'],source_folder=m['folder']))
rows.append(dict(id='soccer-3v3',kind='soccer',robot='k1',task='soccer',policies=['bhuman','bhuman'],source_video=d['soccer_video'],sha256=sha256(ROOT/d['soccer_video'].lstrip('/')),source_folder='results/bhuman-native/studio-match-clock'))
for r in rows:
 src=ROOT/r['source_video'].lstrip('/');name=r['id']+'-'+r['robot']+'-'+r['task']+'.mp4';out=videos/name
 if not out.exists():subprocess.run(['cp','-c',str(src),str(out)],check=True)
 assert sha256(out)==r['sha256'];r.update(video='/'+str(out.relative_to(ROOT)),bytes=out.stat().st_size)
write_json(base/'video-archive.json',dict(recordings=len(rows),counts=dict(comparison=63,combat=4,soccer=1),method='Independent APFS copy-on-write copies, SHA256-verified; original videos and complete source archives retained',videos=rows))
with (base/'recordings.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['id','kind','robot','task','policies','video','sha256','source_folder'])
 for r in rows:w.writerow([r[k] if k!='policies' else ' vs '.join(r[k]) for k in ['id','kind','robot','task','policies','video','sha256','source_folder']])
print('ALL VIDEO ARCHIVE READY',len(rows),flush=True)
