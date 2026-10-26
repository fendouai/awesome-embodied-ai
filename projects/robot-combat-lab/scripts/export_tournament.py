"""Keep a flat, named video archive using APFS copy-on-write copies, SHA256 verified."""
import json,subprocess,csv,argparse
from combatlab.models import ROOT
from combatlab.engine import sha256,write_json
parser=argparse.ArgumentParser();parser.add_argument('--round',default='01');args=parser.parse_args();folder=ROOT/'results/tournaments'/('round-'+args.round);d=json.loads((folder/'index.json').read_text());assert d['state']=='complete';videos=folder/'videos';videos.mkdir(exist_ok=True);entries=[]
with (folder/'matches.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['match','group','blue_policy','amber_policy','blue_score','amber_score','winner_policy','reason','duration_s','video','video_sha256'])
 for m in d['matches']:
  src=ROOT/m['video'].lstrip('/');target=videos/(m['id']+'-'+m['policies'][0]+'-vs-'+m['policies'][1]+'.mp4')
  if not target.exists():subprocess.run(['cp','-c',str(src),str(target)],check=True)
  digest=sha256(target);assert digest==m['video_sha256'];o=m['outcome'];winner='draw' if o['winner'] is None else m['policies'][o['winner']]
  entry=dict(match=m['id'],video='/'+str(target.relative_to(ROOT)),sha256=digest,source_video=m['video'],bytes=target.stat().st_size)
  entries.append(entry);writer.writerow([m['id'],m['group'],*m['policies'],*o['clean_upper_contacts'],winner,o['reason'],o['time'],entry['video'],digest])
write_json(folder/'video-archive.json',dict(method='APFS copy-on-write independent copies; verified identical SHA256; original replay archives retained',videos=entries));print('VIDEO ARCHIVE READY',len(entries),flush=True)
