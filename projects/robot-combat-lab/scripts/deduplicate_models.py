from pathlib import Path
import sys,json,gzip,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from combatlab.storage import encode_model
p=Path(__file__).resolve().parents[1]/'results'
for f in list(p.glob('*/**/scene.mjb.gz')):
 folder=f.parent
 if not (folder/'config.json').exists():continue
 cfg=json.loads((folder/'config.json').read_text());raw=gzip.decompress(f.read_bytes());meta=encode_model(folder,raw,cfg['robots'])
 m=json.loads((folder/'manifest.json').read_text()) if (folder/'manifest.json').exists() else {};m.pop(f.name,None)
 for path in [folder/'model.ref.json',folder/'model.patch.json.gz']:m[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
 (folder/'manifest.json').write_text(json.dumps(m,indent=2));f.unlink();print('EXACT MODEL',folder.name,meta['changed_bytes'],'changed bytes',flush=True)
