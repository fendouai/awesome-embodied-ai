from pathlib import Path
import gzip,hashlib,json,shutil
p=Path(__file__).resolve().parents[1]/'results'
for f in p.glob('*/**/scene.mjb'):
 out=f.with_suffix('.mjb.gz');digest=hashlib.sha256(f.read_bytes()).hexdigest()
 with f.open('rb') as src,gzip.open(out,'wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst)
 assert hashlib.sha256(gzip.decompress(out.read_bytes())).hexdigest()==digest
 manifest=f.parent/'manifest.json'
 if manifest.exists():
  m=json.loads(manifest.read_text());m.pop(f.name,None);m[out.name]=hashlib.sha256(out.read_bytes()).hexdigest();manifest.write_text(json.dumps(m,indent=2))
 f.unlink();print('LOSSLESS',f.parent.name,flush=True)
