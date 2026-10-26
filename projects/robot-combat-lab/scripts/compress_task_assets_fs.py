"""Exact-byte transparent compression of local simulation assets; no deletions."""
from pathlib import Path
import os,hashlib,subprocess,json
from combatlab.models import ROOT
saved=0;entries=[]
for base in ['assets','integrations']:
 for p in (ROOT/base).rglob('*'):
  if not p.is_file() or p.is_symlink() or '.git' in p.parts:continue
  st=p.stat()
  if st.st_size<65536 or st.st_flags&32 or p.suffix in ['.xz','.zip','.gz','.mp4','.png','.jpg','.npz']:continue
  digest=hashlib.sha256(p.read_bytes()).hexdigest();temp=p.with_name('.compact-'+p.name)
  subprocess.run(['ditto','--hfsCompression','--noclone',str(p),str(temp)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  assert hashlib.sha256(temp.read_bytes()).hexdigest()==digest
  gain=(st.st_blocks-temp.stat().st_blocks)*512
  if gain>0:os.replace(temp,p);saved+=gain;entries.append(dict(file=str(p.relative_to(ROOT)),sha256=digest,saved_bytes=gain))
  else:temp.unlink()
(ROOT/'results/asset-compression-audit.json').write_text(json.dumps(dict(saved_bytes=saved,files=entries),indent=2));print('Saved MiB',saved/1024**2,flush=True)
