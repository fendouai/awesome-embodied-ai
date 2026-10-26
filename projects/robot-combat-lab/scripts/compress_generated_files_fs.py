"""macOS transparent compression of task-generated artifacts, exact logical bytes retained."""
from pathlib import Path
import os,subprocess,json,hashlib
from combatlab.models import ROOT
paths=[p for p in (ROOT/'results').rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in ['.json','.wav','.csv']]
build=ROOT/'integrations/bhuman/build-mac'
if build.exists():paths += list(build.rglob('*.o'))
audit=[];saved=0
for p in paths:
 st=p.stat()
 if st.st_size<65536 or st.st_flags&32:continue
 original=hashlib.sha256(p.read_bytes()).hexdigest();temp=p.with_name('.fs-compress-'+p.name)
 subprocess.run(['ditto','--hfsCompression','--noclone',str(p),str(temp)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 assert hashlib.sha256(temp.read_bytes()).hexdigest()==original
 after=temp.stat()
 if after.st_blocks<st.st_blocks:
  os.replace(temp,p);gain=(st.st_blocks-after.st_blocks)*512;saved+=gain;audit.append(dict(file=str(p.relative_to(ROOT)),sha256=original,logical_bytes=st.st_size,physical_before=st.st_blocks*512,physical_after=after.st_blocks*512))
 else:temp.unlink()
(ROOT/'results/filesystem-compression-audit.json').write_text(json.dumps(dict(files=len(audit),saved_bytes=saved,method='ditto --hfsCompression; exact SHA256 logical-content verification; original paths/formats unchanged',entries=audit),indent=2));print('EXACT-BYTE TRANSPARENT COMPRESSION',len(audit),'files, saved',round(saved/1024**2,1),'MiB',flush=True)
