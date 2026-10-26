"""Transactional per-reference lossless cache recompression, preserving every scene SHA."""
from pathlib import Path
import json,lzma,os,gc
from combatlab.models import ROOT
from combatlab.storage import load_binary,decompress_base,digest
from combatlab.engine import write_json

def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,path)

if __name__=='__main__':
 groups={};audit_path=ROOT/'results/cache-repack-xz9-audit.json';audit=json.loads(audit_path.read_text()) if audit_path.exists() else []
 for ref in (ROOT/'results').rglob('model.ref.json'):
  m=json.loads(ref.read_text());base=(ref.parent.parents[1]/m['base']).resolve()
  if m.get('base_preset')==9:continue
  groups.setdefault((base.parent,base.name.split('.')[0],m['base_uncompressed_sha256']),[]).append((ref,m,base))
 for (cache,key,rawsha),refs in sorted(groups.items(),key=lambda x:-max(r[2].stat().st_size for r in x[1])):
  first=refs[0][2];compressed=first.read_bytes();raw=decompress_base(compressed);assert digest(raw)==rawsha
  for ref,m,base in refs:assert digest(base.read_bytes())==m['base_compressed_sha256']
  packed=lzma.compress(raw,preset=9);assert digest(lzma.decompress(packed))==rawsha;packedsha=digest(packed);target=cache/(key+'.xz9-'+packedsha[:12]+'.mjb.xz');tmp=target.with_suffix('.xz.pending')
  if not target.exists():
   with open(tmp,'wb') as stream:
    for start in range(0,len(packed),1024*1024):stream.write(packed[start:start+1024*1024])
    stream.flush();os.fsync(stream.fileno())
   os.replace(tmp,target)
  old_bases=set();metadata=[]
  for number,(ref,m,base) in enumerate(refs):
   before=dict(m);m.update(base=str(target.relative_to(ref.parent.parents[1])),base_compressed_sha256=packedsha,base_codec='xz',base_preset=9);atomic_json(ref,m)
   assert digest(load_binary(ref.parent))==m['scene_uncompressed_sha256']
   manifest=ref.parent/'manifest.json'
   if manifest.exists():hashes=json.loads(manifest.read_text());hashes[ref.name]=digest(ref.read_bytes());atomic_json(manifest,hashes)
   metadata.append(dict(ref=str(ref.relative_to(ROOT)),before=before,after=m));old_bases.add(base)
  saved=sum(b.stat().st_size for b in old_bases if b!=target)-len(packed)
  audit.append(dict(new_base=str(target.relative_to(ROOT)),raw_sha256=rawsha,saved_bytes=saved,references=metadata));atomic_json(audit_path,audit)
  live_bases={(p.parent.parents[1]/json.loads(p.read_text())['base']).resolve() for p in (ROOT/'results').rglob('model.ref.json')}
  for base in old_bases:
   if base!=target and base not in live_bases:base.unlink()
  print('XZ9 FULL-BINARY VERIFIED',key,len(refs),'refs saved',round(saved/1024**2,1),'MiB',flush=True);del raw,packed,compressed;gc.collect()
 print('COMPLETE total_saved_MiB',sum(a['saved_bytes'] for a in audit)/1024**2,flush=True)
