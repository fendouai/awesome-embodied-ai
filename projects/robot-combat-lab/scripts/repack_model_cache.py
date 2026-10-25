"""Losslessly repack derived compiled-model bases; preserve all physical/model bytes."""
from pathlib import Path
import gzip,lzma,json,hashlib,os,time,argparse
from combatlab.models import ROOT
from combatlab.storage import load_binary,digest
from combatlab.engine import write_json

if __name__=='__main__':
 args=argparse.ArgumentParser();args.add_argument('--limit',type=int,default=10);args=args.parse_args();groups={};audit=[]
 for ref in (ROOT/'results').rglob('model.ref.json'):
  meta=json.loads(ref.read_text());base=(ref.parent.parents[1]/meta['base']).resolve()
  if base.suffix=='.gz' and base.exists():groups.setdefault(base,[]).append((ref,meta))
 for base,refs in sorted(groups.items(),key=lambda x:-x[0].stat().st_size)[:args.limit]:
  original=base.read_bytes();raw=gzip.decompress(original);rawhash=digest(raw);assert all(m['base_uncompressed_sha256']==rawhash and m['base_compressed_sha256']==digest(original) for _,m in refs)
  packed=lzma.compress(raw,preset=6);assert lzma.decompress(packed)==raw
  if len(packed)>=len(original):continue
  target=base.with_suffix('.xz');tmp=target.with_suffix('.xz.partial');tmp.write_bytes(packed);os.replace(tmp,target);newsha=digest(packed);changed=[]
  for ref,meta in refs:
   before=dict(meta);meta['base']=str(target.relative_to(ref.parent.parents[1]));meta['base_compressed_sha256']=newsha;meta['base_codec']='xz';ref.write_text(json.dumps(meta,indent=2));assert digest(load_binary(ref.parent))==meta['scene_uncompressed_sha256']
   manifest=ref.parent/'manifest.json'
   if manifest.exists():
    hashes=json.loads(manifest.read_text());hashes[ref.name]=digest(ref.read_bytes());write_json(manifest,hashes)
   changed.append(dict(ref=str(ref.relative_to(ROOT)),before=before,after=meta))
  # Only the derived compression representation is retired after full binary verification.
  saved=len(original)-len(packed);audit.append(dict(base_before=str(base.relative_to(ROOT)),base_after=str(target.relative_to(ROOT)),raw_sha256=rawhash,saved_bytes=saved,references=changed));write_json(ROOT/'results/cache-repack-audit.json',audit);base.unlink();print('REPACK VERIFIED',base.name,'references',len(refs),'saved',round(saved/1024**2,1),'MiB',flush=True)
 print('TOTAL LOSSLESS SAVINGS',round(sum(x['saved_bytes'] for x in audit)/1024**2,1),'MiB',flush=True)
