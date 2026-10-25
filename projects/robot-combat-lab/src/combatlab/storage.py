"""Lossless full-model storage using a shared compressed base and exact byte patches.

A scene is reconstructed byte-for-byte and checked against SHA-256 before loading.
No simulation frames or mesh precision are discarded.
"""
from pathlib import Path
import gzip,lzma,hashlib,json,base64,os,tempfile
import numpy as np,mujoco

def decompress_base(b):return lzma.decompress(b) if b.startswith(bytes.fromhex("fd377a585a00")) else gzip.decompress(b)

def digest(b):return hashlib.sha256(b).hexdigest()
def encode_model(folder,binary,group):
    folder=Path(folder);cache=folder.parents[1]/'model-cache';cache.mkdir(parents=True,exist_ok=True)
    key='-'.join(group)+'-'+str(len(binary));base=cache/(key+'.mjb.gz')
    candidates=sorted(cache.glob(key+'*.mjb.xz'))
    if candidates:base=candidates[0]
    if not base.exists():
        fd,temp=tempfile.mkstemp(prefix='.model-base-',dir=cache)
        try:
            with os.fdopen(fd,'wb') as stream:stream.write(gzip.compress(binary,compresslevel=3))
            try:os.link(temp,base)
            except FileExistsError:pass
        finally:os.unlink(temp)
    original=decompress_base(base.read_bytes());assert len(original)==len(binary)
    diffs=np.flatnonzero(np.frombuffer(original,dtype=np.uint8)!=np.frombuffer(binary,dtype=np.uint8))
    runs=[]
    if len(diffs):
        for run in np.split(diffs,np.flatnonzero(np.diff(diffs)>1)+1):
            lo,hi=int(run[0]),int(run[-1])+1;runs.append([lo,base64.b64encode(binary[lo:hi]).decode()])
    patch=dict(size=len(binary),runs=runs);out=folder/'model.patch.json.gz';out.write_bytes(gzip.compress(json.dumps(patch,separators=(',',':')).encode(),compresslevel=3))
    meta=dict(format='exact-byte-patch-v1',base=str(base.relative_to(folder.parents[1])),base_compressed_sha256=digest(base.read_bytes()),base_uncompressed_sha256=digest(original),scene_uncompressed_sha256=digest(binary),patch_sha256=digest(out.read_bytes()),uncompressed_bytes=len(binary),changed_bytes=int(len(diffs)),base_root=str(folder.parents[1].resolve()))
    # base_root is a locator hint only; loader discovers the relocatable results root.
    (folder/'model.ref.json').write_text(json.dumps(meta,indent=2));assert load_binary(folder)==binary
    return meta

def load_binary(folder):
    folder=Path(folder)
    if (folder/'model.ref.json').exists():
        meta=json.loads((folder/'model.ref.json').read_text());base=folder.parents[1]/meta['base'];compressed=base.read_bytes();assert digest(compressed)==meta['base_compressed_sha256']
        raw=bytearray(decompress_base(compressed));assert digest(raw)==meta['base_uncompressed_sha256']
        patchbytes=(folder/'model.patch.json.gz').read_bytes();assert digest(patchbytes)==meta['patch_sha256'];patch=json.loads(gzip.decompress(patchbytes));assert len(raw)==patch['size']
        for offset,data in patch['runs']:
            value=base64.b64decode(data);raw[offset:offset+len(value)]=value
        raw=bytes(raw);assert digest(raw)==meta['scene_uncompressed_sha256'];return raw
    if (folder/'scene.mjb.gz').exists():return gzip.decompress((folder/'scene.mjb.gz').read_bytes())
    return (folder/'scene.mjb').read_bytes()

def load_model(folder):return mujoco.MjModel.from_binary_path('scene.mjb',{'scene.mjb':load_binary(folder)})
