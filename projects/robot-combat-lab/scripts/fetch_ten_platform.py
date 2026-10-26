from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,hashlib,json
ROOT=Path(__file__).resolve().parents[1];jobs=json.loads((ROOT/'research/ten-platform/download-plan.json').read_text())
def fetch(j):
 p=ROOT/'assets/ten-platform'/j['dest'];p.parent.mkdir(parents=True,exist_ok=True);url=f"https://raw.githubusercontent.com/{j['repo']}/{j['commit']}/{j['path']}";data=p.read_bytes() if p.exists() else urllib.request.urlopen(url,timeout=45).read();blob=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest();assert blob==j['sha'],j['path'];p.write_bytes(data);return dict(repository=j['repo'],commit=j['commit'],path=j['path'],local=str(p.relative_to(ROOT)),git_blob_sha1=blob,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),source=url)
with ThreadPoolExecutor(max_workers=5) as pool:manifest=list(pool.map(fetch,jobs))
(ROOT/'assets/ten-platform/manifest.json').write_text(json.dumps(manifest,indent=2));print('FETCHED',len(manifest),sum(x['bytes'] for x in manifest))
