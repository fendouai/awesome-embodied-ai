"""One-time external preservation, exact bytes and symlink targets; originals untouched."""
import os,json,shutil,uuid,subprocess,sys
from pathlib import Path
from combatlab.models import ROOT
from combatlab.research_archive import workspace,event,atomic_json,digest,now,run_recorded
root=workspace();dest=root/'01_baseline/project';dest.mkdir(parents=True,exist_ok=True);log=root/'08_manifests/baseline-files.jsonl';count=0;total=0
skip={'.venv','.git','__pycache__','.pytest_cache'}
event('baseline_copy_started',source=str(ROOT),destination=str(dest),excluded_directories=sorted(skip),interpretation='A verified preservation copy of existing artifacts. Historical commands not recorded at the time are not reconstructed as if they were contemporaneous logs.')
with log.open('a') as audit:
 for current,dirs,files in os.walk(ROOT,followlinks=False):
  dirs[:]=[d for d in dirs if d not in skip]
  for name in list(dirs):
   if (Path(current)/name).is_symlink():files.append(name);dirs.remove(name)
  for name in files:
   p=Path(current)/name
   if name=='.DS_Store' or name.endswith('.pyc'):continue
   rel=p.relative_to(ROOT);out=dest/rel;out.parent.mkdir(parents=True,exist_ok=True)
   if p.is_symlink():
    target=os.readlink(p)
    if not out.exists() and not out.is_symlink():out.symlink_to(target)
    assert out.is_symlink() and os.readlink(out)==target;row=dict(path=str(rel),kind='symlink',target=target,recorded_utc=now())
   else:
    for attempt in range(3):
     before=digest(p);temp=out.with_name('.copy-'+out.name)
     if out.exists() and digest(out)==before:break
     shutil.copy2(p,temp);after=digest(temp)
     if before==after==digest(p):os.replace(temp,out);break
     temp.unlink()
    else:raise RuntimeError('Source changed repeatedly during preservation: '+str(p))
    assert digest(out)==digest(p);count+=1;total+=out.stat().st_size;row=dict(path=str(rel),kind='file',bytes=out.stat().st_size,sha256=digest(out),source_mtime_ns=p.stat().st_mtime_ns,recorded_utc=now())
   audit.write(json.dumps(row,ensure_ascii=False)+'\n');audit.flush()
   if count and count%500==0:print('PRESERVED',count,round(total/1024**3,2),'GiB',flush=True)
run_recorded([sys.executable,'-m','pip','freeze'],root/'07_logs/baseline-pip-freeze.txt',cwd=ROOT)
summary=dict(state='complete',source=str(ROOT),destination=str(dest),verified_files=count,logical_bytes=total,inventory_sha256=digest(log),finished_utc=now(),excluded_directories=sorted(skip));atomic_json(root/'08_manifests/baseline-summary.json',summary);event('baseline_copy_completed',**summary);print(json.dumps(summary),flush=True)
