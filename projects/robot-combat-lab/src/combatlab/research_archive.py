"""Append-only research provenance and atomic, checksummed experiment artifacts."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,uuid,hashlib,subprocess,sys,platform,fcntl
from .models import ROOT
CONFIG=ROOT/'external-workspace.json'
def now():return datetime.now(timezone.utc).isoformat()
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
def workspace():
 p=Path(json.loads(CONFIG.read_text())['root'])
 if not p.is_dir() or not (p/'archive-identity.json').exists():raise RuntimeError('Research disk missing; refuse fallback to internal disk')
 return p

def event(kind,**fields):
 root=workspace();row=dict(event_id=uuid.uuid4().hex,utc=now(),kind=kind,**fields)
 with (root/'07_logs/events.jsonl').open('a') as f:
  fcntl.flock(f.fileno(),fcntl.LOCK_EX);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno());fcntl.flock(f.fileno(),fcntl.LOCK_UN)
 return row

def atomic_json(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
 with temp.open('w') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)

def inventory(folder,output=None):
 folder=Path(folder);rows=[]
 for p in sorted(folder.rglob('*')):
  if p.is_file() and not p.is_symlink() and p!=output:rows.append(dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,sha256=digest(p)))
 if output:atomic_json(output,dict(created_utc=now(),files=rows))
 return rows

def run_recorded(args,log,**kwargs):
 log=Path(log);log.parent.mkdir(parents=True,exist_ok=True);event('command_started',argv=list(map(str,args)),cwd=str(kwargs.get('cwd',ROOT)),log=str(log));start=now()
 try:
  with log.open('ab') as f:done=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,**kwargs)
  event('command_finished',argv=list(map(str,args)),started_utc=start,returncode=done.returncode,log=str(log),log_sha256=digest(log));done.check_returncode();return done
 except BaseException as error:event('command_failed',argv=list(map(str,args)),error=repr(error),log=str(log));raise

def snapshot_sources(run_folder):
 run_folder=Path(run_folder);out=run_folder/'sources';out.mkdir(parents=True,exist_ok=True)
 import shutil
 for top in ['src','scripts','tests']:
  shutil.copytree(ROOT/top,out/top,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 inventory(out,run_folder/'source-manifest.json');atomic_json(run_folder/'runtime.json',dict(python=sys.version,platform=platform.platform(),executable=sys.executable,recorded_utc=now()))
 event('source_snapshot',path=str(out),manifest_sha256=digest(run_folder/'source-manifest.json'))
