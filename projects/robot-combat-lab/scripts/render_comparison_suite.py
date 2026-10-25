from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,subprocess,sys,os
from combatlab.models import ROOT
from combatlab.engine import write_json

def job(row):
 folder=ROOT/row['folder']
 if (folder/'comparison.mp4').exists():return row['folder']
 with (folder/'render.log').open('w') as log:
  subprocess.run([sys.executable,str(ROOT/'scripts/render_comparison.py'),str(folder)],check=True,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'MUJOCO_GL':'cgl'})
 return row['folder']
if __name__=='__main__':
 suite=json.loads((ROOT/'results/comparisons/suite.json').read_text());rows=suite['rows']
 assert suite['state']=='complete'
 with ThreadPoolExecutor(max_workers=2) as pool:
  for i,f in enumerate(as_completed([pool.submit(job,row) for row in rows])):print('VIDEO',i+1,'/',len(rows),f.result(),flush=True)
 write_json(ROOT/'results/comparisons/video-status.json',dict(complete=True,videos=len(rows)))
