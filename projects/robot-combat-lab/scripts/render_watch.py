import os,sys,time,json
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','cgl');ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from combatlab.render import render_video
expected=26;finished=set()
while len(finished)<expected:
 folders=sorted((ROOT/'results/matches').glob('*')) if (ROOT/'results/matches').exists() else []
 ready=[f for f in folders if (f/'result.json').exists() and (f/'manifest.json').exists()]
 for f in ready:
  if f.name in finished:continue
  if not (f/'battle.mp4').exists():
   start=time.time();print('RENDER',len(finished)+1,'/',expected,f.name,flush=True);render_video(f);print('VIDEO COMPLETE',f.name,round(time.time()-start,1),'s',flush=True)
  finished.add(f.name)
 if len(finished)<expected:time.sleep(2)
print('All full videos complete',flush=True)
