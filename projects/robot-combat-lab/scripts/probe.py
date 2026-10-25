import os
os.environ.setdefault('MUJOCO_GL','cgl')
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import mujoco,numpy as np
from combatlab.models import scene_xml,make_actors,initialize,configure_servos,ROOT
from combatlab.policies import POLICIES,torques,gravity_body

def run(keys,policy_keys,seconds=6.,distance=4.,cmd=(0,0,0)):
 st=time.time();xml=scene_xml(keys);m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);actors=make_actors(m,keys);initialize(m,d,actors,distance=distance);pol=[POLICIES[k](a) for k,a in zip(policy_keys,actors)];configure_servos(m,actors);init=[a.nominal_z for a in actors];mins=init.copy()
 for step in range(round(seconds/m.opt.timestep)):
  for i,(a,policy) in enumerate(zip(actors,pol)):
   if step%10==0:policy.update(m,d,cmd,d.time)
   d.ctrl[a.actuator_ids]=policy.target+(policy.feedforward/a.kp if hasattr(policy,'feedforward') else 0)
  mujoco.mj_step(m,d)
  for i,a in enumerate(actors):mins[i]=min(mins[i],float(d.qpos[a.free_qadr+2]))
 ends=[]
 for a in actors:
  tilt=float(np.degrees(np.arccos(np.clip(-gravity_body(d.qpos[a.free_qadr+3:a.free_qadr+7])[2],-1,1))))
  ends.append(dict(z=float(d.qpos[a.free_qadr+2]),x=float(d.qpos[a.free_qadr]),tilt=tilt))
 print(json.dumps(dict(keys=keys,policies=policy_keys,init_z=init,min_z=mins,end=ends,mass=[float(m.body_mass[a.body_ids].sum()) for a in actors],seconds=time.time()-st)),flush=True)
 return m,d,actors
if __name__=='__main__':
 for keys,pol in [(('g1','h1'),('pd_stand','pd_stand')),(('h1_2','g1'),('pd_stand','pd_stand')),(('g1_full','g1_full'),('unitree_rl_gym','unitree_rl_lab'))]:
  try:run(keys,pol)
  except Exception as e:print(type(e).__name__,str(e),flush=True);raise
