import sys,time,json
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src')]
from combatlab.models import *
from combatlab.policies import *

def test(keys,dist,lean,seconds=10.):
 m=mujoco.MjModel.from_xml_string(scene_xml(keys));d=mujoco.MjData(m);aa=make_actors(m,keys);initialize(m,d,aa,distance=dist);configure_servos(m,aa);pp=[PDStand(a) for a in aa];first=None;falls=[None,None];maxf=0
 for step in range(round(seconds/.002)):
  for a,po in zip(aa,pp):
   if step%10==0:po.update(m,d,[-lean*np.clip((d.time-1.5)/2,0,1),0,0],d.time)
   d.ctrl[a.actuator_ids]=po.target+po.feedforward/a.kp
  mujoco.mj_step(m,d)
  force=0
  for c in d.contact[:d.ncon]:
   b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
   if (b1 in aa[0].body_ids and b2 in aa[1].body_ids) or (b2 in aa[0].body_ids and b1 in aa[1].body_ids):
    f=np.zeros(6);mujoco.mj_contactForce(m,d,list(d.contact).index(c),f) if False else None
    if first is None:first=d.time
  for i,a in enumerate(aa):
   tilt=np.degrees(np.arccos(np.clip(-gravity_body(d.qpos[a.free_qadr+3:a.free_qadr+7])[2],-1,1)))
   if falls[i] is None and (d.qpos[a.free_qadr+2]<a.nominal_z*.55 or tilt>65):falls[i]=d.time
  if any(x is not None for x in falls):break
 print(json.dumps(dict(keys=keys,dist=dist,lean=lean,first_contact=first,falls=falls,init_contacts=0,endtime=d.time)),flush=True)
if __name__=='__main__':
 for dist in [.40,.45,.50]:
  for lean in [.03,.05,.08]:test(('g1','h1'),dist,lean)
