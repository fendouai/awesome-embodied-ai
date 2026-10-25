"""Pinned rl_sar quadruped policy ports; joint names, history and model limits explicit."""
from pathlib import Path
from collections import deque
import copy,json,xml.etree.ElementTree as ET
import numpy as np,torch,yaml,mujoco
from .models import ROOT,RobotSpec,ROBOTS,Actor,scene_xml,configure_servos
from .policies import gravity_body
from .engine import STATE_SPEC,write_json,sha256
from .storage import encode_model
BASE=ROOT/'assets/expansion';SAR=BASE/'rl_sar/policy'

for key in ['go2','a1']:
 cfg=yaml.safe_load((SAR/key/'base.yaml').read_text())[key]
 stub=BASE/f'{key}-actor.yaml'
 if not stub.exists():stub.write_text(yaml.safe_dump(dict(default_angles=cfg['default_dof_pos'],kps=cfg['fixed_kp'],kds=cfg['fixed_kd'])))
 source=BASE/f'menagerie/unitree_{key}/{key}.xml';tree=ET.parse(source);root=tree.getroot();compiler=root.find('compiler');compiler.set('meshdir',str((source.parent/compiler.get('meshdir','')).resolve()));compiler.set('texturedir',str((source.parent/compiler.get('texturedir','')).resolve()))
 for child in list(root.find('worldbody')):
  if child.tag!='body':root.find('worldbody').remove(child)
 canonical=BASE/f'{key}-canonical.xml';tree.write(canonical,encoding='unicode')
 ROBOTS[key]=RobotSpec(key,'Unitree '+key.upper(),canonical,stub,'Pinned Menagerie rigid-body model; rl_sar policy sim-to-sim port; original non-body lights omitted from composed scene')

class QuadPolicy:
 def __init__(self,a,key):
  self.a=a;robot=a.spec.key;variant={'go2_him':'himloco','go2_robot_lab':'robot_lab','a1_vq':'legged_gym'}[key];self.key=key;c=yaml.safe_load((SAR/robot/variant/'config.yaml').read_text())[robot+'/'+variant];self.cfg=c;self.path=SAR/robot/variant/c['model_name'];self.net=torch.jit.load(str(self.path),map_location='cpu').eval()
  base=yaml.safe_load((SAR/robot/'base.yaml').read_text())[robot];sdk=base['joint_names'];self.training_names=[sdk[i] for i in c['joint_mapping']];self.mp=np.array([a.names.index(n) for n in self.training_names]);self.default=np.asarray(c['default_dof_pos']);a.default[self.mp]=self.default;a.kp[self.mp]=c['rl_kp'];a.kd[self.mp]=c['rl_kd'];self.scale=np.asarray(c['action_scale']);self.last_action=np.zeros(12,dtype=np.float32);self.hist=deque([np.zeros(45,dtype=np.float32) for _ in range(max(c['observations_history'],default=0)+1)],maxlen=max(c['observations_history'],default=0)+1);self.target=a.default.copy()
  a.torque_limits[self.mp]=np.minimum(a.torque_limits[self.mp],np.asarray(c['torque_limits']))
 def update(self,m,d,cmd,t):
  a=self.a;c=self.cfg;values=dict(ang_vel=d.qvel[a.free_vadr+3:a.free_vadr+6]*c['ang_vel_scale'],gravity_vec=gravity_body(d.qpos[a.free_qadr+3:a.free_qadr+7]),commands=np.asarray(cmd)*c['commands_scale'],dof_pos=(d.qpos[a.qadr][self.mp]-self.default)*c['dof_pos_scale'],dof_vel=d.qvel[a.vadr][self.mp]*c['dof_vel_scale'],actions=self.last_action)
  prop=np.clip(np.concatenate([values[n] for n in c['observations']]),-c['clip_obs'],c['clip_obs']).astype(np.float32);self.hist.appendleft(prop);obs=np.concatenate([self.hist[i] for i in c['observations_history']]) if c['observations_history'] else prop;self.last_observation=obs.copy()
  with torch.inference_mode():out=self.net(torch.from_numpy(obs)[None]).numpy().reshape(-1)
  if out.shape!=(12,) or not np.isfinite(out).all():raise ValueError('Invalid quadruped action')
  self.last_action=np.clip(out,c['clip_actions_lower'],c['clip_actions_upper']);self.target=a.default.copy();self.target[self.mp]=self.default+self.last_action*self.scale;return self.target


def quad_actors(m,robot):
 actors=[]
 for side in range(2):
  prefix=f'p{side}/';b=np.array([i for i in range(1,m.nbody) if (m.body(i).name or '').startswith(prefix)]);j=np.array([i for i in range(m.njnt) if m.jnt_bodyid[i] in b and m.jnt_type[i]!=mujoco.mjtJoint.mjJNT_FREE]);u=np.array([next(u for u in range(m.nu) if m.actuator_trnid[u,0]==jj) for jj in j]);free=next(i for i in range(m.njnt) if m.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[i] in b);limits=[]
  for jj,uu in zip(j,u):
   if m.jnt_actfrclimited[jj]:limit=max(abs(m.jnt_actfrcrange[jj]))
   elif m.actuator_forcelimited[uu]:limit=max(abs(m.actuator_forcerange[uu]))
   elif m.actuator_ctrllimited[uu]:limit=max(abs(m.actuator_ctrlrange[uu]))
   else:raise ValueError('Missing source actuator limit')
   limits.append(limit)
  names=[m.joint(i).name[len(prefix):] for i in j];actors.append(Actor(prefix,ROBOTS[robot],int(b[0]),free,int(m.jnt_qposadr[free]),int(m.jnt_dofadr[free]),b,j,m.jnt_qposadr[j].copy(),m.jnt_dofadr[j].copy(),u,np.array(limits),np.zeros(len(j)),np.full(len(j),20.),np.full(len(j),.5),names,.27))
 return actors
