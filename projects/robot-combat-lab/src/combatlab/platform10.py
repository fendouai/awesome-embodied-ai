"""Source-pinned B2/wheel-legged, Go1 adaptation and G1 motion-tracking ports."""
from pathlib import Path
from collections import deque
import json,xml.etree.ElementTree as ET
import numpy as np,torch,yaml,mujoco
from scipy.spatial.transform import Rotation,Slerp
from .models import ROOT,ROBOTS,RobotSpec
from .policies import gravity_body
P=ROOT/'assets/ten-platform';SAR=P/'rl_sar/policy'
NEW_ROBOTS=['go1','b2','go2w','b2w'];LEGGED_ROBOTS=['go2','a1']+NEW_ROBOTS
for key in NEW_ROBOTS:
 source=P/f'menagerie/unitree_go1/go1.xml' if key=='go1' else P/f'unitree_mujoco/unitree_robots/{key}/{key}.xml'
 tree=ET.parse(source);r=tree.getroot();compiler=r.find('compiler')
 for prop in ['meshdir','texturedir']:compiler.set(prop,str((source.parent/compiler.get(prop,'')).resolve()))
 for item in list(r.find('worldbody')):
  if item.tag!='body':r.find('worldbody').remove(item)
 canonical=P/f'{key}-canonical.xml';tree.write(canonical,encoding='unicode');stub=P/f'{key}-actor.yaml'
 if not stub.exists():stub.write_text(yaml.safe_dump(dict(default_angles=[0]*12,kps=[20]*12,kds=[.5]*12)))
 ROBOTS[key]=RobotSpec(key,'Unitree '+{'go1':'Go1','b2':'B2','go2w':'Go2-W','b2w':'B2-W'}[key],canonical,stub,'Pinned manufacturer/Menagerie model; CPU transfer with source policy contract; wheels use velocity targets')

class SarPolicy:
 def __init__(self,a,key):
  self.a=a;robot=a.spec.key;self.key=key;c=yaml.safe_load((SAR/robot/'robot_lab/config.yaml').read_text())[robot+'/robot_lab'];self.cfg=c;self.path=SAR/robot/'robot_lab'/c['model_name'];self.paths=[self.path];self.net=torch.jit.load(str(self.path),map_location='cpu').eval();base=yaml.safe_load((SAR/robot/'base.yaml').read_text())[robot];self.training_names=[base['joint_names'][i] for i in c['joint_mapping']];physical=[name if name in a.names else name.replace('_foot_joint','_wheel_joint') for name in self.training_names];self.physical_names=physical;self.mp=np.array([a.names.index(n) for n in physical]);self.n=c['num_of_dofs'];self.default=np.asarray(c['default_dof_pos']);a.default[self.mp]=self.default;a.kp[self.mp]=c['rl_kp'];a.kd[self.mp]=c['rl_kd'];self.scale=np.asarray(c['action_scale']);self.last_action=np.zeros(self.n,dtype=np.float32);self.target=a.default.copy();self.wheels=np.asarray(c['wheel_indices'],dtype=int);a.torque_limits[self.mp]=np.minimum(a.torque_limits[self.mp],c['torque_limits'])
 def update(self,m,d,cmd,t):
  a=self.a;c=self.cfg;q=(d.qpos[a.qadr][self.mp]-self.default).copy();q[self.wheels]=0
  values=dict(ang_vel=d.qvel[a.free_vadr+3:a.free_vadr+6]*c['ang_vel_scale'],gravity_vec=gravity_body(d.qpos[a.free_qadr+3:a.free_qadr+7]),commands=np.asarray(cmd)*c['commands_scale'],dof_pos=q*c['dof_pos_scale'],dof_vel=d.qvel[a.vadr][self.mp]*c['dof_vel_scale'],actions=self.last_action);obs=np.clip(np.concatenate([values[n] for n in c['observations']]),-c['clip_obs'],c['clip_obs']).astype(np.float32);self.last_observation=obs.copy()
  with torch.inference_mode():out=self.net(torch.from_numpy(obs)[None]).numpy()[0]
  assert len(out)==self.n and np.isfinite(out).all();self.last_action=np.clip(out,c['clip_actions_lower'],c['clip_actions_upper']);target=self.default+self.last_action*self.scale;target[self.wheels]=self.last_action[self.wheels]*self.scale[self.wheels];self.target[self.mp]=target;return self.target

class WalkTheseWays:
 key='go1_wtw'
 def __init__(self,a):
  assert a.spec.key=='go1';self.a=a;self.cfg=json.loads((ROOT/'research/ten-platform/wtw-parameters-readable.json').read_text())['Cfg'];c=self.cfg;self.training_names=[f'{leg}_{joint}_joint' for leg in ['FL','FR','RL','RR'] for joint in ['hip','thigh','calf']];self.mp=np.array([a.names.index(n) for n in self.training_names]);self.default=np.array([c['init_state']['default_joint_angles'][n] for n in self.training_names]);a.default[self.mp]=self.default;a.kp[self.mp]=20.;a.kd[self.mp]=.5
  folder=P/'wtw/runs/gait-conditioned-agility/pretrain-v0/train/025417.456545/checkpoints';self.path=folder/'body_latest.jit';self.adapt_path=folder/'adaptation_module_latest.jit';self.paths=[self.path,self.adapt_path];self.net=torch.jit.load(str(self.path),map_location='cpu').eval();self.adapt=torch.jit.load(str(self.adapt_path),map_location='cpu').eval();self.hist=deque([np.zeros(70,dtype=np.float32) for _ in range(30)],maxlen=30);self.first=True;self.last_action=np.zeros(12,dtype=np.float32);self.older_action=self.last_action.copy();self.target=a.default.copy();self.scale=np.full(12,.25);self.scale[[0,3,6,9]]*=.5
 def update(self,m,d,cmd,t):
  a=self.a;s=self.cfg['obs_scales'];commands=np.array([*cmd,0.,3.,.5,0.,0.,.5,.08,0.,0.,.25,.40,0.]);cs=np.array([s[n] for n in ['lin_vel','lin_vel','ang_vel','body_height_cmd','gait_freq_cmd','gait_phase_cmd','gait_phase_cmd','gait_phase_cmd','gait_phase_cmd','footswing_height_cmd','body_pitch_cmd','body_roll_cmd','stance_width_cmd','stance_length_cmd','aux_reward_cmd']]);phase=(t*3)%1.;clock=np.sin(2*np.pi*np.array([phase+.5,phase,phase,phase+.5]));obs=np.r_[gravity_body(d.qpos[a.free_qadr+3:a.free_qadr+7]),commands*cs,(d.qpos[a.qadr][self.mp]-self.default)*s['dof_pos'],d.qvel[a.vadr][self.mp]*s['dof_vel'],self.last_action,self.older_action,clock].astype(np.float32);assert len(obs)==70
  if not self.first:self.hist.append(np.clip(obs,-100,100))
  self.first=False;history=np.concatenate(self.hist).astype(np.float32);self.last_observation=history.copy()
  with torch.inference_mode():latent=self.adapt(torch.from_numpy(history)[None]);out=self.net(torch.cat([torch.from_numpy(history)[None],latent],-1)).numpy()[0]
  self.last_latent=latent.numpy()[0].copy();self.older_action=self.last_action.copy();self.last_action=np.clip(out,-10,10);self.target[self.mp]=self.default+self.last_action*self.scale;return self.target

class BeyondMimic:
 key='g1_beyondmimic'
 def __init__(self,a):
  assert a.spec.key=='g1_full';self.a=a;folder=SAR/'g1/whole_body_tracking/dance_102';self.cfg=yaml.safe_load((folder/'config.yaml').read_text())['g1/whole_body_tracking/dance_102'];c=self.cfg;base=yaml.safe_load((SAR/'g1/base.yaml').read_text())['g1'];self.sdk=base['joint_names'];self.mapping=np.asarray(c['joint_mapping']);self.training_names=[self.sdk[i] for i in self.mapping];self.mp=np.array([a.names.index(n) for n in self.training_names]);self.default=np.asarray(c['default_dof_pos']);a.default[self.mp]=self.default;a.kp[self.mp]=c['rl_kp'];a.kd[self.mp]=c['rl_kd'];a.torque_limits[self.mp]=np.minimum(a.torque_limits[self.mp],c['torque_limits']);self.path=folder/c['model_name'];self.motion_path=folder/c['motion_file'];self.paths=[self.path,self.motion_path];self.net=torch.jit.load(str(self.path),map_location='cpu').eval();self.motion=np.loadtxt(self.motion_path,delimiter=',');self.qref=self.motion[:,7:];self.vref=np.diff(self.qref,axis=0)*60;self.vref=np.vstack([self.vref,self.vref[-1]]);self.length=len(self.motion)/60;self.last_action=np.zeros(29,dtype=np.float32);self.target=a.default.copy();self.initial_q=a.default.copy();self.initial_q[self.mp]=self.qref[0,self.mapping];self.align=None
 def anchor(self,quat,q):return Rotation.from_quat(quat)*Rotation.from_euler('ZXY',q[12:15])
 def update(self,m,d,cmd,t):
  a=self.a;c=self.cfg;u=np.clip(t/self.length,0,1)*(len(self.motion)-1);i=int(u);j=min(i+1,len(self.motion)-1);blend=u-i;q=self.qref[i]*(1-blend)+self.qref[j]*blend;v=self.vref[i]*(1-blend)+self.vref[j]*blend
  if i==j:root=Rotation.from_quat(self.motion[i,3:7])
  else:root=Slerp([0,1],Rotation.from_quat(self.motion[[i,j],3:7]))([blend])[0]
  ref=root*Rotation.from_euler('ZXY',q[12:15]);base_q=d.qpos[a.free_qadr+3:a.free_qadr+7];root_rot=Rotation.from_quat(base_q[[1,2,3,0]]);sdk_q=np.array([d.qpos[a.qadr[a.names.index(n)]] for n in self.sdk]);cur=root_rot*Rotation.from_euler('ZXY',sdk_q[12:15])
  if self.align is None:self.align=Rotation.from_euler('z',cur.as_euler('xyz')[2]-self.anchor(self.motion[0,3:7],self.qref[0]).as_euler('xyz')[2])
  rel=(cur.inv()*self.align*ref).as_matrix()[:,:2].ravel();self.current_reference=q[self.mapping].copy();obs=np.r_[q[self.mapping],v[self.mapping],rel,d.qvel[a.free_vadr+3:a.free_vadr+6],d.qpos[a.qadr][self.mp]-self.default,d.qvel[a.vadr][self.mp],self.last_action].astype(np.float32);assert len(obs)==154;self.last_observation=np.clip(obs,-100,100)
  with torch.inference_mode():out=self.net(torch.from_numpy(self.last_observation)[None]).numpy()[0]
  self.last_action=np.clip(out,c['clip_actions_lower'],c['clip_actions_upper']);self.target[self.mp]=self.default+self.last_action*np.asarray(c['action_scale']);return self.target

NEW_POLICIES={'b2_robot_lab':SarPolicy,'go2w_robot_lab':SarPolicy,'b2w_robot_lab':SarPolicy,'go1_wtw':WalkTheseWays,'g1_beyondmimic':BeyondMimic}
BASE_POLICY={'b2':'b2_robot_lab','go2w':'go2w_robot_lab','b2w':'b2w_robot_lab','go1':'go1_wtw'}
def create(a,key):return SarPolicy(a,key) if key.endswith('_robot_lab') else NEW_POLICIES[key](a)
def configure_wheels(m,actors):
 for a in actors:
  for i,u in enumerate(a.actuator_ids):
   if a.kp[i]==0:
    m.actuator_gainprm[u,0]=a.kd[i];m.actuator_biasprm[u,1]=0.;m.actuator_biasprm[u,2]=-a.kd[i]
