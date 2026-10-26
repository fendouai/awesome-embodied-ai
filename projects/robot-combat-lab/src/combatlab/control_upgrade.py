"""Source-specific public checkpoint adapters. No force/pose assistance.

Kept separate so the original physical experiment and fingerprints remain intact.
"""
from collections import deque
from pathlib import Path
import hashlib
import json,xml.etree.ElementTree as ET
import numpy as np,onnxruntime as ort,torch,yaml
from .models import ROOT,GYM,ROBOTS
from .policies import POLICIES,gravity_body

ASSETS=ROOT/'assets/control-candidates'
G1_NAMES=['left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint','left_ankle_pitch_joint','left_ankle_roll_joint','right_hip_pitch_joint','right_hip_roll_joint','right_hip_yaw_joint','right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint','waist_yaw_joint','waist_roll_joint','waist_pitch_joint','left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint','right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint']
DEFAULT=np.array([-.312,0,0,.669,-.363,0]*2+[0,0,0]+[.2,.2,0,.6,0,0,0]+[.2,-.2,0,.6,0,0,0])

class OnnxControl:
 def load(self,path):
  self.path=path;opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
  self.net=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider']);self.input=self.net.get_inputs()[0].name
 def infer(self,obs):
  self.last_observation=np.asarray(obs,dtype=np.float32).reshape(-1)
  shape=self.net.get_inputs()[0].shape[-1]
  if isinstance(shape,int) and shape!=len(self.last_observation):raise ValueError(f'Expected {shape} inputs, got {len(self.last_observation)}')
  self.last_action=self.net.run(None,{self.input:self.last_observation[None]})[0].reshape(-1)
  if not np.isfinite(self.last_action).all():raise ValueError('Nonfinite checkpoint action')
  return self.last_action
 def map_g1(self,a):
  if a.spec.key!='g1_full':raise ValueError('This checkpoint requires the G1 29-DoF body')
  self.a=a;self.mp=np.array([a.names.index(n) for n in G1_NAMES]);self.target=a.default.copy()

class Holosoma(OnnxControl):
 key='holosoma_fastsac';algorithm='fastsac'
 def __init__(self,a):
  self.map_g1(a);folder=ASSETS/'holosoma/src/holosoma_inference';self.cfg=yaml.safe_load((folder/'holosoma_cpp/config/inference/g1-29dof-loco.yaml').read_text());self.load(folder/f'holosoma_inference/models/loco/g1_29dof/{self.algorithm}_g1_29dof.onnx');meta=self.net.get_modelmeta().custom_metadata_map
  names=json.loads(meta['dof_names']);self.mp=np.array([a.names.index(n) for n in names]);self.default=np.array(self.cfg['robot']['default_dof_angles']);a.default[self.mp]=self.default;a.kp[self.mp]=json.loads(meta['kp']);a.kd[self.mp]=json.loads(meta['kd']);self.last_action=np.zeros(29);self.target=a.default.copy();self.phase=np.array([0.,np.pi]);self.phase_time=0.;self.standing=False
 def update(self,m,d,cmd,t):
  a=self.a
  if np.linalg.norm(cmd[:2])<.01 and abs(cmd[2])<.01:self.phase[:]=np.pi;self.standing=True
  elif self.standing:self.phase[:]=[0,np.pi];self.standing=False
  else:self.phase=(self.phase+2*np.pi*max(0,t-self.phase_time))%(2*np.pi)
  self.phase_time=t;phase=self.phase;terms=dict(base_ang_vel=d.qvel[a.free_vadr+3:a.free_vadr+6],projected_gravity=gravity_body(d.qpos[a.free_qadr+3:a.free_qadr+7]),command_lin_vel=cmd[:2],command_ang_vel=[cmd[2]],dof_pos=d.qpos[a.qadr][self.mp]-self.default,dof_vel=d.qvel[a.vadr][self.mp],actions=self.last_action,sin_phase=np.sin(phase),cos_phase=np.cos(phase));obs=np.concatenate([np.asarray(terms[k])*self.cfg['observation']['obs_scales'][k] for k in sorted(terms)])
  action=np.clip(self.infer(obs),-100,100);self.last_action=action;self.target[self.mp]=self.default+action*.25;return self.target
class HolosomaPPO(Holosoma):key='holosoma_ppo';algorithm='ppo'

class AMPRecovery(OnnxControl):
 key='amp_recovery'
 def __init__(self,a):
  self.map_g1(a);self.load(ASSETS/'wbc_fsm/model/loco/Unitree-G1-AMP-Flat_model_30000.onnx');self.default=DEFAULT.copy();a.default[self.mp]=self.default
  # Constants and ordered motor classes from State_MJAmp.h/mathTypes.h.
  arm=np.array([.025101925,.025101925,.010177520,.025101925,.00721945,.00721945]*2+[.010177520,.00721945,.00721945]+[.003609725]*5+[.0021812]*2+[.003609725]*5+[.0021812]*2);freq=10*2*np.pi
  a.kp[self.mp]=arm*freq**2;a.kd[self.mp]=4*arm*freq;limits=np.array([139,139,88,139,50,50]*2+[88,50,50]+[25]*5+[10]*2+[25]*5+[10]*2)
  self.scale=.25*limits/a.kp[self.mp];self.last_action=np.zeros(29);self.hist=None;self.target=a.default.copy()
 def update(self,m,d,cmd,t):
  a=self.a;obs=np.r_[d.qvel[a.free_vadr+3:a.free_vadr+6],gravity_body(d.qpos[a.free_qadr+3:a.free_qadr+7]),cmd,d.qpos[a.qadr][self.mp]-self.default,d.qvel[a.vadr][self.mp],self.last_action].astype(np.float32)
  if self.hist is None:self.hist=deque([obs.copy() for _ in range(4)],maxlen=4)
  else:self.hist.append(obs)
  action=np.clip(self.infer(np.clip(np.concatenate(self.hist),-100,100)),-100,100);self.last_action=action;self.target[self.mp]=self.default+action*self.scale;return self.target

class NvidiaWBC(OnnxControl):
 key='nvidia_balance';variant='Balance'
 def __init__(self,a):
  self.map_g1(a);folder=ASSETS/'GR00T-WholeBodyControl/decoupled_wbc/sim2mujoco/resources/robots/g1';self.cfg=yaml.safe_load((folder/'g1_gear_wbc.yaml').read_text());self.load(folder/f'policy/GR00T-WholeBodyControl-{self.variant}.onnx');self.default=np.r_[self.cfg['default_angles'],np.zeros(14)];a.default[self.mp]=self.default;a.kp[self.mp]=np.r_[self.cfg['kps'],np.full(14,100.)];a.kd[self.mp]=np.r_[self.cfg['kds'],np.full(14,.5)];self.hist=deque([np.zeros(86) for _ in range(6)],maxlen=6);self.last_action=np.zeros(15);self.target=a.default.copy()
 def update(self,m,d,cmd,t):
  a=self.a;obs=np.r_[np.asarray(cmd)*np.array(self.cfg['cmd_scale']),self.cfg['height_cmd'],self.cfg['rpy_cmd'],d.qvel[a.free_vadr+3:a.free_vadr+6]*.5,gravity_body(d.qpos[a.free_qadr+3:a.free_qadr+7]),d.qpos[a.qadr][self.mp]-self.default,d.qvel[a.vadr][self.mp]*.05,self.last_action];self.hist.append(obs);action=self.infer(np.concatenate(self.hist));self.target[self.mp[:15]]=self.default[:15]+action*.25;return self.target
class NvidiaWalk(NvidiaWBC):key='nvidia_walk';variant='Walk'

class NvidiaAdaptive(NvidiaWBC):
 """Use the source deployment's Balance/Walk command switch, shared history."""
 key='nvidia_wbc'
 def __init__(self,a):
  super().__init__(a);self.balance_net=self.net;self.balance_path=self.path
  self.load(self.path.with_name('GR00T-WholeBodyControl-Walk.onnx'));self.walk_net=self.net;self.walk_path=self.path;self.paths=[self.balance_path,self.walk_path]
 def update(self,m,d,cmd,t):
  moving=np.linalg.norm(cmd)>.05;self.net=self.walk_net if moving else self.balance_net;self.path=self.walk_path if moving else self.balance_path
  return super().update(m,d,cmd,t)

class UnitreeFullBodyGym:
 """Official leg policy mapped BY JOINT NAME onto the articulated full body.
 Upper joints remain servo-held; this is a documented transfer, not native training.
 """
 key='unitree_full_gym'
 def __init__(self,a):
  legacy='h1_2' if a.spec.key=='h1_2_full' else 'h1' if a.spec.key=='h1_full' else None
  if legacy is None:raise ValueError('This adapter supports H1/H1-2 only')
  self.a=a;self.cfg=yaml.safe_load(ROBOTS[legacy].config.read_text());self.n=self.cfg['num_actions'];names=[j.get('name') for j in ET.parse(ROBOTS[legacy].xml).getroot().findall('.//worldbody//joint') if j.get('type')!='free'][:self.n];self.mp=np.array([a.names.index(n) for n in names]);self.default=np.array(self.cfg['default_angles']);a.default[self.mp]=self.default;a.kp[self.mp]=self.cfg['kps'];a.kd[self.mp]=self.cfg['kds'];self.path=GYM/f'deploy/pre_train/{legacy}/motion.pt';self.net=torch.jit.load(str(self.path),map_location='cpu');self.net.eval();self.last_action=np.zeros(self.n,dtype=np.float32);self.target=a.default.copy()
 def update(self,m,d,cmd,t):
  a=self.a;c=self.cfg;obs=np.r_[d.qvel[a.free_vadr+3:a.free_vadr+6]*c['ang_vel_scale'],gravity_body(d.qpos[a.free_qadr+3:a.free_qadr+7]),np.asarray(cmd)*c['cmd_scale'],(d.qpos[a.qadr][self.mp]-self.default)*c['dof_pos_scale'],d.qvel[a.vadr][self.mp]*c['dof_vel_scale'],self.last_action,np.sin(2*np.pi*t/.8),np.cos(2*np.pi*t/.8)].astype(np.float32);self.last_observation=obs.copy()
  with torch.inference_mode():self.last_action=self.net(torch.from_numpy(obs)[None]).numpy().reshape(-1)
  self.target[self.mp]=self.default+self.last_action*c['action_scale'];return self.target

NEW_POLICIES={p.key:p for p in [Holosoma,HolosomaPPO,AMPRecovery,NvidiaWBC,NvidiaWalk,NvidiaAdaptive,UnitreeFullBodyGym]}
POLICIES.update(NEW_POLICIES)
LABELS={'nvidia_wbc':'NVIDIA WBC · Balance/Walk native switch','holosoma_fastsac':'Amazon Holosoma FastSAC','holosoma_ppo':'Amazon Holosoma PPO','amp_recovery':'AMP locomotion + recovery','nvidia_balance':'NVIDIA WBC Balance','nvidia_walk':'NVIDIA WBC Walk','unitree_full_gym':'Unitree RL Gym · full-body joint mapping'}

# Capture the loaded source, so later edits cannot mislabel a running experiment.
ADAPTER_SOURCE=Path(__file__).read_bytes()
ADAPTER_FINGERPRINT=hashlib.sha256(ADAPTER_SOURCE).hexdigest()
