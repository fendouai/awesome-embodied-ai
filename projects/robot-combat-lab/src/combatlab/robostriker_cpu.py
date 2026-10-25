"""Pinned RoboStriker Stage3 inference transplanted to native MuJoCo/CPU.

This is a sim-to-sim port under validation, NOT a reproduction claim of the paper.
No scripted punch, root force, pose animation or NVIDIA velocity policy is used.
"""
from pathlib import Path
from collections import deque
import xml.etree.ElementTree as ET
import hashlib,json
import numpy as np,torch,mujoco
from torch import nn
from .models import ROOT,ROBOTS,RobotSpec,GYM
from .policies import POLICIES

P=ROOT/'assets/adversarial-candidates/RoboStriker-Preview'
ROBOT_KEY='g1_robostriker'
ROBOTS[ROBOT_KEY]=RobotSpec(ROBOT_KEY,'G1 / RoboStriker source URDF',P/'description/native-g1.xml',GYM/'deploy/deploy_mujoco/configs/g1.yaml','Pinned RoboStriker source URDF converted by MuJoCo; source gains/armatures applied; PhysX-to-MuJoCo transfer under validation')

def native_orders():
    root=ET.parse(P/'description/main.urdf').getroot();children={};joint_of={}
    for joint in root.findall('joint'):
        parent=joint.find('parent').get('link');child=joint.find('child').get('link');children.setdefault(parent,[]).append((child,joint.get('type'),joint.get('name')));joint_of[child]=joint
    bodies=['pelvis'];joints=[];queue=['pelvis']
    while queue:
        parent=queue.pop(0)
        for child,kind,name in children.get(parent,[]):
            if kind=='fixed':
                # Source fixed leaves are merged by both URDF importers.
                if children.get(child):raise ValueError('Fixed non-leaf needs explicit source-frame reconciliation')
                continue
            bodies.append(child);joints.append(name);queue.append(child)
    if len(bodies)!=30 or len(joints)!=29:raise ValueError('Source body/joint count mismatch')
    return bodies,joints
BODY_ORDER,JOINT_ORDER=native_orders()

class MLP(nn.Module):
    def __init__(self,state,prefix):
        super().__init__();keys=sorted([k for k in state if k.startswith(prefix+'.') and k.endswith('.weight')],key=lambda k:int(k.split('.')[-2]));layers=[]
        for i,k in enumerate(keys):
            w=state[k];layer=nn.Linear(w.shape[1],w.shape[0]);layer.weight.data.copy_(w);layer.bias.data.copy_(state[k[:-6]+'bias']);layers.append(layer)
            if i<len(keys)-1:layers.append(nn.ELU())
        self.layers=nn.Sequential(*layers)
    def forward(self,x):return self.layers(x)

class Stage3CPU(nn.Module):
    def __init__(self):
        super().__init__();self.raw=torch.load(P/'model_3000.pt',map_location='cpu',weights_only=True);s=self.raw['model_state_dict'];self.ego=MLP(s,'taskpolicy_ego');self.opponent=MLP(s,'taskpolicy_opponent');self.prior=MLP(s,'prior');self.decoder=MLP(s,'decoder');self.eval()
    def norm(self,x,key):
        stat=self.raw[key+'_state_dict'];return (x-stat['_mean'])/(stat['_std']+1e-2)
    @torch.inference_mode()
    def infer(self,side,task,history,decoder):
        task=self.norm(torch.as_tensor(task,dtype=torch.float32).reshape(1,192),'taskpolicy_obs_normalizer_ego' if side==0 else 'taskpolicy_obs_normalizer_opponent');history=self.norm(torch.as_tensor(history,dtype=torch.float32).reshape(1,900),'prior_obs_normalizer');decoder=self.norm(torch.as_tensor(decoder,dtype=torch.float32).reshape(1,90),'decoder_obs_normalizer');z=(self.ego if side==0 else self.opponent)(task)+self.prior(history)[:,:32];z=nn.functional.normalize(z,dim=-1);return self.decoder(torch.cat([decoder,z],-1)).numpy()[0].copy(),z.numpy()[0].copy()
NETWORK=None

class RoboStriker:
    key='robostriker_ego';side_net=0
    def __init__(self,a):
        if a.spec.key!=ROBOT_KEY:raise ValueError('RoboStriker port requires its own source-URDF profile')
        global NETWORK
        if NETWORK is None:NETWORK=Stage3CPU()
        self.net=NETWORK;self.a=a;self.path=P/'model_3000.pt';self.mp=np.array([a.names.index(n) for n in JOINT_ORDER]);self.last_action=np.zeros(29,dtype=np.float32);self.last_latent=np.zeros(32);self.hist=None;self.target=a.default.copy();self.body_ids=None
        frequency=10*2*np.pi;self.native_armatures=[]
        for i,name in enumerate(a.names):
            if 'hip_roll' in name or 'knee' in name:arm=.025101925;limit=139
            elif 'hip_' in name or name=='waist_yaw_joint':arm=.010177520;limit=88
            elif 'ankle_' in name or name in ['waist_roll_joint','waist_pitch_joint']:arm=2*.003609725;limit=50
            elif 'wrist_pitch' in name or 'wrist_yaw' in name:arm=.00425;limit=5
            else:arm=.003609725;limit=25
            self.native_armatures.append(arm);a.kp[i]=arm*frequency**2;a.kd[i]=4*arm*frequency;a.torque_limits[i]=limit;a.default[i]=-.312 if 'hip_pitch' in name else .669 if 'knee' in name else -.363 if 'ankle_pitch' in name else .6 if 'elbow' in name else .2 if 'shoulder_pitch' in name or name=='left_shoulder_roll_joint' else -.2 if name=='right_shoulder_roll_joint' else 0
        self.native_armatures=np.array(self.native_armatures);self.scale=.25*a.torque_limits/a.kp;self.target=a.default.copy()
    def inputs(self,m,d):
        a=self.a;side=int(a.prefix[1]);other=f'p{1-side}/';R=d.xmat[a.root_id].reshape(3,3);root=d.qpos[a.free_qadr:a.free_qadr+3]
        if self.body_ids is None:self.body_ids=np.array([m.body(a.prefix+n).id for n in BODY_ORDER])
        prop=np.r_[d.qvel[a.free_vadr+3:a.free_vadr+6],d.qpos[a.qadr][self.mp]-a.default[self.mp],d.qvel[a.vadr][self.mp],self.last_action].astype(np.float32)
        points=(d.xpos[self.body_ids]-root)@R
        wrists=np.array([d.xpos[m.body(a.prefix+n+'_wrist_yaw_link').id] for n in ['left','right']]);enemy_wrists=np.array([d.xpos[m.body(other+n+'_wrist_yaw_link').id] for n in ['left','right']]);torso=d.xpos[m.body(a.prefix+'torso_link').id];enemy_torso=d.xpos[m.body(other+'torso_link').id]
        task=np.r_[prop,points.ravel(),((wrists-enemy_torso)@R).ravel(),((enemy_wrists-torso)@R).ravel()].astype(np.float32)
        if self.hist is None:self.hist=deque([prop.copy() for _ in range(10)],maxlen=10)
        else:self.hist.append(prop.copy())
        return task,np.concatenate(self.hist).astype(np.float32),prop
    def update(self,m,d,cmd,t):
        task,history,decoder=self.inputs(m,d);self.last_observation=np.r_[task,history,decoder];self.last_action,self.last_latent=self.net.infer(self.side_net,task,history,decoder)
        if not np.isfinite(self.last_action).all():raise ValueError('Nonfinite source Stage3 output')
        self.target=self.a.default.copy();self.target[self.mp]+=self.last_action*self.scale[self.mp];return self.target
class RoboStrikerOpponent(RoboStriker):key='robostriker_opponent';side_net=1
POLICIES.update({p.key:p for p in [RoboStriker,RoboStrikerOpponent]})

SOURCE=Path(__file__).read_bytes();FINGERPRINT=hashlib.sha256(SOURCE).hexdigest()
def configure_native(sim):
    for a,p in zip(sim.actors,sim.policies):
        sim.m.dof_armature[a.vadr]=p.native_armatures;sim.m.dof_damping[a.vadr]=0;sim.m.dof_frictionloss[a.vadr]=0
        sim.m.jnt_actfrclimited[a.joint_ids]=True;sim.m.jnt_actfrcrange[a.joint_ids,0]=-a.torque_limits;sim.m.jnt_actfrcrange[a.joint_ids,1]=a.torque_limits
        sim.d.qpos[a.free_qadr+2]=.76
    mujoco.mj_forward(sim.m,sim.d)

from .endurance import ContinuousSimulation,EnduranceConfig
from .engine import write_json,sha256

class RoboStrikerSimulation(ContinuousSimulation):
    """Same ongoing physical encounter protocol, driven by both trained Stage3 actors."""
    def __init__(self,cfg,parameters=None):
        super().__init__(cfg,parameters);configure_native(self)
        self.state_history=[];self.motor_controls=[];self.hw_history=[];self.limit_history=[];self.frames=[];self.events=[];self.observations=[[],[]];self.policy_actions=[[],[]];self.observation_times=[]
        self.record_state();self.record_hardware();self.record_frame()
    def record_frame(self):
        super().record_frame()
        if hasattr(self,'hardware'):
            f=self.frames[-1];f['backend']='robostriker_cpu_port';f['policy_role']=list(self.cfg.policies);f['opponent_sensing']='simulator_ground_truth';f['source_version']=FINGERPRINT
    def save(self,base=None):
        folder,result=super().save(base or ROOT/'results/robostriker-cpu');(folder/'robostriker_cpu.py').write_bytes(SOURCE)
        write_json(folder/'source-contract.json',dict(repository='yinkangning0124/RoboStriker-Preview',commit='52f54137770ff632391dd091e973ceb098ba3f98',checkpoint_sha256=sha256(P/'model_3000.pt'),urdf_sha256=sha256(P/'description/main.urdf'),body_order=BODY_ORDER,joint_order=JOINT_ORDER,task_observations=192,prior_history_observations=900,decoder_observations=90,latent_width=32,low_level_actions=29,normalization_epsilon=.01,latent_composition='normalize(task_policy(task_norm) + prior(history_norm).mean)',action_target='source_default_joint_pos + decoded_action*(0.25*source_effort_limit/source_stiffness)',joint_gain_source='robots/g1.py from pinned RoboStriker',physics='MuJoCo CPU sim-to-sim port; not independently source-Isaac replay validated',scripted_strikes=False,hidden_root_forces=False))
        result['adversarial_backend']=dict(name='RoboStriker Stage3 CPU port',validated_reproduction=False,source_sha256=FINGERPRINT,scripted_strikes=False);write_json(folder/'result.json',result);write_json(folder/'manifest.json',{f.name:sha256(f) for f in folder.iterdir() if f.is_file() and f.name!='manifest.json'});return folder,result
    @classmethod
    def restore(cls,folder):
        folder=Path(folder)
        if sha256(folder/'robostriker_cpu.py')!=FINGERPRINT:raise ValueError('RoboStriker port source mismatch')
        return super().restore(folder)

# Freeze the complete loaded file, including the encounter/checkpoint wrapper.
SOURCE=Path(__file__).read_bytes();FINGERPRINT=hashlib.sha256(SOURCE).hexdigest()
