from __future__ import annotations
from collections import deque
import numpy as np
import torch, onnxruntime as ort, yaml
import mujoco
from scipy.optimize import nnls
from .models import GYM,LAB

torch.set_num_threads(1)

def gravity_body(q):
 w,x,y,z=q
 return np.array([2*(-z*x+w*y),-2*(z*y+w*x),1-2*(w*w+z*z)])

class PDStand:
    """Same joint-space PD formula and source neutral pose; no root forces."""
    key='pd_stand'
    def __init__(self,actor):self.a=actor;self.target=actor.default.copy();self.last_action=np.zeros(len(actor.default));self.feedforward=np.zeros(len(actor.default));self.reach_enabled=True
    def update(self,m,d,cmd,t):
        a=self.a;self.target=a.default.copy();columns=[];joint_rows=[]
        for contact in d.contact[:d.ncon]:
            g1,g2=int(contact.geom1),int(contact.geom2);b1,b2=int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2])
            body=b2 if b1==0 and b2 in a.body_ids else b1 if b2==0 and b1 in a.body_ids else None
            if body is None or 'ankle' not in m.body(body).name:continue
            jac=np.zeros((3,m.nv));rot=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,rot,contact.pos,body)
            columns.append(jac[2,a.free_vadr:a.free_vadr+6]);joint_rows.append(jac[2,a.vadr])
        self.feedforward=np.zeros(len(a.default))
        if columns:
            M=np.stack(columns,axis=1);rhs=d.qfrc_bias[a.free_vadr:a.free_vadr+6]
            prior=np.full(M.shape[1],max(0.,rhs[2])/M.shape[1])
            regularization=.02
            forces=nnls(np.vstack([M,regularization*np.eye(M.shape[1])]),np.r_[rhs,regularization*prior],maxiter=200)[0]
            self.feedforward=np.clip(d.qfrc_bias[a.vadr]-np.stack(joint_rows).T@forces,-a.torque_limits,a.torque_limits)
        g=gravity_body(d.qpos[a.free_qadr+3:a.free_qadr+7]);pitch=np.arcsin(np.clip(g[0],-1,1));roll=np.arctan2(-g[1],-g[2]);omega=d.qvel[a.free_vadr+3:a.free_vadr+6]
        for j,name in enumerate(a.names):
            if 'ankle_pitch' in name or name in ['left_ankle_joint','right_ankle_joint']:self.target[j]+=np.clip(.15*pitch+.025*omega[1]+cmd[0],-.2,.2)
            if 'ankle_roll' in name:self.target[j]+=np.clip(.15*roll+.025*omega[0],-.2,.2)
        if not any('ankle_roll' in name for name in a.names):
            for j,name in enumerate(a.names):
                if 'hip_roll' in name:self.target[j]+=np.clip(.15*roll+.025*omega[0],-.2,.2)
        reach=np.clip((t-1.5)/2,0,1) if self.reach_enabled else 0.
        for j,name in enumerate(a.names):
            if 'shoulder_pitch' in name:self.target[j]=a.default[j]-.9*reach
            if 'elbow' in name:
                sign=-1 if a.default[j]<0 else 1
                self.target[j]=(1-reach)*a.default[j]+reach*.12*sign
        return self.target

class GymMotion:
    """Official unitree_rl_gym TorchScript deployment observation and PD convention."""
    key='unitree_rl_gym'
    def __init__(self,actor):
        self.a=actor;self.cfg=yaml.safe_load(actor.spec.config.read_text());self.n=self.cfg['num_actions'];k='g1' if actor.spec.key.startswith('g1') else actor.spec.key
        self.path=GYM/f'deploy/pre_train/{k}/motion.pt';self.net=torch.jit.load(str(self.path),map_location='cpu');self.net.eval();self.last_action=np.zeros(self.n,dtype=np.float32);self.target=actor.default.copy()
    def update(self,m,d,cmd,t):
        a=self.a;c=self.cfg;n=self.n;q=d.qpos[a.qadr[:n]];v=d.qvel[a.vadr[:n]]
        obs=np.concatenate([d.qvel[a.free_vadr+3:a.free_vadr+6]*c['ang_vel_scale'],gravity_body(d.qpos[a.free_qadr+3:a.free_qadr+7]),np.array(cmd)*c['cmd_scale'],(q-np.array(c['default_angles']))*c['dof_pos_scale'],v*c['dof_vel_scale'],self.last_action,[np.sin(2*np.pi*t/.8),np.cos(2*np.pi*t/.8)]]).astype(np.float32)
        self.last_observation=obs.copy()
        with torch.inference_mode():action=self.net(torch.from_numpy(obs).unsqueeze(0)).cpu().numpy().reshape(-1)
        if len(action)!=n or not np.isfinite(action).all():raise RuntimeError('Invalid official motion-policy output')
        self.last_action=action;self.target=a.default.copy();self.target[:n]=action*c['action_scale']+np.array(c['default_angles'])
        return self.target

class LabVelocity:
    """Actual Unitree ONNX deployment packages, preserving declared order/history/scale."""
    key='unitree_rl_lab'
    variant='lab'
    def __init__(self,actor):
        if len(actor.joint_ids)!=29:raise ValueError('Checkpoint requires a G1 29-DoF body')
        self.a=actor
        if self.variant=='lab':folder=LAB/'deploy/robots/g1_29dof/config/policy/velocity/v0'
        else:folder=LAB.parent/'mjlab/deploy/robots/g1/config/policy/velocity/v0'
        self.path=folder/'exported/policy.onnx';self.cfg=yaml.safe_load((folder/'params/deploy.yaml').read_text());self.mp=np.array(self.cfg['joint_ids_map']);self.default=np.array(self.cfg['default_joint_pos']);self.last_action=np.zeros(29,dtype=np.float32)
        actor.kp=np.array(self.cfg['stiffness']);actor.kd=np.array(self.cfg['damping'])
        opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
        self.net=ort.InferenceSession(str(self.path),sess_options=opt,providers=['CPUExecutionProvider']);self.input=self.net.get_inputs()[0].name;self.hist=None;self.target=actor.default.copy()
    def update(self,m,d,cmd,t):
        a=self.a;mapping={
          'base_ang_vel':d.qvel[a.free_vadr+3:a.free_vadr+6],
          'projected_gravity':gravity_body(d.qpos[a.free_qadr+3:a.free_qadr+7]),
          'velocity_commands':np.array(cmd),
          'joint_pos_rel':d.qpos[a.qadr][self.mp]-self.default,
          'joint_vel_rel':d.qvel[a.vadr][self.mp],
          'last_action':self.last_action,
        }
        terms=[];lengths=[]
        for name,c in self.cfg['observations'].items():
            if name=='gait_phase':
                period=c['params']['period'];value=np.array([np.sin(2*np.pi*t/period),np.cos(2*np.pi*t/period)])
            else:value=mapping[name]
            if c.get('clip') is not None:value=np.clip(value,*c['clip'])
            value=np.asarray(value)*np.array(c.get('scale',1.))
            terms.append(value.astype(np.float32));lengths.append(c.get('history_length',1))
        if self.hist is None:self.hist=[deque([v.copy() for _ in range(n)],maxlen=n) for v,n in zip(terms,lengths)]
        else:
            for h,v in zip(self.hist,terms):h.append(v.copy())
        obs=np.concatenate([np.concatenate(list(h)) for h in self.hist]).astype(np.float32)[None,:];self.last_observation=obs[0].copy()
        action=self.net.run(None,{self.input:obs})[0].reshape(-1)
        if len(action)!=29 or not np.isfinite(action).all():raise RuntimeError('Invalid official ONNX output')
        self.last_action=action;self.target=np.zeros(29);self.target[self.mp]=self.default+action*np.array(self.cfg['actions']['JointPositionAction']['scale'])
        return self.target

class MjlabVelocity(LabVelocity):
    key='unitree_rl_mjlab';variant='mjlab'

POLICIES={'pd_stand':PDStand,'unitree_rl_gym':GymMotion,'unitree_rl_lab':LabVelocity,'unitree_rl_mjlab':MjlabVelocity}

def torques(actor,d,target):
    raw=actor.kp*(target-d.qpos[actor.qadr])-actor.kd*d.qvel[actor.vadr]
    return np.clip(raw,-actor.torque_limits,actor.torque_limits)
