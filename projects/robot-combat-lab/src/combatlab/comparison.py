"""Compatible deployment A/B trials. Same body/command/pulse, no universal combat score."""
from pathlib import Path
import json,time,hashlib
from dataclasses import dataclass
import numpy as np,mujoco
from . import control_upgrade
from .models import ROOT,ROBOTS,scene_xml,make_actors,configure_servos
from .policies import POLICIES,gravity_body
from .quadruped import QuadPolicy,quad_actors
from . import platform10
from .engine import STATE_SPEC,write_json,sha256
from .storage import encode_model

COMPAT={'g1_full':['nvidia_wbc','amp_recovery','holosoma_fastsac','holosoma_ppo','unitree_rl_lab','unitree_rl_mjlab','pd_stand'],'h1_full':['unitree_full_gym','pd_stand'],'h1_2_full':['unitree_full_gym','pd_stand'],'go2':['go2_robot_lab','go2_him','pd_stand'],'a1':['a1_vq','pd_stand']}
COMPAT.update({'go1':['go1_wtw','pd_stand'],'b2':['b2_robot_lab','pd_stand'],'go2w':['go2w_robot_lab','pd_stand'],'b2w':['b2w_robot_lab','pd_stand']})
COMPAT['g1_full'].append('g1_beyondmimic')
LABELS={**control_upgrade.LABELS,'unitree_rl_lab':'Unitree RL Lab PPO','unitree_rl_mjlab':'Unitree RL MjLab PPO','pd_stand':'Neutral joint PD (reference)','go2_robot_lab':'Go2 Robot Lab PPO','go2_him':'Go2 HIM history/estimator','a1_vq':'A1 VQ history-conditioned policy'}
LABELS.update({'go1_wtw':'Go1 Walk These Ways / adaptation','b2_robot_lab':'B2 Robot Lab PPO','go2w_robot_lab':'Go2-W Robot Lab PPO','b2w_robot_lab':'B2-W Robot Lab PPO','g1_beyondmimic':'G1 BeyondMimic reference tracking'})
class NeutralPD:
 key='pd_stand'
 def __init__(self,a):self.a=a;self.target=a.default.copy();self.last_action=np.zeros(len(a.names))
 def update(self,m,d,cmd,t):return self.target

def validate(robot,left,right,task):
 if robot not in COMPAT or left not in COMPAT[robot] or right not in COMPAT[robot]:raise ValueError('Unsupported robot/policy pair')
 if task not in ['tracking','push','motion']:raise ValueError('Unsupported comparison task')
 if task=='motion' and (robot!='g1_full' or not {left,right}.issubset({'g1_beyondmimic','pd_stand'})):raise ValueError('Reference tracking requires matching motion policy or reference PD')
 if task!='motion' and 'g1_beyondmimic' in [left,right]:raise ValueError('Motion policy does not accept velocity commands')
 if left==right:raise ValueError('Choose two different deployment packages')

class Comparison:
 def __init__(self,robot,left,right,task='push',seed=0,seconds=8.):
  validate(robot,left,right,task);self.robot=robot;self.keys=[left,right];self.task=task;self.seed=seed;self.duration=seconds;self.dt=.002;self.control_dt=.02;self.xml=scene_xml((robot,robot),self.dt,.8);self.m=mujoco.MjModel.from_xml_string(self.xml);self.d=mujoco.MjData(self.m);quad=robot in platform10.LEGGED_ROBOTS;self.actors=quad_actors(self.m,robot) if quad else make_actors(self.m,(robot,robot));self.policies=[]
  for key,a in zip(self.keys,self.actors):
   if robot in platform10.NEW_ROBOTS or key=='g1_beyondmimic':
    p=platform10.create(a,platform10.BASE_POLICY[robot] if key=='pd_stand' else key)
    if key=='pd_stand':p=NeutralPD(a)
   elif quad:
    # Common source initial pose even for the neutral PD reference.
    base_key='go2_robot_lab' if robot=='go2' else 'a1_vq';p=QuadPolicy(a,base_key if key=='pd_stand' else key)
    if key=='pd_stand':p=NeutralPD(a)
   else:p=NeutralPD(a) if key=='pd_stand' else POLICIES[key](a)
   self.policies.append(p)
  configure_servos(self.m,self.actors);platform10.configure_wheels(self.m,self.actors)
  motion_initial=next((p.initial_q for p in self.policies if hasattr(p,'initial_q')),None)
  if motion_initial is not None:
   for p,a in zip(self.policies,self.actors):
    if isinstance(p,NeutralPD):a.default=motion_initial.copy();p.target=motion_initial.copy()
  rng=np.random.default_rng(seed)
  for side,a in enumerate(self.actors):
   q=a.free_qadr;self.d.qpos[q:q+3]=[0,(-1 if side==0 else 1)*1.5,.29 if quad else 1.3];self.d.qpos[q+3:q+7]=[1,0,0,0];self.d.qpos[a.qadr]=(motion_initial if motion_initial is not None else a.default)+np.random.default_rng(seed).normal(0,.002,len(a.names))
  mujoco.mj_forward(self.m,self.d)
  # Humanoid model root placement uses actual collision vertices (not bounding spheres).
  # Apply the same geometric ground-clearance rule to every morphology.
  for a in self.actors:
   low=100.
   for g in range(self.m.ngeom):
    if self.m.geom_bodyid[g] not in a.body_ids or not self.m.geom_contype[g]:continue
    kind=self.m.geom_type[g];R=self.d.geom_xmat[g].reshape(3,3);pos=self.d.geom_xpos[g];size=self.m.geom_size[g]
    if kind==mujoco.mjtGeom.mjGEOM_MESH:
     mid=self.m.geom_dataid[g];verts=self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid]+self.m.mesh_vertnum[mid]];z=(verts@R.T+pos)[:,2].min()
    elif kind==mujoco.mjtGeom.mjGEOM_BOX:z=pos[2]-abs(R[2])@size
    elif kind in [mujoco.mjtGeom.mjGEOM_CAPSULE,mujoco.mjtGeom.mjGEOM_CYLINDER]:z=pos[2]-abs(R[2,2])*size[1]-size[0]
    else:z=pos[2]-size[0]
    low=min(low,float(z))
   self.d.qpos[a.free_qadr+2]+=.003-low
  mujoco.mj_forward(self.m,self.d)
  for a in self.actors:a.nominal_z=float(self.d.qpos[a.free_qadr+2])
  self.mass=[float(self.m.body_mass[a.body_ids].sum()) for a in self.actors];assert abs(self.mass[0]-self.mass[1])<1e-9
  self.start=[self.d.qpos[a.free_qadr:a.free_qadr+3].copy() for a in self.actors];self.frames=[];self.states=[];self.controls=[];self.force_log=[];self.io=[[],[]];self.actions=[[],[]];self.io_times=[];self.energy=np.zeros(2);self.max_tilt=np.zeros(2);self.fallen=np.zeros(2,dtype=bool);self.fall_time=np.zeros(2);self.errors=[[],[]];self.motion_errors=[[],[]];self.max_torque_violation=0.;self.delivered=np.zeros((2,3));self.steps=0;self.cmd=np.zeros(3);self.terminal=None;self.record()
 def update(self):
  self.cmd=np.array([.3 if self.task=='tracking' and self.d.time>=1.5 else 0,0,0])
  for i,(a,p) in enumerate(zip(self.actors,self.policies)):
   p.update(self.m,self.d,self.cmd,self.d.time);self.d.ctrl[a.actuator_ids]=p.target
   if hasattr(p,'last_observation'):self.io[i].append(p.last_observation.copy());self.actions[i].append(p.last_action.copy())
  if self.task=='motion':
   motion=next(p for p in self.policies if hasattr(p,'current_reference'))
   for side,a in enumerate(self.actors):self.motion_errors[side].append(float(np.sqrt(np.mean((self.d.qpos[a.qadr][motion.mp]-motion.current_reference)**2))))
  self.io_times.append(float(self.d.time))
 def record(self):
  state=np.empty(mujoco.mj_stateSize(self.m,STATE_SPEC));mujoco.mj_getState(self.m,self.d,state,STATE_SPEC);self.states.append(state);self.controls.append(self.d.ctrl.copy())
 def frame(self):
  actors=[]
  for side,a in enumerate(self.actors):
   q=self.d.qpos[a.free_qadr:a.free_qadr+7];tilt=float(np.degrees(np.arccos(np.clip(-gravity_body(q[3:])[2],-1,1))));self.max_tilt[side]=max(self.max_tilt[side],tilt);bad=tilt>65 or q[2]<a.nominal_z*.55;self.fall_time[side]=self.fall_time[side]+(max(0.,float(self.d.time)-self.frames[-1]['time']) if self.frames else 0.) if bad else 0;self.fallen[side]|=self.fall_time[side]>=.3
   velocity=self.d.xmat[a.root_id].reshape(3,3).T@self.d.qvel[a.free_vadr:a.free_vadr+3]
   if self.d.time>=2:self.errors[side].append(float(abs(velocity[0]-self.cmd[0])))
   actors.append(dict(robot=a.spec.label,policy=self.keys[side],state='FALLEN' if bad else 'UNSTABLE' if tilt>25 else 'MOVING' if abs(velocity[0])>.04 else 'STANDING',base_position=q[:3].copy(),tilt_deg=tilt,height_m=float(q[2]),speed_m_s=float(np.linalg.norm(velocity[:2])),energy_positive_j=float(self.energy[side]),torque_utilization=float(np.max(abs(self.d.actuator_force[a.actuator_ids])/a.torque_limits)),command=self.cmd.copy()))
  
  if self.task=='motion':
   for side in range(2):actors[side]['reference_joint_rmse_rad']=self.motion_errors[side][-1] if self.motion_errors[side] else None
  self.frames.append(dict(time=float(self.d.time),physics_index=len(self.states)-1,actors=actors,contact_force_n=0.,contact_impulse_ns=0.,peak_force_n=0.,contacts=[],terminal=self.terminal,external_push_n=float(np.linalg.norm(self.d.xfrc_applied[:,0:3],axis=1).max())))
 def run(self):
  self.update();self.controls[0]=self.d.ctrl.copy();self.frame()
  while self.d.time<self.duration-1e-8:
   if self.steps and self.steps%10==0:self.update()
   self.d.xfrc_applied[:]=0
   if self.task=='push' and 3-1e-9<=self.d.time<3.1-1e-9:
    for i,a in enumerate(self.actors):self.d.xfrc_applied[a.root_id,1]=self.mass[i]*.25/.1;self.delivered[i,1]+=self.mass[i]*.25/.1*self.dt
   self.force_log.append(np.r_[self.d.time,*[self.d.xfrc_applied[a.root_id,:3] for a in self.actors]])
   old=self.d.time;mujoco.mj_step(self.m,self.d);self.steps+=1
   if self.d.time<=old or not np.isfinite(self.d.qpos).all():self.terminal=dict(reason='numerical_abort',winner=None,time=old);break
   for i,a in enumerate(self.actors):
    self.energy[i]+=np.maximum(self.d.actuator_force[a.actuator_ids]*self.d.qvel[a.vadr],0).sum()*self.dt;self.max_torque_violation=max(self.max_torque_violation,float(np.max(abs(self.d.actuator_force[a.actuator_ids])-a.torque_limits)))
   self.record()
   if self.steps%10==0:mujoco.mj_forward(self.m,self.d);self.frame()
  self.d.xfrc_applied[:]=0;self.terminal=self.terminal or dict(reason='test_complete',winner=None,time=float(self.d.time));self.frame();return self
 def save(self):
  mid=f'compare-{self.robot}-{self.task}-s{self.seed}-'+str(time.time_ns());folder=ROOT/'results/comparisons'/mid;folder.mkdir(parents=True)
  (folder/'scene.xml').write_text(self.xml);mujoco.mj_saveModel(self.m,str(folder/'scene.mjb'),None);encode_model(folder,(folder/'scene.mjb').read_bytes(),(self.robot,self.robot));(folder/'scene.mjb').unlink()
  np.savez_compressed(folder/'trajectory.npz',integration_states=np.stack(self.states),controls=np.stack(self.controls),timestep=self.dt,state_spec=int(STATE_SPEC),qpos_width=self.m.nq,qvel_width=self.m.nv)
  io=dict(time=self.io_times)
  for side in range(2):
   if self.io[side]:io.update({f'p{side}_observations':np.stack(self.io[side]),f'p{side}_actions':np.stack(self.actions[side])})
  np.savez_compressed(folder/'policy_io.npz',**io);np.savez_compressed(folder/'disturbance.npz',samples=np.asarray(self.force_log),columns=['time','left_fx','left_fy','left_fz','right_fx','right_fy','right_fz']);np.savez_compressed(folder/'contacts.npz',samples=np.empty((0,15)));write_json(folder/'telemetry.json',self.frames);write_json(folder/'events.json',[])
  results=[]
  for i,(a,p) in enumerate(zip(self.actors,self.policies)):
   tail=[f['actors'][i] for f in self.frames if f['time']>=self.duration-.5];results.append(dict(policy=self.keys[i],label=LABELS[self.keys[i]],mass_kg=self.mass[i],controlled_dof=len(a.names),fell=bool(self.fallen[i]),final_stable=all(v['tilt_deg']<20 and v['height_m']>.7*a.nominal_z for v in tail),max_tilt_deg=float(self.max_tilt[i]),velocity_mae_m_s=float(np.mean(self.errors[i])) if self.errors[i] and self.task!='motion' else None,reference_joint_rmse_rad=float(np.sqrt(np.mean(np.square(self.motion_errors[i])))) if self.motion_errors[i] else None,positive_work_j=float(self.energy[i]),distance_m=float(np.linalg.norm(self.d.qpos[a.free_qadr:a.free_qadr+2]-self.start[i][:2])),delivered_impulse_ns=self.delivered[i].copy(),kp=a.kp,kd=a.kd,torque_limits_nm=a.torque_limits,policy_file=str(p.path.relative_to(ROOT)) if hasattr(p,'path') else None,policy_sha256=sha256(p.path) if hasattr(p,'path') else None,policy_files=[dict(file=str(f.relative_to(ROOT)),sha256=sha256(f)) for f in getattr(p,'paths',[p.path] if hasattr(p,'path') else [])],actuator_modes=['velocity' if a.kp[j]==0 else 'position' for j in range(len(a.names))],joint_names=a.names,training_joint_names=getattr(p,'training_names',None)))
  result=dict(match_id=mid,track='comparison',rated=False,robots=[self.robot]*2,policies=self.keys,task=self.task,seed=self.seed,duration_s=float(self.d.time),result=self.terminal,actors=results,numerical_warnings=int(sum(w.number for w in self.d.warning)),torque_violation_nm=max(0.,self.max_torque_violation),fairness=dict(same_robot=True,same_physics_timestep=True,same_command_schedule=True,same_seed=True,native_controller_gains_preserved=True,push='lateral 0.1s at3s, impulse=mass*0.25 m/s',normalized_impulse_delta_v_m_s=.25 if self.task=='push' else 0,not_a_combat_ranking=True),physics=dict(engine='MuJoCo',version=mujoco.__version__,timestep_s=self.dt,control_dt_s=self.control_dt),continuous=None)
  write_json(folder/'result.json',result);write_json(folder/'config.json',dict(robot=self.robot,policies=self.keys,task=self.task,seed=self.seed,seconds=self.duration));(folder/'comparison.py').write_bytes(LOADED_COMPARISON_SOURCE);(folder/'quadruped.py').write_bytes(LOADED_QUAD_SOURCE);write_json(folder/'manifest.json',{p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'});
  from .ten_provenance import snapshot
  snapshot(folder,self.keys)
  return folder,result

# Freeze loaded source for an in-flight run when the workspace is edited later.
LOADED_COMPARISON_SOURCE=Path(__file__).read_bytes()
LOADED_QUAD_SOURCE=(Path(__file__).parent/"quadruped.py").read_bytes()
