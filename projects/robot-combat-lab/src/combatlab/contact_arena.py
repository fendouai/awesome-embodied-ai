"""Opponent-conditioned contact environment over source locomotion policies.
All high-level actions are learned commands/residuals, applied solely via joint actuators.
"""
from pathlib import Path
import json,copy,xml.etree.ElementTree as ET
import numpy as np,mujoco
from .models import ROOT,ROBOTS,Actor,scene_xml,configure_servos
from .comparison import NeutralPD
from .policies import POLICIES,gravity_body
from .quadruped import QuadPolicy
from . import platform10
from .k1_policy import register,K1Walk
from .research_archive import workspace,atomic_json,event,digest
from .engine import STATE_SPEC

BODIES=['g1_full','h1_full','h1_2_full','k1','go2','a1','go1','b2','go2w','b2w']
OBS_DIM=131;ACTION_DIM=32

def bind(m,side,p):
 pre=f'p{side}/';b=np.array([i for i in range(1,m.nbody) if (m.body(i).name or '').startswith(pre)]);j=np.array([m.joint(pre+n).id for n in p['names']]);u=np.array([next(k for k in range(m.nu) if m.actuator_trnid[k,0]==jid) for jid in j]);free=next(i for i in range(m.njnt) if m.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[i] in b)
 return Actor(pre,ROBOTS[p['key']],int(m.jnt_bodyid[free]),free,int(m.jnt_qposadr[free]),int(m.jnt_dofadr[free]),b,j,m.jnt_qposadr[j].copy(),m.jnt_dofadr[j].copy(),u,np.array(p['limits']),np.array(p['default']),np.array(p['kp']),np.array(p['kd']),p['names'])

def base_policy(a,key):
 if key=='bhuman_k1_walk':return K1Walk(a)
 if a.spec.key in platform10.NEW_ROBOTS:return platform10.create(a,key)
 if a.spec.key in ['go2','a1']:return QuadPolicy(a,key)
 return POLICIES[key](a)

class RawTrace:
 def __init__(self,folder,m,metadata):
  self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False);self.m=m;self.index=0;self.rows=[];self.controls=[];self.episode=[];self.aux=[];self.contacts=[];self.motor_forces=[];self.step=0
  mujoco.mj_saveModel(m,str(self.folder/'scene.mjb'),None);atomic_json(self.folder/'metadata.json',dict(**metadata,format_version=2,actor_perception_mode='PRIVILEGED_SIMULATOR_STATE',camera_pixels_used=False,real_sensor_equivalence=False,contact_columns=['global_physics_frame','step_end_time_s','geom1','geom2','world_x_m','world_y_m','world_z_m','normal_force_n','tangent1_force_n','tangent2_force_n','world_normal_x','world_normal_y','world_normal_z','pre_step_relative_approach_m_s'],mujoco_version=mujoco.__version__,state_spec=int(STATE_SPEC),scene_sha256=digest(self.folder/'scene.mjb'),sampling='Every physics step; post-step integration state and actual actuator control. Reset states recorded with new episode number. Native K1 filtered motor velocities recorded separately.'))
 def record(self,d,episode,aux,contacts=()):
  state=np.empty(mujoco.mj_stateSize(self.m,STATE_SPEC));mujoco.mj_getState(self.m,d,state,STATE_SPEC);self.rows.append(state);self.controls.append(d.ctrl.copy());self.episode.append(episode);self.aux.append(aux.copy());self.motor_forces.append(d.actuator_force.copy())
  for row in contacts:self.contacts.append([self.step,*row])
  self.step+=1
  if len(self.rows)>=2000:self.flush()
 def flush(self):
  if not self.rows:return
  path=self.folder/f'physics-{self.index:06}.npz';np.savez_compressed(path,integration_states=np.array(self.rows),controls=np.array(self.controls),episode=np.array(self.episode),native_motor_velocity=np.array(self.aux),actuator_force=np.array(self.motor_forces),contacts=np.array(self.contacts).reshape(-1,14));row=dict(chunk=self.index,file=path.name,sha256=digest(path),frames=len(self.rows),last_global_frame=self.step-1)
  with (self.folder/'chunks.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
  self.index+=1;self.rows=[];self.controls=[];self.episode=[];self.aux=[];self.contacts=[];self.motor_forces=[]

def contact_xml(keys,profiles):
 """Preserve each source self-collision bit relation; add separate cross-body tags."""
 r=ET.fromstring(scene_xml(tuple(keys),.002,.8))
 for side,profile in enumerate(profiles):
  prefix=f'p{side}/';shift=1+14*side;tag=1<<(29+side);other=1<<(30-side)
  for body in r.findall('.//worldbody//body'):
   name=body.get('name','')
   if not name.startswith(prefix):continue
   source=profile['source_collision_groups'].get(name[len(prefix):],[])
   for ordinal,g in enumerate(body.findall('geom')):
    if ordinal>=len(source) or '_contact_proxy' in g.get('name',''):
     # Existing massless visual-hand proxy is inter-body/ground only.
     if g.get('contype','0')!='0':g.set('contype',str(tag));g.set('conaffinity',str(1|other))
     continue
    old=source[ordinal];ct=old['contype'];ca=old['conaffinity']
    if ct==0 and ca==0:g.set('contype','0');g.set('conaffinity','0');continue
    if ct<0 or ct>=1<<14:raise ValueError('Source collision bit width requires a dedicated mapping')
    g.set('contype',str((ct<<shift)|tag));g.set('conaffinity',str(((ca&0x3fff)<<shift)|1|other))
 ground=r.find("worldbody/geom[@name='ground']");ground.set('conaffinity',str((1<<29)|(1<<30)));return ET.tostring(r,encoding='unicode')

class ContactArena:
 def __init__(self,left,right=None,seed=0,trace_folder=None,episode_s=6,initial_jitter=False,reward_version=1,yaw_jitter=.04,lateral_jitter=.02,gap_range=(.3,.5),scene_transform=None):
  register();self.keys=[left,right or left];self.profiles=[json.loads((workspace()/'09_models/profiles'/(k+'.json')).read_text()) for k in self.keys];self.xml=contact_xml(self.keys,self.profiles);self.xml=scene_transform(self.xml,self.keys) if scene_transform else self.xml;self.m=mujoco.MjModel.from_xml_string(self.xml);self.d=mujoco.MjData(self.m);self.dt=.002;self.control_dt=.02;self.rng=np.random.default_rng(seed);self.seed=seed;self.episode_s=episode_s;self.initial_jitter=initial_jitter;self.reward_version=reward_version;self.yaw_jitter=yaw_jitter;self.lateral_jitter=lateral_jitter;self.gap_range=gap_range;self.episode=-1;self.actors=[bind(self.m,i,p) for i,p in enumerate(self.profiles)];self.policies=[base_policy(a,p['base_policy']) for a,p in zip(self.actors,self.profiles)];configure_servos(self.m,self.actors);platform10.configure_wheels(self.m,self.actors)
  for a,p in zip(self.actors,self.profiles):
   if p['key']=='k1':
    for i,u in enumerate(a.actuator_ids):self.m.actuator_gainprm[u,:]=0;self.m.actuator_gainprm[u,0]=1;self.m.actuator_biasprm[u,:]=0;self.m.actuator_ctrlrange[u]=[-a.torque_limits[i],a.torque_limits[i]]
  self.trace=None;self.physics_observer=None
  self.reset()
  if trace_folder:
   self.trace=RawTrace(trace_folder,self.m,dict(robots=self.keys,seed=seed,profiles=self.profiles,observation_dim=OBS_DIM,action_dim=ACTION_DIM,episode_s=episode_s,initial_jitter=initial_jitter,reward_version=reward_version,yaw_jitter=yaw_jitter,lateral_jitter=lateral_jitter,gap_range=gap_range,source_self_collision_bit_relations_preserved=True,external_root_forces=False,scripted_strikes=False));(self.trace.folder/'scene.xml').write_text(self.xml);self.record()
 def reset(self):
  mujoco.mj_resetData(self.m,self.d);self.episode+=1;self.steps=0;self.last_actions=np.zeros((2,ACTION_DIM));self.filtered=np.zeros((2,29));self.last_targets=[a.default.copy() for a in self.actors];self.bad_time=np.zeros(2);self.total_contacts=np.zeros(2);self.last_contact=np.full(2,-1.);self.native_updates=[]
  # Fresh native policy histories are essential; checkpoint loading is cached by providers.
  if self.episode:
   self.policies=[base_policy(a,p['base_policy']) for a,p in zip(self.actors,self.profiles)]
  for side,a in enumerate(self.actors):
   q=a.free_qadr;self.d.qpos[q:q+3]=[(-1 if side==0 else 1)*1.3,0,1.5];self.d.qpos[q+3:q+7]=[1,0,0,0] if side==0 else [0,0,0,1];self.d.qpos[a.qadr]=a.default+(self.rng.normal(0,.002,len(a.names)) if self.initial_jitter else 0)
   if self.initial_jitter:
    yaw=(0 if side==0 else np.pi)+self.rng.uniform(-self.yaw_jitter,self.yaw_jitter);self.d.qpos[q+3:q+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)];self.d.qpos[q+1]=self.rng.uniform(-self.lateral_jitter,self.lateral_jitter)
  mujoco.mj_forward(self.m,self.d);bounds=[]
  for a in self.actors:
   low=100.;xmin=100.;xmax=-100.
   for g in range(self.m.ngeom):
    if self.m.geom_bodyid[g] not in a.body_ids or not self.m.geom_contype[g]:continue
    pos=self.d.geom_xpos[g];R=self.d.geom_xmat[g].reshape(3,3);sz=self.m.geom_size[g];kind=self.m.geom_type[g]
    if kind==mujoco.mjtGeom.mjGEOM_MESH:
     mid=self.m.geom_dataid[g];verts=self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid]+self.m.mesh_vertnum[mid]]@R.T+pos;lo=verts.min(0);hi=verts.max(0)
    else:
     ext=abs(R)@sz if kind==mujoco.mjtGeom.mjGEOM_BOX else (abs(R[:,2])*sz[1]+sz[0] if kind in [mujoco.mjtGeom.mjGEOM_CAPSULE,mujoco.mjtGeom.mjGEOM_CYLINDER] else np.full(3,sz[0]));lo=pos-ext;hi=pos+ext
    low=min(low,lo[2]);xmin=min(xmin,lo[0]);xmax=max(xmax,hi[0])
   self.d.qpos[a.free_qadr+2]+=.003-low;a.nominal_z=float(self.d.qpos[a.free_qadr+2]);bounds.append((xmin-self.d.qpos[a.free_qadr],xmax-self.d.qpos[a.free_qadr]))
  gap=self.rng.uniform(*self.gap_range) if self.initial_jitter else .35;distance=bounds[0][1]-bounds[1][0]+gap
  for side,a in enumerate(self.actors):self.d.qpos[a.free_qadr]=(-.5 if side==0 else .5)*distance
  mujoco.mj_forward(self.m,self.d);self.previous_distance=self.distance();self.initial_contact_distance=self.previous_distance-gap;self.native_updates=[]
  if self.trace:self.record()
  return self.observations()
 def distance(self):return float(np.linalg.norm(self.d.qpos[self.actors[1].free_qadr:self.actors[1].free_qadr+2]-self.d.qpos[self.actors[0].free_qadr:self.actors[0].free_qadr+2]))
 def observations(self):
  rows=[]
  for side,a in enumerate(self.actors):
   b=self.actors[1-side];R=self.d.xmat[a.root_id].reshape(3,3);q=self.d.qpos[a.free_qadr:a.free_qadr+7];pad=lambda v:np.pad(v,(0,29-len(v)));rel=R.T@(self.d.qpos[b.free_qadr:b.free_qadr+3]-q[:3]);vel=R.T@self.d.qvel[a.free_vadr:a.free_vadr+3];ov=R.T@self.d.qvel[b.free_vadr:b.free_vadr+3];one=np.eye(10)[BODIES.index(a.spec.key)];other=np.eye(10)[BODIES.index(b.spec.key)];obs=np.r_[gravity_body(q[3:]),vel/3,self.d.qvel[a.free_vadr+3:a.free_vadr+6]/5,pad(self.d.qpos[a.qadr]-a.default)/np.pi,pad(self.d.qvel[a.vadr])/10,self.last_actions[side],rel/5,ov/3,q[2]/max(.1,a.nominal_z),self.d.qpos[b.free_qadr+2]/max(.1,b.nominal_z),gravity_body(self.d.qpos[b.free_qadr+3:b.free_qadr+7]),self.total_contacts[side]/100,one,other];assert len(obs)==OBS_DIM;rows.append(np.clip(obs,-10,10))
  return np.array(rows,dtype=np.float32)
 def action_mask(self):return np.array([[1.]*(3+len(a.names))+[0.]*(29-len(a.names)) for a in self.actors],dtype=np.float32)
 def record(self,contacts=()):
  if self.trace:self.trace.record(self.d,self.episode,self.filtered,contacts)
 def step(self,raw_actions):
  actions=np.tanh(np.asarray(raw_actions,dtype=float))*self.action_mask();reward=np.zeros(2);registered=np.zeros(2);all_contacts=[];normal_impulse=0.;contact_physics_steps=0;normal_peak=0.
  for _ in range(10):
   for side,(a,p) in enumerate(zip(self.actors,self.policies)):
    if self.steps%round(getattr(p,'control_dt',.02)/self.dt)==0:
     scale=np.array([.12,.05,.3]) if a.spec.key=='k1' else np.array([.5,.3,1.]);cmd=actions[side,:3]*scale;p.update(self.m,self.d,cmd,self.d.time);self.last_targets[side]=p.target.copy();self.native_updates.append(dict(time=float(self.d.time),side=side,observation=np.asarray(getattr(p,'last_observation',[])).tolist(),native_action=np.asarray(getattr(p,'last_action',[])).tolist(),command=cmd.tolist(),command_units='step_m_step_rad' if a.spec.key=='k1' else 'm_s_rad_s',target=p.target.tolist()))
    scales=np.array([.25 if any(w in n.lower() for w in ['shoulder','elbow','wrist']) else .5 if 'wheel' in n.lower() else .035 for n in a.names]);target=self.last_targets[side]+actions[side,3:3+len(a.names)]*scales
    for i,j in enumerate(a.joint_ids):
     if self.m.jnt_limited[j]:target[i]=np.clip(target[i],*self.m.jnt_range[j])
    if a.spec.key=='k1':
     params=self.profiles[side]['native_motor_parameters'];lp=np.array([float(v['velocity_low_pass']) for v in params]);assert all(float(v['delay_s'])==0 for v in params);self.filtered[side,:len(a.names)]=lp*self.d.qvel[a.vadr]+(1-lp)*self.filtered[side,:len(a.names)];self.d.ctrl[a.actuator_ids]=np.clip(a.kp*(target-self.d.qpos[a.qadr])-a.kd*self.filtered[side,:len(a.names)],-a.torque_limits,a.torque_limits)
    else:self.d.ctrl[a.actuator_ids]=target
   assert not self.d.xfrc_applied.any() and not self.d.qfrc_applied.any();pre=self.d.qvel.copy();mujoco.mj_step(self.m,self.d);self.steps+=1;rows=[];scoring=[]
   for ci,c in enumerate(self.d.contact[:self.d.ncon]):
    b1=int(self.m.geom_bodyid[c.geom1]);b2=int(self.m.geom_bodyid[c.geom2]);s1=0 if b1 in self.actors[0].body_ids else 1 if b1 in self.actors[1].body_ids else None;s2=0 if b2 in self.actors[0].body_ids else 1 if b2 in self.actors[1].body_ids else None
    if s1 is None or s2 is None or s1==s2:continue
    f=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,ci,f);j1=np.zeros((3,self.m.nv));j2=np.zeros_like(j1);rot=np.zeros_like(j1);mujoco.mj_jac(self.m,self.d,j1,rot,c.pos,b1);mujoco.mj_jac(self.m,self.d,j2,rot,c.pos,b2);speed1=float((j1@pre)@c.frame[:3]);speed2=float(-(j2@pre)@c.frame[:3]);impulse=max(0,float(f[0]))*self.dt
    for side,speed in [(s1,speed1),(s2,speed2)]:
     scoring.append(dict(side=side,normal_force_n=max(0.,float(f[0])),approach_m_s=max(0.,speed)))
     if speed>.05:registered[side]+=impulse
    rows.append([float(self.d.time),int(c.geom1),int(c.geom2),*c.pos,*f[:3],*c.frame[:3],float((j1@pre-j2@pre)@c.frame[:3])])
   force=sum(max(0.,r[6]) for r in rows);normal_impulse+=force*self.dt;contact_physics_steps+=int(bool(rows));normal_peak=max(normal_peak,force);self.record(rows);all_contacts.extend(rows)
   if self.steps%10==0:mujoco.mj_forward(self.m,self.d)
   if self.physics_observer is not None and self.physics_observer(self,scoring):break
  distance=self.distance();progress=np.clip(self.previous_distance-distance,-.2,.2);self.previous_distance=distance;good=[];reward_terms=[]
  for side,a in enumerate(self.actors):
   tilt=float(np.degrees(np.arccos(np.clip(-gravity_body(self.d.qpos[a.free_qadr+3:a.free_qadr+7])[2],-1,1))));up=tilt<55 and self.d.qpos[a.free_qadr+2]>.55*a.nominal_z;good.append(up);self.bad_time[side]=0 if up else self.bad_time[side]+self.control_dt;self.total_contacts[side]+=registered[side];terms=dict(progress=2*float(progress),upright=.05*self.control_dt*float(up),instability=-.2*self.control_dt*float(not up),action_change=-.002*float(np.mean((actions[side]-self.last_actions[side])**2)),initiated_contact=.02*float(min(registered[side],10)) if up else 0.,alignment=0.,proximity=0.,contact_presence=0.)
   if self.reward_version>=2 and up:
    opponent=self.actors[1-side];relative=self.d.xmat[a.root_id].reshape(3,3).T@(self.d.qpos[opponent.free_qadr:opponent.free_qadr+3]-self.d.qpos[a.free_qadr:a.free_qadr+3]);terms['alignment']=.15*self.control_dt*float(np.clip(relative[0]/max(distance,.1),-1,1));terms['proximity']=.1*self.control_dt*float(np.exp(-max(0.,distance-self.initial_contact_distance)/.5));terms['contact_presence']=.1*self.control_dt*bool(all_contacts)
   reward[side]=sum(terms.values());reward_terms.append(terms)
  fallen=self.bad_time>=.3
  if fallen.any():reward+=3*(fallen[::-1].astype(float)-fallen.astype(float))
  self.last_actions=actions;terminated=bool(fallen.any() or not np.isfinite(self.d.qpos).all() or any(w.number for w in self.d.warning));truncated=self.d.time>=self.episode_s-1e-8;info=dict(time=float(self.d.time),progress_m=float(progress),initiated_contact_impulse_ns=registered.tolist(),fallen=fallen.tolist(),distance_m=distance,contacts=len(all_contacts),contact_normal_impulse_ns=normal_impulse,contact_physics_steps=contact_physics_steps,contact_normal_peak_n=normal_peak,native_updates=self.native_updates,reward_terms=reward_terms,reward_version=self.reward_version);self.native_updates=[]
  return self.observations(),reward.astype(np.float32),terminated,truncated,info
 def close(self):
  if self.trace:self.trace.flush()
