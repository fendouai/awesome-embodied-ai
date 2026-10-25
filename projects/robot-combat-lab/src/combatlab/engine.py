from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,time,platform,gzip,shutil
import numpy as np
import mujoco
from .models import ROOT,ROBOTS,scene_xml,make_actors,initialize,configure_servos,calibrate_stance
from .policies import POLICIES,gravity_body

STATE_SPEC=mujoco.mjtState.mjSTATE_INTEGRATION

def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)

def write_json(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=jsonable))

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

PHYSICS_SOURCES={name:(Path(__file__).parent/name).read_bytes() for name in ['engine.py','models.py','policies.py']}
SOFTWARE_FINGERPRINT=hashlib.sha256(b''.join(name.encode()+data for name,data in PHYSICS_SOURCES.items())).hexdigest()

@dataclass
class MatchConfig:
    match_id:str
    track:str='ai'
    robots:tuple[str,str]=('g1_full','g1_full')
    policies:tuple[str,str]=('unitree_rl_lab','unitree_rl_mjlab')
    seed:int=0
    friction:float=.8
    timestep:float=.002
    control_dt:float=.02
    duration:float=14.
    settle:float=1.5
    postroll:float=1.5
    distance:float=2.2
    speed:float=.35
    lean:float=.035
    ring_half_width:float=2.5
    fall_tilt_deg:float=65.
    fall_height_ratio:float=.55
    fall_hold:float=.30
    rated:bool=True
    note:str=''
    pose_jitter:float=.008
    position_jitter:float=.012

class Simulation:
    def __init__(self,cfg:MatchConfig):
        self.cfg=cfg;self.xml=scene_xml(cfg.robots,cfg.timestep,cfg.friction);self.m=mujoco.MjModel.from_xml_string(self.xml);self.d=mujoco.MjData(self.m);self.actors=make_actors(self.m,cfg.robots);self.policies=[POLICIES[k](a) for k,a in zip(cfg.policies,self.actors)];configure_servos(self.m,self.actors)
        if cfg.track=='robot':
            initialize(self.m,self.d,self.actors,distance=4.,seed=0)
            calibrate_stance(self.m,self.d,self.actors)
            for p,a in zip(self.policies,self.actors):p.target=a.default.copy()
        distance=cfg.distance
        # An identical geometric clearance search prevents interpenetrating initial states.
        for _ in range(80):
            initialize(self.m,self.d,self.actors,distance=distance,seed=cfg.seed)
            rng=np.random.default_rng(cfg.seed)
            dy=float(rng.uniform(-cfg.position_jitter,cfg.position_jitter))
            for side,a in enumerate(self.actors):
                yaw=(0. if side==0 else np.pi)+float(rng.uniform(-cfg.pose_jitter,cfg.pose_jitter))
                self.d.qpos[a.free_qadr+1]=dy if side==0 else -dy
                self.d.qpos[a.free_qadr+3:a.free_qadr+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
            mujoco.mj_forward(self.m,self.d)
            if not self.inter_contacts():break
            distance+=.01
        else:raise RuntimeError('Could not find a penetration-free initial placement')
        self.actual_distance=distance
        self.step_count=0;self.control_every=round(cfg.control_dt/cfg.timestep);self.energy=np.zeros(2);self.abswork=np.zeros(2);self.impulse=0.;self.peak_force=0.;self.first_contact=None;self.contact_steps=0;self.fall_timers=np.zeros(2);self.events=[];self.frames=[];self.state_history=[];self.motor_controls=[];self.observations=[[],[]];self.policy_actions=[[],[]];self.observation_times=[];self.contacts_history=[];self.terminal=None;self.recording_enabled=True;self.last_state_label=['INITIALIZING']*2;self.command=np.zeros((2,3));self.max_tilt=np.zeros(2);self.external_root_force_peak=0.;self.torque_violation=0.;self.precontact_fall=False;self.result_time=None
        self.update_control();mujoco.mj_forward(self.m,self.d);self.record_state();self.record_frame()
    def inter_contacts(self):
        result=[];m=self.m;d=self.d;aa=self.actors
        for k,c in enumerate(d.contact[:d.ncon]):
            b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            if (b1 in aa[0].body_ids and b2 in aa[1].body_ids) or (b2 in aa[0].body_ids and b1 in aa[1].body_ids):
                f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f)
                result.append(dict(index=k,pos=c.pos.copy(),force=float(np.linalg.norm(f[:3])),force_frame=f.copy(),normal=c.frame[:3].copy(),geom1=int(c.geom1),geom2=int(c.geom2),distance=float(c.dist)))
        return result
    def desired_command(self,side):
        c=self.cfg;d=self.d;a=self.actors[side];other=self.actors[1-side]
        if self.terminal is not None:return np.zeros(3)
        if c.track=='robot':return np.array([-c.lean*np.clip((d.time-c.settle)/2,0,1),0,0])
        if d.time<c.settle:return np.zeros(3)
        xy=d.qpos[other.free_qadr:other.free_qadr+2]-d.qpos[a.free_qadr:a.free_qadr+2]
        R=d.xmat[a.root_id].reshape(3,3);local=R[:2,:2].T@xy;angle=float(np.arctan2(local[1],local[0]));dist=float(np.linalg.norm(xy))
        return np.array([c.speed*max(.15,np.cos(angle)),np.clip(local[1]*.3,-.1,.1),np.clip(angle*1.2,-.2,.2)])
    def update_control(self,manual=None):
        for i,(a,p) in enumerate(zip(self.actors,self.policies)):
            cmd=np.array(manual[i]) if manual is not None and manual[i] is not None else self.desired_command(i)
            cmd=np.clip(cmd,[-.5,-.3,-.2],[.6,.3,.2]);self.command[i]=cmd
            p.update(self.m,self.d,cmd,self.d.time)
            self.d.ctrl[a.actuator_ids]=p.target+(p.feedforward/a.kp if hasattr(p,'feedforward') else 0)
            if self.recording_enabled and hasattr(p,'last_observation'):
                self.observations[i].append(p.last_observation.copy());self.policy_actions[i].append(p.last_action.copy())
        if self.recording_enabled and any(hasattr(p,'last_observation') for p in self.policies):self.observation_times.append(float(self.d.time))
    def record_state(self):
        if not self.recording_enabled:return
        state=np.empty(mujoco.mj_stateSize(self.m,STATE_SPEC));mujoco.mj_getState(self.m,self.d,state,STATE_SPEC);self.state_history.append(state);self.motor_controls.append(self.d.ctrl.copy())
    def record_frame(self):
        d=self.d;m=self.m;c=self.cfg;contacts=self.inter_contacts();total=sum(x['force'] for x in contacts);actors=[]
        for i,a in enumerate(self.actors):
            q=d.qpos[a.free_qadr:a.free_qadr+7];g=gravity_body(q[3:7]);tilt=float(np.degrees(np.arccos(np.clip(-g[2],-1,1))));self.max_tilt[i]=max(self.max_tilt[i],tilt)
            support=0
            for cc in d.contact[:d.ncon]:
                b1,b2=int(m.geom_bodyid[cc.geom1]),int(m.geom_bodyid[cc.geom2])
                if (b1==0 and b2 in a.body_ids and 'ankle' in m.body(b2).name) or (b2==0 and b1 in a.body_ids and 'ankle' in m.body(b1).name):support+=1
            ratio=float(np.max(np.abs(d.actuator_force[a.actuator_ids])/a.torque_limits));vel=d.qvel[a.free_vadr:a.free_vadr+3];fallen=self.fall_timers[i]>=c.fall_hold
            label='FALLEN' if fallen else 'OUTSIDE' if np.max(np.abs(q[:2]))>c.ring_half_width else 'UNSTABLE' if tilt>25 else 'CONTACT' if total>5 else 'MOVING' if np.linalg.norm(vel[:2])>.06 else 'STANDING'
            if label!=self.last_state_label[i]:
                self.events.append(dict(time=float(d.time),type='state',side=i,state=label));self.last_state_label[i]=label
            actors.append(dict(robot=a.spec.label,policy=c.policies[i],state=label,base_position=q[:3].copy(),base_quat=q[3:7].copy(),com=d.subtree_com[a.root_id].copy(),tilt_deg=tilt,height_m=float(q[2]),nominal_height_m=a.nominal_z,speed_m_s=float(np.linalg.norm(vel[:2])),energy_positive_j=float(self.energy[i]),mechanical_work_abs_j=float(self.abswork[i]),torque_utilization=ratio,support_contacts=support,command=self.command[i].copy(),joint_torque_nm=d.actuator_force[a.actuator_ids].copy()))
        frame=dict(time=float(d.time),physics_index=len(self.state_history)-1,actors=actors,contact_force_n=total,contact_impulse_ns=float(self.impulse),peak_force_n=float(self.peak_force),contacts=[dict(pos=x['pos'],force_n=x['force'],normal=x['normal']) for x in contacts[:12]],terminal=self.terminal)
        self.frames.append(frame)
    def advance(self,seconds,manual=None):
        steps=round(seconds/self.cfg.timestep)
        for _ in range(steps):
            if self.step_count and self.step_count%self.control_every==0:self.update_control(manual)
            old_time=self.d.time;mujoco.mj_step(self.m,self.d);self.step_count+=1
            if not np.isfinite(self.d.qpos).all() or self.d.time<old_time:
                self.terminal=dict(reason='numerical_failure',winner=None,time=float(old_time));break
            for i,a in enumerate(self.actors):
                power=self.d.actuator_force[a.actuator_ids]*self.d.qvel[a.vadr];self.energy[i]+=np.maximum(power,0).sum()*self.cfg.timestep;self.abswork[i]+=np.abs(power).sum()*self.cfg.timestep
                ratio=np.abs(self.d.actuator_force[a.actuator_ids])-a.torque_limits;self.torque_violation=max(self.torque_violation,float(np.max(ratio)))
                self.external_root_force_peak=max(self.external_root_force_peak,float(np.max(np.abs(self.d.xfrc_applied[a.body_ids]))))
                tilt=np.degrees(np.arccos(np.clip(-gravity_body(self.d.qpos[a.free_qadr+3:a.free_qadr+7])[2],-1,1)))
                below=self.d.qpos[a.free_qadr+2]<a.nominal_z*self.cfg.fall_height_ratio
                self.fall_timers[i]=self.fall_timers[i]+self.cfg.timestep if (below or tilt>self.cfg.fall_tilt_deg) else 0.
            contacts=self.inter_contacts();force=sum(x['force'] for x in contacts);self.impulse+=force*self.cfg.timestep;self.peak_force=max(self.peak_force,force)
            if contacts:
                self.contact_steps+=1
                if self.first_contact is None:self.first_contact=float(self.d.time);self.events.append(dict(time=self.first_contact,type='first_contact',force_n=force))
                if self.recording_enabled:
                    for x in contacts:self.contacts_history.append([self.d.time,x['geom1'],x['geom2'],*x['pos'],*x['force_frame'],*x['normal']])
            if self.terminal is None and self.d.time>=self.cfg.settle:
                lost=[self.fall_timers[i]>=self.cfg.fall_hold or np.max(np.abs(self.d.qpos[a.free_qadr:a.free_qadr+2]))>self.cfg.ring_half_width for i,a in enumerate(self.actors)]
                if any(lost):
                    self.precontact_fall=self.first_contact is None
                    self.terminal=dict(reason='double_loss' if all(lost) else 'fall' if any(self.fall_timers>=self.cfg.fall_hold) else 'boundary_exit',winner=None if all(lost) else (1 if lost[0] else 0),time=float(self.d.time),contact_before_result=self.first_contact is not None)
                elif self.d.time>=self.cfg.duration:self.terminal=dict(reason='time_limit',winner=None,time=float(self.d.time),contact_before_result=self.first_contact is not None)
                if self.terminal:self.events.append(dict(time=float(self.d.time),type='result',result=self.terminal));self.result_time=float(self.d.time)
            if self.step_count%self.control_every==0:mujoco.mj_forward(self.m,self.d)
            self.record_state()
            if self.step_count%self.control_every==0:self.record_frame()
            if self.terminal and (self.d.time-self.terminal['time']>=self.cfg.postroll or self.terminal['reason']=='numerical_failure'):return False
        return True
    def run(self):
        while self.advance(.2):pass
        return self
    def save(self,base=None):
        source_dir=ROOT/'results/software'/SOFTWARE_FINGERPRINT
        source_dir.mkdir(parents=True,exist_ok=True)
        for name,data in PHYSICS_SOURCES.items():
            target=source_dir/name
            if not target.exists():target.write_bytes(data)
        folder=(Path(base) if base else ROOT/'results'/'matches')/self.cfg.match_id;folder.mkdir(parents=True,exist_ok=True)
        (folder/'scene.xml').write_text(self.xml);mujoco.mj_saveModel(self.m,str(folder/'scene.mjb'),None)
        from .storage import encode_model
        encode_model(folder,(folder/'scene.mjb').read_bytes(),self.cfg.robots)
        (folder/'scene.mjb').unlink()
        states=np.stack(self.state_history);controls=np.stack(self.motor_controls)
        np.savez_compressed(folder/'trajectory.npz',integration_states=states,controls=controls,timestep=self.cfg.timestep,state_spec=int(STATE_SPEC),qpos_width=self.m.nq,qvel_width=self.m.nv)
        arrays={'time':np.array(self.observation_times)}
        for i in [0,1]:
            if self.observations[i]:arrays[f'p{i}_observations']=np.stack(self.observations[i]);arrays[f'p{i}_actions']=np.stack(self.policy_actions[i])
        np.savez_compressed(folder/'policy_io.npz',**arrays)
        np.savez_compressed(folder/'contacts.npz',samples=np.array(self.contacts_history,dtype=float).reshape(-1,15),columns=np.array(['time','geom1','geom2','x','y','z','fn','ft1','ft2','tn','tt1','tt2','nx','ny','nz']))
        write_json(folder/'telemetry.json',self.frames);write_json(folder/'events.json',self.events);write_json(folder/'config.json',asdict(self.cfg))
        info=[]
        for a,p in zip(self.actors,self.policies):
            info.append(dict(robot=a.spec.label,key=a.spec.key,model_scope=a.spec.scope,mass_kg=float(self.m.body_mass[a.body_ids].sum()),controlled_dof=len(a.joint_ids),nominal_root_height=a.nominal_z,joint_names=a.names,kp=a.kp,kd=a.kd,torque_limits_nm=a.torque_limits,policy=self.cfg.policies[len(info)],policy_file=str(p.path.relative_to(ROOT)) if hasattr(p,'path') else None,policy_sha256=sha256(p.path) if hasattr(p,'path') else None))
        numerical_warning_count=int(sum(w.number for w in self.d.warning))
        result=dict(software_fingerprint=SOFTWARE_FINGERPRINT,provenance_sha256=sha256(ROOT/'assets/provenance.json'),match_id=self.cfg.match_id,track=self.cfg.track,rated=self.cfg.rated and self.first_contact is not None and not self.precontact_fall and self.terminal['reason']!='numerical_failure',configured_rated=self.cfg.rated,robots=self.cfg.robots,policies=self.cfg.policies,seed=self.cfg.seed,friction=self.cfg.friction,actual_initial_distance=self.actual_distance,duration_s=float(self.d.time),result=self.terminal,first_contact_s=self.first_contact,contact_duration_s=self.contact_steps*self.cfg.timestep,contact_impulse_ns=self.impulse,peak_contact_force_n=self.peak_force,positive_mechanical_work_j=self.energy,max_tilt_deg=self.max_tilt,external_root_force_peak=self.external_root_force_peak,max_torque_limit_violation_nm=max(0.,self.torque_violation),numerical_warnings=numerical_warning_count,actors=info,physics=dict(engine='MuJoCo',version=mujoco.__version__,timestep_s=self.cfg.timestep,gravity=self.m.opt.gravity,integrator='implicitfast',solver='Newton',iterations=int(self.m.opt.iterations),self_collision='source self contacts retained; added massless convex rubber-hand cross/ground proxies',derived_hand_colliders=[self.m.geom(i).name for i in range(self.m.ngeom) if (self.m.geom(i).name or '').endswith('_contact_proxy')],no_hidden_root_forces=True,ground_friction=self.cfg.friction),archive=dict(physics_frames=len(states),control_frames=len(self.frames),state_spec=int(STATE_SPEC),replay='restore complete integration state; never reconstruct from animation'),created_unix=time.time())
        write_json(folder/'result.json',result)
        hashes={f.name:sha256(f) for f in folder.iterdir() if f.is_file() and f.name!='manifest.json'};write_json(folder/'manifest.json',hashes)
        return folder,result
