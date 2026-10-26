"""Continuous contact simulation with honest unknown damage and resumable chunks."""
from dataclasses import dataclass,asdict
from pathlib import Path
from collections import deque
import hashlib,json,copy,shutil
import numpy as np,mujoco,torch
from . import control_upgrade
from .models import ROOT
from .engine import Simulation,MatchConfig,STATE_SPEC,write_json,sha256
from .policies import gravity_body
from .hardware import HardwareMonitor,ThermalParameters

ENDURANCE_SOURCE=Path(__file__).read_bytes()
HARDWARE_SOURCE=(Path(__file__).parent/'hardware.py').read_bytes()
ENDURANCE_FINGERPRINT=hashlib.sha256(ENDURANCE_SOURCE+HARDWARE_SOURCE+control_upgrade.ADAPTER_SOURCE).hexdigest()

@dataclass
class EnduranceConfig(MatchConfig):
    track: str='continuous'
    rated: bool=False
    max_sim_time: float=120.
    chunk_seconds: float=30.

def pack(x):
    if isinstance(x,np.ndarray):return dict(array=x.tolist(),dtype=str(x.dtype))
    if isinstance(x,deque):return dict(deque=[pack(v) for v in x],maxlen=x.maxlen)
    if isinstance(x,list):return [pack(v) for v in x]
    if isinstance(x,(float,int,bool,str)) or x is None:return x
    raise TypeError(type(x).__name__)
def unpack(x):
    if isinstance(x,dict) and 'array' in x:return np.asarray(x['array'],dtype=x['dtype'])
    if isinstance(x,dict) and 'deque' in x:return deque([unpack(v) for v in x['deque']],maxlen=x['maxlen'])
    if isinstance(x,list):return [unpack(v) for v in x]
    return x

class ContinuousSimulation(Simulation):
    def __init__(self,cfg:EnduranceConfig,parameters=None):
        # Base constructor calls record_frame, before our monitor is constructed.
        super().__init__(cfg);self.hardware=HardwareMonitor(self.m,self.actors,parameters);self.run_id=cfg.match_id;self.parts=[];self.chunk_start=float(self.d.time);self.hw_history=[];self.limit_history=[];self.on_chunk_saved=None;self.save_on_chunks=True
        self.frames=[];self.record_frame();self.record_hardware()

    def record_hardware(self):
        self.hw_history.append(self.hardware.temperature.copy());self.limit_history.append(self.m.actuator_forcerange.copy())

    def record_frame(self):
        super().record_frame()
        if hasattr(self,'hardware'):
            f=self.frames[-1];f['protocol']='continuous';f['hardware']=[self.hardware.summary(i) for i in [0,1]];f['damage_prediction_available']=bool(self.hardware.parameters);f['fall_is_terminal']=False

    def advance(self,seconds,manual=None):
        if self.terminal is not None:return False
        for _ in range(round(seconds/self.cfg.timestep)):
            if self.step_count and self.step_count%self.control_every==0:self.update_control(manual)
            old=float(self.d.time);mujoco.mj_step(self.m,self.d);self.step_count+=1
            if not np.isfinite(self.d.qpos).all() or self.d.time<old:
                self.terminal=dict(reason='numerical_abort',winner=None,time=old,hardware_damage=False);return False
            for i,a in enumerate(self.actors):
                power=self.d.actuator_force[a.actuator_ids]*self.d.qvel[a.vadr];self.energy[i]+=np.maximum(power,0).sum()*self.cfg.timestep;self.abswork[i]+=np.abs(power).sum()*self.cfg.timestep
                self.torque_violation=max(self.torque_violation,float(np.max(abs(self.d.actuator_force[a.actuator_ids])-a.torque_limits)))
                self.external_root_force_peak=max(self.external_root_force_peak,float(np.max(abs(self.d.xfrc_applied[a.body_ids]))))
                tilt=np.degrees(np.arccos(np.clip(-gravity_body(self.d.qpos[a.free_qadr+3:a.free_qadr+7])[2],-1,1)));low=self.d.qpos[a.free_qadr+2]<a.nominal_z*self.cfg.fall_height_ratio
                self.fall_timers[i]=self.fall_timers[i]+self.cfg.timestep if low or tilt>self.cfg.fall_tilt_deg else 0.
            contacts=self.inter_contacts();force=sum(c['force'] for c in contacts);self.impulse+=force*self.cfg.timestep;self.peak_force=max(self.peak_force,force)
            if contacts:
                self.contact_steps+=1
                if self.first_contact is None:self.first_contact=float(self.d.time);self.events.append(dict(type='first_contact',time=self.first_contact))
                for c in contacts:self.contacts_history.append([self.d.time,c['geom1'],c['geom2'],*c['pos'],*c['force_frame'],*c['normal']])
            self.hardware.step(self.d.actuator_force.copy(),self.cfg.timestep,float(self.d.time))
            lost=self.hardware.loss_of_all_actuation()
            if any(lost):self.terminal=dict(reason='parameterized_all_actuators_failed',winner=None if all(lost) else 1 if lost[0] else 0,time=float(self.d.time),basis='unvalidated parameterized motor model',real_hardware_damage_confirmed=False)
            elif self.step_count%500==0 and shutil.disk_usage(ROOT).free<512*1024**2:self.terminal=dict(reason='storage_pause',winner=None,time=float(self.d.time),hardware_damage=False,resumable=True)
            elif self.d.time>=self.cfg.max_sim_time-1e-9:self.terminal=dict(reason='budget_pause',winner=None,time=float(self.d.time),hardware_damage=False,resumable=True)
            if self.step_count%self.control_every==0:mujoco.mj_forward(self.m,self.d)
            self.record_state();self.record_hardware()
            if self.step_count%self.control_every==0 or self.terminal:self.record_frame()
            if self.terminal:return False
            if self.save_on_chunks and self.d.time-self.chunk_start>=self.cfg.chunk_seconds-1e-9:
                folder,_=self.save(ROOT/'results/continuous')
                if self.on_chunk_saved:self.on_chunk_saved(folder)
        return True

    def resume(self,additional_seconds=120.):
        if self.terminal and self.terminal['reason'] not in ['budget_pause','manual_pause','storage_pause']:raise ValueError('Only paused runs may resume')
        self.cfg.max_sim_time=float(self.d.time)+additional_seconds;self.terminal=None

    def runtime(self):
        policy=[]
        for p in self.policies:
            fields={k:pack(getattr(p,k)) for k in ['last_action','target','feedforward','hist','phase','phase_time','standing'] if hasattr(p,k)}
            buffers={k:pack(v.detach().numpy()) for k,v in p.net.named_buffers()} if isinstance(getattr(p,'net',None),torch.nn.Module) else {}
            policy.append(dict(fields=fields,buffers=buffers))
        return dict(run_id=self.run_id,parts=self.parts,step_count=self.step_count,energy=self.energy.tolist(),abswork=self.abswork.tolist(),impulse=self.impulse,peak_force=self.peak_force,first_contact=self.first_contact,contact_steps=self.contact_steps,fall_timers=self.fall_timers.tolist(),max_tilt=self.max_tilt.tolist(),last_state_label=self.last_state_label,command=self.command.tolist(),terminal=self.terminal,nominal_heights=[a.nominal_z for a in self.actors],policy=policy,hardware=self.hardware.state())

    def save(self,base=None):
        base=Path(base or ROOT/'results/continuous');part=f'{self.run_id}-part{len(self.parts)+1:04d}';original_id=self.cfg.match_id;actual_terminal=self.terminal
        self.cfg.match_id=part
        if self.terminal is None:self.terminal=dict(reason='checkpoint',winner=None,time=float(self.d.time),hardware_damage=False,resumable=True)
        self.record_frame()
        folder,result=super().save(base);self.cfg.match_id=original_id;self.terminal=actual_terminal
        np.savez_compressed(folder/'hardware.npz',temperature_c=np.asarray(self.hw_history),actuator_forcerange=np.asarray(self.limit_history))
        (folder/'endurance.py').write_bytes(ENDURANCE_SOURCE);(folder/'hardware.py').write_bytes(HARDWARE_SOURCE);(folder/'control_upgrade.py').write_bytes(control_upgrade.ADAPTER_SOURCE)
        self.parts.append(dict(path=str(folder.relative_to(ROOT)),start=self.chunk_start,end=float(self.d.time)))
        write_json(folder/'runtime.json',self.runtime());write_json(folder/'continuous-config.json',asdict(self.cfg));result['continuous']=dict(run_id=self.run_id,part=len(self.parts),start_s=self.chunk_start,end_s=float(self.d.time),fall_is_terminal=False,time_limit_is_loss=False,damage_prediction='parameterized_unvalidated' if self.hardware.parameters else 'not_calibrated',endurance_fingerprint=ENDURANCE_FINGERPRINT,hardware=[self.hardware.summary(i) for i in [0,1]],resume_checkpoint='runtime.json');write_json(folder/'result.json',result)
        write_json(folder/'manifest.json',{p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'})
        write_json(base/f'{self.run_id}-index.json',dict(run_id=self.run_id,parts=self.parts,terminal=self.terminal,damage_prediction=result['continuous']['damage_prediction']))
        # Keep the boundary state. New segment starts at the exact previous endpoint.
        self.state_history=self.state_history[-1:];self.motor_controls=self.motor_controls[-1:];self.hw_history=self.hw_history[-1:];self.limit_history=self.limit_history[-1:];self.frames=[];self.events=[];self.contacts_history=[];self.observations=[[],[]];self.policy_actions=[[],[]];self.observation_times=[];self.chunk_start=float(self.d.time);self.record_frame()
        return folder,result

    @classmethod
    def restore(cls,folder):
        from .storage import load_model
        from .models import make_actors
        folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
        for name,h in manifest.items():
            if sha256(folder/name)!=h:raise ValueError(f'Archive hash mismatch: {name}')
        result=json.loads((folder/'result.json').read_text())
        if result['continuous']['endurance_fingerprint']!=ENDURANCE_FINGERPRINT:raise ValueError('Loaded endurance source differs from checkpoint')
        config=EnduranceConfig(**json.loads((folder/'continuous-config.json').read_text()));runtime=json.loads((folder/'runtime.json').read_text());params={k:ThermalParameters(**v) for k,v in runtime['hardware']['parameters'].items()};s=cls(config,params)
        s.m=load_model(folder);s.d=mujoco.MjData(s.m);s.actors=make_actors(s.m,tuple(config.robots));s.policies=[control_upgrade.POLICIES[k](a) for k,a in zip(config.policies,s.actors)];s.hardware=HardwareMonitor(s.m,s.actors,params)
        raw=np.load(folder/'trajectory.npz');mujoco.mj_setState(s.m,s.d,raw['integration_states'][-1],STATE_SPEC)
        for key in ['run_id','parts','step_count','impulse','peak_force','first_contact','contact_steps','last_state_label','terminal']:setattr(s,key,runtime[key])
        for key in ['energy','abswork','fall_timers','max_tilt','command']:setattr(s,key,np.asarray(runtime[key]))
        for a,height in zip(s.actors,runtime['nominal_heights']):a.nominal_z=height
        for p,state in zip(s.policies,runtime['policy']):
            for k,v in state['fields'].items():setattr(p,k,unpack(v))
            if state['buffers']:
                buffers=dict(p.net.named_buffers())
                with torch.no_grad():
                    for k,v in state['buffers'].items():buffers[k].copy_(torch.from_numpy(unpack(v)))
        s.hardware.restore(runtime['hardware']);mujoco.mj_forward(s.m,s.d);s.state_history=[];s.motor_controls=[];s.hw_history=[];s.limit_history=[];s.frames=[];s.events=[];s.contacts_history=[];s.observations=[[],[]];s.policy_actions=[[],[]];s.observation_times=[];s.chunk_start=float(s.d.time);s.record_state();s.record_hardware();s.record_frame();return s
