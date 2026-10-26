"""Declared torso-force tests; no hidden stabilizing force. Screening != final validation."""
from pathlib import Path
import json,hashlib,time
import numpy as np,mujoco
from . import control_upgrade
from .engine import Simulation,MatchConfig,write_json,sha256
from .models import ROOT
PUSH_SOURCE=Path(__file__).read_bytes()
OUT=ROOT/'results/control-upgrade';OUT.mkdir(exist_ok=True)
DIRECTIONS={'forward':np.array([1.,0,0]),'backward':np.array([-1.,0,0]),'left':np.array([0.,1,0]),'right':np.array([0.,-1,0])}

def run_trial(robot,policy,direction='forward',impulse=0.,seed=0,walk=0.,archive=False,tag='screen',duration=8.):
 mid=f'{tag}-{robot}-{policy}-{direction}-j{impulse:g}-s{seed}-v{walk:g}'.replace('.','p')
 cfg=MatchConfig(mid,track='push_validation',robots=(robot,robot),policies=(policy,'pd_stand'),duration=duration,postroll=0,distance=8.,fall_hold=100.,ring_half_width=100.,rated=False,seed=seed,pose_jitter=.008 if seed else 0,position_jitter=0,note='Explicit 0.1s horizontal torso force at t=3s. Left=candidate, right=old neutral PD baseline; separate bodies receive mirrored body-frame disturbance. Not a combat rank.')
 s=Simulation(cfg);s.recording_enabled=archive
 for p in s.policies:
  if hasattr(p,'reach_enabled'):p.reach_enabled=False
 # PD reference uses the same old geometric calibration, applied only to the baseline actor.
 from .models import calibrate_stance,initialize
 calibrate_stance(s.m,s.d,[s.actors[1]])
 s.policies[1].target=s.actors[1].default.copy()
 if policy=='pd_stand':calibrate_stance(s.m,s.d,[s.actors[0]]);s.policies[0].target=s.actors[0].default.copy()
 # Apply calibrated starting reference before any physics; preserve initial orientation/placement.
 for a in s.actors:s.d.qpos[a.qadr]=a.default
 mujoco.mj_forward(s.m,s.d);s.update_control([[0,0,0],[0,0,0]])
 s.frames=[];s.state_history=[];s.motor_controls=[];s.observations=[[],[]];s.policy_actions=[[],[]];s.observation_times=[];s.record_state();s.record_frame()
 torso=[next((b for b in a.body_ids if s.m.body(b).name.endswith('/torso_link')),a.root_id) for a in s.actors]
 pulse_start=3.;pulse_duration=.1;direction_vec=DIRECTIONS[direction];push_directions=[]
 for a in s.actors:
  R=np.empty(9);mujoco.mju_quat2Mat(R,s.d.qpos[a.free_qadr+3:a.free_qadr+7]);push_directions.append(R.reshape(3,3)@direction_vec)
 force_log=[];delivered=np.zeros((2,3));full_force=[];fell=[False,False];timers=np.zeros(2);max_tilt=np.zeros(2);max_post=np.zeros(2);ready=[True,True];initial=[s.d.qpos[a.free_qadr:a.free_qadr+3].copy() for a in s.actors];stance_foot=[{},{}];steps=[0,0]
 while s.d.time<duration-1e-8:
  t=s.d.time;force=np.zeros((2,3))
  if pulse_start-1e-8<=t<pulse_start+pulse_duration-1e-8:
   for i,a in enumerate(s.actors):
    force[i]=push_directions[i]*(impulse/pulse_duration)
    s.d.xfrc_applied[torso[i],:3]=force[i];delivered[i]+=force[i]*cfg.timestep
  else:s.d.xfrc_applied[:]=0
  # Direction is frozen to initial world orientation in the stored schedule below.
  if archive:full_force.append(np.r_[t,force.ravel()])
  if archive:s.advance(cfg.timestep,[[walk if t>=1.5 else 0,0,0],[0,0,0]])
  else:
   if s.step_count%10==0:s.update_control([[walk if t>=1.5 else 0,0,0],[0,0,0]])
   mujoco.mj_step(s.m,s.d);s.step_count+=1
   if s.d.time<t or not np.isfinite(s.d.qpos).all():s.terminal=dict(reason='numerical_failure');break
   if s.step_count%10==0:mujoco.mj_forward(s.m,s.d);s.record_frame()
  if s.terminal and s.terminal['reason']=='numerical_failure':break
  if not archive and s.step_count%10:continue
  for i,a in enumerate(s.actors):
   from .policies import gravity_body
   q=s.d.qpos[a.free_qadr:a.free_qadr+7];tilt=np.degrees(np.arccos(np.clip(-gravity_body(q[3:])[2],-1,1)));max_tilt[i]=max(max_tilt[i],tilt)
   bad=tilt>65 or q[2]<.55*a.nominal_z;timers[i]=timers[i]+(cfg.timestep if archive else cfg.control_dt) if bad else 0.
   if t<3 and (tilt>20 or q[2]<.7*a.nominal_z):ready[i]=False
   if t>=3:
    max_post[i]=max(max_post[i],tilt)
    if timers[i]>=.3:fell[i]=True
   if s.step_count%10==0:
    for b in a.body_ids:
     if 'ankle_roll_link' not in s.m.body(b).name and not s.m.body(b).name.endswith('ankle_link'):continue
     name=s.m.body(b).name;pos=s.d.xpos[b].copy()
     if name not in stance_foot[i]:stance_foot[i][name]=[pos.copy(),False]
     prev,air=stance_foot[i][name]
     if pos[2]>prev[2]+.035:stance_foot[i][name][1]=True
     elif air and pos[2]<prev[2]+.02:
      if np.linalg.norm(pos[:2]-prev[:2])>.04:steps[i]+=1
      stance_foot[i][name]=[pos.copy(),False]
  if s.step_count%10==0:force_log.append(np.r_[s.d.time,force.ravel()])
 s.d.xfrc_applied[:]=0
 tele=s.frames;actors=[]
 for i,a in enumerate(s.actors):
  tail=[f['actors'][i] for f in tele if f['time']>=duration-.8];stable=bool(tail) and all(x['tilt_deg']<20 and x['height_m']>.7*a.nominal_z and (walk>0 or x['speed_m_s']<.35) for x in tail)
  actors.append(dict(policy=cfg.policies[i],ready_before_push=ready[i],fell_after_push=fell[i],recovered_final=stable,survived=ready[i] and not fell[i] and stable and (s.terminal or {}).get('reason')!='numerical_failure',max_tilt_deg=float(max_tilt[i]),max_post_push_tilt_deg=float(max_post[i]),final_tilt_deg=tele[-1]['actors'][i]['tilt_deg'],foot_relocations=steps[i],displacement_m=float(np.linalg.norm(s.d.qpos[a.free_qadr:a.free_qadr+2]-initial[i][:2])),mass_kg=float(s.m.body_mass[a.body_ids].sum()),positive_work_j=float(s.energy[i])))
 result=dict(id=mid,robot=robot,policy=policy,direction=direction,impulse_ns=impulse,force_n=impulse/pulse_duration,pulse_duration_s=pulse_duration,pulse_start_s=pulse_start,seed=seed,walk_command_m_s=walk,duration_s=float(s.d.time),actors=actors,numerical_warnings=int(sum(x.number for x in s.d.warning)),torque_violation_nm=max(0,s.torque_violation),force_definition='Horizontal direction fixed from the initial body yaw; constant magnitude for 0.1s. Full schedule saved. No stabilization force.',software_fingerprint=control_upgrade.ADAPTER_FINGERPRINT,engine_fingerprint=__import__('combatlab.engine',fromlist=['SOFTWARE_FINGERPRINT']).SOFTWARE_FINGERPRINT)
 folder=OUT/('archives' if archive else 'screen')/mid;folder.mkdir(parents=True,exist_ok=True)
 if archive:
  s.save(OUT/'archives');np.savez_compressed(folder/'disturbance.npz',samples=np.asarray(full_force),columns=['time','left_fx','left_fy','left_fz','right_fx','right_fy','right_fz']);(folder/'control_upgrade.py').write_bytes(control_upgrade.ADAPTER_SOURCE);(folder/'push_benchmark.py').write_bytes(PUSH_SOURCE)
 else:
  np.savez_compressed(folder/'telemetry.npz',time=[f['time'] for f in tele],tilt=[[a['tilt_deg'] for a in f['actors']] for f in tele],height=[[a['height_m'] for a in f['actors']] for f in tele],speed=[[a['speed_m_s'] for a in f['actors']] for f in tele],force_schedule=np.asarray(force_log))
 write_json(folder/'push-result.json',result)
 if archive:
  manifest=json.loads((folder/'manifest.json').read_text());manifest.update({f.name:sha256(f) for f in folder.iterdir() if f.is_file() and f.name!='manifest.json'});write_json(folder/'manifest.json',manifest)
 return result

def screen():
 from concurrent.futures import ProcessPoolExecutor,as_completed
 candidates={'g1_full':['amp_recovery','holosoma_fastsac','holosoma_ppo','nvidia_balance','nvidia_walk','unitree_rl_lab','unitree_rl_mjlab'],'h1_full':['unitree_full_gym'],'h1_2_full':['unitree_full_gym']};jobs=[]
 for robot,policies in candidates.items():
  for policy in policies:
   jobs.extend([(robot,policy,'forward',0.,0,v) for v in [0.,.35]])
   jobs.extend([(robot,policy,d,j,0,0.) for d in DIRECTIONS for j in [10.,30.,60.]])
 results=[]
 with ProcessPoolExecutor(max_workers=2) as pool:
  pending={pool.submit(_job,args):args for args in jobs}
  for future in as_completed(pending):
   r=future.result();results.append(r);print('PUSH',len(results),'/',len(jobs),r['robot'],r['policy'],r['direction'],r['impulse_ns'],r['actors'][0]['survived'],flush=True);write_json(OUT/'screen-results.json',results)
 return results

def _job(args):return run_trial(*args)
