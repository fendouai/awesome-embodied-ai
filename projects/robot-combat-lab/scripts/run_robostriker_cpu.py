from pathlib import Path
import argparse,json,time
import numpy as np
from combatlab.robostriker_cpu import RoboStrikerSimulation,ROBOT_KEY
from combatlab.endurance import EnduranceConfig
from combatlab.models import ROOT
from combatlab.engine import write_json

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=20);p.add_argument('--timestep',type=float,default=.005);p.add_argument('--friction',type=float,default=1.);p.add_argument('--seed',type=int,default=0);p.add_argument('--distance',type=float,default=2.);p.add_argument('--resume');args=p.parse_args()
 if args.resume:s=RoboStrikerSimulation.restore(args.resume);s.resume(args.seconds)
 else:s=RoboStrikerSimulation(EnduranceConfig('robostriker-cpu-'+str(time.time_ns()),track='robostriker_cpu_port',robots=(ROBOT_KEY,ROBOT_KEY),policies=('robostriker_ego','robostriker_opponent'),seed=args.seed,timestep=args.timestep,control_dt=.02,friction=args.friction,distance=args.distance,duration=args.seconds,max_sim_time=args.seconds,chunk_seconds=30,pose_jitter=0 if not args.seed else .008,position_jitter=0 if not args.seed else .012))
 # The base protocol keeps falling nonterminal and records every native physics step.
 s.save_on_chunks=False
 start=time.time()
 while s.advance(.1):
  if s.step_count%1000==0:print('STRIKER',round(s.d.time,2),'tilt',s.max_tilt.round(1).tolist(),'contact',round(s.contact_steps*s.cfg.timestep,2),flush=True)
 folder,result=s.save(ROOT/'results/robostriker-cpu');write_json(ROOT/'results/robostriker-cpu/latest.json',dict(folder=str(folder.relative_to(ROOT)),parts=s.parts,result=result,wall_time_s=time.time()-start,native_match_reproduction=False,notes='MuJoCo sim-to-sim CPU port of pinned trained Stage3 actors. Physics records, not animation. No scripted attack or hidden stabilizing force.'))
 print('PORT ENCOUNTER SAVED',folder,'tilt',s.max_tilt.tolist(),'contact',s.first_contact,result['result'],flush=True)
