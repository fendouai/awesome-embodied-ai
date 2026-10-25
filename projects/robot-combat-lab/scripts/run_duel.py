import time,argparse
from combatlab.duel import DuelSimulation,DuelRules
from combatlab.robostriker_cpu import ROBOT_KEY
from combatlab.endurance import EnduranceConfig
from combatlab.models import ROOT
from combatlab.engine import write_json
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--limit',type=float,default=90.);p.add_argument('--seed',type=int,default=0);p.add_argument("--swap",action="store_true");p.add_argument("--left",choices=["robostriker_ego","robostriker_opponent"],default=None);p.add_argument("--right",choices=["robostriker_ego","robostriker_opponent"],default=None);a=p.parse_args();rules=DuelRules(limit_s=a.limit);s=DuelSimulation(EnduranceConfig('duel-'+str(time.time_ns()),track='robostriker_cpu_port',robots=(ROBOT_KEY,ROBOT_KEY),policies=(a.left or ('robostriker_opponent' if a.swap else 'robostriker_ego'),a.right or ('robostriker_ego' if a.swap else 'robostriker_opponent')),seed=a.seed,timestep=.005,control_dt=.02,friction=1.,distance=2.,duration=a.limit,max_sim_time=a.limit,pose_jitter=0 if a.seed==0 else .008,position_jitter=0 if a.seed==0 else .012),rules)
 interrupted=None
 try:
  while s.advance(.1):
   if s.step_count%1000==0:print('DUEL',round(s.d.time,1),'tilt',s.max_tilt.round(1).tolist(),s.referee.status(s.d.time),flush=True)
 except Exception as error:
  interrupted=error;s.referee.outcome=None;s.terminal=dict(reason='adjudication_error',winner=None,time=float(s.d.time),hardware_damage=False,error=str(error));s.record_frame()
 folder,result=s.save();target='latest.json' if result['duel']['complete'] else 'latest-attempt.json';write_json(ROOT/'results/duels'/target,dict(folder=str(folder.relative_to(ROOT)),result=result));print('DUEL_SAVED',folder,result['duel'],flush=True)
 if interrupted:raise interrupted
 if not result['duel']['complete']:raise SystemExit('Incomplete simulation: no competition winner published')
