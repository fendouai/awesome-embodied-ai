"""Run both released NCP boxing characters with recorded stochastic inference."""
import argparse
from combatlab.ncp_cpu import NCPEncounter
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=20);p.add_argument('--seed',type=int,default=0);p.add_argument('--greedy',action='store_true');a=p.parse_args()
 s=NCPEncounter(seed=a.seed,stochastic=not a.greedy)
 while s.d.time+1e-9<a.seconds:
  if not s.advance(min(1.,a.seconds-s.d.time)):break
  print('NCP',round(s.d.time,2),'tilt',s.max_tilt.round(2).tolist(),flush=True)
 folder,result=s.save();print(folder,result['result'],'contact',result['contact_duration_s'],flush=True)
