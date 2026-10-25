import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from combatlab.push_benchmark import run_trial,OUT,DIRECTIONS
from combatlab.engine import write_json

def job(args):return run_trial(*args,tag='refine')
if __name__=='__main__':
 jobs=[('g1_full',p,d,j,0,0.) for p in ['amp_recovery','holosoma_fastsac','holosoma_ppo','nvidia_balance','nvidia_walk','unitree_rl_lab','unitree_rl_mjlab'] for d in DIRECTIONS for j in [40.,50.]]
 rows=[]
 with ProcessPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(job,a) for a in jobs]):
   r=f.result();rows.append(r);print('REFINE',len(rows),'/',len(jobs),r['policy'],r['direction'],r['impulse_ns'],r['actors'][0]['survived'],flush=True);write_json(OUT/'refine-results.json',rows)
