from concurrent.futures import ProcessPoolExecutor,as_completed
from combatlab.push_benchmark import run_trial,OUT,DIRECTIONS
from combatlab.engine import write_json

def job(args):return run_trial(*args,archive=True,tag='holdout')
if __name__=='__main__':
 jobs=[(r,p,d,30.,seed,0.) for r,p in [('g1_full','nvidia_wbc'),('h1_full','unitree_full_gym'),('h1_2_full','unitree_full_gym')] for seed in [1,2] for d in DIRECTIONS]
 jobs.extend([('g1_full','nvidia_wbc',d,40.,seed,0.) for seed in [1,2] for d in DIRECTIONS]);jobs.append(('g1_full','nvidia_wbc','forward',60.,1,0.))
 results=[]
 with ProcessPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(job,a) for a in jobs]):
   r=f.result();results.append(r);print('VALIDATE',len(results),'/',len(jobs),r['robot'],r['direction'],r['impulse_ns'],r['actors'][0]['survived'],flush=True);write_json(OUT/'validation-results.json',results)
