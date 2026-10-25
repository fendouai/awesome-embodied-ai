import argparse,json,numpy as np
from combatlab.contact_arena import ContactArena
from combatlab.research_archive import workspace,event,atomic_json
p=argparse.ArgumentParser();p.add_argument('robot');a=p.parse_args();out=workspace()/'03_training'/('adapter-probe-'+a.robot);out.mkdir(exist_ok=True);env=ContactArena(a.robot,trace_folder=out/'raw',episode_s=5);frames=[]
try:
 obs=env.observations();assert obs.shape==(2,131)
 for i in range(250):
  obs,r,done,trunc,info=env.step(np.zeros((2,32)));frames.append({k:v for k,v in info.items() if k!='native_updates'})
  if done or trunc:break
 atomic_json(out/'probe.json',dict(robot=a.robot,steps=len(frames),last=frames[-1],numerical_warnings=int(sum(w.number for w in env.d.warning)),external_root_force_peak=float(abs(env.d.xfrc_applied).max()),state='adapter_probe_not_trained',frames=frames));event('contact_adapter_probed',robot=a.robot,steps=len(frames),last=frames[-1]);print(a.robot,len(frames),frames[-1],flush=True)
finally:env.close()
