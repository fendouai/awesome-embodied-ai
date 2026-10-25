import argparse
from combatlab.contact_training import train
p=argparse.ArgumentParser();p.add_argument('robot');p.add_argument('--iterations',type=int,default=24);p.add_argument('--steps',type=int,default=256);p.add_argument('--seed',type=int,default=0);p.add_argument('--initial-checkpoint');p.add_argument('--receipt');p.add_argument('--episode-seconds',type=float,default=6);p.add_argument('--reward-version',type=int,default=1);p.add_argument('--navigation-auxiliary',action='store_true');a=p.parse_args();folder=train(a.robot,a.iterations,a.steps,a.seed,a.initial_checkpoint,a.episode_seconds,a.reward_version,a.navigation_auxiliary);print(folder,flush=True)
if a.receipt:
 from combatlab.research_archive import atomic_json
 atomic_json(a.receipt,dict(folder=str(folder)))
