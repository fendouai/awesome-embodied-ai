import argparse
from combatlab.ncp_duel import NCPDuel
from combatlab.duel import DuelRules
from combatlab.models import ROOT
from combatlab.engine import write_json
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--limit',type=float,default=90);p.add_argument('--seed',type=int,default=0);p.add_argument("--swap",action="store_true");p.add_argument("--left",choices=["106","100"],default=None);p.add_argument("--right",choices=["106","100"],default=None);a=p.parse_args();left=a.left or ("100" if a.swap else "106");right=a.right or ("106" if a.swap else "100");s=NCPDuel(a.seed,DuelRules(limit_s=a.limit),checkpoints=(f"boxing_{left}.pth",f"boxing_{right}.pth"),policy_seeds=(a.seed+(1 if left=="100" else 0),a.seed+(1 if right=="100" else 0)))
 while s.advance(.1):
  if s.steps%1200==0:print('NCP_DUEL',round(s.d.time,1),s.referee.status(s.d.time),flush=True)
 folder,result=s.save();write_json(ROOT/'results/ncp-duels'/('latest.json' if s.referee.outcome else 'latest-attempt.json'),dict(folder=str(folder.relative_to(ROOT)),result=result));print('NCP_DUEL_SAVED',folder,result['duel'],flush=True)
 if not s.referee.outcome:raise SystemExit('Incomplete simulation: no competition winner published')
