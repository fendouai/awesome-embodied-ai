import argparse
from combatlab.navigation_pretrain import pretrain
from combatlab.research_archive import atomic_json
p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--receipt');p.add_argument('--updates',type=int,default=1500);a=p.parse_args();out=pretrain(a.checkpoint,a.updates);print(out,flush=True)
if a.receipt:atomic_json(a.receipt,dict(checkpoint=str(out)))
