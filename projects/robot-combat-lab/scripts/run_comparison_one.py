import argparse
from combatlab.comparison import Comparison
from combatlab.comparison_analysis import analyze
from combatlab.engine import write_json
from combatlab.models import ROOT
p=argparse.ArgumentParser();p.add_argument('robot');p.add_argument('left');p.add_argument('right');p.add_argument('task');p.add_argument('receipt');a=p.parse_args()
s=Comparison(a.robot,a.left,a.right,a.task,0).run();folder,result=s.save();analyze(folder);write_json(ROOT/a.receipt,dict(folder=str(folder.relative_to(ROOT)),result=result));print('SAVED',folder,flush=True)
