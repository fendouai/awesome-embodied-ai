from combatlab import control_upgrade
from combatlab.engine import Simulation,MatchConfig,write_json,sha256
from combatlab.models import ROOT
from pathlib import Path
import json
if __name__=='__main__':
 rows=[]
 for robot,left,right in [('g1_full','nvidia_wbc','nvidia_wbc'),('g1_full','nvidia_wbc','amp_recovery'),('h1_full','unitree_full_gym','unitree_full_gym'),('h1_2_full','unitree_full_gym','unitree_full_gym')]:
  mid=f'contact-{robot}-{left}-{right}'
  sim=Simulation(MatchConfig(mid,track='upgrade_contact',robots=(robot,robot),policies=(left,right),rated=False,duration=12,speed=.35,note='Upgraded actual learned motion policies; exploratory contact trial, not a body ranking.'));sim.run();folder,result=sim.save(ROOT/'results/control-upgrade/contact')
  (folder/'control_upgrade.py').write_bytes(Path(control_upgrade.__file__).read_bytes());manifest=json.loads((folder/'manifest.json').read_text());manifest['control_upgrade.py']=sha256(folder/'control_upgrade.py');write_json(folder/'manifest.json',manifest);rows.append(result);write_json(ROOT/'results/control-upgrade/contact-results.json',rows);print('CONTACT',robot,left,right,result['first_contact_s'],result['result'],flush=True)
