"""Record exact combat controller coverage for the ten installed robot products."""
import json
from pathlib import Path
from combatlab.models import ROOT
from combatlab.catalog import ROBOTS,CONTROLS
for roundid in ['01','02']:
 p=ROOT/f'results/tournaments/round-{roundid}/index.json'
 if not p.exists():continue
 d=json.loads(p.read_text());d.setdefault('coverage',{})['robot_model_matrix']=[dict(robot=r['name'],robot_id=r['id'],compatible_deployments=[c['id'] for c in CONTROLS if r['id'] in c['robots']],adversarial_deployments=[c['id'] for c in CONTROLS if r['id'] in c['robots'] and 'combat' in c['tasks']],status='included' if 'combat' in r['tasks'] else 'excluded',reason='RoboStriker trained attack policy is available' if 'combat' in r['tasks'] else 'Only locomotion, balance, motion-tracking or soccer deployments are available; no compatible opponent-targeting attack policy') for r in ROBOTS]
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2));temp.replace(p)
