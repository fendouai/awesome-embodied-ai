import json
from itertools import combinations
from combatlab.models import ROOT
from combatlab.comparison import COMPAT,validate
from combatlab.engine import sha256

def test_all_distinct_compatible_pairs_and_tasks_are_covered():
 base=ROOT/'results/all-combinations/round-01';d=json.loads((base/'index.json').read_text());assert d['state']=='complete'
 expected=set()
 for robot,policies in COMPAT.items():
  for left,right in combinations(policies,2):
   for task in ['tracking','push','motion']:
    try:validate(robot,left,right,task)
    except ValueError:continue
    expected.add((robot,frozenset((left,right)),task))
 actual={(j['robot'],frozenset(j['policies']),j['task']) for j in d['jobs']}
 assert len(d['jobs'])==len(actual)==len(expected)==63 and actual==expected
 assert all(j['state']=='complete' and j['is_physical_combat'] is False for j in d['jobs'])
 audit=json.loads((base/'audit.json').read_text());assert audit['state']=='complete' and len(audit['jobs'])==63
 for j in audit['jobs']:
  assert j['verified'] and j['max_qpos_error']<2e-7 and len(j['sampled_native_step_indices'])==3

def test_every_recording_has_a_preserved_independent_copy():
 base=ROOT/'results/all-combinations/round-01';d=json.loads((base/'video-archive.json').read_text());assert d['recordings']==68 and len(d['videos'])==68
 assert len({r['video'] for r in d['videos']})==68
 for r in d['videos']:
  out=ROOT/r['video'].lstrip('/');source=ROOT/r['source_video'].lstrip('/')
  assert out.stat().st_ino!=source.stat().st_ino
  assert sha256(out)==sha256(source)==r['sha256'] and out.stat().st_size>10000
