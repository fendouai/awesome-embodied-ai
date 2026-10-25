"""Both seat assignments must use the declared real policies, with replayable videos."""
import json,subprocess
import numpy as np,torch
from combatlab.models import ROOT
from combatlab.engine import sha256

def test_round_robin_coverage_and_video_integrity():
 d=json.loads((ROOT/'results/tournaments/round-01/index.json').read_text());assert d['state']=='complete' and len(d['matches'])==4
 for group in ['g1','ncp']:
  matches=[m for m in d['matches'] if m['group']==group];assert len(matches)==2
  assert matches[0]['policies']==list(reversed(matches[1]['policies']))
 for m in d['matches']:
  f=ROOT/m['folder'];result=json.loads((f/'result.json').read_text());assert result['policies']==m['policies']
  assert m['state']=='complete' and m['verification']['verified'] and m['verification']['max_qpos_replay_error']<2e-7
  assert m['verification']['trajectory_sha256']==sha256(f/'trajectory.npz')
  assert m['video_sha256']==sha256(ROOT/m['video'].lstrip('/'))
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name','-of','json',str(ROOT/m['video'].lstrip('/'))]))
  assert float(probe['format']['duration'])>=m['outcome']['time']+5.9
  assert {s['codec_name'] for s in probe['streams']}=={'h264','aac'}
  assert result['numerical_warnings']==0 and result['external_root_force_peak']==0
  assert m['outcome']['hardware_damage'] is False

def test_swapped_policies_match_saved_neural_actions():
 from combatlab.robostriker_cpu import Stage3CPU
 from combatlab.ncp_cpu import NCPPolicy
 d=json.loads((ROOT/'results/tournaments/round-01/index.json').read_text());net=Stage3CPU()
 for m in d['matches']:
  if m['leg']!=2:continue
  folder=ROOT/m['folder'];raw=np.load(folder/'policy_io.npz')
  for side,key in enumerate(m['policies']):
   obs=raw[f'p{side}_observations'];actions=raw[f'p{side}_actions']
   for idx in [0,len(obs)//2,len(obs)-1]:
    if m['group']=='g1':
     task,hist,decoder=np.split(obs[idx],[192,1092]);out,_=net.infer(0 if key.endswith('_ego') else 1,task,hist,decoder)
    else:
     p=NCPPolicy(0,key.removeprefix('ncp_')+'.pth');prop=p.norm(obs[idx,:323],'prop');embedding=p.embedding[[int(raw[f'p{side}_codes'][idx])]]
     with torch.inference_mode():out=p.low(torch.cat([prop,embedding],-1)).numpy()[0]+raw[f'p{side}_noise'][idx]
    np.testing.assert_allclose(out,actions[idx],atol=1e-6,rtol=1e-6)
  if m['group']=='ncp':assert json.loads((folder/'config.json').read_text())['policy_seeds']==[1,0]
