"""Actual published outcomes must match raw replay verification and playable video."""
import json,subprocess
from combatlab.models import ROOT
from combatlab.engine import sha256

def test_published_duels_have_verified_results_and_visible_endings():
 data=json.loads((ROOT/'results/duel-index.json').read_text())
 assert {m['key'] for m in data['matches']}=={'g1','ncp'}
 for match in data['matches']:
  folder=ROOT/match['folder'].lstrip('/');outcome=match['verdict']['outcome'];verification=match['verification']
  assert match['verdict']['complete'] and verification['verified']
  assert verification['trajectory_sha256']==sha256(folder/'trajectory.npz')
  assert verification['score']==outcome['clean_upper_contacts'] and verification['winner']==outcome['winner']
  events=[e for e in match['events'] if e['type']=='scoring_contact']
  assert [sum(e['side']==i for e in events) for i in [0,1]]==verification['score']
  assert outcome['hardware_damage'] is False
  info=json.loads((folder/'duel-video.json').read_text());assert info['physics_unchanged'] and info['final_frame_hold_s']==6
  assert info['video_sha256']==sha256(folder/info['video'])
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name','-of','json',str(folder/info['video'])]))
  assert {s['codec_name'] for s in probe['streams']}=={'h264','aac'}
  assert abs(float(probe['format']['duration'])-info['source_duration_s']-6)<.1
