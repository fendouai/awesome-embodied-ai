"""Published match matrix, result metadata, and replay-video integrity."""
import json
import subprocess
from collections import Counter
from combatlab.models import ROOT
from combatlab.engine import sha256

def test_all_compatible_attack_policy_pairings_are_archived():
    data=json.loads((ROOT/'results/tournaments/round-02/index.json').read_text())
    assert data['state']=='complete' and len(data['matches'])==8
    expected={
        'g1':Counter({('robostriker_ego','robostriker_ego'):1,('robostriker_ego','robostriker_opponent'):1,('robostriker_opponent','robostriker_ego'):1,('robostriker_opponent','robostriker_opponent'):1}),
        'ncp':Counter({('ncp_boxing_106','ncp_boxing_106'):1,('ncp_boxing_106','ncp_boxing_100'):1,('ncp_boxing_100','ncp_boxing_106'):1,('ncp_boxing_100','ncp_boxing_100'):1}),
    }
    for group,want in expected.items():
        matches=[m for m in data['matches'] if m['group']==group]
        assert Counter(tuple(m['policies']) for m in matches)==want
    for match in data['matches']:
        path=ROOT/match['published_folder'].lstrip('/')
        result=json.loads((path/'duel-result.json').read_text());proof=json.loads((path/'duel-verification.json').read_text())
        assert match['state']=='complete' and proof['verified']
        assert proof['score']==match['outcome']['clean_upper_contacts'] and proof['winner']==match['outcome']['winner']
        assert result['outcome']['hardware_damage'] is False
        video=ROOT/match['video'].lstrip('/');assert sha256(video)==match['video_sha256']
        info=json.loads((path/'duel-video.json').read_text())
        assert info['physics_unchanged'] and info['video_sha256']==sha256(video)
        streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name,width,height','-of','json',str(video)]))['streams']
        assert {'h264'}=={s['codec_name'] for s in streams} and streams[0]['width']==1280

def test_visual_camperception_record_is_truthfully_labeled():
    match=json.loads((ROOT/'results/tournaments/round-02/vision-match.json').read_text())
    assert match['perception']['fast_ground_truth_mode'] is False
    assert match['official_regulation_completed'] is False
    video=ROOT/match['video'].lstrip('/')
    assert sha256(video)==match['source_video_sha256']
