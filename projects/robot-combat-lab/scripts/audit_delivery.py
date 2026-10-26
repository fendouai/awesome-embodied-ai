"""Final package audit, including actual interactive command traces and full videos."""
from pathlib import Path
import json,hashlib,re,subprocess,urllib.request
import numpy as np
from combatlab.models import ROOT
from combatlab.engine import SOFTWARE_FINGERPRINT,write_json

p=ROOT;v=json.loads((p/'results/verification.json').read_text());assert v['passed'] and v['require_videos'] and len(v['matches'])==26
board=json.loads((p/'results/leaderboards.json').read_text());assert board['leaderboards']['robot']['rated_matches']==18 and board['leaderboards']['ai']['rated_matches']==8
assert all(x['rated'] for x in board['matches'])
health=json.load(urllib.request.urlopen('http://127.0.0.1:8767/api/health'));assert health['software_fingerprint']==SOFTWARE_FINGERPRINT==v['software_fingerprint']
browser=json.loads((p/'evidence/browser-check.json').read_text());assert not browser['errors'];assert all(not x['overflow'] and x['contentBottom']<x['footerTop'] for x in browser['layout'])
controls=[]
for text in browser['live']:
    sid=re.search(r'live-\d+',text).group();folder=p/'results/live-sessions'/sid;r=json.loads((folder/'result.json').read_text());info=json.loads((folder/'interaction.json').read_text());tele=json.loads((folder/'telemetry.json').read_text());cmd=np.array([[a['command'] for a in f['actors']] for f in tele]);assert np.isfinite(cmd).all();assert np.max(cmd[:,0,0])>=.49
    if info['mode']=='human-human':assert np.max(cmd[:,1,0])>=.49
    assert not r['rated'] and r['track']=='interactive';manifest=json.loads((folder/'manifest.json').read_text())
    for name,sha in manifest.items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(folder/'battle.mp4')]))
    assert abs(float(probe['format']['duration'])-r['duration_s'])<.1
    controls.append(dict(session=sid,mode=info['mode'],duration_s=r['duration_s'],frames=r['archive']['physics_frames'],recorded_command_max=cmd.max(axis=0).tolist(),complete_video=True))
assert {c['mode'] for c in controls}=={'human-ai','human-human'}
report=json.loads((p/'reports/report-manifest.json').read_text());assert report['physical_dataset']==hashlib.sha256((p/'results/leaderboards.json').read_bytes()).hexdigest();assert (p/'reports/report.pdf').stat().st_size>100000
assert len(list((p/'results/matches').glob('*/battle.mp4')))==26
sensitivity=json.loads((p/'results/sensitivity.json').read_text());assert len(sensitivity['checks'])==2 and all(c['winner_agrees'] for c in sensitivity['checks'])
out=dict(passed=True,official_matches=26,body_rated=18,motion_model_rated=8,complete_videos=26,source_fingerprint=SOFTWARE_FINGERPRINT,live_source_matches_archive=True,interactive_sessions=controls,report_pages=report['pages'],browser_errors=[],pdf_and_report_dataset_verified=True,physical_verification='results/verification.json',browser_verification='evidence/browser-check.json',limitations='Protocol-specific 3-body/2-motion-policy simulation; no real hardware or VLA/combat-trained claims.')
write_json(p/'results/delivery-audit.json',out);print(json.dumps(out,ensure_ascii=False,indent=2))
