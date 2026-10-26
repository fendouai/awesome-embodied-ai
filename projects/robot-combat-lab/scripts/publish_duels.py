import json
from combatlab.models import ROOT
from combatlab.engine import write_json, sha256
entries=[]
for key,title,group in [('g1','G1 · RoboStriker 自主格斗','duels'),('ncp','NCP · 数字人拳击','ncp-duels')]:
 latest=json.loads((ROOT/'results'/group/'latest.json').read_text());folder=ROOT/latest['folder']
 if not (folder/'duel-video.json').exists():continue
 verdict=json.loads((folder/'duel-result.json').read_text());verification=json.loads((folder/'duel-verification.json').read_text())
 assert verdict['complete'] and verification['verified']
 assert verification['trajectory_sha256']==sha256(folder/'trajectory.npz')
 assert verification['winner']==verdict['outcome']['winner'] and verification['score']==verdict['outcome']['clean_upper_contacts']
 entries.append(dict(key=key,title=title,folder='/'+latest['folder'],verdict=verdict,verification=verification,events=json.loads((folder/'referee-events.json').read_text())))
 if (folder/'commentary/commentary.json').exists():
  entries[-1].update(commentary_video='/'+latest['folder']+'/commentary/match-commentated.zh-CN.mp4',captions='/'+latest['folder']+'/commentary/captions.zh-CN.vtt',commentary_script='/'+latest['folder']+'/commentary/script.zh-CN.txt')
write_json(ROOT/'results/duel-index.json',dict(matches=entries))
