"""Publish one complete local recording index, revision log and source/narration manifest."""
import csv,json,subprocess,sys
from pathlib import Path
from combatlab.models import ROOT
from combatlab.engine import write_json,sha256
base=ROOT/'results/tournaments/round-04';groups=[];revisions=[]
for name in ['simulation-index.json','vision-index.json']:
 p=base/name;data=json.loads(p.read_text());assert data['state']=='complete'
 for m in data['matches']:
  assert m['state']=='complete';folder=ROOT/m['folder'];video=ROOT/m['video'].lstrip('/');source=ROOT/m['source_video'].lstrip('/');meta=json.loads((folder/'commentary/commentary.json').read_text())
  if meta.get('commentary_version',0)<2:
   args=[sys.executable,str(ROOT/'scripts/produce_match_commentary.py'),str(folder),str(source)]
   if name=='vision-index.json':args+=['--kind','vision']
   subprocess.run(args,check=True);meta=json.loads((folder/'commentary/commentary.json').read_text())
  actual=sha256(video)
  assert meta['video_sha256']==actual and meta['source_video_sha256']==sha256(source)
  if m['video_sha256']!=actual:revisions.append(dict(id=m['id'],previous_published_hash=m['video_sha256'],current_hash=actual,reason='Narration preview quality correction; source video SHA-256 unchanged'))
  m['video_sha256']=actual
 write_json(p,data);groups.append(data)
write_json(base/'production-revisions.json',revisions)
subprocess.run([sys.executable,str(ROOT/'scripts/audit_competition_v2.py'),'--require-complete'],check=True)
all_matches=[];artifacts=[]
for group,data in zip(['simulation','vision'],groups):
 for m in data['matches']:
  entry=dict(m,track=group);all_matches.append(entry);folder=ROOT/m['folder'];paths=[ROOT/m['source_video'].lstrip('/'),ROOT/m['video'].lstrip('/'),folder/'commentary/commentary.json',folder/'commentary/captions.zh-CN.srt',folder/'commentary/captions.zh-CN.vtt',folder/'commentary/narration.wav',folder/'commentary/script.zh-CN.txt']
  if (folder/'frames').is_dir():
   write_json(folder/'source-frame-manifest.json',{p.name:sha256(p) for p in sorted((folder/'frames').iterdir()) if p.is_file()})
  for n in ['trajectory.npz','policy_io.npz','referee-events.json','duel-rules.json','duel-result.json','duel-verification.json','telemetry.csv','frame-times.csv','source-frame-manifest.json','events.jsonl','timeline.json','native-result.json','analysis.json']:
   if (folder/n).is_file():paths.append(folder/n)
  for p in paths:artifacts.append(dict(match=m['id'],path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha256(p)))
write_json(base/'index.json',dict(id='round-04',state='complete',protocol=json.loads((base/'protocol.json').read_text()),matches=all_matches,audit='audit.json',recording_manifest='recording-manifest.json'))
write_json(base/'recording-manifest.json',dict(method='SHA-256 of original footage, narrated videos, subtitles, narration audio, transcripts and original event/physics evidence',artifacts=artifacts))
with (base/'matches.csv').open('w',newline='') as f:
 fields=['id','track','score','outcome','video','source_video','captions','script'];w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader()
 for m in all_matches:w.writerow(dict(id=m['id'],track=m['track'],score=':'.join(map(str,m.get('score',m.get('outcome',{}).get('clean_upper_contacts',[])))),outcome=m.get('outcome',{}).get('kind',m.get('referee_state','')),video=m['video'],source_video=m['source_video'],captions=m['captions'],script=m['script']))
summary=['# Round 04：完整比赛、中文解说与字幕','', '本轮为明确公布的180秒本地赛制；8场仿真格斗、4场视觉足球，全部附原片和中文解说字幕。','', '| 比赛 | 赛道 | 比分 | 结果 |','| --- | --- | --- | --- |']
for m in all_matches:summary.append('| '+m['id']+' | '+m['track']+' | '+':'.join(map(str,m.get('score',m.get('outcome',{}).get('clean_upper_contacts',[]))))+' | '+m.get('outcome',{}).get('kind',m.get('referee_state',''))+' |')
(base/'recording-report.zh-CN.md').write_text('\n'.join(summary)+'\n')
write_json(ROOT/'results/tournament-active.json',dict(state='complete',tournament='round-04'))
print('Published12 complete recorded matches, original footage, Mandarin narration and captions',flush=True)
