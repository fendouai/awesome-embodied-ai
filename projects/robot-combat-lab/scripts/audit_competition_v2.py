"""Audit source integrity, complete referee results, narration audio/captions and opening visibility."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from combatlab.models import ROOT
from combatlab.engine import write_json,sha256
from combatlab.commentary import probe
p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');p.add_argument('--track',choices=['all','simulation','vision'],default='all');a=p.parse_args();base=ROOT/'results/tournaments/round-04';reports=[];pending=[]
for name,count in [('simulation-index.json',8),('vision-index.json',4)]:
 if a.track!='all' and not name.startswith(a.track):continue
 data=json.loads((base/name).read_text());assert len(data['matches'])==count
 for m in data['matches']:
  if m['state']!='complete':pending.append(m['id']);continue
  folder=ROOT/m['folder'];video=ROOT/m['video'].lstrip('/');source=ROOT/m['source_video'].lstrip('/');meta=json.loads((folder/'commentary/commentary.json').read_text());info=probe(video)
  codecs={(s['codec_type'],s['codec_name']) for s in info['streams']};assert {('video','h264'),('audio','aac'),('subtitle','mov_text')}<=codecs
  assert meta['source_video_sha256']==sha256(source) and meta['video_sha256']==sha256(video)==m['video_sha256'];assert meta['physics_speed_changed'] is False
  cues=meta['cues'];assert cues and all(c['start']<c['end'] for c in cues);assert all(x['end']<=y['start'] for x,y in zip(cues,cues[1:]));assert cues[-1]['end']<=float(info['format']['duration'])+.1
  for c in cues:
   actual=float(probe(folder/'commentary'/c['audio'])['format']['duration']);assert abs(actual-(c['end']-c['start']))<.02
  assert (folder/'commentary/captions.zh-CN.srt').is_file() and (folder/'commentary/captions.zh-CN.vtt').is_file()
  pixels=subprocess.check_output(['ffmpeg','-v','error','-ss','2','-i',str(video),'-frames:v','1','-vf','crop=iw:ih*0.7:0:0,scale=64:32','-pix_fmt','rgb24','-f','rawvideo','-']);frame=np.frombuffer(pixels,dtype=np.uint8);assert frame.mean()>8 and frame.std()>10,'Opening is blank/black'
  if name=='simulation-index.json':
   proof=json.loads((folder/'duel-verification.json').read_text());verdict=json.loads((folder/'duel-result.json').read_text());assert proof['verified'] and verdict['complete'];assert verdict['rules']['limit_s']==180;assert proof['score']==m['outcome']['clean_upper_contacts']
  else:
   native=json.loads((folder/'native-result.json').read_text());assert native['state']=='FINISHED' and native['period_seconds']==180
   timeline=json.loads((folder/'timeline.json').read_text());last=[0,0];goal_events=[]
   for event in timeline:
    if event['score']!=last:goal_events.append(event);last=event['score']
   assert all(event in [c['evidence'] for c in cues] for event in goal_events),'A native goal was omitted from commentary'
  reports.append(dict(id=m['id'],state='passed',video_sha256=sha256(video),source_sha256=sha256(source),cue_count=len(cues),duration_seconds=float(info['format']['duration'])))
result=dict(scope=a.track,state='complete' if not pending else 'in_progress',passed=len(reports),pending=pending,matches=reports);write_json(base/('audit.json' if a.track=='all' else 'audit-'+a.track+'.json'),result);print(json.dumps(dict(state=result['state'],passed=len(reports),pending=len(pending)),ensure_ascii=False))
if a.require_complete and pending:raise SystemExit('Matches remain unfinished')
