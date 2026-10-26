"""Compact public showcase; originals, speech WAVs and full physics stay locally archived."""
import json,shutil,subprocess,os
from pathlib import Path
from combatlab.models import ROOT
from combatlab.engine import write_json,sha256
base=ROOT/'results/tournaments/round-04';data=json.loads((base/'index.json').read_text());audit=json.loads((base/'audit.json').read_text());assert data['state']=='complete' and audit['passed']==12
out=base/'published';videos=out/'videos';meta_dir=out/'matches';videos.mkdir(parents=True,exist_ok=True);meta_dir.mkdir(exist_ok=True)
entries=[]
for m in data['matches']:
 folder=ROOT/m['folder'];source=ROOT/m['video'].lstrip('/');target=videos/(m['id']+'.zh-CN.mp4')
 if not target.exists():
  try:os.link(source,target)
  except OSError:shutil.copy2(source,target)
 if source.stat().st_size>15*1024*1024 and (not target.exists() or sha256(target)==m['video_sha256']):
  pending=target.with_suffix('.pending.mp4')
  subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(source),'-map','0','-c:v','libx264','-preset','fast','-threads','2','-b:v','400k','-maxrate','600k','-bufsize','1200k','-pix_fmt','yuv420p','-c:a','copy','-c:s','copy','-movflags','+faststart',str(pending)],check=True)
  pending.replace(target)
 assert target.stat().st_size<16*1024*1024
 public_probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(target)]))
 assert {x['codec_type'] for x in public_probe['streams']}=={'video','audio','subtitle'}
 subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(target),'-f','null','-'],check=True)
 target_meta=meta_dir/m['id'];target_meta.mkdir(exist_ok=True)
 for filename in ['captions.zh-CN.srt','captions.zh-CN.vtt','script.zh-CN.txt']:shutil.copy2(folder/'commentary'/filename,target_meta/filename)
 meta=json.loads((folder/'commentary/commentary.json').read_text());assert abs(float(meta['probe']['format']['duration'])-float(public_probe['format']['duration']))<0.05;meta['original_metadata_sha256']=sha256(folder/'commentary/commentary.json');meta['source_video']='original footage retained in local archive';meta['video']=target.name;meta['local_narrated_video_sha256']=meta['video_sha256'];meta['video_sha256']=sha256(target);meta['public_encoding']='H.264 publication derivative at400kbps when original exceeds15MiB; source timing, resolution, audio and captions retained; local master unchanged';meta['probe']=public_probe;meta['probe']['format']['filename']=target.name;write_json(target_meta/'commentary.json',meta)
 for filename in ['duel-result.json','duel-rules.json','duel-verification.json','referee-events.json','native-result.json','analysis.json','timeline.json']:
  if (folder/filename).exists():shutil.copy2(folder/filename,target_meta/filename)
 poster=videos/(m['id']+'.jpg');t=meta['intro_hold_seconds']+min(8.,float(meta['probe']['format']['duration'])/4)
 if m['track']=='vision':
  timeline=json.loads((folder/'timeline.json').read_text());playing=next(e['time'] for e in timeline if e['state']==3);t=meta['intro_hold_seconds']+playing+15
 subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(t),'-i',str(target),'-frames:v','1',str(poster)],check=True)
 score=m.get('score',m.get('outcome',{}).get('clean_upper_contacts',[]));entry=dict(id=m['id'],track=m['track'],score=score,outcome=m.get('outcome',m.get('referee_state')),video='videos/'+target.name,poster='videos/'+poster.name,record='matches/'+m['id'],video_sha256=sha256(target),original_video_sha256=meta['source_video_sha256']);entries.append(entry)
for filename in ['protocol.json','official-source-record.json','native-runtime.json','audit.json','production-revisions.json','production-attempts.json']:
 if (base/filename).exists():shutil.copy2(base/filename,out/filename)
write_json(out/'index.json',dict(state='complete',matches=entries,originals_retained_locally=True,full_physics_and_speech_wav_retained_locally=True,website_kept_local=True))
for lang in ['zh-CN','en']:
 zh=lang=='zh-CN';title='# Round 04：完整比赛、中文解说和字幕' if zh else '# Round 04: complete matches with Mandarin commentary and captions';lines=[title,'',('8场MuJoCo格斗与4场B-Human相机视觉足球。均使用明确公布的180秒本地计分/比赛窗口；格斗可因恢复条件提前结束。每段解说MP4包含完整原比赛过程与明确记录的首尾静帧，中文配音、烧录字幕以及独立SRT/VTT与解说稿。' if zh else 'Eight MuJoCo boxing bouts and four B-Human camera-perception soccer matches use a declared local180-second scoring/playing window. Boxing can end under recovery-based technical rules. Narrated MP4s retain the full match with explicitly recorded opening/closing freeze frames, Mandarin audio, burned captions, SRT/VTT and transcripts.'),'',('官方参考赛时分别记录于protocol.json：RoboStriker源配置50秒，NCP约16.67秒，HSL两个10分钟半场。本轮本地成绩不冒充官方基准复现。G1、NCP数字人和足球分别呈现，不跨任务排名。' if zh else 'Source timing references are recorded separately in protocol.json: RoboStriker50 seconds, NCP about16.67 seconds, and HSL two ten-minute halves. Local scores are not source benchmark reproduction claims. G1, NCP digital characters and soccer remain separate tracks.'),'',('| 比赛 | 赛道 | 接触统计 / 进球 | 裁判结果 | 视频 | 字幕与解说稿 |' if zh else '| Match | Track | Contact counts / goals | Referee outcome | Video | Captions and transcript |'),'| --- | --- | --- | --- | --- | --- |']
 for m in entries:
  outcome=m['outcome']
  if isinstance(outcome,dict):
   verdict=('技术平局' if zh else 'Technical draw') if outcome['kind']=='technical_draw' else ('平局' if zh else 'Draw') if outcome['winner'] is None else (('蓝方胜' if zh else 'Blue wins') if outcome['winner']==0 else ('橙方胜' if zh else 'Amber wins'))
   verdict+=f" · {outcome['time']:.2f}s"
  else:verdict=('平局' if zh else 'Draw') if m['score'][0]==m['score'][1] else ('红队胜' if zh else 'Red wins') if m['score'][0]>m['score'][1] else ('白队胜' if zh else 'White wins')
  lines.append('| '+m['id']+' | '+m['track']+' | '+':'.join(map(str,m['score']))+' | '+verdict+' | ['+('中文解说' if zh else 'Mandarin commentary')+']('+m['video']+') | [SRT]('+m['record']+'/captions.zh-CN.srt) · [VTT]('+m['record']+'/captions.zh-CN.vtt) · ['+('解说稿' if zh else 'Transcript')+']('+m['record']+'/script.zh-CN.txt) |')
 lines+=['',('完整原片、配音成片母版、48kHz配音WAV和详细物理归档保留在本机；公开包的大文件另行压缩，保持完整时序、720p画面与音频字幕，哈希单独记录。网站按要求暂留本机。audit.json记录视频、配音字幕和原始证据检查结果。' if zh else 'Original footage, narrated masters,48kHz narration WAVs and detailed physics archives remain local. Large publication videos are separately compressed with full timing,720p frames,audio and captions retained; derivative hashes are recorded. This is a compact public showcase; the website is kept local as requested. audit.json records the video/audio/caption and evidence checks.')]
 for track,choose in [('G1',next(m for m in entries if m['id']=='g1-robostriker_ego-robostriker_opponent')),('NCP',next(m for m in entries if m['id']=='ncp-106-100')),('B-Human',next(m for m in entries if m['id']=='vision-01'))]:lines+=['',f'## {track}','',f'[![{track}]({choose["poster"]})]({choose["video"]})']
 (out/('README.md' if zh else 'README.en.md')).write_text('\n'.join(lines)+'\n')
write_json(out/'publication-audit.json',dict(passed=len(entries),checks=['complete decoder pass for each public video','video,audio and subtitle streams present','duration differs from local narrated master by less than0.05s','each video under16MiB','local narrated masters unchanged'],local_master_audit='audit.json'))
write_json(out/'manifest.json',{str(p.relative_to(out)):dict(sha256=sha256(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'})
print('Public package ready:',out,len(entries),'videos',flush=True)
