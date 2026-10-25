"""Package reproducible round metadata and flat replay URLs while keeping bulk trajectories local."""
import json,shutil
from pathlib import Path
from combatlab.models import ROOT
from combatlab.engine import sha256,write_json
rounddir=ROOT/'results/tournaments/round-02';data=json.loads((rounddir/'index.json').read_text());base=rounddir/'match-data';base.mkdir(exist_ok=True);videos=rounddir/'videos';videos.mkdir(exist_ok=True)
keep=['result.json','duel-result.json','duel-rules.json','referee-events.json','duel-verification.json','duel-video.json']
for m in data['matches']:
 source=ROOT/m['folder'];target=base/m['id'];target.mkdir(exist_ok=True)
 for name in keep:
  p=source/name
  if p.exists():shutil.copy2(p,target/name)
 source_video=ROOT/m['video'].lstrip('/');stem=Path(m['video']).name
 # Stage flat video archive targets are already copies; use stable links in the index.
 target_video=next((p for p in videos.glob(m['id']+'-*.mp4')),None)
 if target_video is None:target_video=videos/(m['id']+'-'+stem)
 if not target_video.exists():shutil.copy2(source_video,target_video)
 p=source/'duel-poster.jpg';target_poster=videos/(m['id']+'-poster.jpg')
 if p.exists() and not target_poster.exists():shutil.copy2(p,target_poster)
 m['source_folder']=m.get('source_folder',m['folder']);m.setdefault('source_video',m['video']);m['published_folder']='/'+str(target.relative_to(ROOT));m['video']='/'+str(target_video.relative_to(ROOT));m['poster']='/'+str(target_poster.relative_to(ROOT)) if target_poster.exists() else None
 if sha256(target_video)!=m['video_sha256']:raise RuntimeError('video hash mismatch: '+m['id'])
vision=data['vision_match'];vsource=ROOT/'results/bhuman-native/vision-native-capture-20261025';vtarget=base/'vision-bhuman-k1-3v3';vtarget.mkdir(exist_ok=True)
for name in ['vision-result.json','match.csv']:
 shutil.copy2(vsource/name,vtarget/name)
vision.update(source_folder=vision.get('folder'),published_folder='/'+str(vtarget.relative_to(ROOT)),source_video=vision['video'],video='/'+str((videos/'vision-bhuman-k1-3v3.mp4').relative_to(ROOT)),poster='/'+str((videos/'vision-bhuman-k1-3v3-poster.jpg').relative_to(ROOT)),result='/'+str((vtarget/'vision-result.json').relative_to(ROOT)),telemetry='/'+str((vtarget/'match.csv').relative_to(ROOT)))
if sha256(ROOT/vision['video'].lstrip('/'))!=vision['source_video_sha256']:raise RuntimeError('vision video hash mismatch')
write_json(rounddir/'index.json',data)
archive=json.loads((rounddir/'video-archive.json').read_text());by={x['match']:x for x in archive['videos']}
for m in data['matches']:
 by[m['id']]['video']=m['video'];by[m['id']]['sha256']=m['video_sha256'];by[m['id']]['poster']=m['poster']
vision_info=by['vision-bhuman-k1-3v3'];vision_info.update(video=vision['video'],poster=vision['poster'],sha256=vision['source_video_sha256'],bytes=(ROOT/vision['video'].lstrip('/')).stat().st_size)
write_json(rounddir/'video-archive.json',archive)
print('Packaged',len(data['matches']),'simulation replays and one vision recording')
