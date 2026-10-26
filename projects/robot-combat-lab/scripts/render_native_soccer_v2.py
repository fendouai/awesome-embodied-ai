"""Render a single uninterrupted native visual match using monotonic physics steps."""
import argparse,bisect,csv,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from combatlab.commentary import rgba_video,probe
from combatlab.engine import write_json,sha256
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();folder=a.folder.resolve()
result=json.loads((folder/'native-result.json').read_text());tele=np.loadtxt(folder/'telemetry.csv',delimiter=',');rows=list(csv.reader((folder/'frame-times.csv').open()))
ready=tele[tele[:,3]>=1][0,0];rows=[r for r in rows if int(r[0])>=ready];walls=np.array([int(r[0]) for r in rows]);sim=np.array([float(r[2]) for r in rows]);assert len(rows)>100 and np.all(np.diff(sim)>0),'reset or missing camera timeline: not publishable'
times=(sim-sim[0])/1000.;duration=float(times[-1]);states={0:'INITIAL',1:'READY',2:'SET',3:'PLAYING',4:'FINISHED'}
wall_to_time=lambda wall:float(np.interp(wall,walls,times))
timeline=[];last=None
for r in tele:
 if r[0]<walls[0]:continue
 key=tuple(r[2:6].astype(int));
 if key!=last:timeline.append(dict(time=wall_to_time(r[0]),remaining_seconds=int(r[2]),state=int(r[3]),score=[int(r[4]),int(r[5])],native_wall_ms=int(r[0]),native_sim_ms=int(r[1])));last=key
write_json(folder/'timeline.json',timeline)
playing=tele[tele[:,3]==3];span=float(np.linalg.norm(np.ptp(playing[:,6:8],axis=0))) if len(playing) else 0
write_json(folder/'analysis.json',dict(active_ball_span_m=span,active_samples=len(playing),source_frames=len(rows),monotonic_physics_clock=True,quality='interactive' if span>=.1 else 'insufficient_ball_interaction'))
concat=folder/'frames.ffconcat'
lines=['ffconcat version 1.0']
for i,r in enumerate(rows):
 path=str(folder/'frames'/r[1]).replace("'","'\\''");lines.append(f"file '{path}'");lines.append(f'duration {times[i+1]-times[i]:.6f}' if i+1<len(rows) else 'duration 0.05')
concat.write_text('\n'.join(lines)+'\n')
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',22);cache={}
def hud(t):
 k=max(0,bisect.bisect_right([v['time'] for v in timeline],t)-1);v=timeline[k];key=(k,round(t))
 if key not in cache:
  im=Image.new('RGBA',(1280,64),(5,13,23,245));d=ImageDraw.Draw(im);d.text((18,18),'BOTSTEST / K1 3v3 / RGB VISION',font=font,fill='#9ce2ef');d.text((530,18),f'RED {v["score"][0]} : {v["score"][1]} WHITE',font=font,fill='#ffd18e');d.text((880,18),f'{states[v["state"]]} / {max(0,v["remaining_seconds"]):03d}s',font=font,fill='white');cache[key]=np.asarray(im)
 return cache[key]
layer=folder/'hud.mov';rgba_video(layer,duration+1,1280,64,hud,4)
video=folder/'source.mp4';subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(concat),'-i',str(layer),'-filter_complex','[0:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[base];[base][1:v]overlay=0:0:eof_action=repeat[v]','-map','[v]','-r','24','-c:v','libx264','-preset','veryfast','-crf','20','-threads','2','-pix_fmt','yuv420p','-t',str(duration),'-movflags','+faststart',str(video)],check=True)
layer.unlink();result.update(video=video.name,video_sha256=sha256(video),probe=probe(video),source_duration_s=duration,active_ball_span_m=span,source_frames=len(rows),official_regulation=False,clock='monotonic native simulationStep*0.002s; no reset stitching')
write_json(folder/'result.json',result);print('VISION_RENDERED',folder,result['red'],result['white'],span,flush=True)
