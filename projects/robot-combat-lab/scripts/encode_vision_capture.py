"""Encode native SimRobot screenshots at their measured wall-clock intervals."""
from pathlib import Path
import json,subprocess
import numpy as np
from PIL import Image
from combatlab.models import ROOT
from combatlab.engine import write_json,sha256
folder=ROOT/'results/bhuman-native/vision-capture-20261025';capture=json.loads((folder/'capture-times.json').read_text());frames=capture['frames'];lines=['ffconcat version 1.0']
for i,item in enumerate(frames):
 next_ms=frames[i+1]['finished_ms'] if i+1<len(frames) else capture['finished_ms']
 lines.extend([f"file '{item['file']}'",'duration '+str(max(.001,(next_ms-item['finished_ms'])/1000))])
lines.append(f"file '{frames[-1]['file']}'");(folder/'frames.ffconcat').write_text('\n'.join(lines)+'\n')
video=folder/'vision-match.mp4'
subprocess.run(['ffmpeg','-y','-loglevel','error','-safe','0','-f','concat','-i',str(folder/'frames.ffconcat'),'-vf','fps=12','-fps_mode','cfr','-r','12','-c:v','libx264','-crf','20','-preset','veryfast','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
Image.open(folder/frames[len(frames)//2]['file']).save(folder/'vision-poster.jpg',quality=92)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height,avg_frame_rate','-of','json',str(video)]))
unique=len({sha256(folder/x['file']) for x in frames});duration=(frames[-1]['finished_ms']-frames[0]['finished_ms'])/1000
# Native scoreboard text is recorded by SimRobot; result is the score at cutoff.
result=dict(kind='camera_perception_visual_soccer_exhibition',state='recorded_segment',official_regulation_completed=False,scene='MacAcceptance3v3M ros3; normal-mode Game3v3M con; 3v3 K1',perception=dict(camera='SimRobot CameraColor RGB virtual-camera sensor',controller='B-Human normal camera-to-percept pipeline',not_fast_ground_truth=True),simulator=dict(engine='SimRobot Core3/MuJoCo 3.3.5',configured_step_s=.002,first_visible_step=44787,last_visible_step=111218,observed_physics_duration_s=(111218-44787)*.002,run='native local simulator'),source='Native SimRobot application screenshots captured by CUA; no synthetic frames',source_frames=len(frames),unique_frames=unique,capture_wall_duration_s=duration,source_capture_fps=(len(frames)-1)/duration,video=video.name,video_sha256=sha256(video),video_probe=probe,temporal_interpolation=False,score_at_final_frame=dict(red=2,white=0,source='Visible native scoreboard; red team led 2:0 at capture cutoff'), exhibition_judgement=dict(winner_at_cutoff='red',criteria='Native B-Human goals visible at timed segment cutoff',full_regulation_winner=False),limitations=['This is an exhibition capture segment, not a completed 10-minute regulation match','No scoring or referee event was inserted; the shown 2:0 is the native scoreboard at the capture cutoff','No match-wide official winner is claimed for a truncated regulation game','Native application audio was not captured'])
write_json(folder/'result.json',result)
write_json(folder/'manifest.json',{str(p.relative_to(folder)):sha256(p) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json'})
print(json.dumps(result,ensure_ascii=False,indent=2))
