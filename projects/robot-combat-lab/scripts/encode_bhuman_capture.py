"""Encode actual native-window samples at their measured wall-clock cadence."""
from pathlib import Path
import json,subprocess,hashlib
from PIL import Image
from combatlab.models import ROOT
from combatlab.engine import write_json,sha256
folder=ROOT/'results/bhuman-native/capture-20261025';capture=json.loads((folder/'capture-times.json').read_text());frames=capture['frames'];lines=['ffconcat version 1.0']
for i,frame in enumerate(frames):
    next_time=frames[i+1]['finished_ms'] if i+1<len(frames) else capture['finished_ms']
    lines.extend(["file '"+frame['file']+"'",'duration '+str(max(.001,(next_time-frame['finished_ms'])/1000))])
lines.append("file '"+frames[-1]['file']+"'");(folder/'frames.ffconcat').write_text('\n'.join(lines)+'\n')
# Spatial crop only, keeping both goals, players and scoreboard. No motion interpolation.
subprocess.run(['ffmpeg','-y','-loglevel','error','-safe','0','-f','concat','-i',str(folder/'frames.ffconcat'),'-vf','crop=960:430:30:200,scale=1280:-2','-fps_mode','vfr','-c:v','libx264','-crf','19','-preset','veryfast','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'battle.mp4')],check=True)
Image.open(folder/frames[len(frames)//2]['file']).crop((30,200,990,630)).save(folder/'poster.jpg',quality=95)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height,nb_frames,avg_frame_rate','-of','json',str(folder/'battle.mp4')]))
unique=len(set(sha256(folder/f['file']) for f in frames));duration=(frames[-1]['finished_ms']-frames[0]['finished_ms'])/1000
result=dict(backend='B-Human native SimRobot',source='actual native window recorded using CUA screenshots',scene='MacAcceptance3v3M',robot_count=6,robot_type='K1',perception='Fast ground truth',physics='MuJoCo 3.3.5, source 2ms step; native viewport recording is not full-state replay',capture_frames=len(frames),unique_frames=unique,capture_duration_s=duration,capture_average_fps=(len(frames)-1)/duration,video_probe=probe,temporal_interpolation=False,speed='Measured wall-clock frame intervals retained; no speedup',spatial_crop=[30,200,960,430],limitations=['Approximately 4 captured frames/s, distinct from 500Hz physics','Not a complete match or physics-state archive','Referee resets follow original soccer rules; not continuous combat'],video_sha256=sha256(folder/'battle.mp4'))
write_json(folder/'result.json',result);write_json(folder/'manifest.json',{str(p.relative_to(folder)):sha256(p) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json'})
write_json(folder.parent/'latest.json',dict(folder=str(folder.relative_to(ROOT)),result=result));print(json.dumps(result,ensure_ascii=False,indent=2))
