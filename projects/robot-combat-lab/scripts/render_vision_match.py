"""Broadcast genuine camera-based B-Human SimRobot frames, aligned to native physics time."""
from pathlib import Path
import json,bisect,subprocess,wave
import numpy as np
from PIL import Image,ImageDraw
from combatlab.models import ROOT
from combatlab.render import font
from combatlab.engine import write_json,sha256

folder=ROOT/'results/bhuman-native/vision-native-capture-20261025';video=folder/'vision-final.mp4'
index=[x.split(',') for x in (folder/'frame-times.csv').read_text().splitlines() if x.strip()]
tele=np.loadtxt(folder/'match.csv',delimiter=',');walls=np.array([int(x[0]) for x in index]);sim=np.array([float(x[2]) for x in index]);start=sim[0];duration=(sim[-1]-start)/1000.;frames=round(duration*30)+1
rowids=np.maximum(0,np.searchsorted(tele[:,0],walls,side='right')-1);rows=tele[rowids]
stream=folder/'vision-silent.mp4'
cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-','-an','-c:v','libx264','-crf','19','-preset','veryfast','-pix_fmt','yuv420p',str(stream)]
proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);last=-1
phase_names={0:'INITIAL',1:'READY',2:'SET',3:'PLAYING',4:'FINISHED'}
for i in range(frames):
 t=min(duration,i/30);k=max(0,bisect.bisect_right(sim,start+t*1000)-1);row=rows[k]
 if k!=last:
  field=Image.open(folder/'frames'/index[k][1]).convert('RGB').resize((1280,672),Image.Resampling.LANCZOS);last=k
 canvas=Image.new('RGB',(1280,720),'#080f18');canvas.paste(field,(0,48));d=ImageDraw.Draw(canvas)
 def txt(x,y,s,size=17,c='#dce7f3',bold=False):d.text((x,y),str(s),font=font(size,bold),fill=c)
 d.rectangle((0,0,1280,48),fill='#080f18');txt(22,14,'BOTSTEST  /  B-HUMAN K1 3v3  /  RGB CAMERA PERCEPTION',17,'#88d9ea',True)
 txt(575,12,f'RED  {int(row[4])} : {int(row[5])}  WHITE',24,'#ffd28e',True);txt(1125,13,f'{duration-t:04.1f}s',18,'#edf4fa',True)
 d.rectangle((0,704,1280,720),fill='#080f18');txt(20,705,'SIMROBOT / B-HUMAN NORMAL CAMERA PIPELINE  ·  REAL PHYSICS FRAMEBUFFER  ·  NO INTERPOLATED FRAMES',10,'#aec0cf')
 if i==frames-1:canvas.save(folder/'vision-poster.jpg',quality=94)
 proc.stdin.write(np.asarray(canvas).tobytes())
# Hold the native final scoreboard visibly for four seconds; these are presentation frames only.
final=canvas.copy();d=ImageDraw.Draw(final);d.rounded_rectangle((340,250,940,440),radius=15,fill='#071321',outline='#e3ba79',width=3)
d.text((395,270),'VISUAL EXHIBITION / CAPTURE CUTOFF',font=font(24,True),fill='#a8c5d7')
d.text((395,323),f"RED  {int(rows[-1,4])} : {int(rows[-1,5])}  WHITE",font=font(40,True),fill='#ffd18e')
d.text((395,385),'B-HUMAN REFEREE SCORE · REGULATION CONTINUES',font=font(17),fill='#b8c8d6')
final_array=np.asarray(final)
for _ in range(30*4):proc.stdin.write(final_array.tobytes())
proc.stdin.close();assert proc.wait()==0
subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(stream),'-c:v','copy','-movflags','+faststart',str(video)],check=True);stream.unlink()
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height,avg_frame_rate,nb_frames','-of','json',str(video)]))
result=dict(kind='camera_perception_visual_soccer_exhibition',state='captured_score_at_cutoff',scene='MacAcceptance3v3M.ros3 / normal-mode Game3v3M.con',robot_count=6,robot_type='Booster K1',teams=['B-Human','B-Team'],perception=dict(camera='SimRobot CameraColor virtual RGB camera',controller='B-Human normal camera-to-percept modules including ECImageProvider and RobotBallPerceptor',fast_ground_truth_mode=False,source_configuration='Game3v3M'),physics=dict(engine='native SimRobot Core3 / MuJoCo 3.3.5',step_length_s=.002,source='Actual native framebuffer timestamps aligned to simulation physics clock'),score=dict(red=int(rows[-1,4]),white=int(rows[-1,5]),from_native_referee=True,official_full_regulation_completed=False),source_frames=len(index),unique_frames=len({sha256(folder/'frames'/x[1]) for x in index}),source_simulation_duration_s=duration,source_wall_capture_duration_s=(walls[-1]-walls[0])/1000.,source_capture_fps=(len(index)-1)/duration,video=video.name,video_sha256=sha256(video),video_probe=probe,temporal_interpolation=False,temporal_resampling='Nearest native framebuffer sample at each 30fps physics-clock output tick; four-second final-frame hold',audio='No application audio captured',limitations=['A camera-perception exhibition segment, not a completed ten-minute regulation match','2:0 is the native B-Human referee score at capture cutoff','No scripted ball motion, goals, manual score changes or extra forces'])
write_json(folder/'vision-result.json',result);manifest=json.loads((folder/'manifest.json').read_text()) if (folder/'manifest.json').exists() else {}
for p in folder.rglob('*'):
 if p.is_file() and p.name!='manifest.json':manifest[str(p.relative_to(folder))]=sha256(p)
write_json(folder/'manifest.json',manifest);write_json(folder.parent/'vision-result-latest.json',dict(folder=str(folder.relative_to(ROOT)),result=result));print(json.dumps(result,ensure_ascii=False,indent=2))
