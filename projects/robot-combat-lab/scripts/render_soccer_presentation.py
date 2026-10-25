"""Measured native soccer broadcast: source frames + native referee/ball telemetry."""
from pathlib import Path
import json,bisect,wave,subprocess
import numpy as np
from PIL import Image,ImageDraw
from combatlab.models import ROOT
from combatlab.render import font
from combatlab.engine import write_json,sha256
folder=ROOT/'results/bhuman-native/studio-match-clock';index=[line.split(',') for line in (folder/'frame-times.csv').read_text().splitlines()];frame_wall=np.array([int(x[0]) for x in index]);frame_times=np.array([float(x[2]) for x in index]);tele=np.loadtxt(folder/'match.csv',delimiter=',');start=frame_times[0];duration=(frame_times[-1]-start)/1000.;n=round(duration*30)+1
# Align each rendered native frame with its preceding controller telemetry.
frame_rows=tele[np.maximum(0,np.searchsorted(tele[:,0],frame_wall,side='right')-1)]
speed=np.full(len(frame_rows),np.nan)
for i in range(1,len(frame_rows)):
 dt=(frame_times[i]-frame_times[i-1])/1000;delta=np.linalg.norm(frame_rows[i,6:8]-frame_rows[i-1,6:8]);same_ref=np.array_equal(frame_rows[i,3:6],frame_rows[i-1,3:6])
 if dt>0 and dt<.2 and delta<.6 and same_ref and frame_rows[i,3]==3:speed[i]=delta/dt
# Synthetic referee/goal cues only, never fabricate kick/contact events.
sr=48000;audio=np.zeros(round((duration+.2)*sr));events=[]
for i in range(1,len(frame_rows)):
 goal=bool(np.any(frame_rows[i,4:6]>frame_rows[i-1,4:6]));phase=bool(frame_rows[i,3]!=frame_rows[i-1,3]);t=(frame_times[i]-start)/1000
 if t<0 or not (goal or phase):continue
 events.append(dict(time_s=t,type='goal_score_change' if goal else 'referee_phase_change',score=frame_rows[i,4:6].astype(int).tolist(),phase=int(frame_rows[i,3])))
 length=.55 if goal else .20;u=np.arange(round(length*sr))/sr
 tone=.16*(np.sin(2*np.pi*660*u)+.5*np.sin(2*np.pi*880*u)+.3*np.sin(2*np.pi*1100*u))*np.exp(-u*5) if goal else .14*np.sin(2*np.pi*2100*u+2*np.sin(2*np.pi*30*u))*np.minimum(u*80,1)*np.exp(-u*15)
 k=round(t*sr);count=min(len(tone),len(audio)-k)
 if count>0:audio[k:k+count]+=tone[:count]
# Low-level movement cue follows valid measured ball speed; never a fake kick.
for frame in range(round(duration*30)):
 t=frame/30;frame_idx=max(0,bisect.bisect_right(frame_times,start+t*1000)-1);v=speed[frame_idx]
 if not np.isfinite(v) or v<.15:continue
 k=round(t*sr);count=min(round(sr/30),len(audio)-k);u=np.arange(count)/sr+t
 audio[k:k+count]+=.035*np.sqrt(min(v,8))*np.sin(2*np.pi*(180+min(v,8)*32)*u)*np.sin(np.pi*np.arange(count)/max(count-1,1))**2
with wave.open(str(folder/'referee-sonification.wav'),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr);f.writeframes((np.clip(audio,-.8,.8)*32767).astype('<i2').tobytes())
cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-','-an','-c:v','libx264','-crf','19','-preset','veryfast','-pix_fmt','yuv420p',str(folder/'presentation-silent.mp4')];p=subprocess.Popen(cmd,stdin=subprocess.PIPE);last=-1;field=None
for i in range(n):
 t=min(duration,i/30);now=start+t*1000;idx=max(0,bisect.bisect_right(frame_times,now)-1);row=frame_rows[idx]
 if idx!=last:
  image=Image.open(folder/'frames'/index[idx][1]).convert('RGB');w,h=image.size
  # This is framebuffer content without toolbars; retain the complete playing field.
  field=image.crop((20,20,min(w,1010),min(h,455))).resize((1280,562),Image.Resampling.LANCZOS);last=idx
 canvas=Image.new('RGB',(1280,720),'#080f18');canvas.paste(field,(0,90));d=ImageDraw.Draw(canvas)
 def text(x,y,s,size=18,c='#dce7f3',bold=False):d.text((x,y),s,font=font(size,bold),fill=c)
 phase={0:'INITIAL',1:'READY',2:'SET',3:'PLAYING',4:'FINISHED'}.get(int(row[3]),'UNKNOWN');remaining=max(0,int(row[2]));score=[int(x) for x in row[4:6]]
 text(25,14,'B-HUMAN  /  K1 3v3',22,'#8fdaec',True);text(25,49,'NATIVE PHYSICS / GROUND-TRUTH FAST MODE',12)
 text(485,10,f'RED  {score[0]} : {score[1]}  WHITE',30,bold=True);text(536,50,f'{remaining//60:02d}:{remaining%60:02d}   {phase}',17,'#e5c997')
 text(1030,18,f'{t:05.2f} s',27,bold=True);text(1030,53,'SIMULATION TIME',12)
 d.rectangle((0,652,1280,720),fill='#080f18');ball_speed=f'{speed[idx]:.2f} m/s' if np.isfinite(speed[idx]) else 'UNKNOWN / RESET';text(25,666,'BALL SPEED  '+ball_speed,20,'#ffcf91',True);text(460,666,f'BALL XY  {row[6]:+.2f} / {row[7]:+.2f} m',19);text(950,666,'6 ROBOTS / 2 TEAMS',18)
 recent=next((e for e in reversed(events) if 0<=t-e['time_s']<.7),None)
 if recent:
  d.rounded_rectangle((390,100,890,146),7,fill='#0e1c2c');text(410,108,'GOAL / REFEREE SCORE UPDATE' if recent['type'].startswith('goal') else 'REFEREE / '+phase,19,'#ffcf91',True)
 text(25,698,'Position-derived ball speed; referee resets excluded. Synthetic ball-speed / referee cues; not microphone sound.',11,'#91a6ba')
 p.stdin.write(np.asarray(canvas).tobytes())
 if i==120:canvas.save(folder/'presentation-poster.jpg',quality=95)
p.stdin.close();assert p.wait()==0
subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(folder/'presentation-silent.mp4'),'-i',str(folder/'referee-sonification.wav'),'-c:v','copy','-c:a','aac','-b:a','128k','-shortest','-movflags','+faststart',str(folder/'presentation.mp4')],check=True)
# Remove only this script's transient silent duplicate; all source captures remain.
(folder/'presentation-silent.mp4').unlink()
result=dict(backend='B-Human native studio recording',duration_s=duration,source_frames=len(index),source_fps=(len(index)-1)/duration,encoded_fps=30,timing='physics clock; render cost does not change playback speed',wall_capture_duration_s=(frame_wall[-1]-frame_wall[0])/1000,telemetry_samples=len(tele),telemetry_columns=['wall_ms','controller_clock_ms','secs_remaining','phase','red_score','white_score','ball_x_m','ball_y_m']+[f'robot{i}_{a}_m' for i in range(6) for a in ['x','y']],score_source='Original native referee data',ball_speed_source='position-derived estimate from controller positions aligned to native framebuffer samples / MuJoCo physics-time delta',speed_unknown_rule='non-playing phase, referee change, dt outside (0,.2)s, or position discontinuity >=.6m',synthetic_referee_audio=True,ball_speed_sonification=True,events=events,physics_unchanged=True,temporal_interpolation=False,frame_selection='latest captured frame at recorded physics-clock timestamp; each frame tagged with native MuJoCo time; no invented motion',video_sha256=sha256(folder/'presentation.mp4'))
write_json(folder/'presentation.json',result);write_json(folder/'manifest.json',{str(x.relative_to(folder)):sha256(x) for x in folder.rglob('*') if x.is_file() and x.name!='manifest.json'});write_json(folder.parent/'presentation-latest.json',dict(folder=str(folder.relative_to(ROOT)),result=result));print(json.dumps(result,indent=2))
