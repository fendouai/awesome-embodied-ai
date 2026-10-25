"""Synchronized split replay, source integration states; no pose animation."""
from pathlib import Path
import os,sys,json,bisect,subprocess,wave
os.environ.setdefault('MUJOCO_GL','cgl')
import numpy as np,mujoco
from PIL import Image,ImageDraw
from combatlab.models import ROOT
from combatlab.engine import STATE_SPEC,write_json,sha256
from combatlab.storage import load_model
from combatlab.render import font,COLORS

def render(folder):
 folder=Path(folder);result=json.loads((folder/'result.json').read_text());tele=json.loads((folder/'telemetry.json').read_text());raw=np.load(folder/'trajectory.npz');states=raw['integration_states'];dt=float(raw['timestep']);m=load_model(folder);d=mujoco.MjData(m);r=mujoco.Renderer(m,720,640);opt=mujoco.MjvOption();camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;quad=result['robots'][0] in ['go2','a1','go1','b2','go2w','b2w'];duration=float(result['duration_s']);disturbance=np.load(folder/'disturbance.npz')['samples'];times=[v['time'] for v in tele]
 visual_bodies=set(int(m.geom_bodyid[i]) for i in range(m.ngeom) if m.geom_contype[i]==0 and m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH)
 for i in range(m.ngeom):
  name=m.body(m.geom_bodyid[i]).name or ''
  if name.startswith(('p0/','p1/')):
   side=int(name[1]);hidden=m.geom_contype[i]!=0 and int(m.geom_bodyid[i]) in visual_bodies;m.geom_group[i]=5 if hidden else 2+side
  elif m.geom(i).name=='ground':m.geom_matid[i]=-1;m.geom_rgba[i]=[.20,.24,.28,1]
 if m.nmat:m.mat_shininess[:]=.4;m.mat_specular[:]=.4
 opt.geomgroup[:]=1;opt.geomgroup[5]=0
 audio=np.zeros(round((duration+.1)*48000))
 if result['task']=='push':
  u=np.arange(7200)/48000;tone=.24*np.sin(2*np.pi*95*u)*np.exp(-u*28);audio[144000:144000+len(tone)]+=tone
 with wave.open(str(folder/'test-cue.wav'),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(48000);f.writeframes((audio*32767).astype('<i2').tobytes())
 out=folder/'comparison.mp4';tmp=folder/'comparison-silent.mp4';p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','24','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','21','-pix_fmt','yuv420p',str(tmp)],stdin=subprocess.PIPE)
 try:
  for frame in range(round(duration*24)+1):
   t=min(duration,frame/24);idx=min(len(states)-1,round(t/dt));mujoco.mj_setState(m,d,states[idx],STATE_SPEC);mujoco.mj_forward(m,d);v=tele[max(0,bisect.bisect_right(times,t)-1)];panels=[]
   for side in range(2):
    pos=v['actors'][side]['base_position'];camera.lookat[:]=[pos[0],pos[1],(.4 if result['robots'][0] in ['b2','b2w'] else .25 if result['robots'][0]=='go2w' else .2) if quad else min(.85,max(.18,pos[2]*.8)) if (result['robots'][0] in ['h1_full','h1_2_full'] or result['task']=='motion') else .65];camera.distance=(2.5 if result['robots'][0] in ['b2','b2w'] else 1.8 if result['robots'][0]=='go2w' else 1.55) if quad else 4.0 if (result['robots'][0] in ['h1_full','h1_2_full'] or result['task']=='motion') else 3.2;camera.azimuth=132;camera.elevation=-15;opt.geomgroup[2]=side==0;opt.geomgroup[3]=side==1;r.update_scene(d,camera=camera,scene_option=opt)
    if result['task']=='push' and 3<=t<3.1:
     scene=r.scene;force=disturbance[min(round(t/dt),len(disturbance)-1),1+side*3:4+side*3];norm=np.linalg.norm(force)
     if norm>0:
      g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([1,.4,.1,1],dtype=np.float32));tip=np.asarray(pos);mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.02,tip-force/norm*.45,tip);scene.ngeom+=1
    panels.append(r.render().copy())
   image=Image.fromarray(np.concatenate(panels,axis=1));draw=ImageDraw.Draw(image)
   draw.rectangle((0,0,1280,108),fill='#080f18');draw.rectangle((0,554,1280,720),fill='#080f18');draw.line((640,0,640,720),fill='#3a5166',width=2)
   def text(x,y,s,size=18,c='#dce7f3',bold=False):draw.text((x,y),s,font=font(size,bold),fill=c)
   for side in range(2):
    x=22+side*640;a=v['actors'][side];summary=result['actors'][side];color=COLORS[side];text(x,12,a['robot'],23,color,True);label=summary['label'];text(x,48,label,17);text(x,77,f'{result["task"].upper()} / SEED {result["seed"]} / {t:05.2f}s',13)
    text(x,569,f'TILT {a["tilt_deg"]:.1f} deg   SPEED {a["speed_m_s"]:.2f} m/s',20,color,True);text(x,602,f'WORK {a["energy_positive_j"]:.1f} J   JOINT LOAD {a["torque_utilization"]*100:.0f}%',17)
    if result['task']=='push':text(x,635,f'PULSE: mass x 0.25 m/s / 0.1s  |  J={np.linalg.norm(summary["delivered_impulse_ns"]):.2f} N s',14)
    elif result['task']=='motion':text(x,635,f'JOINT REF ERROR  {a["reference_joint_rmse_rad"]:.4f} rad',17)
    else:text(x,635,'COMMAND: +0.30 m/s after1.5s / same body and timeline',14)
    text(x,672,'NO INTER-ROBOT CONTACT / SYNTHETIC TEST CUE / NO HEALTH SCORE',10,'#8ea6ba')
   p.stdin.write(np.asarray(image).tobytes())
   if frame==min(84,round(duration*12)):image.save(folder/'comparison-poster.jpg',quality=93)
  p.stdin.close();assert p.wait()==0
  subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(tmp),'-i',str(folder/'test-cue.wav'),'-c:v','copy','-c:a','aac','-shortest','-movflags','+faststart',str(out)],check=True)
 finally:
  r.close();raw.close()
  if p.poll() is None:p.terminate()
  if tmp.exists():tmp.unlink()
 manifest=json.loads((folder/'manifest.json').read_text())
 for name in ['comparison.mp4','comparison-poster.jpg','test-cue.wav']:manifest[name]=sha256(folder/name)
 write_json(folder/'manifest.json',manifest);return out
if __name__=='__main__':print(render(Path(sys.argv[1])),flush=True)
