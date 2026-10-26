"""Full, uncut recordings derived from preserved physical states; no pose animation."""
from pathlib import Path
import json,subprocess,wave,bisect
import numpy as np,mujoco
from PIL import Image,ImageDraw
from .engine import STATE_SPEC
from .render import font
from .research_archive import atomic_json,digest,event

NAMES={'g1_full':'UNITREE G1','h1_full':'UNITREE H1','h1_2_full':'UNITREE H1-2','k1':'BOOSTER K1','go2':'UNITREE GO2','a1':'UNITREE A1','go1':'UNITREE GO1','b2':'UNITREE B2','go2w':'UNITREE GO2-W','b2w':'UNITREE B2-W'}
COLORS=['#77d9ff','#ffbc77']
class PhysicalRecording:
 def __init__(self,folder):
  self.folder=Path(folder);raw=self.folder/'raw';self.metadata=json.loads((raw/'metadata.json').read_text());self.result=json.loads((self.folder/'result.json').read_text());self.m=mujoco.MjModel.from_binary_path(str(raw/'scene.mjb'));self.d=mujoco.MjData(self.m);states=[];contacts=[]
  for path in sorted(raw.glob('physics-*.npz')):
   with np.load(path) as p:states.append(p['integration_states']);contacts.append(p['contacts']);assert len(set(p['episode'].tolist()))==1
  self.states=np.concatenate(states);self.contacts=np.concatenate(contacts);self.times=self.states[:,0];self.duration=float(self.times[-1]);self.dt=float(self.m.opt.timestep);self.roots=[next(j for j in range(self.m.njnt) if self.m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and (self.m.body(self.m.jnt_bodyid[j]).name or '').startswith(f'p{s}/')) for s in range(2)];self.qadr=[int(self.m.jnt_qposadr[j]) for j in self.roots];self.vadr=[int(self.m.jnt_dofadr[j]) for j in self.roots];self.force=np.zeros(len(self.states))
  for row in self.contacts:
   idx=int(row[0])
   if 0<=idx<len(self.force):self.force[idx]+=max(0.,row[7])
  self.impulse=np.cumsum(self.force)*self.dt;self.io=json.loads((self.folder/'control-io.json').read_text()) if (self.folder/'control-io.json').exists() else [];self.io_times=[r['time'] for r in self.io];self.initiative=np.cumsum([r['initiated_contact_impulse_ns'] for r in self.io],axis=0) if self.io else np.zeros((1,2));self.renderer=mujoco.Renderer(self.m,720,1280);self.opt=mujoco.MjvOption();self.cam=mujoco.MjvCamera();self.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
  visual={int(self.m.geom_bodyid[g]) for g in range(self.m.ngeom) if self.m.geom_contype[g]==0 and self.m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH}
  for g in range(self.m.ngeom):
   if int(self.m.geom_bodyid[g]) in visual and self.m.geom_contype[g]:self.m.geom_group[g]=5
  self.opt.geomgroup[5]=0
 def frame(self,t):
  idx=int(np.clip(np.searchsorted(self.times,t,side='right')-1,0,len(self.times)-1));mujoco.mj_setState(self.m,self.d,self.states[idx],STATE_SPEC);mujoco.mj_forward(self.m,self.d);pos=[self.d.qpos[q:q+3].copy() for q in self.qadr];center=np.mean(pos,axis=0);sep=np.linalg.norm(pos[0][:2]-pos[1][:2]);self.cam.lookat[:]=[center[0],center[1],max(.2,center[2]*.8)];self.cam.distance=max(2.8,2.2+sep*1.2,max(p[2] for p in pos)*4);self.cam.azimuth=95;self.cam.elevation=-15;self.renderer.update_scene(self.d,camera=self.cam,scene_option=self.opt);im=Image.fromarray(self.renderer.render().copy());draw=ImageDraw.Draw(im)
  def txt(x,y,s,size=18,c='#e9f3ff',bold=False):draw.text((x,y),s,font=font(size,bold),fill=c)
  draw.rectangle((0,0,1280,105),fill='#08121e');draw.rectangle((0,570,1280,720),fill='#08121e')
  for side in range(2):
   x=24 if side==0 else 826;key=self.metadata['robots'][side];txt(x,18,NAMES.get(key,key.upper()),26,COLORS[side],True);txt(x,57,'LEARNED OPPONENT-CONDITIONED CONTROL',12);q=self.d.qpos[self.qadr[side]:self.qadr[side]+7];tilt=np.degrees(np.arccos(np.clip(2*(q[3]**2+q[6]**2)-1,-1,1)));speed=np.linalg.norm(self.d.qvel[self.vadr[side]:self.vadr[side]+2]);txt(x,585,f'TILT {tilt:.1f} deg / BASE {q[2]:.3f} m',19,COLORS[side]);txt(x,623,f'SPEED {speed:.2f} m/s',20);i=max(0,min(len(self.initiative)-1,bisect.bisect_right(self.io_times,t)-1));txt(x,659,f'INITIATED CONTACT {self.initiative[i,side]:.2f} N s',15)
  txt(538,16,f'{min(t,self.duration):05.2f} s',30,bold=True);txt(515,59,'SIM-STATE PERCEPTION',13);txt(450,585,f'CONTACT {self.force[idx]:,.0f} N',22,'#ffcf91',True);txt(450,626,f'NORMAL IMPULSE {self.impulse[idx]:.1f} N s',17);txt(450,661,'RECORDED PHYSICS / NO HIDDEN ROOT FORCES',11);txt(24,696,'BOTSTEST / EVALUATION, NOT A TOURNAMENT RESULT / SYNTHETIC CONTACT AUDIO',11,'#91aabd')
  if t>=self.duration-1e-7:
   passed=self.result.get('continuous_engagement',False);draw.rounded_rectangle((270,255,1010,445),12,fill='#0a1928',outline='#e4bd83',width=2);txt(310,278,'CONTACT GATE PASSED' if passed else 'CONTACT GATE NOT PASSED',32,'#ffcf91',True);txt(310,333,'HELD-OUT INITIAL CONDITION / NO RANDOM EXPLORATION',17);txt(310,373,'FINAL FRAME HOLD / NOT HARDWARE DAMAGE',17);txt(310,408,f"SOURCE CHECKPOINT {self.result.get('checkpoint_sha256','')[:16]}",14)
  return np.asarray(im)
 def close(self):self.renderer.close()

def render(folder):
 folder=Path(folder);out=folder/'video';out.mkdir(exist_ok=True);r=PhysicalRecording(folder);fps=24;hold=2;silent=out/'silent.mp4';audio=out/'synthetic-contact.wav';video=out/'full.mp4';sr=48000;sound=np.zeros(round((r.duration+hold+1)*sr));active=np.flatnonzero(r.force>3);rng=np.random.default_rng(2030)
 if len(active):
  for ids in np.split(active,np.flatnonzero(np.diff(active)*r.dt>.08)+1):
   i=round(r.times[int(ids[0])]*sr);n=4800;u=np.arange(n)/sr;peak=float(r.force[ids].max());amp=min(.5,.05+.05*np.log10(peak));pulse=amp*(np.sin(2*np.pi*95*u)*np.exp(-30*u)+np.convolve(rng.normal(0,1,n),np.ones(8)/8,'same')*np.exp(-50*u));sound[i:i+n]+=pulse[:min(n,len(sound)-i)]
 with wave.open(str(audio),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr);f.writeframes((np.tanh(sound)*28000).astype('<i2').tobytes())
 p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','22','-preset','veryfast','-pix_fmt','yuv420p',str(silent)],stdin=subprocess.PIPE)
 try:
  count=round(r.duration*fps)+1;stills=set([0,count//4,count//2,3*count//4,count-1])
  for i in range(count):
   pix=r.frame(min(i/fps,r.duration));p.stdin.write(pix.tobytes())
   if i in stills:Image.fromarray(pix).save(out/f'frame-{i:05}.jpg',quality=92)
   if i==count//4:Image.fromarray(pix).save(out/'poster.jpg',quality=92)
  final=r.frame(r.duration)
  for _ in range(fps*hold):p.stdin.write(final.tobytes())
  p.stdin.close();assert p.wait()==0
  subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(silent),'-i',str(audio),'-c:v','copy','-c:a','aac','-b:a','96k','-shortest','-movflags','+faststart',str(video)],check=True);atomic_json(out/'manifest.json',dict(video=video.name,video_sha256=digest(video),source_scene_sha256=digest(folder/'raw/scene.mjb'),source_chunks_manifest_sha256=digest(folder/'raw/chunks.jsonl'),source_result_sha256=digest(folder/'result.json'),physics_duration_s=r.duration,final_frame_hold_s=hold,fps=fps,synthetic_audio=True,physics_unchanged=True,full_uncut_episode=True));event('evaluation_video_rendered',folder=str(folder),video=str(video),sha256=digest(video));return video
 finally:
  r.close()
  if p.poll() is None:p.terminate()
  silent.unlink(missing_ok=True)
