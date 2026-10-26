"""Presentation-only replay: measured contacts, restrained studio rendering and sonification.
Never edits model dynamics, trajectory, policy actions or outcome. Audio is synthetic.
"""
from pathlib import Path
import bisect,json,subprocess,wave
import numpy as np,mujoco
from PIL import Image,ImageDraw
from .render import ReplayRenderer,font,COLORS
from .engine import STATE_SPEC,write_json,sha256

class ContactTrack:
    def __init__(self,folder):
        raw=np.load(Path(folder)/'contacts.npz');self.rows=raw['samples'];traj=np.load(Path(folder)/'trajectory.npz');self.dt=float(traj['timestep']);self.start=float(traj['integration_states'][0,0]);self.duration=float(traj['integration_states'][-1,0]-self.start);traj.close()
        self.force=np.zeros(round(self.duration/self.dt)+1)
        for row in self.rows:
            i=round((row[0]-self.start)/self.dt)
            if 0<=i<len(self.force):self.force[i]+=np.linalg.norm(row[6:9])
        self.impulse=np.cumsum(self.force)*self.dt;active=np.flatnonzero(self.force>3);self.events=[]
        if len(active):
            for ids in np.split(active,np.flatnonzero(np.diff(active)*self.dt>.08)+1):
                lo,hi=int(ids[0]),int(ids[-1]);self.events.append(dict(start_s=lo*self.dt,end_s=hi*self.dt,peak_n=float(self.force[lo:hi+1].max()),impulse_ns=float(self.force[lo:hi+1].sum()*self.dt)))
    def at(self,t,window=1/30):
        i=int(np.clip(round(t/self.dt),0,len(self.force)-1));lo=max(0,i-round(window/self.dt));return float(self.force[lo:i+1].max()),float(self.impulse[i])
    def sound(self,out):
        sr=48000;y=np.zeros(round((self.duration+.1)*sr));rng=np.random.default_rng(2030)
        for e in self.events:
            n=round(sr*.15);t=np.arange(n)/sr;noise=rng.normal(0,1,n);noise=np.convolve(noise,np.ones(12)/12,'same');amp=min(.6,.11+.10*np.log10(1+e['peak_n']));pulse=amp*(.65*np.sin(2*np.pi*(85*t-100*t*t))*np.exp(-t*36)+.6*noise*np.exp(-t*52));i=round(e['start_s']*sr);count=min(n,len(y)-i)
            if count>0:y[i:i+count]+=pulse[:count]
        y=np.tanh(y)*.78
        with wave.open(str(out),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr);f.writeframes((y*32767).astype('<i2').tobytes())
        return int(np.count_nonzero(y))

class BroadcastRenderer(ReplayRenderer):
    def __init__(self,folder):
        super().__init__(folder);self.contact=ContactTrack(folder);self.lookat_height=.62
        # Only shading/appearance fields; no geometry, mass, contact or servo edits.
        self.m.vis.headlight.ambient[:]=[.24,.26,.30];self.m.vis.headlight.diffuse[:]=[.58,.58,.58];self.m.vis.headlight.specular[:]=[.32,.32,.32]
        self.m.light_diffuse[:]=[[.95,.92,.85] if i==0 else [.42,.56,.75] for i in range(self.m.nlight)]
        if self.m.nmat:self.m.mat_specular[:]=.45;self.m.mat_shininess[:]=.42;self.m.mat_reflectance[:]=.035
        for i in range(self.m.ngeom):
            n=self.m.body(self.m.geom_bodyid[i]).name or '';g=self.m.geom(i).name or ''
            if n.startswith(('p0/','p1/')):
                side=int(n[1]);dark=any(k in n for k in ['ankle','wrist','elbow','hip_yaw','waist'])
                self.m.geom_rgba[i]=[.12,.15,.19,1] if dark else ([.75,.80,.85,1] if side==0 else [.42,.47,.53,1])
                if 'head' in n or 'torso' in n or n.endswith('/pelvis'):self.m.geom_rgba[i]=[.76,.82,.87,1] if side==0 else [.48,.53,.60,1]
            if g=='ground':self.m.geom_matid[i]=-1;self.m.geom_rgba[i]=[.21,.25,.30,1]
        # Hide collision overlays where source visual meshes exist. Physical
        # colliders remain enabled and unchanged; geom groups affect this viewer.
        if self.result['track']=='robostriker_cpu_port':
            visual_bodies=set(int(self.m.geom_bodyid[i]) for i in range(self.m.ngeom) if self.m.geom_contype[i]==0 and self.m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH)
            for i in range(self.m.ngeom):
                if int(self.m.geom_bodyid[i]) in visual_bodies and self.m.geom_contype[i]!=0:self.m.geom_group[i]=5
            self.opt.geomgroup[5]=0
        # Existing MuJoCo geometry is retained; digital humans stay visibly identified.
    def frame(self,t,**kwargs):
        ti=max(0,bisect.bisect_right(self.times,self.start_time+t)-1);v=self.tele[min(ti,len(self.tele)-1)];separation=np.linalg.norm(np.asarray(v['actors'][0]['base_position'])[:2]-np.asarray(v['actors'][1]['base_position'])[:2]);distance=max(3.15,2.4+separation*.65)
        pixels=super().frame(t,azimuth=92,elevation=-12,distance=distance,hud=False);return self.overlay(pixels,v,t)
    def overlay(self,pixels,v,t):
        im=Image.fromarray(pixels).convert('RGBA');over=Image.new('RGBA',im.size);d=ImageDraw.Draw(over)
        def txt(x,y,s,size=17,c='#dce7f3',bold=False):d.text((x,y),s,font=font(size,bold),fill=c)
        def panel(box):d.rounded_rectangle(box,radius=7,fill=(8,15,24,237),outline=(55,74,90,190))
        is_ncp=self.result['track']=='ncp_cpu_port';panel((18,14,480,100));panel((800,14,1262,100));panel((518,14,762,83))
        for side,a in enumerate(v['actors']):
            x=34 if side==0 else 816;color=COLORS[side];name=('NCP BOXER '+a['policy'].rsplit('_',1)[-1]) if is_ncp else ('UNITREE G1 / '+a['policy'].removeprefix('robostriker_').upper());txt(x,23,name,23,color,True);txt(x,59,'DIGITAL HUMAN' if is_ncp else 'SOURCE URDF / 29 JOINTS',12);txt(x+275,61,a['state'],14,color,True)
        txt(553,23,f'{v["time"]:05.2f} s',28,bold=True);txt(545,59,'RECORDED PHYSICS / 1x',12)
        force,impulse=self.contact.at(t);episodes=[e for e in self.contact.events if e['start_s']<=t];recent=next((e for e in reversed(episodes) if t-e['start_s']<.7),None)
        if recent:
            panel((488,106,792,166));txt(510,115,'CONTACT EPISODE',17,'#ffce8d',True);txt(510,141,f'PEAK {self.contact.force[round(recent["start_s"]/self.dt):max(round(recent["start_s"]/self.dt)+1,round(t/self.dt)+1)].max():,.0f} N',14)
        panel((18,559,1262,705))
        for side,a in enumerate(v['actors']):
            x=34 if side==0 else 850;color=COLORS[side];txt(x,570,f'TILT {a["tilt_deg"]:4.1f} deg',21,color,True);txt(x+205,574,f'BASE {a["height_m"]:.3f} m',15)
            txt(x,608,'JOINT LOAD / TORQUE LIMIT',12);ratio=min(1,a['torque_utilization']);d.rounded_rectangle((x,632,x+370,639),3,fill='#263746');d.rounded_rectangle((x,632,x+max(1,370*ratio),639),3,fill=color);txt(x+305,606,f'{100*a["torque_utilization"]:.0f}%',17,color)
            txt(x,654,f'SPEED {a["speed_m_s"]:.2f} m/s   WORK {a["energy_positive_j"]:.0f} J',15)
        txt(470,569,f'{force:,.0f} N',27,'#ffcf91',True);txt(470,603,'CONTACT PEAK / LAST 33ms',11);txt(470,627,f'IMPULSE {impulse:.1f} N s',17);txt(470,655,f'EPISODES {len(episodes):02d}  /  NOT HIT SCORE',11)
        # Three-second force history; fixed per-record range so peaks stay comparable.
        end=min(round(t/self.dt),len(self.contact.force)-1);lo=max(0,end-round(3/self.dt));ys=self.contact.force[lo:end+1];peak=max(1,float(self.contact.force.max()));xs=np.linspace(670,824,len(ys));pts=[(float(x),651-70*float(y)/peak) for x,y in zip(xs,ys)]
        if len(pts)>1:d.line(pts,fill='#efba78',width=2)
        txt(677,660,'FORCE / 3 s',11);txt(24,534,'BOTSTEST  /  STUDIO REPLAY',12,'#a8bdcd',True);txt(24,688,'Synthetic contact sonification. No recorded physical sound. No calibrated hardware damage.',11,'#8da5b8')
        return np.asarray(Image.alpha_composite(im,over).convert('RGB'))

def render_presentation(folder):
    folder=Path(folder);r=BroadcastRenderer(folder);r.contact.sound(folder/'contact-sonification.wav');temp=folder/'presentation-silent.mp4';out=folder/'presentation.mp4';p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-','-an','-c:v','libx264','-crf','19','-preset','veryfast','-pix_fmt','yuv420p',str(temp)],stdin=subprocess.PIPE)
    try:
        for i in range(round(r.duration*30)+1):
            pix=r.frame(min(i/30,r.duration));p.stdin.write(pix.tobytes())
            if i==120:Image.fromarray(pix).save(folder/'presentation-poster.jpg',quality=95)
        p.stdin.close();assert p.wait()==0
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(temp),'-i',str(folder/'contact-sonification.wav'),'-c:v','copy','-c:a','aac','-b:a','128k','-shortest','-movflags','+faststart',str(out)],check=True)
        write_json(folder/'presentation.json',dict(source_trajectory_sha256=sha256(folder/'trajectory.npz'),source_contacts_sha256=sha256(folder/'contacts.npz'),source_policy_io_sha256=sha256(folder/'policy_io.npz'),video=out.name,video_sha256=sha256(out),fps=30,physics_unchanged=True,shading_only=True,synthetic_contact_audio=True,audio_is_physical_recording=False,contact_episode_gap_s=.08,contact_audio_gate_n=3.,events=r.contact.events))
        manifest=json.loads((folder/'manifest.json').read_text())
        for name in ['presentation.mp4','presentation-poster.jpg','presentation.json','contact-sonification.wav']:manifest[name]=sha256(folder/name)
        write_json(folder/'manifest.json',manifest)
    finally:
        r.close()
        if p.poll() is None:p.terminate()
        if temp.exists():temp.unlink()
    return out
