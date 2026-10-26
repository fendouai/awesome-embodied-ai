from __future__ import annotations
import os
os.environ.setdefault('MUJOCO_GL','cgl')
from pathlib import Path
import json,subprocess,bisect,gzip
import numpy as np
import mujoco
from functools import lru_cache
from PIL import Image,ImageDraw,ImageFont
from .engine import STATE_SPEC,write_json,sha256
from .storage import load_model

FONT_DIR=Path('/System/Library/Fonts/Supplemental')
@lru_cache(maxsize=64)
def font(n,bold=False):
    for path in [FONT_DIR/('Arial Bold.ttf' if bold else 'Arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]:
        if path.exists():return ImageFont.truetype(str(path),n)
    return ImageFont.load_default()
COLORS=[(92,224,250),(255,172,103)]
NAMES={'robostriker_ego':'RoboStriker Stage3 / ego','robostriker_opponent':'RoboStriker Stage3 / opponent','ncp_boxing_106':'NCP / published boxer 106','ncp_boxing_100':'NCP / published boxer 100','unitree_rl_lab':'Unitree RL Lab / G1 v0','unitree_rl_mjlab':'Unitree RL MjLab / G1 v0','pd_stand':'Common PD + gravity compensation','nvidia_wbc':'NVIDIA WBC / Balance + Walk','unitree_full_gym':'Unitree RL Gym / full-body map','amp_recovery':'AMP locomotion + recovery','holosoma_fastsac':'Holosoma FastSAC','holosoma_ppo':'Holosoma PPO'}

class ReplayRenderer:
    def __init__(self,folder,width=1280,height=720):
        self.folder=Path(folder);self.m=load_model(self.folder);
        if os.environ.get("COMBATLAB_RENDER_LOD")=="1":
            from .presentation_lod import load_display_model
            self.m=load_display_model(self.folder)
            self.m.vis.quality.shadowsize=1024
            self.m.vis.quality.offsamples=0
        self.d=mujoco.MjData(self.m);self.raw=np.load(self.folder/'trajectory.npz');self.states=self.raw['integration_states'];self.tele=json.loads((self.folder/'telemetry.json').read_text());self.times=[x['time'] for x in self.tele];self.result=json.loads((self.folder/'result.json').read_text());self.dt=float(self.raw['timestep']);self.start_time=float(self.states[0,0]);self.duration=float(self.states[-1,0]-self.start_time);self.width=width;self.height=height;self.renderer=mujoco.Renderer(self.m,height,width);self.opt=mujoco.MjvOption()
        self.opt.geomgroup[:]=1
        self.push=json.loads((self.folder/'push-result.json').read_text()) if (self.folder/'push-result.json').exists() else None;self.disturbance=np.load(self.folder/'disturbance.npz')['samples'] if (self.folder/'disturbance.npz').exists() else None;self.current_push=np.zeros(3)
        self.cam=mujoco.MjvCamera();self.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    def frame(self,t,azimuth=105,elevation=-18,distance=4.6,hud=True):
        idx=int(np.clip(round(t/self.dt),0,len(self.states)-1));mujoco.mj_setState(self.m,self.d,self.states[idx],STATE_SPEC);mujoco.mj_forward(self.m,self.d)
        ti=max(0,min(len(self.tele)-1,bisect.bisect_right(self.times,float(self.d.time))-1));v=self.tele[ti]
        pos=np.mean([a['base_position'] for a in v['actors']],axis=0);self.cam.lookat[:]=[pos[0],pos[1],getattr(self,"lookat_height",.78)];self.cam.distance=distance;self.cam.azimuth=azimuth;self.cam.elevation=elevation
        self.renderer.update_scene(self.d,camera=self.cam,scene_option=self.opt)
        # Contact markers are measured world-space locations; decorative only.
        scene=self.renderer.scene
        if self.disturbance is not None:
            fi=min(max(idx-1,0),len(self.disturbance)-1);self.current_push=self.disturbance[fi,1:4];norm=np.linalg.norm(self.current_push)
            if norm>0 and scene.ngeom<scene.maxgeom:
                body=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,'p0/torso_link');tip=self.d.xipos[body];g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([1.,.25,.1,1.],dtype=np.float32));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.025,tip-self.current_push/norm*.65,tip);scene.ngeom+=1
        for c in v['contacts'][:8]:
            if scene.ngeom>=scene.maxgeom:break
            g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.024,0,0]),np.array(c['pos']),np.eye(3).ravel(),np.array([1.,.32,.14,.9],dtype=np.float32));scene.ngeom+=1
        if self.push:
            panels=[]
            for side in [0,1]:
                q=v['actors'][side]['base_position'];self.cam.lookat[:]=[q[0],q[1],.65];self.cam.distance=3.6;self.cam.azimuth=105;self.cam.elevation=-18;self.renderer.update_scene(self.d,camera=self.cam,scene_option=self.opt);scene=self.renderer.scene
                force=self.disturbance[fi,1+3*side:4+3*side];mag=np.linalg.norm(force)
                if mag>0 and scene.ngeom<scene.maxgeom:
                    body=self.m.body(f'p{side}/torso_link').id;tip=self.d.xipos[body];g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([1.,.25,.1,1.],dtype=np.float32));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.025,tip-force/mag*.65,tip);scene.ngeom+=1
                panels.append(self.renderer.render().copy()[:,self.width//4:self.width*3//4])
            pixels=np.concatenate(panels,axis=1);pixels[:,self.width//2-1:self.width//2+1]=[25,55,78]
        else:pixels=self.renderer.render().copy()
        return self.hud(pixels,v,ti) if hud else pixels
    def hud(self,pixels,v,ti):
        im=Image.fromarray(pixels).convert('RGBA');overlay=Image.new('RGBA',im.size,(0,0,0,0));dr=ImageDraw.Draw(overlay);W,H=im.size;s=W/1280
        def rect(box,fill,outline=None,width=1):dr.rectangle(tuple(int(x*s) for x in box),fill=fill,outline=outline,width=width)
        def text(x,y,t,size=18,color=(225,237,249),bold=False):dr.text((int(x*s),int(y*s)),str(t),font=font(max(10,int(size*s)),bold),fill=color)
        rect((0,0,1280,110),(6,13,23,238));rect((0,570,1280,720),(6,13,23,239));rect((0,109,1280,112),(53,88,116,255))
        text(24,10,'BOTSTEST  /  ROBOT COMBAT LAB',15,(128,159,187),True);text(990,10,'PHYSICS / NOT ANIMATION',14,(128,159,187))
        for i,a in enumerate(v['actors']):
            x=24 if i==0 else 780;c=COLORS[i];text(x,34,a['robot'].split(' / ')[0],25,c,True);text(x,68,NAMES.get(a['policy'],a['policy']),15,(174,195,214));text(x+290,38,a['state'],18,c,True)
        text(568,29,f'{v["time"]:05.2f} s',26,(242,249,255),True);text(554,68,({'robostriker_cpu_port':'ROBOSTRIKER / CPU','ncp_cpu_port':'NCP / CPU'}.get(self.result['track'],self.result['track'].upper()+' PROTOCOL')),13,(145,169,193))
        for i,a in enumerate(v['actors']):
            x=24 if i==0 else 810;c=COLORS[i]
            text(x,584,f'TILT  {a["tilt_deg"]:5.1f} deg',20,c,True);text(x+230,584,f'HEIGHT {a["height_m"]:.3f} m',18)
            rect((x,614,x+418,622),(33,49,65,255));rect((x,614,x+418*min(a['tilt_deg']/65,1),622),(*c,255))
            text(x,632,f'TORQUE LIMIT  {a["torque_utilization"]*100:5.1f}%',17);text(x+235,632,f'SPEED {a["speed_m_s"]:.2f} m/s',17)
            text(x,662,f'POSITIVE MECH. WORK  {a["energy_positive_j"]:.1f} J',17);text(x,691,'Tilt bar: 0-65 deg  |  No fictional health/damage',12,(118,147,174))
        text(480,583,(f'PUSH {np.linalg.norm(self.current_push):,.0f} N' if self.push else f'CONTACT {v["contact_force_n"]:,.0f} N'),21,(255,212,139),True);text(480,615,f'IMPULSE {(float(np.linalg.norm(self.disturbance[:max(0,round(v["time"]/self.dt)),1:4],axis=1).sum()*self.dt) if self.push else v["contact_impulse_ns"]):,.1f} N s',17);text(480,645,f'PEAK {(self.push["force_n"] if self.push and v["time"]>=3 else 0 if self.push else v["peak_force_n"]):,.0f} N',17);text(480,685,f'MuJoCo {mujoco.__version__} / dt {self.dt*1000:g}ms / recorded state replay',12,(118,147,174))
        if self.push:
            text(25,131,f'EXPLICIT TORSO PUSH: {self.push["force_n"]:g} N x 0.1 s = {self.push["impulse_ns"]:g} N s  /  {self.push["direction"].upper()}',18,(255,202,123),True)
            text(25,158,'SPLIT VIEWS: candidate / old neutral PD | same scene/time | arrows: direction only',14,(200,214,230))
        if self.result.get('continuous'):
            text(25,132,'CONTINUOUS CONTACT / FALL IS NOT A LOSS',19,(255,202,123),True)
            text(25,159,'HARDWARE DAMAGE: UNCALIBRATED' if self.result['continuous']['damage_prediction']=='not_calibrated' else 'PARAMETERIZED THERMAL MODEL / NOT REAL-HARDWARE DAMAGE EVIDENCE',16,(218,198,156))
        terminal=v.get('terminal')
        if terminal:
            winner=terminal['winner'];label=('PUSH TEST PASS' if self.push['actors'][0]['survived'] else 'PUSH TEST FAIL') if self.push else 'DRAW' if winner is None else ('CYAN' if winner==0 else 'AMBER')+' WINS BY RULE'
            if self.result.get('continuous'):label='PAUSED / NO WINNER' if winner is None else 'PARAMETERIZED MODEL FAILURE'
            rect((362,473,918,551),(5,13,24,228));text(391,482,label+'  /  '+terminal['reason'].replace('_',' ').upper(),22,(244,231,206),True);text(392,519,('Checkpoint saved; pause is not hardware failure.' if self.result.get('continuous') else 'No sustained fall + stable final window. See test protocol.' if self.push else 'Outcome derived from height, tilt, boundary and time.'),15,(167,190,210))
        else:
            entries=[e for e in self.result.get('events',[]) if 0<=v['time']-e.get('time',0)<1]
            if v['contact_force_n']>5:text(525,530,'CONTACT ACTIVE',20,(255,202,123),True)
        return np.asarray(Image.alpha_composite(im,overlay).convert('RGB'))
    def close(self):self.renderer.close();self.raw.close()


def render_video(folder,fps=24,width=1280,height=720,camera='main'):
    folder=Path(folder);out=folder/('battle.mp4' if camera=='main' else 'battle-overhead.mp4');r=ReplayRenderer(folder,width,height)
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for i in range(int(np.ceil(r.duration*fps))+1):
            t=min(i/fps,r.duration);params=dict(azimuth=105,elevation=-18,distance=4.7) if camera=='main' else dict(azimuth=90,elevation=-78,distance=5.5)
            pixels=r.frame(t,**params);process.stdin.write(pixels.tobytes())
            if i==min(48,int(r.duration*fps)//2):Image.fromarray(pixels).save(folder/'poster.jpg',quality=90)
        process.stdin.close();code=process.wait()
        if code:raise RuntimeError(f'ffmpeg exited {code}')
    finally:
        r.close()
        if process.poll() is None:process.terminate();process.wait()
    manifest=json.loads((folder/'manifest.json').read_text());manifest[out.name]=sha256(out)
    if (folder/'poster.jpg').exists():manifest['poster.jpg']=sha256(folder/'poster.jpg')
    write_json(folder/'manifest.json',manifest)
    return out
