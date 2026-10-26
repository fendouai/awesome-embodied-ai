"""Result overlays over unchanged recorded physics; explicit six-second freeze at end."""
import json, subprocess, wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .presentation import BroadcastRenderer
from .render import font
from .engine import write_json, sha256

class DuelRenderer(BroadcastRenderer):
    def __init__(self, folder):
        super().__init__(folder)
        self.verdict=json.loads((self.folder/'duel-result.json').read_text())['outcome']
        self.events=json.loads((self.folder/'referee-events.json').read_text())
    def overlay(self,pixels,v,t):
        im=Image.fromarray(super().overlay(pixels,v,t));d=ImageDraw.Draw(im)
        events=[e for e in self.events if e['type']=='scoring_contact' and e['time']<=t+1e-8]
        scores=[sum(e['side']==i for e in events) for i in range(2)]
        d.rounded_rectangle((480,90,800,185),8,fill='#08131e',outline='#638097')
        d.text((505,99),f'{scores[0]:02d}   :   {scores[1]:02d}',font=font(42,True),fill='#f4e6c7')
        d.text((505,151),'VALID UPPER-BODY CONTACTS',font=font(13),fill='#afc4d4')
        if events and 0<=t-events[-1]['time']<.8:
            e=events[-1];d.text((500,195),('BLUE' if e['side']==0 else 'AMBER')+' +1 / VALID CONTACT',font=font(17,True),fill='#ffcc83')
        counts=v.get('duel',{}).get('recovery_count_s',[None,None])
        for side,count in enumerate(counts):
            if count is not None:d.text((40+side*810,110),f'RECOVERY COUNT {count:.1f} s',font=font(21,True),fill='#ffbb77')
        if t>=self.duration-1e-7:
            o=self.verdict;winner=o['winner'];label='DRAW' if winner is None else ('BLUE' if winner==0 else 'AMBER')+' WINS'
            d.rounded_rectangle((280,245,1000,480),14,fill='#08131e',outline='#cba76a',width=3)
            d.text((330,263),label,font=font(54,True),fill='#ffce83')
            d.text((330,331),f"FINAL  {o['clean_upper_contacts'][0]} : {o['clean_upper_contacts'][1]}",font=font(38,True),fill='#f1f6ff')
            d.text((330,387),f"CONTACT DECISION / {self.result['duel']['rules']['limit_s']:g} s" if o['kind']=='contact_decision' else o['kind'].upper(),font=font(23,True),fill='#b7d0e2')
            d.text((330,427),'FINAL FRAME HOLD / NOT HARDWARE DAMAGE',font=font(16),fill='#9cb5c8')
        return np.asarray(im)

def render_duel(folder):
    folder=Path(folder);r=DuelRenderer(folder);fps=24;hold=6
    silent=folder/'duel-silent.mp4';audio=folder/'duel-audio.wav';out=folder/'duel-full.mp4'
    r.contact.sound(audio)
    with wave.open(str(audio),'rb') as f:sr=f.getframerate();samples=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').copy()
    sound=np.zeros(round((r.duration+hold+1)*sr),dtype=np.float64);sound[:len(samples)]=samples
    # Clearly synthetic bell at the adjudicated ending; no recorded microphone audio.
    ts=np.arange(sr)/sr;bell=(np.sin(2*np.pi*880*ts)+.35*np.sin(2*np.pi*1320*ts))*np.exp(-5*ts)*6000
    off=round(r.duration*sr);sound[off:off+sr]+=bell
    with wave.open(str(audio),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr);f.writeframes(np.clip(sound,-32767,32767).astype('<i2').tobytes())
    p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','22','-preset','veryfast','-pix_fmt','yuv420p',str(silent)],stdin=subprocess.PIPE)
    try:
        for i in range(round(r.duration*fps)+1):
            pix=r.frame(min(i/fps,r.duration));p.stdin.write(pix.tobytes())
            if i==fps*8:Image.fromarray(pix).save(folder/'duel-poster.jpg',quality=90)
            if i%(fps*15)==0:print(folder.name,round(i/fps),'s',flush=True)
        final=r.frame(r.duration);Image.fromarray(final).save(folder/'duel-final.jpg',quality=92)
        for _ in range(fps*hold):p.stdin.write(final.tobytes())
        p.stdin.close();assert p.wait()==0
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(silent),'-i',str(audio),'-c:v','copy','-c:a','aac','-b:a','96k','-shortest','-movflags','+faststart',str(out)],check=True)
        metadata=dict(video=out.name,source_trajectory_sha256=sha256(folder/'trajectory.npz'),video_sha256=sha256(out),fps=fps,source_duration_s=r.duration,final_frame_hold_s=hold,physics_unchanged=True,synthetic_audio=True,outcome=r.verdict)
        write_json(folder/'duel-video.json',metadata)
        manifest=json.loads((folder/'manifest.json').read_text())
        for name in ['duel-full.mp4','duel-poster.jpg','duel-final.jpg','duel-video.json']:manifest[name]=sha256(folder/name)
        write_json(folder/'manifest.json',manifest)
    finally:
        r.close()
        if p.poll() is None:p.terminate()
        silent.unlink(missing_ok=True);audio.unlink(missing_ok=True)
    return out
