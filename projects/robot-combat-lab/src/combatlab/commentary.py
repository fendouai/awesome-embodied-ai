"""Evidence-timed Mandarin commentary and readable subtitles for recorded matches."""
from __future__ import annotations
import bisect,json,math,subprocess,wave,re
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .engine import sha256,write_json

CJK=Path('/System/Library/Fonts/STHeiti Medium.ttc')

def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))

def rgba_video(path,duration,width,height,draw_frame,fps=4):
    p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgba','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','qtrle','-threads','2',str(path)],stdin=subprocess.PIPE)
    try:
        for i in range(math.ceil(duration*fps)+1):p.stdin.write(draw_frame(i/fps).tobytes())
        p.stdin.close()
        if p.wait():raise RuntimeError('RGBA overlay encoder failed')
    finally:
        if p.poll() is None:p.terminate()

def speech(text,path):
    txt=path.with_suffix('.txt')
    if path.exists() and txt.exists() and txt.read_text()==text:
        with wave.open(str(path)) as f:return np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').copy()
    txt.write_text(text)
    aiff=path.with_suffix('.aiff')
    subprocess.run(['say','-v','Tingting','-r','210','-f',str(txt),'-o',str(aiff)],check=True)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(aiff),'-ar','48000','-ac','1','-c:a','pcm_s16le',str(path)],check=True)
    with wave.open(str(path)) as f:samples=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').copy()
    aiff.unlink(missing_ok=True)
    return samples

def timestamp(seconds,web=False):
    ms=round(seconds*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02}:{m:02}:{s:02}{"." if web else ","}{ms:03}'

def wrap_text(text,limit=29):
    lines=[];line=''
    for token in re.findall(r'[A-Za-z0-9]+|.',text):
        if len(line)+len(token)>limit and line:lines.append(line);line=''
        line+=token
    if line:lines.append(line)
    return '\n'.join(lines)

def narrate(folder,source,kind='boxing'):
    folder=Path(folder);source=Path(source);out=folder/'commentary';out.mkdir(exist_ok=True)
    metadata=probe(source);source_duration=float(metadata['format']['duration'])
    candidates=[]
    if kind=='boxing':
        result=json.loads((folder/'duel-result.json').read_text());outcome=result['outcome'];events=json.loads((folder/'referee-events.json').read_text());duration=outcome['time'];limit=result['rules']['limit_s']
        model=json.loads((folder/'result.json').read_text());roles=[p.rsplit('_',1)[-1] for p in model['policies']]
        identities=(f'NCP数字人拳击，蓝方模型{roles[0]}，橙方模型{roles[1]}。' if model['track']=='ncp_cpu_port' else f'G1机器人对抗，蓝方策略{roles[0]}，橙方策略{roles[1]}。')
        intro=f'{identities}本场上限{int(limit)}秒。手部接触上身，满足速度和冲量阈值才记分，同时记录倒地与恢复。'
        previous_score=[0,0]
        for t in np.arange(15,duration,24):
            hits=[e for e in events if e['type']=='scoring_contact' and e['time']<=t];score=[sum(e.get('side')==i for e in hits) for i in [0,1]]
            delta=[score[i]-previous_score[i] for i in [0,1]];previous_score=score
            candidates.append(dict(time=float(t),text=f'比赛进行到{int(t)}秒。有效上身接触，蓝方{score[0]}次，橙方{score[1]}次。这一统计区间新增{delta[0]}比{delta[1]}次。',evidence={'type':'score_snapshot','simulation_time':float(t),'score':score,'interval_delta':delta}))
        for e in events:
            side='蓝方' if e.get('side')==0 else '橙方'
            if e['type']=='down_confirmed':candidates.append(dict(time=e['time'],priority=3,text=f'{side}达到倒地确认条件，裁判开始记录恢复过程。',evidence=e))
            if e['type']=='recovered':candidates.append(dict(time=e['time'],priority=3,text=f'{side}已经满足站稳要求，恢复确认完成。',evidence=e))
        a,b=outcome['clean_upper_contacts'];winner=outcome['winner'];verdict='双方平局' if winner is None else ('蓝方获胜' if winner==0 else '橙方获胜')
        reason={'contact_decision':'限时接触计分判定','technical_draw':'恢复计数达到技术平局条件','technical_stoppage':'一方恢复失败，技术终止','draw':'没有形成有效计分差异，平局'}.get(outcome['kind'],'公布的本地裁判规则判定')
        if outcome['reason']=='double_recovery_timeout':reason='双方恢复失败，技术平局'
        elif outcome['reason']=='unresolved_recovery_at_extension_limit':reason='恢复延长计数到达上限，技术平局'
        final=f'比赛在{duration:.1f}秒结束。有效上身接触{a}比{b}，{verdict}。结果依据是{reason}。这些是仿真比赛数据，不能解释成真实硬件损坏。'
        evidence={'type':'match_ended','outcome':outcome}
    else:
        result=json.loads((folder/'native-result.json').read_text());duration=source_duration
        if (folder/'analysis.json').exists():result['ball_span_m']=json.loads((folder/'analysis.json').read_text())['active_ball_span_m']
        intro=f'这里是B Human对B Team的三对三视觉足球。六台K1通过相机感知比赛。本场采用{result["period_seconds"]}秒本地赛制，完整保留准备、开球、比赛和裁判终局。'
        timeline=json.loads((folder/'timeline.json').read_text());state_names={0:'初始化',1:'准备入场和定位',2:'开球准备',3:'比赛进行',4:'裁判终场'}
        last_score=[0,0];last_state=None
        for e in timeline:
            if e['state']!=last_state:
                candidates.append(dict(time=e['time'],priority=1,text=f'裁判状态进入{state_names.get(e["state"],"比赛状态")}。',evidence=e));last_state=e['state']
            if e['score']!=last_score:
                candidates.append(dict(time=e['time'],priority=3,text=f'原生裁判更新比分，红队{e["score"][0]}比白队{e["score"][1]}。',evidence=e));last_score=e['score']
        for t in np.arange(45,max(45,duration-12),35):
            e=timeline[max(0,bisect.bisect_right([v['time'] for v in timeline],t)-1)]
            candidates.append(dict(time=float(t),text=f'当前比分，红队{e["score"][0]}比白队{e["score"][1]}。裁判记录剩余{max(0,e["remaining_seconds"])}秒。',evidence=e))
        a,b=result['red'],result['white'];verdict='平局' if a==b else '红队获胜' if a>b else '白队获胜'
        action='记录到了足球的位置变化' if result.get('ball_span_m',0)>=.1 else '足球几乎没有位移，说明本场有效争球不足，不能据此评价球队强弱'
        final=f'原生裁判已经结束本场比赛，最终红队{a}比白队{b}，{verdict}。{action}。本场采用{result["period_seconds"]}秒本地赛制，完整官方比赛通常包含两个十分钟半场。'
        evidence={'type':'native_referee_finished','result':result}
    opening=speech(intro,out/'intro.wav');intro_duration=len(opening)/48000+1.0
    cues=[dict(start=.3,end=.3+len(opening)/48000,text=intro,evidence={'type':'protocol','kind':kind},audio='intro.wav')]
    mixed=[(.3,opening)];last_end=intro_duration
    priority_times=[c['time'] for c in candidates if c.get('priority',0)>=3]
    for number,c in enumerate(sorted(candidates,key=lambda x:(x['time'],-x.get('priority',0)))):
        if c.get('priority',0)<3 and any(0<=event_time-c['time']<12 for event_time in priority_times):continue
        start=intro_duration+c['time']
        if start<last_end+1.0 or (c['time']>duration-6 and c.get('priority',0)<3):continue
        samples=speech(c['text'],out/f'cue-{number:03}.wav');length=len(samples)/48000
        if start+length>intro_duration+duration and c.get('priority',0)<3:continue
        cues.append(dict(start=start,end=start+length,text=c['text'],evidence=c['evidence'],audio=f'cue-{number:03}.wav'));mixed.append((start,samples));last_end=start+length
    ending=speech(final,out/'final.wav');ending_start=max(intro_duration+duration+.4,last_end+1)
    if kind=='vision':
        expected=[c['evidence'] for c in candidates if c.get('priority')==3]
        spoken=[c['evidence'] for c in cues]
        if any(e not in spoken for e in expected):raise ValueError('A logged goal update was omitted from the narration')
    cues.append(dict(start=ending_start,end=ending_start+len(ending)/48000,text=final,evidence=evidence,audio='final.wav'));mixed.append((ending_start,ending))
    full_duration=max(intro_duration+source_duration,ending_start+len(ending)/48000+1.5);tail=max(0,full_duration-intro_duration-source_duration)
    wave_samples=np.zeros(math.ceil(full_duration*48000),dtype=np.float64)
    for start,samples in mixed:
        offset=round(start*48000);wave_samples[offset:offset+len(samples)]+=samples
    audio=out/'narration.wav'
    with wave.open(str(audio),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(48000);f.writeframes(np.clip(wave_samples,-32767,32767).astype('<i2').tobytes())
    srt=out/'captions.zh-CN.srt';srt.write_text('\n\n'.join(f'{i+1}\n{timestamp(c["start"])} --> {timestamp(c["end"])}\n{wrap_text(c["text"])}' for i,c in enumerate(cues))+'\n')
    (out/'captions.zh-CN.vtt').write_text('WEBVTT\n\n'+'\n\n'.join(f'{timestamp(c["start"],True)} --> {timestamp(c["end"],True)}\n{wrap_text(c["text"])}' for c in cues)+'\n')
    (out/'script.zh-CN.txt').write_text('\n\n'.join(f'[{c["start"]:.2f}–{c["end"]:.2f}] {c["text"]}' for c in cues))
    # Sparse alpha overlay works on FFmpeg builds that do not include libass.
    subtitle=out/'subtitle-layer.mov';font=ImageFont.truetype(str(CJK),29);starts=[c['start'] for c in cues];cache={}
    def overlay(t):
        k=bisect.bisect_right(starts,t)-1;k=k if k>=0 and t<cues[k]['end'] else -1
        if k not in cache:
            im=Image.new('RGBA',(1280,150));d=ImageDraw.Draw(im)
            if k>=0:
                lines=wrap_text(cues[k]['text']).splitlines();lines=lines[:4];h=len(lines)*34+22;d.rounded_rectangle((18,148-h,1262,148),radius=8,fill=(3,9,17,225))
                for j,line in enumerate(lines):d.text((36,156-h+j*34),line,font=font,fill='white')
            cache[k]=np.asarray(im)
        return cache[k]
    rgba_video(subtitle,full_duration,1280,150,overlay,4)
    output=out/'match-commentated.zh-CN.mp4'
    pending_output=out/'match-commentated.zh-CN.pending.mp4'
    cmd=['ffmpeg','-y','-loglevel','error','-i',str(source),'-i',str(subtitle),'-i',str(audio),'-i',str(srt)]
    filters=f'[0:v]tpad=start_mode=clone:stop_mode=clone:start_duration={intro_duration:.4f}:stop_duration={tail:.4f}[base];[base][1:v]overlay=0:H-h:eof_action=repeat[v]'
    has_audio=any(x['codec_type']=='audio' for x in metadata['streams'])
    if has_audio:
        filters+=f';[0:a]adelay={round(intro_duration*1000)}:all=1,volume=0.18[a0];[a0][2:a]amix=inputs=2:duration=longest:normalize=0[a]'
    cmd+=['-filter_complex',filters,'-map','[v]','-map','[a]' if has_audio else '2:a','-map','3:s','-c:v','libx264','-preset','veryfast','-crf','20','-threads','2','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-c:s','mov_text','-metadata:s:s:0','language=zho','-disposition:s:0','0','-t',f'{full_duration:.4f}','-movflags','+faststart',str(pending_output)]
    subprocess.run(cmd,check=True)
    pending_output.replace(output)
    subtitle.unlink(missing_ok=True)
    data=dict(commentary_version=2,kind=kind,voice='Tingting',engine='macOS local speech synthesis',language='zh-CN',source_video=str(source),source_video_sha256=sha256(source),video=output.name,video_sha256=sha256(output),intro_hold_seconds=intro_duration,outro_extension_seconds=tail,physics_speed_changed=False,subtitles='burned and embedded Chinese captions; external SRT/VTT also supplied',cues=cues,probe=probe(output))
    write_json(out/'commentary.pending.json',data)
    (out/'commentary.pending.json').replace(out/'commentary.json')
    return output
