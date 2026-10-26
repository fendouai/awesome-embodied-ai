from __future__ import annotations
import os
os.environ.setdefault('MUJOCO_GL','cgl')
from pathlib import Path
import base64,io,json,re,threading,time,subprocess,sys,hashlib
from collections import OrderedDict
import numpy as np,mujoco
from PIL import Image
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse,Response,HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel,Field
from .models import ROOT
from .engine import Simulation,MatchConfig,write_json,SOFTWARE_FINGERPRINT,jsonable,sha256
from .render import ReplayRenderer
from . import control_upgrade
from .endurance import ContinuousSimulation,EnduranceConfig,ENDURANCE_FINGERPRINT

app=FastAPI(title='BotsTest Robot Combat Lab',docs_url='/api/docs')
app.add_middleware(GZipMiddleware,minimum_size=1000)
def ui_revision():return hashlib.sha256(b''.join((ROOT/'web'/name).read_bytes() for name in ['index.html','app.js','style.css'])).hexdigest()[:12]
@app.middleware('http')
async def fresh_ui(request,call_next):
    response=await call_next(request)
    if request.url.path=='/' or request.url.path.startswith(('/web/','/api/')):
        response.headers['Cache-Control']='no-store, max-age=0';response.headers['X-UI-Revision']=ui_revision()
    return response
MATCHES=ROOT/'results/matches';CACHE=OrderedDict();LOCK=threading.RLock();LIVE={}
def match_folder(mid):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',mid):raise HTTPException(400,'Invalid match id')
    f=MATCHES/mid
    if not (f/'result.json').exists():raise HTTPException(404,'Match not found')
    return f

# Native backends are isolated from the historical neutral-PD experiment.
PORT_JOB={'state':'idle'}
def tournament_running():
    p=ROOT/'results/tournament-active.json'
    return p.exists() and json.loads(p.read_text()).get('state')=='running'

PORT_SCRIPTS={'robostriker':'run_robostriker_cpu.py','ncp':'run_ncp_cpu.py'}

@app.get('/api/tournament')
async def tournament_status():
    choices=[ROOT/'results/tournaments/round-02/index.json',ROOT/'results/tournaments/round-01/index.json']
    selected=None
    for p in choices:
        if p.exists() and json.loads(p.read_text()).get('state') in ['running','complete']:
            selected=p;break
    if selected is None and choices[1].exists():selected=choices[1]
    return dict(tournament=json.loads(selected.read_text()) if selected else dict(state='not_started',matches=[]))

@app.get('/api/duels')
async def duel_status():
    path=ROOT/'results/duel-index.json'
    return dict(**(json.loads(path.read_text()) if path.exists() else {'matches':[]}),job=PORT_JOB)

@app.post('/api/duels/run/{backend}')
async def duel_run(backend:str):
    import shutil
    scripts={'g1':('run_duel.py','duels'),'ncp':('run_ncp_duel.py','ncp-duels')}
    if backend not in scripts:raise HTTPException(400,'Unsupported duel backend')
    with LOCK:
        if tournament_running() or PORT_JOB['state']=='running' or COMPARE_JOB['state']=='running':raise HTTPException(409,'A CPU rollout is already running')
        if shutil.disk_usage(ROOT).free<512*1024**2:raise HTTPException(507,'Storage reserve below 512 MiB; existing evidence has been preserved')
        PORT_JOB.clear();PORT_JOB.update(state='running',backend=backend,started=time.time(),seconds=180,stage='simulation')
        def run():
            try:
                script,group=scripts[backend];env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'MUJOCO_GL':'cgl'}
                log_path=ROOT/'results'/('duel-rerun-'+backend+'.log')
                with log_path.open('w') as log:
                    def execute(args,timeout=1800):
                        subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=True)
                    execute([ROOT/'scripts'/script,'--limit','180'])
                    latest=json.loads((ROOT/'results'/group/'latest.json').read_text());folder=ROOT/latest['folder']
                    PORT_JOB['stage']='verification';execute([ROOT/'scripts/verify_duel.py',folder])
                    PORT_JOB['stage']='rendering';execute([ROOT/'scripts/render_duel.py',folder])
                    PORT_JOB['stage']='commentary';execute([ROOT/'scripts/produce_match_commentary.py',folder,folder/'duel-full.mp4'])
                    execute([ROOT/'scripts/publish_duels.py'])
                PORT_JOB.update(state='completed',stage='published',finished=time.time())
            except Exception as error:PORT_JOB.update(state='error',error=str(error),finished=time.time())
        threading.Thread(target=run,daemon=True).start()
    return PORT_JOB

@app.get('/api/adversarial')
async def adversarial_status():
    entries={}
    for key,name in [('robostriker','robostriker-cpu'),('ncp','ncp-cpu')]:
        path=ROOT/'results'/name/'latest.json'
        entries[key]=json.loads(path.read_text()) if path.exists() else None
        if entries[key]:
            archive=ROOT/entries[key]['folder'];ready=(archive/'presentation.json').exists()
            entries[key].update(video_name='presentation.mp4' if ready else 'battle.mp4',poster_name='presentation-poster.jpg' if ready else 'poster.jpg')
    return dict(backends=entries,job=PORT_JOB,bhuman=json.loads((ROOT/'integrations/bhuman/acceptance.json').read_text()) if (ROOT/'integrations/bhuman/acceptance.json').exists() else None)

@app.post('/api/adversarial/run/{backend}')
async def adversarial_run(backend:str):
    import shutil
    if backend not in PORT_SCRIPTS:raise HTTPException(400,'Unsupported backend')
    with LOCK:
        if tournament_running() or PORT_JOB['state']=='running' or COMPARE_JOB['state']=='running':raise HTTPException(409,'A CPU rollout is already running')
        if shutil.disk_usage(ROOT).free<512*1024**2:raise HTTPException(507,'Storage reserve below 512 MiB; existing evidence has been preserved')
        PORT_JOB.clear();PORT_JOB.update(state='running',backend=backend,started=time.time(),seconds=20)
        def run():
            try:
                log_path=ROOT/'results'/('rerun-'+backend+'.log')
                with log_path.open('w') as log:
                    done=subprocess.run([sys.executable,str(ROOT/'scripts'/PORT_SCRIPTS[backend]),'--seconds','20'],env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'MUJOCO_GL':'cgl'},stdout=log,stderr=subprocess.STDOUT,timeout=240)
                    if done.returncode:raise RuntimeError('Simulation failed; inspect '+str(log_path.relative_to(ROOT)))
                    latest=json.loads((ROOT/'results'/('robostriker-cpu' if backend=='robostriker' else 'ncp-cpu')/'latest.json').read_text())
                    rendered=subprocess.run([sys.executable,str(ROOT/'scripts/render_presentation.py'),str(ROOT/latest['folder'])],env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'MUJOCO_GL':'cgl'},stdout=log,stderr=subprocess.STDOUT,timeout=240)
                    if rendered.returncode:raise RuntimeError('Physics saved; video rendering failed')
                PORT_JOB.update(state='complete',finished=time.time(),folder=latest['folder'])
            except Exception as error:PORT_JOB.update(state='failed',error=str(error),finished=time.time())
        threading.Thread(target=run,daemon=True).start()
        return PORT_JOB

# Registered capabilities and isolated A/B trials use strict robot/policy compatibility.
COMPARE_JOB={'state':'idle'}
class CompareRequest(BaseModel):
    robot:str
    left:str
    right:str
    task:str='push'
    seed:int=Field(default=0,ge=0,le=9999)

@app.get('/api/catalog')
async def robot_catalog():
    from .catalog import catalog
    return catalog()

@app.get('/api/comparisons')
async def comparison_status():
    source=ROOT/'results/comparisons/suite.json'
    suite=json.loads(source.read_text()) if source.exists() else dict(state='empty',rows=[],trials=0)
    for row in suite['rows']:
        archive=ROOT/row['folder'];manifest=archive/'manifest.json'
        try:row['video_ready']=(archive/'comparison.mp4').exists() and manifest.exists() and 'comparison.mp4' in json.loads(manifest.read_text())
        except json.JSONDecodeError:row['video_ready']=False
        analysis=archive/'analysis.json'
        if analysis.exists():row['analysis']=json.loads(analysis.read_text())
    return dict(suite=suite,job=COMPARE_JOB)

@app.post('/api/comparisons/run')
async def comparison_run(config:CompareRequest):
    from .comparison import validate,Comparison
    import shutil
    try:validate(config.robot,config.left,config.right,config.task)
    except ValueError as error:raise HTTPException(400,str(error))
    with LOCK:
        if tournament_running() or COMPARE_JOB['state']=='running' or PORT_JOB['state']=='running':raise HTTPException(409,'A local CPU job is running')
        if shutil.disk_usage(ROOT).free<512*1024**2:raise HTTPException(507,'Storage reserve below512MiB')
        COMPARE_JOB.clear();COMPARE_JOB.update(state='running',configuration=config.model_dump(),started=time.time())
        def run():
            try:
                folder,result=Comparison(config.robot,config.left,config.right,config.task,config.seed).run().save()
                from .comparison_analysis import analyze
                analyze(folder)
                with (folder/'render.log').open('w') as log:
                    done=subprocess.run([sys.executable,str(ROOT/'scripts/render_comparison.py'),str(folder)],env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'MUJOCO_GL':'cgl'},stdout=log,stderr=subprocess.STDOUT,timeout=240)
                    if done.returncode:raise RuntimeError('Physics saved; comparison video rendering failed')
                path=ROOT/'results/comparisons/suite.json';suite=json.loads(path.read_text()) if path.exists() else dict(rows=[],state='complete')
                suite['rows'].append(dict(folder=str(folder.relative_to(ROOT)),result=result));suite['trials']=len(suite['rows']);temp=path.with_suffix('.pending.json');write_json(temp,suite);os.replace(temp,path)
                COMPARE_JOB.update(state='complete',folder=str(folder.relative_to(ROOT)),finished=time.time())
            except Exception as error:COMPARE_JOB.update(state='failed',error=str(error),finished=time.time())
        threading.Thread(target=run,daemon=True).start()
        return COMPARE_JOB

@app.get('/api/health')
async def health():return dict(status='ok',mujoco=mujoco.__version__,software_fingerprint=SOFTWARE_FINGERPRINT,controller_adapter_fingerprint=control_upgrade.ADAPTER_FINGERPRINT,endurance_fingerprint=ENDURANCE_FINGERPRINT,ui_revision=ui_revision())
@app.get('/api/control-upgrade')
async def upgrade():
    path=ROOT/'results/control-upgrade/summary.json'
    return json.loads(path.read_text()) if path.exists() else dict(status='running',rows=[],selected={},videos=[])
@app.get('/api/continuous')
async def continuous():
    path=ROOT/'results/continuous/summary.json'
    return json.loads(path.read_text()) if path.exists() else dict(status='ready',damage_prediction='not_calibrated',parts=[])
@app.get('/api/leaderboards')
async def leaderboards():
    f=ROOT/'results/leaderboards.json'
    return json.loads(f.read_text()) if f.exists() else dict(matches=[],leaderboards={},status='Experiments running')
@app.get('/api/protocol')
async def protocol():
    return dict(authenticity=(ROOT/'REQUIREMENTS.md').read_text(),provenance=json.loads((ROOT/'assets/provenance.json').read_text()),preflight=json.loads((ROOT/'results/preflight.json').read_text()) if (ROOT/'results/preflight.json').exists() else None)
@app.get('/api/matches/{mid}/telemetry')
async def telemetry(mid:str):return FileResponse(match_folder(mid)/'telemetry.json',media_type='application/json')
@app.get('/api/matches/{mid}/events')
async def events(mid:str):return FileResponse(match_folder(mid)/'events.json',media_type='application/json')
@app.get('/api/frame/{mid}')
async def frame(mid:str,t:float=0,azimuth:float=45,elevation:float=-18,distance:float=4.6):
    folder=match_folder(mid)
    with LOCK:
        if mid not in CACHE:
            if len(CACHE)>=2:
                _,old=CACHE.popitem(last=False);old.close()
            CACHE[mid]=ReplayRenderer(folder,960,540)
        r=CACHE[mid];CACHE.move_to_end(mid)
        pix=r.frame(max(0,t),azimuth=azimuth%360,elevation=float(np.clip(elevation,-89,-3)),distance=float(np.clip(distance,2,9)),hud=False)
        b=io.BytesIO();Image.fromarray(pix).save(b,format='JPEG',quality=85)
    return Response(b.getvalue(),media_type='image/jpeg',headers={'Cache-Control':'no-store'})

class LiveStart(BaseModel):
    protocol:str='continuous'
    max_sim_time:float=Field(default=120.,ge=1.,le=600.)
    robot:str='g1_full'
    mode:str='ai'
    left_policy:str='unitree_rl_lab'
    right_policy:str='unitree_rl_mjlab'
    seed:int=20261024
class LiveStep(BaseModel):
    session_id:str
    left:list[float]|None=None
    right:list[float]|None=None
    dt:float=Field(default=.04,ge=.02,le=.1)
class LiveStop(BaseModel):session_id:str

def live_image(entry,azimuth=45):
    s=entry['sim'];r=entry['renderer'];cam=entry['cam'];p=np.mean([s.d.qpos[a.free_qadr:a.free_qadr+3] for a in s.actors],axis=0);cam.lookat[:]=[p[0],p[1],.8];cam.distance=4.6;cam.azimuth=azimuth;cam.elevation=-18;r.update_scene(s.d,camera=cam);im=r.render();b=io.BytesIO();Image.fromarray(im).save(b,format='JPEG',quality=82);return base64.b64encode(b.getvalue()).decode()
@app.post('/api/live/start')
async def live_start(c:LiveStart):
    if c.protocol not in ['continuous','finite']:raise HTTPException(400,'Unknown protocol')
    if c.mode not in ['ai','human-ai','human-human']:raise HTTPException(400,'Unknown mode')
    allowed={'g1_full':['unitree_rl_lab','unitree_rl_mjlab','amp_recovery','holosoma_fastsac','holosoma_ppo','nvidia_balance','nvidia_walk','nvidia_wbc'],'h1_full':['unitree_full_gym'],'h1_2_full':['unitree_full_gym']}
    if c.robot not in allowed or c.left_policy not in allowed[c.robot] or c.right_policy not in allowed[c.robot]:raise HTTPException(400,'Checkpoint is incompatible with selected robot')
    with LOCK:
        for entry in list(LIVE.values()):_save_live(entry)
        LIVE.clear();sid='live-'+str(time.time_ns());cfg_args=dict(match_id=sid,robots=(c.robot,c.robot),policies=(c.left_policy,c.right_policy),seed=c.seed,duration=c.max_sim_time if c.protocol=='continuous' else 20.,rated=False,note='Continuous physical contact; fall/boundary/time are not hardware failure. Missing damage parameters remain unknown.' if c.protocol=='continuous' else 'Finite diagnostic contact session, not hardware damage.')
        cfg=EnduranceConfig(**cfg_args,max_sim_time=c.max_sim_time) if c.protocol=='continuous' else MatchConfig(**cfg_args,track='interactive')
        sim=ContinuousSimulation(cfg) if c.protocol=='continuous' else Simulation(cfg)
        if isinstance(sim,ContinuousSimulation):sim.on_chunk_saved=_render_saved
        renderer=mujoco.Renderer(sim.m,540,960);cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;entry=dict(id=sid,sim=sim,renderer=renderer,cam=cam,mode=c.mode,protocol=c.protocol,saved=False);LIVE[sid]=entry
        return dict(session_id=sid,image=live_image(entry),frame=json.loads(json.dumps(sim.frames[-1],default=jsonable)),mode=c.mode)
def _save_live(entry):
    if entry['saved']:return
    sim=entry['sim']
    if sim.terminal is None:sim.terminal=dict(reason='manual_pause' if isinstance(sim,ContinuousSimulation) else 'manual_stop',winner=None,time=float(sim.d.time),contact_before_result=sim.first_contact is not None);sim.record_frame()
    folder,result=sim.save(ROOT/'results/live-sessions');(folder/'control_upgrade.py').write_bytes(control_upgrade.ADAPTER_SOURCE);write_json(folder/'controller-deployment.json',dict(adapter_sha256=control_upgrade.ADAPTER_FINGERPRINT,actors=[dict(policy=sim.cfg.policies[i],weights=[dict(file=str(path.relative_to(ROOT)),sha256=sha256(path)) for path in getattr(policy,'paths',[policy.path] if hasattr(policy,'path') else [])]) for i,policy in enumerate(sim.policies)]));write_json(folder/'interaction.json',dict(mode=entry['mode'],protocol=entry.get('protocol','finite'),input='Velocity setpoints in telemetry.json; actual neural-policy observations/actions in policy_io.npz',ranked=False));manifest=json.loads((folder/'manifest.json').read_text());manifest['interaction.json']=sha256(folder/'interaction.json');manifest['control_upgrade.py']=sha256(folder/'control_upgrade.py');manifest['controller-deployment.json']=sha256(folder/'controller-deployment.json');write_json(folder/'manifest.json',manifest);entry['renderer'].close();entry['saved']=True;entry['folder']=folder
    _render_saved(folder)
def _render_saved(folder):
    with open(folder/'video-render.log','w') as log:subprocess.Popen([sys.executable,str(ROOT/'scripts/render_one.py'),str(folder)],env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'MUJOCO_GL':'cgl'},stdout=log,stderr=subprocess.STDOUT)
@app.post('/api/live/step')
async def live_step(c:LiveStep):
    with LOCK:
        if c.session_id not in LIVE:raise HTTPException(404,'Session missing')
        e=LIVE[c.session_id]
        if e['saved']:return dict(finished=True,archive=str(e['folder'].relative_to(ROOT)))
        for command in [c.left,c.right]:
            if command is not None and (len(command)!=3 or not np.isfinite(command).all()):raise HTTPException(400,'Command must have three finite setpoints')
        manual=[None,None]
        if e['mode']=='human-ai':manual=[c.left or [0,0,0],None]
        elif e['mode']=='human-human':manual=[c.left or [0,0,0],c.right or [0,0,0]]
        alive=e['sim'].advance(c.dt,manual);data=dict(image=live_image(e),frame=json.loads(json.dumps(e['sim'].frames[-1],default=jsonable)),finished=not alive)
        if not alive:_save_live(e);data['archive']=str(e['folder'].relative_to(ROOT))
        return data
@app.post('/api/live/resume')
async def live_resume(c:LiveStop):
    with LOCK:
        if c.session_id not in LIVE:raise HTTPException(404,'Session missing')
        e=LIVE[c.session_id]
        if not isinstance(e['sim'],ContinuousSimulation) or not e['saved']:raise HTTPException(400,'No paused continuous session')
        try:e['sim'].resume(120.)
        except ValueError as error:raise HTTPException(400,str(error))
        e['renderer']=mujoco.Renderer(e['sim'].m,540,960);e['saved']=False;e['sim'].record_frame()
        return dict(session_id=c.session_id,image=live_image(e),frame=json.loads(json.dumps(e['sim'].frames[-1],default=jsonable)))
@app.post('/api/live/stop')
async def live_stop(c:LiveStop):
    with LOCK:
        if c.session_id not in LIVE:raise HTTPException(404,'Session missing')
        e=LIVE[c.session_id];_save_live(e);return dict(saved=True,archive=str(e['folder'].relative_to(ROOT)))
@app.get('/')
async def index():
    revision=ui_revision();html=(ROOT/'web/index.html').read_text().replace('/web/style.css',f'/web/style.css?v={revision}').replace('/web/app.js',f'/web/app.js?v={revision}')
    return HTMLResponse(html)
@app.get('/api/research')
async def research_status():
    from .research_status import status
    return status()

try:
    from .research_archive import workspace as research_workspace
    app.mount('/research',StaticFiles(directory=research_workspace()),name='external_research')
except (RuntimeError,FileNotFoundError):
    pass

app.mount('/web',StaticFiles(directory=ROOT/'web'),name='web')
app.mount('/results',StaticFiles(directory=ROOT/'results'),name='results')
app.mount('/reports',StaticFiles(directory=ROOT/'reports'),name='reports')

app.mount('/evidence',StaticFiles(directory=ROOT/'evidence'),name='evidence')
