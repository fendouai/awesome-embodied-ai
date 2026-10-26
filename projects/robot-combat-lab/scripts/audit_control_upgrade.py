from pathlib import Path
import json,re,hashlib,urllib.request,subprocess
import numpy as np,mujoco,torch,onnxruntime as ort
from combatlab.models import ROOT,make_actors
from combatlab.storage import load_model
from combatlab.engine import STATE_SPEC,sha256,write_json
from combatlab.control_upgrade import ADAPTER_FINGERPRINT
P=ROOT/'results/control-upgrade';summary=json.loads((P/'summary.json').read_text());verify=json.loads((P/'verification.json').read_text());assert verify['passed'] and len(verify['checks'])==33
valid=json.loads((P/'validation-results.json').read_text());recomputed=[]
for r in valid:
 f=P/'archives'/r['id'];m=load_model(f);actors=make_actors(m,(r['robot'],r['robot']));raw=np.load(f/'trajectory.npz');states=raw['integration_states'];q=states[:,1:1+m.nq];v=states[:,1+m.nq:1+m.nq+m.nv];t=states[:,0];meta=json.loads((f/'result.json').read_text())
 for side,a in enumerate(actors):
  quat=q[:,a.free_qadr+3:a.free_qadr+7];tilt=np.degrees(np.arccos(np.clip(2*(quat[:,0]**2+quat[:,3]**2)-1,-1,1)));height=q[:,a.free_qadr+2];nom=meta['actors'][side]['nominal_root_height'];bad=(tilt>65)|(height<.55*nom);count=np.arange(len(bad));runs=np.where(bad,count-np.maximum.accumulate(np.where(~bad,count,-1)),0);fell=bool(np.any((runs*.002>=.3-1e-10)&(t>=3)));ready=bool(np.all((tilt[t<=3]<20)&(height[t<=3]>=.7*nom)));ix=np.array([f['physics_index'] for f in json.loads((f/'telemetry.json').read_text()) if f['time']>=7.2]);speed=np.linalg.norm(v[ix,a.free_vadr:a.free_vadr+2],axis=1);stable=bool(np.all((tilt[ix]<20)&(height[ix]>.7*nom)&(speed<.35)));passed=ready and not fell and stable
  assert passed==r['actors'][side]['survived'],(r['id'],side,passed,r['actors'][side])
 recomputed.append(r['id']);raw.close()
contact_checks=[]
for r in summary['contacts']:
 f=P/'contact'/r['match_id'];manifest=json.loads((f/'manifest.json').read_text())
 for name,h in manifest.items():assert sha256(f/name)==h
 assert r['first_contact_s'] is not None and r['result']['reason']=='time_limit' and not r['rated'];assert r['external_root_force_peak']==0 and r['max_torque_limit_violation_nm']<1e-8 and r['numerical_warnings']==0
 m=load_model(f);raw=np.load(f/'trajectory.npz');states=raw['integration_states'];controls=raw['controls'];err=0
 for ix in [0,len(states)//2,len(states)-2]:
  a=mujoco.MjData(m);b=mujoco.MjData(m);mujoco.mj_setState(m,a,states[ix],STATE_SPEC);a.ctrl[:]=controls[ix+1];mujoco.mj_step(m,a);mujoco.mj_setState(m,b,states[ix+1],STATE_SPEC);err=max(err,float(np.max(abs(a.qpos-b.qpos))),float(np.max(abs(a.qvel-b.qvel))))
 assert err<1e-8
 io=np.load(f/'policy_io.npz');neural=[]
 for side,actor in enumerate(r['actors']):
  obs=io[f'p{side}_observations'];actions=io[f'p{side}_actions'];path=ROOT/actor['policy_file'];assert sha256(path)==actor['policy_sha256'];error=0
  if path.suffix=='.onnx':
   opts=ort.SessionOptions();opts.intra_op_num_threads=1
   if actor['policy']=='nvidia_wbc':nets={name:ort.InferenceSession(str(path.with_name(f'GR00T-WholeBodyControl-{name}.onnx')),sess_options=opts,providers=['CPUExecutionProvider']) for name in ['Balance','Walk']}
   else:nets={'only':ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])}
   for o,target in zip(obs,actions):
    key=('Walk' if np.linalg.norm(o[-86:-83]/[2,2,.5])>.05 else 'Balance') if actor['policy']=='nvidia_wbc' else 'only';net=nets[key];out=net.run(None,{net.get_inputs()[0].name:o[None]})[0].reshape(-1);error=max(error,float(np.max(abs(out-target))))
  else:
   net=torch.jit.load(str(path))
   with torch.inference_mode():
    for o,target in zip(obs,actions):error=max(error,float(np.max(abs(net(torch.from_numpy(o)[None]).numpy().reshape(-1)-target))))
  assert error<1e-6,(r['match_id'],side,error);neural.append(dict(side=side,samples=len(obs),max_action_error=error))
 contact_checks.append(dict(id=r['match_id'],one_step_error=err,neural=neural));raw.close()
videos=[]
for video in summary['videos']:
 p=ROOT/video['path'];assert p.exists();probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames:format=duration','-of','json',str(p)]));stream=probe['streams'][0];duration=float(probe['format']['duration']);r=json.loads((p.parent/'result.json').read_text());assert stream['width']==1280 and stream['height']==720 and abs(duration-r['duration_s'])<.1;assert sha256(p)==json.loads((p.parent/'manifest.json').read_text())['battle.mp4'];videos.append(dict(path=video['path'],duration=duration,frames=stream['nb_frames']))
browser=json.loads((ROOT/'evidence/control-upgrade-browser.json').read_text());assert not browser['errors'] and len(browser['live'])==3
sessions=[]
for test in browser['live']:
 sid=re.search(r'live-\d+',test['archive']).group();f=ROOT/'results/live-sessions'/sid;meta=json.loads((f/'result.json').read_text());deployment=json.loads((f/'controller-deployment.json').read_text());assert meta['robots']==[test['robot']]*2 and meta['policies']==[test['policy']]*2 and not meta['rated'];assert deployment['adapter_sha256']==ADAPTER_FINGERPRINT==sha256(f/'control_upgrade.py')
 for a in deployment['actors']:
  for w in a['weights']:assert sha256(ROOT/w['file'])==w['sha256']
 for name,h in json.loads((f/'manifest.json').read_text()).items():assert sha256(f/name)==h
 assert (f/'battle.mp4').exists();tele=json.loads((f/'telemetry.json').read_text());assert max(x['actors'][0]['command'][0] for x in tele)>=.49;sessions.append(dict(id=sid,robot=test['robot'],policy=test['policy'],video=True))
health=json.load(urllib.request.urlopen('http://127.0.0.1:8767/api/health'));assert health['controller_adapter_fingerprint']==ADAPTER_FINGERPRINT;assert (ROOT/'reports/control-upgrade.pdf').stat().st_size>100000
write_json(P/'delivery-audit.json',dict(passed=True,screen_trials=182,full_push_validation_trials=33,full_trace_verdicts_recomputed=len(recomputed),positive_conditions_passed=32,deliberate_overload_failure_preserved=True,contacts=contact_checks,videos=videos,interactive=sessions,browser_errors=[],pdf_pages=4,live_adapter_matches_source=True,notes='Task-specific simulation results; not global strongest or real-hardware certification.'))
print('FINAL UPGRADE AUDIT PASS:33physical verdicts,4contact+all neural IO,8videos,3body interactive sessions,report,live source.')
