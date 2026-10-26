"""Incremental exact-file/video and sampled native-step replay audit; no fabricated result."""
import json,subprocess,csv,time
from pathlib import Path
import numpy as np,mujoco
from combatlab.models import ROOT
from combatlab.engine import STATE_SPEC,sha256
from combatlab.storage import load_model
BASE=ROOT/'results/all-combinations/round-01';OUT=BASE/'audit.json'
def audit_job(j):
 folder=ROOT/j['folder'];manifest=json.loads((folder/'manifest.json').read_text());checked={}
 for name in ['trajectory.npz','policy_io.npz','disturbance.npz','comparison.mp4']:
  value=sha256(folder/name);assert manifest[name]==value;checked[name]=value
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height','-of','json',str(folder/'comparison.mp4')]))
 result=json.loads((folder/'result.json').read_text());assert abs(float(probe['format']['duration'])-result['duration_s'])<.15
 assert {s['codec_name'] for s in probe['streams']}=={'h264','aac'}
 m=load_model(folder);raw=np.load(folder/'trajectory.npz');states=raw['integration_states'];controls=raw['controls'];forces=np.load(folder/'disturbance.npz')['samples'];expected=mujoco.MjData(m);roots=[m.jnt_bodyid[k] for k in range(m.njnt) if m.jnt_type[k]==mujoco.mjtJoint.mjJNT_FREE];assert len(roots)==2
 indices=sorted(set([20,len(states)//2,len(states)-2]));error=0.
 for i in indices:
  d=mujoco.MjData(m);mujoco.mj_setState(m,d,states[i],STATE_SPEC);d.ctrl[:]=controls[i+1];d.xfrc_applied[:]=0
  for side,b in enumerate(roots):d.xfrc_applied[b,:3]=forces[i,1+3*side:4+3*side]
  mujoco.mj_step(m,d);mujoco.mj_setState(m,expected,states[i+1],STATE_SPEC);error=max(error,float(np.max(np.abs(d.qpos-expected.qpos))));np.testing.assert_allclose(d.qpos,expected.qpos,atol=2e-7,rtol=1e-7)
 assert result['numerical_warnings']==0 and result['torque_violation_nm']<1e-6
 return dict(id=j['id'],folder=j['folder'],verified=True,file_hashes=checked,video_duration_s=float(probe['format']['duration']),sampled_native_step_indices=indices,max_qpos_error=error,physics_frames=len(states),method='Exact artifact SHA256, ffprobe duration/codecs, three restored native MuJoCo steps with archived controls and explicit push forces; not a full deterministic rerun')
if __name__=='__main__':
 existing=json.loads(OUT.read_text()) if OUT.exists() else dict(jobs=[]);rows={j['id']:j for j in existing['jobs']}
 while True:
  plan=json.loads((BASE/'index.json').read_text())
  for j in plan['jobs']:
   if j['state']!='complete' or j['id'] in rows:continue
   rows[j['id']]=audit_job(j);tmp=OUT.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state='complete' if len(rows)==plan['planned'] else 'running',jobs=list(rows.values())),indent=2));tmp.replace(OUT);print('AUDITED',j['id'],flush=True)
  if plan['state']=='complete':assert len(rows)==plan['planned'];break
  if plan['state']=='error':raise RuntimeError(plan['error'])
  time.sleep(10)
