from pathlib import Path
import json,hashlib,subprocess
import numpy as np,mujoco,torch,onnxruntime as ort
from combatlab.models import ROOT,make_actors
from combatlab.storage import load_model
from combatlab.engine import STATE_SPEC,write_json,sha256
from combatlab.policies import gravity_body

if __name__=='__main__':
 root=ROOT/'results/control-upgrade';rows=json.loads((root/'validation-results.json').read_text());assert len(rows)==33;checks=[]
 for asset in json.loads((ROOT/'research/control-upgrade/candidate-assets.json').read_text()):assert sha256(ROOT/asset['local'])==asset['sha256']
 for r in rows:
  f=root/'archives'/r['id'];manifest=json.loads((f/'manifest.json').read_text())
  for name,h in manifest.items():assert sha256(f/name)==h,(r['id'],name)
  assert sha256(f/'control_upgrade.py')==r['software_fingerprint']
  m=load_model(f);actors=make_actors(m,(r['robot'],r['robot']));raw=np.load(f/'trajectory.npz');states=raw['integration_states'];ctrl=raw['controls'];forces=np.load(f/'disturbance.npz')['samples'];assert len(states)==len(forces)+1;assert np.max(abs(np.diff(states[:,0])-.002))<1e-8
  assert np.max(abs(states[:-1,0]-forces[:,0]))<1e-8
  bodies=[m.body(a.prefix+'torso_link').id for a in actors];flag=int(mujoco.mjtState.mjSTATE_XFRC_APPLIED);offset=mujoco.mj_stateSize(m,int(STATE_SPEC)&(flag-1));external=states[:,offset:offset+m.nbody*6].reshape(len(states),m.nbody,6);expected=np.zeros_like(external)
  for side,b in enumerate(bodies):expected[1:,b,:3]=forces[:,1+3*side:4+3*side]
  assert np.max(abs(external-expected))<1e-8,'Undeclared force'
  impulses=[float(np.linalg.norm(forces[:,1+3*i:4+3*i],axis=1).sum()*.002) for i in [0,1]];assert max(abs(x-r['impulse_ns']) for x in impulses)<1e-6
  active=np.linalg.norm(forces[:,1:4],axis=1)>0
  if active.any():assert np.min(forces[active,0])>=3-1e-8 and np.max(forces[active,0])<3.1+1e-8
  qflag=int(mujoco.mjtState.mjSTATE_QFRC_APPLIED);qoff=mujoco.mj_stateSize(m,int(STATE_SPEC)&(qflag-1));assert np.max(abs(states[:,qoff:qoff+m.nv]))==0
  err=0.
  for ix in [0,50,1499,1500,1549,1550,len(states)//2,len(states)-2]:
   d=mujoco.MjData(m);e=mujoco.MjData(m);mujoco.mj_setState(m,d,states[ix],STATE_SPEC);d.ctrl[:]=ctrl[ix+1];d.xfrc_applied[:]=0
   for side,b in enumerate(bodies):d.xfrc_applied[b,:3]=forces[ix,1+3*side:4+3*side]
   mujoco.mj_step(m,d);mujoco.mj_setState(m,e,states[ix+1],STATE_SPEC);err=max(err,float(np.max(abs(d.qpos-e.qpos))),float(np.max(abs(d.qvel-e.qvel))))
  assert err<1e-8,(r['id'],err)
  assert not r['numerical_warnings'] and r['torque_violation_nm']<1e-8
  io=np.load(f/'policy_io.npz');meta=json.loads((f/'result.json').read_text());weight=ROOT/meta['actors'][0]['policy_file'];assert sha256(weight)==meta['actors'][0]['policy_sha256'];obs=io['p0_observations'];actions=io['p0_actions'];neural=0.
  if weight.suffix=='.onnx':
   opts=ort.SessionOptions();opts.intra_op_num_threads=1;net=ort.InferenceSession(str(weight),sess_options=opts,providers=['CPUExecutionProvider']);name=net.get_inputs()[0].name
   for ix in [0,100,150,len(obs)-1]:neural=max(neural,float(np.max(abs(net.run(None,{name:obs[ix:ix+1]})[0].reshape(-1)-actions[ix]))))
   nsamples=4
  else:
   net=torch.jit.load(str(weight));n=actions.shape[1];warm=np.zeros(9+3*n+2,dtype=np.float32);warm[5]=-1;warm[-1]=1
   with torch.inference_mode():
    first=net(torch.from_numpy(warm)[None]).numpy().reshape(-1);warm[9+2*n:9+3*n]=first;net(torch.from_numpy(warm)[None])
    for o,a in zip(obs,actions):neural=max(neural,float(np.max(abs(net(torch.from_numpy(o)[None]).numpy().reshape(-1)-a))))
   nsamples=len(obs)
  assert neural<1e-6
  checks.append(dict(id=r['id'],exact_source_verified=True,physics_frames=len(states),declared_impulses_ns=impulses,no_undeclared_force=True,one_step_points=8,max_replay_error=err,neural_samples=nsamples,max_neural_error=neural));print('VERIFY UPGRADE',len(checks),'/33',r['id'],flush=True)
 write_json(root/'verification.json',dict(passed=True,checks=checks,scope='All33 validation archives: hashes, loaded-adapter snapshot, all-frame explicit force schedule and zero undeclared forces, eight one-step physics replays including pulse edges. ONNX4IO samples; Torch recurrent full399-observation sequence after two documented deterministic initialization calls. Screening uses20ms measurements, validation stores500Hz complete states.',source_assets_verified=True))
