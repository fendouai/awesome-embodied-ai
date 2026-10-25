"""Readiness and checkpoint smoke checks. This does NOT claim match reproduction."""
from pathlib import Path
import json,hashlib,platform,shutil,subprocess,time
import torch,numpy as np,onnxruntime as ort
from combatlab.models import ROOT
from combatlab.engine import write_json,sha256

def command(args):
 try:
  p=subprocess.run(args,capture_output=True,text=True,timeout=8);return dict(ok=p.returncode==0,output=(p.stdout+p.stderr).strip()[:800])
 except (FileNotFoundError,subprocess.TimeoutExpired) as e:return dict(ok=False,output=type(e).__name__)

def mlp_probe(state,prefix):
 layers=[]
 for k,v in state.items():
  if k.startswith(prefix+'.') and k.endswith('.weight') and v.ndim==2:
   index=int(k[len(prefix)+1:].split('.')[0]);layers.append((index,v,state[k[:-6]+'bias']))
 layers.sort(key=lambda x:x[0]);x=torch.zeros(1,layers[0][1].shape[1]);start=time.perf_counter()
 with torch.inference_mode():
  for i,(_,w,b) in enumerate(layers):
   x=torch.nn.functional.linear(x,w,b)
   if i<len(layers)-1:x=torch.nn.functional.elu(x)
 assert torch.isfinite(x).all()
 return dict(input_width=layers[0][1].shape[1],output_width=x.shape[1],finite=True,elapsed_ms=(time.perf_counter()-start)*1000,scope='ELU MLP subnet on synthetic zero input; NOT physical rollout or end-to-end match')

if __name__=='__main__':
 torch.set_num_threads(1);P=ROOT/'assets/adversarial-candidates';manifest=json.loads((P/'manifest.json').read_text());checks=[]
 for record in manifest:
  file=ROOT/record['local'];assert sha256(file)==record['sha256'];blob=hashlib.sha1(f'blob {file.stat().st_size}\0'.encode()+file.read_bytes()).hexdigest();assert blob==record['git_blob_sha1']
  if file.suffix=='.onnx':
   opt=ort.SessionOptions();opt.intra_op_num_threads=1;net=ort.InferenceSession(str(file),sess_options=opt,providers=['CPUExecutionProvider']);inputs={x.name:np.zeros([i if isinstance(i,int) and i>0 else 1 for i in x.shape],dtype=np.float32) for x in net.get_inputs()};outputs=net.run(None,inputs);assert all(np.isfinite(x).all() for x in outputs);check=dict(kind='ONNX CPU synthetic-input forward',inputs={k:list(v.shape) for k,v in inputs.items()},outputs=[list(v.shape) for v in outputs],finite=True)
  else:
   obj=torch.load(file,map_location='cpu',weights_only=True);state=obj.get('model_state_dict',obj.get('model',obj));tensors={k:v for k,v in state.items() if isinstance(v,torch.Tensor)};assert tensors and all(torch.isfinite(v).all() for v in tensors.values());check=dict(kind='safe tensor-only CPU checkpoint loading',tensor_count=len(tensors),tensor_values=sum(v.numel() for v in tensors.values()),finite=True)
   if 'taskpolicy_ego.0.weight' in state:
    check['subnets']={p:mlp_probe(state,p) for p in ['taskpolicy_ego','taskpolicy_opponent','prior','decoder']};check['normalizer_widths']={k:list(v['_mean'].shape) for k,v in obj.items() if isinstance(v,dict) and '_mean' in v and '_std' in v}
   else:check['observed_contract']=dict(proprioception_width=323,task_width=289,latent_categories=256,latent_embedding_width=64,low_level_actions=28,scope='Identified from actual tensor shapes, not a G1 action mapping')
  checks.append(dict(file=record['local'],repository=record['repository'],commit=record['commit'],sha256=record['sha256'],git_object_verified=True,check=check));print('CHECKPOINT PASS',file.name,flush=True)
 host=dict(platform=platform.platform(),python=platform.python_version(),cuda_available=torch.cuda.is_available(),cuda_devices=torch.cuda.device_count(),free_disk_bytes=shutil.disk_usage(ROOT).free,xcode=command(['xcodebuild','-version']),cmake=shutil.which('cmake'),ninja=shutil.which('ninja'),docker=command(['docker','info','--format','{{.ServerVersion}} {{.Architecture}}']),full_matches_reproduced=0)
 result=dict(date='2026-10-25',host=host,checkpoint_checks=checks,lanes=[dict(id='bhuman',goal='完整机器人足球',native_match_ready=False,stage='K1 walk/kick/get-up ONNX synthetic-input CPU forward passed',blockers=['Full Xcode/CMake build environment missing','Only about600MBdiskfree; complete repo+submodules+build assets exceed safe reserve','K1/T1 scene and full perception/team behaviour not yet run']),dict(id='sports',goal='SMPLOlympics＋NCP多项目/复杂人形',native_match_ready=False,stage='NCP actual boxing checkpoint safely loaded on CPU; SMPLOlympics download/script contract audited',blockers=['Isaac Gym Linux/NVIDIA environment absent','SMPLOlympics SMPL/SMPL-X assets and per-sport trained policies not validated','NCP native humanoid observations/actuation differ from G1']),dict(id='robostriker',goal='G1自主格斗',native_match_ready=False,stage='Stage3 genuine checkpoint verified; both task networks, prior and29-action decoder subnet CPU forwards passed',blockers=['Isaac Lab2.1/Isaac Sim4.5 native runtime requires Linux/NVIDIA','Native assets/motion files and init checkpoint path must be resolved','MuJoCo observation ordering, history, normalization, native gains and learned latent composition need independent sim-to-sim validation'])],scope='Infrastructure/serialization/tensor checks only. No completed native match, no installed Isaac runtime, no gameplay or combat-capability claim.')
 # Keep previously measured physical-rollout acceptance; these tensor probes
 # neither invalidate it nor replace it with a fictitious new rollout.
 previous_path=ROOT/'integrations/readiness.json'
 if previous_path.exists():
  previous=json.loads(previous_path.read_text())
  if 'local_runs' in previous:
   for key in ['local_runs','lanes','status','scope']:result[key]=previous[key]
 write_json(ROOT/'integrations/readiness.json',result)
 print('READINESS SAVED; completed matches=0; no substitution of existing pushing footage.')
