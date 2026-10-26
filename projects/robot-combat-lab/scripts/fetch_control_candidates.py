from pathlib import Path
import json,urllib.request,urllib.parse,concurrent.futures,hashlib
P=Path(__file__).resolve().parents[1];research=P/'research/control-upgrade'
choices={
'ccrpRepo/wbc_fsm':['src/FSM/State_MJAmp.cpp','include/FSM/State_MJAmp.h','config/mjamp.json','model/loco/Unitree-G1-AMP-Flat_model_30000.onnx','README.md'],
'amazon-far/holosoma':['src/holosoma_inference/holosoma_cpp/config/inference/g1-29dof-loco.yaml','src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx','src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/ppo_g1_29dof.onnx','src/holosoma/holosoma/config_values/loco/g1/observation.py','LICENSE'],
'NVlabs/GR00T-WholeBodyControl':['decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py','decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml','decoupled_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml','decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx','decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Walk.onnx','decoupled_wbc/sim2mujoco/resources/robots/g1/policy/NVIDIA Open Model License']}
jobs=[]
for repo,paths in choices.items():
 tree=json.loads((research/(repo.split('/')[-1]+'-tree.json')).read_text());sha=tree['sha']
 for path in paths:jobs.append((repo,sha,path))
def fetch(job):
 repo,sha,path=job;url=f'https://raw.githubusercontent.com/{repo}/{sha}/{urllib.parse.quote(path)}';req=urllib.request.Request(url,headers={'User-Agent':'BotsTest-Control-Research'});data=urllib.request.urlopen(req,timeout=60).read()
 if data.startswith(b'version https://git-lfs'):data=urllib.request.urlopen('https://media.githubusercontent.com/media/'+repo+'/'+sha+'/'+path.replace(' ','%20'),timeout=120).read()
 out=P/'assets/control-candidates'/repo.split('/')[-1]/path;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
 return dict(repository=repo,commit=sha,path=path,local=str(out.relative_to(P)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),source=url)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
 results=list(pool.map(fetch,jobs))
(research/'candidate-assets.json').write_text(json.dumps(results,indent=2));print('FETCHED',len(results),sum(x['bytes'] for x in results))
