import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import mujoco,torch,numpy as np,yaml
from combatlab.models import GYM
from combatlab.policies import gravity_body
m=mujoco.MjModel.from_xml_path(str(GYM/'resources/robots/g1_description/scene.xml'));d=mujoco.MjData(m);c=yaml.safe_load((GYM/'deploy/deploy_mujoco/configs/g1.yaml').read_text());m.opt.timestep=.002
q=np.array(c['default_angles']);d.qpos[7:]=q;d.qpos[2]=.78;action=np.zeros(12);target=q.copy();net=torch.jit.load(str(GYM/'deploy/pre_train/g1/motion.pt'));net.eval();torch.set_num_threads(1)
for step in range(3000):
 d.ctrl[:]=(target-d.qpos[7:])*np.array(c['kps'])-d.qvel[6:]*np.array(c['kds']);mujoco.mj_step(m,d)
 if step%10==9:
  obs=np.r_[d.qvel[3:6]*.25,gravity_body(d.qpos[3:7]),[.4,0,0],d.qpos[7:]-q,d.qvel[6:]*.05,action,np.sin(2*np.pi*d.time/.8),np.cos(2*np.pi*d.time/.8)].astype('float32')
  with torch.inference_mode():action=net(torch.from_numpy(obs)[None,:]).numpy().reshape(-1)
  target=q+action*.25
 if step%500==0:print(step, d.qpos[:7],flush=True)
