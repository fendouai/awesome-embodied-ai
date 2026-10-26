"""CPU port of the pinned B-Human K1 walking observation contract; gait unvalidated."""
from collections import deque
import numpy as np,onnxruntime as ort
from .models import ROOT,ROBOTS,RobotSpec
from .policies import gravity_body
from .research_archive import workspace

def register():
 root=workspace();path=root/'09_models/k1/k1-native.xml';ROBOTS['k1']=RobotSpec('k1','Booster K1 / native collision export',path,path,'Native SimRobot3.3.5 rigid-body/actuator export, CPU walking adapter under validation')

class K1Walk:
 key='bhuman_k1_walk';control_dt=.012
 def __init__(self,a):
  self.a=a;self.path=ROOT/'integrations/bhuman/source/Config/NeuralNets/Walk/K1_walk.onnx';o=ort.SessionOptions();o.intra_op_num_threads=1;o.inter_op_num_threads=1;self.net=ort.InferenceSession(str(self.path),sess_options=o,providers=['CPUExecutionProvider']);self.input=self.net.get_inputs()[0];self.legnames=[s+j for s in ['L','R'] for j in ['HipPitch','HipRoll','HipYaw','KneePitch','AnklePitch','AnkleRoll']];self.mp=np.array([a.names.index(n) for n in self.legnames]);self.offset=a.default[self.mp].copy();self.target=a.default.copy();self.reset()
 def reset(self):self.history=deque(maxlen=10);self.last_action=np.zeros(12);self.raw_frequency=0.;self.phase=0.;self.target=self.a.default.copy();self.last_observation=np.zeros(344)
 def update(self,m,d,cmd,t):
  a=self.a;gravity=gravity_body(d.qpos[a.free_qadr+3:a.free_qadr+7]);gyro=d.qvel[a.free_vadr+3:a.free_vadr+6];q=d.qpos[a.qadr][self.mp];dq=d.qvel[a.vadr][self.mp];frame=np.r_[gravity,gyro,q-self.offset,self.last_action,0.,0.].astype(np.float32)
  if not self.history:
   for _ in range(10):self.history.append(frame.copy())
  else:self.history.append(frame)
  moving=np.linalg.norm(cmd)>1e-4;cycle=[np.cos(2*np.pi*self.phase),np.sin(2*np.pi*self.phase)] if moving else [0.,0.];obs=np.r_[np.concatenate(self.history),cmd,cycle,dq*.1,self.raw_frequency,np.zeros(6)].astype(np.float32);assert len(obs)==344
  self.last_observation=obs.copy();shape=self.input.shape;x=obs if len(shape)==1 else obs[None];out=self.net.run(None,{self.input.name:x})[0].reshape(-1);assert out.shape==(13,) and np.isfinite(out).all();self.last_action=np.clip(out[:12],-2,2);self.target[self.mp]=self.offset+self.last_action;self.raw_frequency=float(np.clip(out[-1],-.5,.5));self.phase=(self.phase+(1.5+self.raw_frequency)*self.control_dt)%1;return self.target
