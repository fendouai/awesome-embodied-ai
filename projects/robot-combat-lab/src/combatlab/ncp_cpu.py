"""NCP physical-character CPU/MuJoCo transfer probe (not a G1 controller).

Uses the published 323+289 observation contract and checkpoint; no scripted strikes.
The source Isaac-Gym PD/contact engine and MuJoCo are NOT asserted equivalent.
"""
from pathlib import Path
import xml.etree.ElementTree as ET,json,hashlib,copy,time
import numpy as np,torch,mujoco
from torch import nn
from scipy.spatial.transform import Rotation
from .models import ROOT,ROBOTS,RobotSpec,scene_xml,GYM
from .engine import STATE_SPEC,write_json,sha256
from .storage import encode_model

BASE=ROOT/'research/adversarial-options/ncp-source/ncp/data/assets/mjcf/amp_humanoid_boxing.xml'
OUT=ROOT/'assets/adversarial-candidates/NCP'
BODY_NAMES=['pelvis','torso','head','right_upper_arm','right_lower_arm','right_hand','left_upper_arm','left_lower_arm','left_hand','right_thigh','right_shin','right_foot','left_thigh','left_shin','left_foot']
INITIAL=np.array([.0962,-.0388,-.0436,-.0695,-.1147,-.2051,-.2319,-.6831,.1655,-2.0966,.2319,-.6831,-.1655,-2.0966,.1199,-.2866,.1201,.6217,-.0489,-.3315,-.2917,-.0191,-.2375,-.1433,.5284,.0246,-.3087,.1618])

def build_model():
    OUT.mkdir(exist_ok=True);m=mujoco.MjModel.from_xml_path(str(BASE));mujoco.mj_saveLastXML(str(OUT/'native-boxer.xml'),m)
    ROBOTS['ncp_boxer']=RobotSpec('ncp_boxer','NCP physical character',OUT/'native-boxer.xml',GYM/'deploy/deploy_mujoco/configs/g1.yaml','Published NCP digital humanoid; not an identified hardware robot')
    return m

def tan_norm(R):return np.r_[R[:,0],R[:,2]]
def exp_features(q,offsets):
    terms=[]
    for lo,hi in zip(offsets[:-1],offsets[1:]):
        v=q[lo:hi];rotvec=v if hi-lo==3 else np.array([0,float(v[0]),0]);terms.append(tan_norm(Rotation.from_rotvec(rotvec).as_matrix()))
    return np.concatenate(terms)

class Dense(nn.Module):
    def __init__(self,state,prefix):
        super().__init__();keys=sorted([k for k in state if k.startswith(prefix) and k.endswith('.weight')],key=lambda k:int(k.split('.')[-2]));layers=[]
        for i,k in enumerate(keys):
            w=state[k];linear=nn.Linear(w.shape[1],w.shape[0]);linear.weight.data.copy_(w);linear.bias.data.copy_(state[k[:-6]+'bias']);layers.append(linear)
            if i<len(keys)-1:layers.append(nn.ReLU())
        self.model=nn.Sequential(*layers)
    def forward(self,x):return self.model(x)

class NCPPolicy:
    def __init__(self,seed,checkpoint="boxing_106.pth"):
        self.path=OUT/checkpoint;self.state=torch.load(self.path,map_location='cpu',weights_only=True)['model'];self.high=Dense(self.state,'a2c_network.pi_net.high_level_policy.mlp.');self.low=Dense(self.state,'a2c_network.pi_net.llc.mlp.');self.embedding=self.state['a2c_network.pi_net.embeddings.weight'];self.std=self.state['a2c_network.pi_net.llc.log_std'].exp();self.rng=torch.Generator().manual_seed(seed)
    def norm(self,x,field):
        s=self.state;prefix='a2c_network.rms_'+field+'.';return torch.clamp((torch.as_tensor(x,dtype=torch.float32)[None]-s[prefix+'moving_mean'])/(s[prefix+'moving_std']+1e-8),-5,5)
    @torch.inference_mode()
    def infer(self,prop,drill,stochastic=True):
        p=self.norm(prop,'prop');d=self.norm(drill,'drill');logits=self.high(torch.cat([p,d],-1));z=torch.multinomial(logits.softmax(-1),1,generator=self.rng).reshape(-1) if stochastic else logits.argmax(-1);embedding=self.embedding[z];mean=self.low(torch.cat([p,embedding],-1));noise=torch.randn(mean.shape,generator=self.rng)*self.std if stochastic else torch.zeros_like(mean);action=(mean+noise).numpy()[0];return action,int(z.item()),noise.numpy()[0]

class NCPEncounter:
    def __init__(self,seed=0,timestep=1/240,stochastic=True,checkpoints=("boxing_106.pth","boxing_100.pth"),policy_seeds=None):
        self.seed=seed;self.started=time.time();source=build_model();tree=ET.parse(OUT/'native-boxer.xml');root=tree.getroot()
        if root.find('asset') is None:ET.SubElement(root,'asset')
        tree.write(OUT/'native-boxer.xml',encoding='unicode')
        self.xml=scene_xml(('ncp_boxer','ncp_boxer'),timestep,1.);self.m=mujoco.MjModel.from_xml_string(self.xml);self.d=mujoco.MjData(self.m);self.dt=timestep;self.control_every=round((1/30)/timestep);self.stochastic=stochastic;self.policy_seeds=list(policy_seeds if policy_seeds is not None else (seed,seed+1));self.policy_names=list(checkpoints);self.policies=[NCPPolicy(s,c) for s,c in zip(self.policy_seeds,self.policy_names)];self.actors=[];self.offsets=[0]
        xml=ET.parse(BASE).getroot()
        for body in xml.findall('.//worldbody//body'):
            n=len(body.findall('joint'))
            if n:self.offsets.append(self.offsets[-1]+n)
        assert self.offsets[-1]==28
        for side in [0,1]:
            prefix=f'p{side}/';ids=np.array([i for i in range(self.m.nbody) if self.m.body(i).name.startswith(prefix)]);joint_ids=np.array([i for i in range(self.m.njnt) if self.m.joint(i).name.startswith(prefix) and self.m.jnt_type[i]!=mujoco.mjtJoint.mjJNT_FREE]);aids=np.array([next(u for u in range(self.m.nu) if self.m.actuator_trnid[u,0]==j) for j in joint_ids]);root=next(j for j in range(self.m.njnt) if self.m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and self.m.body(self.m.jnt_bodyid[j]).name==prefix+'pelvis');qadr=self.m.jnt_qposadr[joint_ids];vadr=self.m.jnt_dofadr[joint_ids];kp=self.m.jnt_stiffness[joint_ids].copy();kd=self.m.dof_damping[vadr].copy();limits=np.abs(self.m.actuator_gear[aids,0]).copy()
            self.m.jnt_stiffness[joint_ids]=0;self.m.dof_damping[vadr]=0
            for i,u in enumerate(aids):
                self.m.actuator_gear[u,:]=0;self.m.actuator_gear[u,0]=1;self.m.actuator_gaintype[u]=mujoco.mjtGain.mjGAIN_FIXED;self.m.actuator_biastype[u]=mujoco.mjtBias.mjBIAS_AFFINE;self.m.actuator_gainprm[u,:]=0;self.m.actuator_gainprm[u,0]=kp[i];self.m.actuator_biasprm[u,:]=0;self.m.actuator_biasprm[u,1]=-kp[i];self.m.actuator_biasprm[u,2]=-kd[i];self.m.actuator_forcelimited[u]=True;self.m.actuator_forcerange[u,:]=[-limits[i],limits[i]];self.m.actuator_ctrllimited[u]=False
            self.actors.append(dict(prefix=prefix,bodies=np.array([self.m.body(prefix+n).id for n in BODY_NAMES]),qadr=qadr,vadr=vadr,aids=aids,root_jid=root,root_bid=self.m.body(prefix+'pelvis').id,free_qadr=int(self.m.jnt_qposadr[root]),free_vadr=int(self.m.jnt_dofadr[root]),kp=kp,kd=kd,limits=limits))
        for side,a in enumerate(self.actors):
            q=a['free_qadr'];self.d.qpos[q:q+3]=[-.525 if side==0 else .525,0,.8739];R=Rotation.from_quat([-.0202,.0282,-.0027,.9994]).as_matrix();yaw=np.arctan2(R[1,0],R[0,0]);R=Rotation.from_euler('z',(np.pi if side else 0)-yaw).as_matrix()@R;quat=Rotation.from_matrix(R).as_quat();self.d.qpos[q+3:q+7]=quat[[3,0,1,2]];self.d.qpos[a['qadr']]=INITIAL
        mujoco.mj_forward(self.m,self.d);self.steps=0;self.frames=[];self.states=[];self.controls=[];self.io=[[],[]];self.actions=[[],[]];self.codes=[[],[]];self.noise=[[],[]];self.contact_rows=[];self.contact_impulse=0.;self.contact_steps=0;self.current_contacts=[];self.current_force=0.;self.peak=0.;self.energy=np.zeros(2);self.max_tilt=np.zeros(2);self.first_contact=None;self.warning_count=0;self.torque_violation=0.;self.terminal=None;self.update();self.record();self.record_frame()
    def body_state(self,a):
        ids=a['bodies'];pos=self.d.xpos[ids].copy();rot=self.d.xmat[ids].reshape(-1,3,3).copy();vel=[];ang=[]
        for b in ids:
            v=np.zeros(6);mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_BODY,int(b),v,0);ang.append(v[:3]);vel.append(v[3:])
        return pos,rot,np.asarray(vel),np.asarray(ang)
    def net_forces(self):
        forces=np.zeros((self.m.nbody,3))
        for k,c in enumerate(self.d.contact[:self.d.ncon]):
            f=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,k,f);world=c.frame.reshape(3,3).T@f[:3];forces[self.m.geom_bodyid[c.geom1]]-=world;forces[self.m.geom_bodyid[c.geom2]]+=world
        return [np.linalg.norm(forces[a['bodies']],axis=1) for a in self.actors]
    def observations(self,side,states,forces):
        a=self.actors[side];b=self.actors[1-side];pos,R,v,w=states[side];op,oR,ov,ow=states[1-side];yaw=np.arctan2(R[0,1,0],R[0,0,0]);H=Rotation.from_euler('z',-yaw).as_matrix();q=self.d.qpos[a['qadr']];dq=self.d.qvel[a['vadr']];oq=self.d.qpos[b['qadr']];odq=self.d.qvel[b['vadr']]
        prop=np.r_[pos[0,2],((pos-pos[0])@H.T)[1:].ravel(),np.concatenate([tan_norm(H@r) for r in R]),(v@H.T).ravel(),(w@H.T).ravel(),exp_features(q,self.offsets),dq]
        diff=op[0]-pos[0];diff[2]=op[0,2];other=op[1:]-pos[0];other[:,2]=op[1:,2];hands=pos[[BODY_NAMES.index('left_hand'),BODY_NAMES.index('right_hand')]];targets=op[[0,1,2]];hand_targets=((targets[None,:,:]-hands[:,None,:]).reshape(-1,3)@H.T).ravel()
        drill=np.r_[H@diff,tan_norm(H@oR[0]),H@ov[0],H@ow[0],exp_features(oq,self.offsets),odq,(other@H.T).ravel(),np.concatenate([tan_norm(H@r) for r in oR[1:]]),hand_targets,forces[side],forces[1-side]]
        assert len(prop)==323 and len(drill)==289
        return prop.astype(np.float32),drill.astype(np.float32)
    def update(self):
        states=[self.body_state(a) for a in self.actors];forces=self.net_forces()
        for side,a in enumerate(self.actors):
            prop,drill=self.observations(side,states,forces);action,z,noise=self.policies[side].infer(prop,drill,self.stochastic);self.d.ctrl[a['aids']]=self.d.qpos[a['qadr']]+action;self.io[side].append(np.r_[prop,drill]);self.actions[side].append(action);self.codes[side].append(z);self.noise[side].append(noise)
    def record(self):
        state=np.empty(mujoco.mj_stateSize(self.m,STATE_SPEC));mujoco.mj_getState(self.m,self.d,state,STATE_SPEC);self.states.append(state);self.controls.append(self.d.ctrl.copy())
    def record_frame(self):
        actors=[]
        for side,a in enumerate(self.actors):
            q=self.d.qpos[a['free_qadr']:a['free_qadr']+7];tilt=float(np.degrees(np.arccos(np.clip(2*(q[3]**2+q[6]**2)-1,-1,1))));self.max_tilt[side]=max(self.max_tilt[side],tilt);actors.append(dict(robot='NCP digital humanoid',policy='ncp_'+self.policy_names[side].removesuffix('.pth'),state='FALLEN' if tilt>65 or q[2]<.45 else 'UNSTABLE' if tilt>25 else 'MOVING',base_position=q[:3].copy(),tilt_deg=tilt,height_m=float(q[2]),speed_m_s=float(np.linalg.norm(self.d.qvel[a['free_vadr']:a['free_vadr']+2])),energy_positive_j=float(self.energy[side]),torque_utilization=float(np.max(abs(self.d.actuator_force[a['aids']])/a['limits'])),code=self.codes[side][-1]))
        self.frames.append(dict(time=float(self.d.time),physics_index=len(self.states)-1,actors=actors,contact_force_n=self.current_force,contact_impulse_ns=self.contact_impulse,peak_force_n=self.peak,contacts=self.current_contacts,terminal=self.terminal))
    def advance(self,seconds):
        for _ in range(round(seconds/self.dt)):
            if self.steps and self.steps%self.control_every==0:self.update()
            old=self.d.time;mujoco.mj_step(self.m,self.d);self.steps+=1
            if self.d.time<old or not np.isfinite(self.d.qpos).all():self.terminal=dict(reason='numerical_abort',winner=None,time=old);return False
            force=0.;self.current_contacts=[]
            for side,a in enumerate(self.actors):
                self.energy[side]+=float(np.maximum(self.d.actuator_force[a['aids']]*self.d.qvel[a['vadr']],0).sum()*self.dt);self.torque_violation=max(self.torque_violation,float(np.max(abs(self.d.actuator_force[a['aids']])-a['limits'])))
            for k,c in enumerate(self.d.contact[:self.d.ncon]):
                b1=self.m.body(self.m.geom_bodyid[c.geom1]).name;b2=self.m.body(self.m.geom_bodyid[c.geom2]).name
                if (b1.startswith('p0/') and b2.startswith('p1/')) or (b1.startswith('p1/') and b2.startswith('p0/')):
                    f=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,k,f);force+=np.linalg.norm(f[:3]);self.current_contacts.append(dict(pos=c.pos.copy(),force_n=float(np.linalg.norm(f[:3])),normal=c.frame[:3].copy()));self.contact_rows.append([self.d.time,c.geom1,c.geom2,*c.pos,*f,*c.frame[:3]])
            self.current_force=float(force);self.contact_steps+=int(force>0);self.contact_impulse+=force*self.dt;self.peak=max(self.peak,force)
            if force and self.first_contact is None:self.first_contact=float(self.d.time)
            self.record()
            if self.steps%self.control_every==0:mujoco.mj_forward(self.m,self.d);self.record_frame()
        return True

    def save(self,base=None):
        """Preserve physics and stochastic policy IO; finite resource pause is not a loss."""
        mid='ncp-cpu-'+str(time.time_ns());folder=Path(base or ROOT/'results/ncp-cpu')/mid;folder.mkdir(parents=True)
        if self.terminal is None:self.terminal=dict(reason='budget_pause',winner=None,time=float(self.d.time),hardware_damage=False)
        self.record_frame();(folder/'scene.xml').write_text(self.xml)
        mujoco.mj_saveModel(self.m,str(folder/'scene.mjb'),None);encode_model(folder,(folder/'scene.mjb').read_bytes(),('ncp_boxer','ncp_boxer'));(folder/'scene.mjb').unlink()
        np.savez_compressed(folder/'trajectory.npz',integration_states=np.stack(self.states),controls=np.stack(self.controls),timestep=self.dt,state_spec=int(STATE_SPEC),qpos_width=self.m.nq,qvel_width=self.m.nv)
        arrays={'time':np.arange(len(self.io[0]))*self.control_every*self.dt}
        for side in (0,1):
            arrays.update({f'p{side}_observations':np.stack(self.io[side]),f'p{side}_actions':np.stack(self.actions[side]),f'p{side}_codes':np.asarray(self.codes[side]),f'p{side}_noise':np.stack(self.noise[side]),f'p{side}_rng_final':self.policies[side].rng.get_state().numpy()})
        np.savez_compressed(folder/'policy_io.npz',**arrays)
        np.savez_compressed(folder/'contacts.npz',samples=np.asarray(self.contact_rows,dtype=float).reshape(-1,15),columns=np.array(['time','geom1','geom2','x','y','z','fn','ft1','ft2','tn','tt1','tt2','nx','ny','nz']))
        write_json(folder/'telemetry.json',self.frames);write_json(folder/'events.json',[])
        write_json(folder/'config.json',dict(seed=self.seed,timestep=self.dt,control_dt=self.control_every*self.dt,stochastic=self.stochastic,policies=self.policy_names,policy_seeds=self.policy_seeds,initial_distance_m=1.05))
        (folder/'ncp_cpu.py').write_bytes(Path(__file__).read_bytes());(folder/'source-character.xml').write_bytes(BASE.read_bytes())
        contract=dict(repository='Tencent-RoboticsX/NCP',commit='42f9af5ecf179443e7d38dfd65241843d27588c2',body_order=BODY_NAMES,proprioception=323,drill=289,action=28,codebook=[256,64],target='current_joint_position + policy_action',sensing='simulator_ground_truth',checkpoints=[dict(file=p.path.name,sha256=sha256(p.path)) for p in self.policies],limitations=['Physical character, not a hardware robot','Isaac Gym to MuJoCo transfer, not source physics parity','Stochastic evaluation; latent code and Gaussian noise saved','No validated hardware failure or competition ranking','Not the SMPLOlympics multi-sport task suite'])
        write_json(folder/'source-contract.json',contract)
        result=dict(match_id=mid,track='ncp_cpu_port',rated=False,robots=['NCP digital humanoid']*2,policies=['ncp_'+name.removesuffix('.pth') for name in self.policy_names],duration_s=float(self.d.time),result=self.terminal,seed=self.seed,first_contact_s=self.first_contact,contact_duration_s=self.contact_steps*self.dt,contact_impulse_ns=self.contact_impulse,peak_contact_force_n=self.peak,max_tilt_deg=self.max_tilt,positive_mechanical_work_j=self.energy,numerical_warnings=int(sum(w.number for w in self.d.warning)),max_torque_limit_violation_nm=max(0.,self.torque_violation),external_root_force_peak=float(np.abs(self.d.xfrc_applied).max()),adversarial_backend=dict(name='NCP CPU physical character transfer',validated_reproduction=False,scripted_strikes=False),continuous=dict(damage_prediction='not_calibrated'),physics=dict(engine='MuJoCo',version=mujoco.__version__,timestep_s=self.dt,control_dt_s=self.control_every*self.dt,no_hidden_root_forces=True,source_engine='Isaac Gym PhysX'),archive=dict(physics_frames=len(self.states),control_frames=len(self.io[0]),replay='Complete integration states, controls, latent code, noise and neural observations'),wall_time_s=time.time()-self.started)
        write_json(folder/'result.json',result);write_json(folder/'manifest.json',{f.name:sha256(f) for f in folder.iterdir() if f.is_file() and f.name!='manifest.json'})
        write_json(folder.parent/'latest.json',dict(folder=str(folder.relative_to(ROOT)),result=result,native_match_reproduction=False))
        return folder,result
