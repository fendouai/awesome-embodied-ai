"""Recorded synthetic goal-direction supervision; teacher is NEVER used at deployment.
Real proprioception is resampled from physical rollouts, relative goal directions are
augmented and explicitly labeled synthetic. Checkpoints and datasets remain linked.
"""
import json,time,uuid
from pathlib import Path
import numpy as np,torch
from .contact_training import ContactPolicy
from .research_archive import workspace,event,atomic_json,snapshot_sources,digest

def teacher_targets(observations):
 rel=observations[:,99:102]*5;x=rel[:,0];y=rel[:,1];angle=torch.atan2(y,x);yaw=.8*torch.clamp(1.5*angle,-1,1);yaw=torch.where((x<0)&(y.abs()<.25),torch.full_like(yaw,.8),yaw);forward=.6*torch.clamp(torch.cos(angle),0,1);lateral=.3*torch.sin(angle);return torch.stack([forward,lateral,yaw],-1)

def pretrain(checkpoint,updates=1500,seed=2026):
 checkpoint=Path(checkpoint);saved=torch.load(checkpoint,map_location='cpu',weights_only=False);robot=saved['config']['robot'];torch.manual_seed(seed);rng=np.random.default_rng(seed);folder=workspace()/'03_training'/('navigation-'+robot+'-'+uuid.uuid4().hex[:8]);folder.mkdir();snapshot_sources(folder);policy=ContactPolicy();policy.load_state_dict(saved['policy']);reference=ContactPolicy();reference.load_state_dict(saved['policy']);reference.eval();rollouts=sorted((checkpoint.parent.parent/'rollouts').glob('iteration-*.npz'));source=[];indices=[]
 for p in rollouts:
  with np.load(p) as a:source.append(a['observations'].reshape(-1,131))
  indices.append(dict(file=str(p),sha256=digest(p)))
 source=np.concatenate(source);n=32768;X=source[rng.integers(0,len(source),n)].copy();angle=rng.uniform(-np.pi,np.pi,n);radius=rng.uniform(.15,4,n);X[:,99]=radius*np.cos(angle)/5;X[:,100]=radius*np.sin(angle)/5;X[:,101]=rng.uniform(-1.3,1.3,n)/5;x=torch.from_numpy(X);targets=teacher_targets(x)
 with torch.no_grad():residual_reference=reference.actor(x)[:,3:]
 np.savez_compressed(folder/'synthetic-navigation-dataset.npz',observations=X,normalized_goal_command_targets=targets.numpy(),residual_reference=residual_reference.numpy());atomic_json(folder/'config.json',dict(robot=robot,seed=seed,updates=updates,teacher='Geometric bearing/forward-velocity teacher with rear tie-break. Supervision only, not a deployed controller or strike script.',source_checkpoint=str(checkpoint),source_checkpoint_sha256=digest(checkpoint),source_physical_rollouts=indices,dataset_kind='SYNTHETIC RELATIVE GOAL AUGMENTATION OF RECORDED PROPRIOCEPTION; NOT PHYSICAL TRAJECTORIES',normalization='Observation99:102 is body-frame relative position /5; output0:3 normalized command, deployment tanh bounds still applied'))
 optimizer=torch.optim.Adam(policy.actor.parameters(),lr=.001);event('navigation_supervision_started',robot=robot,folder=str(folder),source_checkpoint_sha256=digest(checkpoint));started=time.time()
 with (folder/'metrics.jsonl').open('w') as log:
  for i in range(1,updates+1):
   idx=torch.randint(n,(256,));out=policy.actor(x[idx]);nav_loss=(out[:,:3].tanh()-targets[idx]).square().mean();distill=(out[:,3:]-residual_reference[idx]).square().mean();loss=nav_loss+.1*distill;optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.);optimizer.step();row=dict(update=i,navigation_mse=float(nav_loss.detach()),residual_distillation_mse=float(distill.detach()),wall_time_s=time.time()-started);log.write(json.dumps(row)+'\n')
   if i%250==0:
    outpath=folder/f'checkpoint-{i:05}.pt';cfg={**saved['config'],'navigation_pretraining':True,'navigation_teacher_version':1,'robot':robot};torch.save(dict(policy=policy.state_dict(),optimizer=optimizer.state_dict(),config=cfg,iteration=i,status='navigation_supervision_requires_physical_validation',torch_rng=torch.get_rng_state(),numpy_generator_state=rng.bit_generator.state),outpath);event('navigation_supervision_checkpoint',robot=robot,update=i,checkpoint=str(outpath),sha256=digest(outpath));print(robot,i,row,flush=True)
 final=folder/f'checkpoint-{updates:05}.pt';assert final.exists();atomic_json(folder/'latest.json',dict(checkpoint=str(final),sha256=digest(final),robot=robot));return final
