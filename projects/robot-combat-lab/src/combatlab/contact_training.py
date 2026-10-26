"""Small CPU PPO self-play trainer; every rollout and native physical step is archived.
Raw Gaussian actions are transformed by tanh in the environment. PPO ratios use the
same raw variable so its transform Jacobian cancels; inactive joints are masked.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,time,uuid,os
import numpy as np,torch
from torch import nn
from .contact_arena import ContactArena,OBS_DIM,ACTION_DIM
from .research_archive import workspace,event,atomic_json,snapshot_sources,digest

class ContactPolicy(nn.Module):
 def __init__(self):
  super().__init__();self.actor=nn.Sequential(nn.Linear(OBS_DIM,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,ACTION_DIM));self.critic=nn.Sequential(nn.Linear(OBS_DIM,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1));self.logstd=nn.Parameter(torch.full((ACTION_DIM,),-.8));nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.zeros_(self.actor[-1].bias)
 def distribution(self,x):return torch.distributions.Normal(self.actor(x),self.logstd.clamp(-3,0).exp())
 def value(self,x):return self.critic(x).squeeze(-1)

def train(robot,iterations=24,steps_per_iteration=256,seed=0,initial_checkpoint=None,episode_s=6,reward_version=1,navigation_auxiliary=False):
 torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed);root=workspace();run_id='ppo-'+robot+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6];folder=root/'03_training'/run_id;folder.mkdir();(folder/'checkpoints').mkdir();(folder/'rollouts').mkdir();snapshot_sources(folder)
 config=dict(robot=robot,seed=seed,navigation_auxiliary=navigation_auxiliary,algorithm='PPO clipped surrogate / shared-policy symmetric self-play',iterations=iterations,steps_per_iteration=steps_per_iteration,learning_rate=.0003,gamma=.99,gae_lambda=.95,clip=.2,epochs=4,minibatch=128,entropy_coefficient=.005,value_coefficient=.5,max_gradient_norm=.5,observation_dim=OBS_DIM,action_dim=ACTION_DIM,physics_dt=.002,action_dt=.02,episode_s=episode_s,reward_version=reward_version,action_transform='tanh(raw Gaussian); mask inactive joint dimensions',reward='2*distance progress +0.05dt upright -0.2dt unstable -0.002mean(action delta squared) +0.02min(initiated contact impulse,10) when upright; opponent fall +3, own fall -3',reward_v2_additions='When upright: alignment .15dt*cos(opponent bearing), proximity .1dt*exp(-estimated gap/.5), contact presence .1dt. Logged as shaping rewards, not direct controls.',policy_status='training_experiment_not_validated',all_physics_steps_recorded=True,randomized_initial_conditions=True,initial_checkpoint=initial_checkpoint);atomic_json(folder/'config.json',config);env=ContactArena(robot,seed=seed,trace_folder=folder/'raw',episode_s=episode_s,initial_jitter=True,reward_version=reward_version,yaw_jitter=.25 if reward_version>=2 else .04,lateral_jitter=.12 if reward_version>=2 else .02,gap_range=(.3,.6) if reward_version>=2 else (.3,.5));policy=ContactPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=config['learning_rate'])
 if initial_checkpoint:policy.load_state_dict(torch.load(initial_checkpoint,map_location='cpu',weights_only=False)['policy'])
 if navigation_auxiliary:
  from .navigation_pretrain import teacher_targets
 mask=torch.as_tensor(env.action_mask());global_steps=0;episode_records=[];episode_return=np.zeros(2);episode_contacts=np.zeros(2);started=time.time();event('training_started',run_id=run_id,folder=str(folder),config=config)
 def checkpoint(iteration,status):
  target=folder/'checkpoints'/f'iteration-{iteration:05}.pt';temp=target.with_suffix('.tmp');torch.save(dict(policy=policy.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),environment_rng=env.rng.bit_generator.state,iteration=iteration,global_control_steps=global_steps,config=config,status=status,next_episode=env.episode+1,episode_boundary_checkpoint=True),temp);os.replace(temp,target);row=dict(run_id=run_id,robot=robot,iteration=iteration,checkpoint=str(target),sha256=digest(target),status=status);atomic_json(folder/'latest.json',row);event('training_checkpoint',**row);return row
 try:
  checkpoint(0,'untrained');obs=env.observations()
  for iteration in range(1,iterations+1):
   episode_start=len(episode_records)
   if iteration>1:obs=env.reset();episode_return[:]=0;episode_contacts[:]=0
   saved={k:[] for k in ['observations','raw_actions','log_probabilities','values','rewards','terminated','truncated','next_values','episode']};info_rows=[]
   for step in range(steps_per_iteration):
    x=torch.as_tensor(obs)
    with torch.no_grad():distribution=policy.distribution(x);raw=distribution.sample();logp=(distribution.log_prob(raw)*mask).sum(-1);values=policy.value(x)
    next_obs,reward,terminated,truncated,info=env.step(raw.numpy());global_steps+=1;episode_return+=reward;episode_contacts+=np.array(info['initiated_contact_impulse_ns'])
    with torch.no_grad():next_value=policy.value(torch.as_tensor(next_obs)).numpy()
    for key,value in dict(observations=obs,raw_actions=raw.numpy(),log_probabilities=logp.numpy(),values=values.numpy(),rewards=reward,terminated=np.full(2,terminated),truncated=np.full(2,truncated or step==steps_per_iteration-1),next_values=next_value,episode=np.full(2,env.episode)).items():saved[key].append(np.asarray(value).copy())
    info_rows.append(dict(control_step=global_steps,episode=env.episode,**info))
    if terminated or truncated or step==steps_per_iteration-1:
     episode_records.append(dict(episode=env.episode,iteration=iteration,return_by_side=episode_return.tolist(),initiated_contact_impulse_ns=episode_contacts.tolist(),duration_s=info['time'],terminated=terminated,truncated=truncated or step==steps_per_iteration-1,iteration_boundary=step==steps_per_iteration-1,fallen=info['fallen']));episode_return[:]=0;episode_contacts[:]=0
     obs=env.reset() if step<steps_per_iteration-1 else next_obs
    else:obs=next_obs
   arrays={k:np.array(v) for k,v in saved.items()};advantages=np.zeros_like(arrays['rewards']);last=np.zeros(2)
   for t in reversed(range(steps_per_iteration)):
    bootstrap=~arrays['terminated'][t];continue_trajectory=~(arrays['terminated'][t]|arrays['truncated'][t]);delta=arrays['rewards'][t]+config['gamma']*arrays['next_values'][t]*bootstrap-arrays['values'][t];last=delta+config['gamma']*config['gae_lambda']*continue_trajectory*last;advantages[t]=last
   returns=advantages+arrays['values'];arrays.update(advantages=advantages,returns=returns);np.savez_compressed(folder/'rollouts'/f'iteration-{iteration:05}.npz',**arrays);atomic_json(folder/'rollouts'/f'iteration-{iteration:05}-native-io.json',info_rows)
   X=torch.as_tensor(arrays['observations'].reshape(-1,OBS_DIM));A=torch.as_tensor(arrays['raw_actions'].reshape(-1,ACTION_DIM));OLD=torch.as_tensor(arrays['log_probabilities'].reshape(-1));ADV=torch.as_tensor(advantages.reshape(-1));ADV=(ADV-ADV.mean())/(ADV.std()+1e-8);RET=torch.as_tensor(returns.reshape(-1));flatmask=mask.repeat(steps_per_iteration,1);losses=[]
   for epoch in range(config['epochs']):
    ids=torch.randperm(len(X))
    for idx in ids.split(config['minibatch']):
     dist=policy.distribution(X[idx]);logp=(dist.log_prob(A[idx])*flatmask[idx]).sum(-1);ratio=(logp-OLD[idx]).exp();actor=-torch.minimum(ratio*ADV[idx],ratio.clamp(.8,1.2)*ADV[idx]).mean();value_loss=(policy.value(X[idx])-RET[idx]).square().mean();entropy=(dist.entropy()*flatmask[idx]).sum(-1).mean();navigation_loss=(policy.actor(X[idx])[:,:3].tanh()-teacher_targets(X[idx])).square().mean() if navigation_auxiliary else torch.tensor(0.);loss=actor+.5*value_loss-.005*entropy+.2*navigation_loss;optimizer.zero_grad();loss.backward();grad=float(nn.utils.clip_grad_norm_(policy.parameters(),.5));optimizer.step();losses.append(dict(navigation_auxiliary_mse=float(navigation_loss.detach()),actor=float(actor.detach()),critic=float(value_loss.detach()),raw_gaussian_entropy=float(entropy.detach()),gradient_norm_before_clip=grad,approx_kl=float((OLD[idx]-logp).mean().detach())))
   env.trace.flush();current=checkpoint(iteration,'training_experiment_not_validated');metrics=dict(iteration=iteration,global_control_steps=global_steps,mean_reward=float(arrays['rewards'].mean()),mean_training_loss=float(np.mean([l['actor']+l['critic']*.5 for l in losses])),optimizer_updates=losses,episodes=episode_records[episode_start:],episodes_total=len(episode_records),wall_time_s=time.time()-started,checkpoint=current)
   atomic_json(folder/'metrics.json',metrics)
   with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
   print('PPO',robot,iteration,'steps',global_steps,'reward',metrics['mean_reward'],'contacts',sum(sum(e['initiated_contact_impulse_ns']) for e in episode_records),'falls',sum(any(e['fallen']) for e in episode_records),flush=True);event('training_iteration_completed',run_id=run_id,robot=robot,iteration=iteration,control_steps=global_steps,checkpoint_sha256=current['sha256'])
  atomic_json(folder/'status.json',dict(state='training_pass_complete_requires_evaluation',run_id=run_id,robot=robot,iterations=iterations,control_steps=global_steps));event('training_pass_finished',run_id=run_id,robot=robot,folder=str(folder));return folder
 except BaseException as error:
  atomic_json(folder/'status.json',dict(state='interrupted',error=repr(error),control_steps=global_steps));event('training_interrupted',run_id=run_id,error=repr(error));raise
 finally:env.close()
