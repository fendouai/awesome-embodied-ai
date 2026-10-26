import json
import numpy as np,mujoco,torch,pytest
from combatlab.models import ROOT
from combatlab.comparison import Comparison,validate
from combatlab.catalog import catalog
from combatlab.storage import load_model
from combatlab.engine import STATE_SPEC,sha256


def extensions():return json.loads((ROOT/'results/comparisons/ten-extension.json').read_text())['rows']


def test_ten_counts_use_distinct_hardware_and_architectures():
 c=catalog();assert c['robot_count']==10 and c['control_scheme_count']==10
 assert {r['id'] for r in c['robots']}=={'g1_full','h1_full','h1_2_full','k1','go2','a1','go1','b2','go2w','b2w'}
 assert 'mini_cheetah' not in {r['id'] for r in c['robots']}
 assert len({x['scheme'] for x in c['controls'] if x['scheme']!='reference'})==10
 with pytest.raises(ValueError):validate('g1_full','g1_beyondmimic','pd_stand','tracking')
 with pytest.raises(ValueError):validate('go2w','g1_beyondmimic','pd_stand','motion')


def test_wheel_axes_are_velocity_controlled_not_position_locked():
 for robot in ['go2w','b2w']:
  s=Comparison(robot,robot+'_robot_lab','pd_stand',seed=1)
  for a in s.actors:
   wheel=[i for i,n in enumerate(a.names) if 'wheel_joint' in n];assert len(wheel)==4
   for i in wheel:
    u=a.actuator_ids[i];assert a.kp[i]==0 and s.m.actuator_gainprm[u,0]==a.kd[i] and s.m.actuator_biasprm[u,1]==0 and s.m.actuator_biasprm[u,2]==-a.kd[i]
  assert min([c.dist for c in s.d.contact[:s.d.ncon]] or [.003])>=-.001


def test_new_neural_ports_reproduce_actual_saved_actions():
 for robot,policy,task in [('go1','go1_wtw','tracking'),('b2','b2_robot_lab','tracking'),('go2w','go2w_robot_lab','tracking'),('b2w','b2w_robot_lab','tracking'),('g1_full','g1_beyondmimic','motion')]:
  row=next(r for r in extensions() if r['result']['robots'][0]==robot and r['result']['task']==task and r['result']['seed']==0);raw=np.load(ROOT/row['folder']/'policy_io.npz');p=Comparison(robot,policy,'pd_stand',task).policies[0];obs=raw['p0_observations'];actions=raw['p0_actions']
  for i in [0,len(obs)//2,len(obs)-1]:
   x=torch.from_numpy(obs[i])[None]
   with torch.inference_mode():out=(p.net(torch.cat([x,p.adapt(x)],-1)) if robot=='go1' else p.net(x)).numpy()[0]
   clip=10 if robot=='go1' else 100;np.testing.assert_allclose(np.clip(out,-clip,clip),actions[i],atol=1e-5,rtol=1e-5)
  for f in row['result']['actors'][0]['policy_files']:assert sha256(ROOT/f['file'])==f['sha256']


def test_reference_motion_uses_feedback_and_has_zero_root_assistance():
 row=next(r for r in extensions() if r['result']['task']=='motion' and r['result']['seed']==0);folder=ROOT/row['folder'];r=row['result'];assert r['actors'][0]['reference_joint_rmse_rad']<.15 and r['actors'][0]['reference_joint_rmse_rad']<r['actors'][1]['reference_joint_rmse_rad']
 assert not np.any(np.load(folder/'disturbance.npz')['samples'][:,1:]);m=load_model(folder);d=mujoco.MjData(m);raw=np.load(folder/'trajectory.npz');i=501;mujoco.mj_setState(m,d,raw['integration_states'][i],STATE_SPEC);assert not d.xfrc_applied.any() and not d.qfrc_applied.any();d.ctrl[:]=raw['controls'][i+1];mujoco.mj_step(m,d);expected=mujoco.MjData(m);mujoco.mj_setState(m,expected,raw['integration_states'][i+1],STATE_SPEC);np.testing.assert_allclose(d.qpos,expected.qpos,atol=2e-8,rtol=1e-7)


def test_go1_policy_configuration_matches_go1_not_excluded_mini():
 cfg=json.loads((ROOT/'research/ten-platform/wtw-parameters-readable.json').read_text())['Cfg'];assert '/go1/' in cfg['asset']['file'] and cfg['env']['num_observations']==70 and cfg['env']['num_observation_history']==30


def test_wheel_robot_record_replays_one_physics_step_with_declared_push():
 row=next(r for r in extensions() if r['result']['robots'][0]=='go2w' and r['result']['task']=='push' and r['result']['seed']==0);folder=ROOT/row['folder'];m=load_model(folder);d=mujoco.MjData(m);raw=np.load(folder/'trajectory.npz');forces=np.load(folder/'disturbance.npz')['samples'];dt=float(raw['timestep']);i=round(3.04/dt)
 mujoco.mj_setState(m,d,raw['integration_states'][i],STATE_SPEC);d.ctrl[:]=raw['controls'][i+1];d.xfrc_applied[:]=0
 for side in range(2):
  j=next(j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and m.body(m.jnt_bodyid[j]).name.startswith(f'p{side}/'));d.xfrc_applied[m.jnt_bodyid[j],:3]=forces[i,1+side*3:4+side*3]
 mujoco.mj_step(m,d);expected=mujoco.MjData(m);mujoco.mj_setState(m,expected,raw['integration_states'][i+1],STATE_SPEC);np.testing.assert_allclose(d.qpos,expected.qpos,atol=2e-8,rtol=1e-7);np.testing.assert_allclose(d.qvel,expected.qvel,atol=2e-7,rtol=1e-6)
