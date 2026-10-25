"""Verify counts, compatibility, source policy IO and real physical A/B schedules."""
import json
import numpy as np
import torch,mujoco,pytest
from combatlab.models import ROOT
from combatlab.catalog import catalog
from combatlab.comparison import Comparison,validate
from combatlab.quadruped import QuadPolicy
from combatlab.storage import load_model
from combatlab.engine import STATE_SPEC,sha256


def suite():return json.loads((ROOT/'results/comparisons/suite.json').read_text())['rows']

def test_hardware_and_control_counts_are_not_weight_duplicates():
 c=catalog();assert len({r['id'] for r in c['robots']})==10;assert c['control_scheme_count']==10
 assert 'ncp' not in {r['id'] for r in c['robots']}
 assert len({p['scheme'] for p in c['controls'] if p['scheme']!='reference'})==10
 with pytest.raises(ValueError):validate('go2','nvidia_wbc','go2_him','push')
 with pytest.raises(ValueError):validate('k1','bhuman','nvidia_wbc','push')
 with pytest.raises(ValueError):validate('g1_full','nvidia_wbc','nvidia_wbc','tracking')


def test_initial_clearance_and_matched_jitter():
 for robot,left,right in [('go2','go2_robot_lab','go2_him'),('a1','a1_vq','pd_stand'),('h1_full','unitree_full_gym','pd_stand')]:
  s=Comparison(robot,left,right,seed=1)
  assert min([c.dist for c in s.d.contact[:s.d.ncon]] or [.003])>=-.001
  a,b=s.actors;np.testing.assert_allclose(s.d.qpos[a.qadr]-a.default,s.d.qpos[b.qadr]-b.default,rtol=0,atol=1e-12)
  assert abs(s.mass[0]-s.mass[1])<1e-12 and not np.any(s.d.xfrc_applied)


def test_actual_quad_checkpoint_outputs_and_history_order():
 for robot,first,second in [('go2','go2_robot_lab','go2_him'),('a1','a1_vq','pd_stand')]:
  row=next(r for r in suite() if r['result']['robots'][0]==robot and r['result']['task']=='tracking' and r['result']['seed']==0);folder=ROOT/row['folder'];data=np.load(folder/'policy_io.npz');s=Comparison(robot,first,second)
  for side,p in enumerate(s.policies):
   if not isinstance(p,QuadPolicy):continue
   observations=data[f'p{side}_observations'];actions=data[f'p{side}_actions']
   for i in (0,len(observations)//2,len(observations)-1):
    with torch.inference_mode():a=p.net(torch.from_numpy(observations[i])[None]).numpy()[0]
    a=np.clip(a,p.cfg['clip_actions_lower'],p.cfg['clip_actions_upper']);np.testing.assert_allclose(a,actions[i],rtol=1e-5,atol=1e-5)
   if p.cfg['observations_history']:
    block=observations[0].reshape(6,45);old=block[1:] if p.cfg['observations_history'][0]==0 else block[:-1];assert not np.any(old)
   assert len(set(p.training_names))==12 and all(n in p.a.names for n in p.training_names)


def test_explicit_push_impulse_and_archive_one_step():
 rows=suite();assert len(rows)>=40
 for row in rows:
  result=row['result'];assert result['numerical_warnings']==0 and result['torque_violation_nm']<1e-6
  folder=ROOT/row['folder'];force=np.load(folder/'disturbance.npz')['samples'];dt=result['physics']['timestep_s']
  for side,a in enumerate(result['actors']):
   integral=force[:,1+side*3:4+side*3].sum(0)*dt;expected=np.array([0,a['mass_kg']*.25 if result['task']=='push' else 0,0]);np.testing.assert_allclose(integral,expected,atol=1e-8)
  if result['robots'][0] not in ['go2','h1_full']:continue
  if result['seed']!=0 or result['task']!='push':continue
  m=load_model(folder);d=mujoco.MjData(m);raw=np.load(folder/'trajectory.npz');i=round(3.04/dt);mujoco.mj_setState(m,d,raw['integration_states'][i],STATE_SPEC);d.ctrl[:]=raw['controls'][i+1];d.xfrc_applied[:]=0
  for side in [0,1]:
   root=next(j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and m.body(m.jnt_bodyid[j]).name.startswith(f'p{side}/'));d.xfrc_applied[m.jnt_bodyid[root],:3]=force[i,1+side*3:4+side*3]
  mujoco.mj_step(m,d);expected=mujoco.MjData(m);mujoco.mj_setState(m,expected,raw['integration_states'][i+1],STATE_SPEC);np.testing.assert_allclose(d.qpos,expected.qpos,atol=2e-8,rtol=1e-7)


def test_catalog_dofs_and_http_compatibility_guards():
 from fastapi.testclient import TestClient
 from combatlab.server import app
 from combatlab.models import scene_xml,make_actors
 from combatlab.quadruped import quad_actors
 import re
 c=catalog()
 for robot in c['robots']:
  if robot['id']=='k1':
   cfg=(ROOT/'integrations/bhuman/source/Config/Robots/Default/K1/motorParameters.cfg').read_text().split('regular = {',1)[1].split('\n  };',1)[0];limits=[float(x) for x in re.findall(r'maxTorque\s*=\s*([0-9.]+)',cfg)];assert sum(x>0 for x in limits)==robot['dof'];continue
  m=mujoco.MjModel.from_xml_string(scene_xml((robot['id'],robot['id'])));a=(quad_actors(m,robot['id']) if robot['id'] in ['go2','a1','go1','b2','go2w','b2w'] else make_actors(m,(robot['id'],robot['id'])))[0];assert len(a.names)==robot['dof']
 client=TestClient(app);assert client.get('/api/catalog').json()['robot_count']==10
 for payload in [dict(robot='go2',left='nvidia_wbc',right='go2_him'),dict(robot='g1_full',left='nvidia_wbc',right='amp_recovery',task='combat')]:
  assert client.post('/api/comparisons/run',json=payload).status_code==400


def test_readiness_annotations_preserve_trace_hashes():
 for row in suite():
  folder=ROOT/row['folder'];analysis=json.loads((folder/'analysis.json').read_text());assert analysis['source_telemetry_sha256']==sha256(folder/'telemetry.json')
  if row['result']['robots'][0] in ['h1_full','h1_2_full','g1_full'] and row['result']['task']=='push' and row['result']['policies'][1]=='pd_stand':assert analysis['actors'][1]['fell_before_push']


def test_cpu_jobs_cannot_overlap_between_backends(monkeypatch):
 from fastapi.testclient import TestClient
 from combatlab import server
 client=TestClient(server.app)
 monkeypatch.setattr(server,'COMPARE_JOB',dict(state='running'));monkeypatch.setattr(server,'PORT_JOB',dict(state='idle'))
 assert client.post('/api/adversarial/run/robostriker').status_code==409
 monkeypatch.setattr(server,'COMPARE_JOB',dict(state='idle'));monkeypatch.setattr(server,'PORT_JOB',dict(state='running'))
 assert client.post('/api/comparisons/run',json=dict(robot='go2',left='go2_robot_lab',right='go2_him')).status_code==409


def test_split_comparisons_have_no_cross_robot_contacts():
 models={}
 for row in suite():
  folder=ROOT/row['folder'];robot=row['result']['robots'][0]
  if robot not in models:
   m=load_model(folder);side=np.array([0 if (m.body(m.geom_bodyid[g]).name or '').startswith('p0/') else 1 if (m.body(m.geom_bodyid[g]).name or '').startswith('p1/') else -1 for g in range(m.ngeom)]);models[robot]=(m,side)
  m,side=models[robot];d=mujoco.MjData(m);raw=np.load(folder/'trajectory.npz')
  for state in raw['integration_states'][::10]:
   mujoco.mj_setState(m,d,state,STATE_SPEC);mujoco.mj_forward(m,d)
   for c in d.contact[:d.ncon]:assert not (side[c.geom1]>=0 and side[c.geom2]>=0 and side[c.geom1]!=side[c.geom2])
