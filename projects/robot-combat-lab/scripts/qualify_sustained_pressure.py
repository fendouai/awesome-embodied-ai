"""A separately reported low-speed pushing qualification, with matched zero-action ablation.
Does not overwrite or relabel the original dynamic-impact gate. Fresh validation seeds
are separate from100-102 used to diagnose the low-speed contact case.
"""
import json,os,sys
from pathlib import Path
import numpy as np,mujoco
from combatlab.models import ROOT
from combatlab.engine import STATE_SPEC
from combatlab.research_archive import workspace,run_recorded,atomic_json,event,digest,now
root=workspace();index=root/'00_index/pressure-qualification.json';criteria=dict(version=1,method='SUSTAINED_ACTIVE_PUSHING, not repeated impact/striking',defined_after_diagnosis_of_seeds_100_102=True,validation_seeds=[200,201,202,203,204],thresholds=dict(each_agent_initial_directed_approach_m=.05,last3s_physics_contact_duty=.9,last3s_each_agent_positive_native_forward_command_fraction=.8,native_forward_command_threshold_m_s=.05,mean_normal_force_above_zero_baseline_n=1.,normal_force_ratio_to_baseline=5.,minimum_mean_normal_force_n=1.),requirements=['Same model checkpoint, seed and initial qpos/qvel for learned and zero-high-level-control cases','Zero case keeps the identical native gait controller and source physical model','Complete12s, no falls/numerical warnings/root forces','Actual archived actuator forces within source limits','All5fresh validation seeds pass; no change to original dynamic gate'],interpretation='Slow bodies can deliberately approach and maintain motor-driven normal pressure while relative impact speed approaches zero. This is qualified as pushing, not punching, damage or a dynamic-impact gate pass.')
atomic_json(root/'00_index/pressure-criteria.json',criteria);event('sustained_pressure_validation_protocol_recorded',criteria_sha256=digest(root/'00_index/pressure-criteria.json'),validation_seeds=criteria['validation_seeds']);nav=json.loads((root/'00_index/navigation-refinement.json').read_text());data=dict(state='running',criteria=criteria,robots=[])
def source(folder):
 meta=json.loads((folder/'raw/metadata.json').read_text());m=mujoco.MjModel.from_binary_path(str(folder/'raw/scene.mjb'));states=[];contacts=[];forces=[]
 for file in sorted((folder/'raw').glob('physics-*.npz')):
  with np.load(file) as a:states.append(a['integration_states']);contacts.append(a['contacts']);forces.append(a['actuator_force'])
 states=np.concatenate(states);contacts=np.concatenate(contacts);forces=np.concatenate(forces);qadr=[int(m.jnt_qposadr[next(j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and (m.body(m.jnt_bodyid[j]).name or '').startswith(f'p{s}/'))]) for s in range(2)];d0=mujoco.MjData(m);d1=mujoco.MjData(m);mujoco.mj_setState(m,d0,states[0],STATE_SPEC);mujoco.mj_setState(m,d1,states[-1],STATE_SPEC);start=np.array([d0.qpos[q:q+2] for q in qadr]);end=np.array([d1.qpos[q:q+2] for q in qadr]);direction=(start[1]-start[0])/np.linalg.norm(start[1]-start[0]);approach=[float((end[0]-start[0])@direction),float((end[1]-start[1])@(-direction))];tail=contacts[contacts[:,1]>=float(states[-1,0])-3-1e-8];duty=len(np.unique(tail[:,0]))/round(3/m.opt.timestep);normal=float(tail[:,7].sum()*m.opt.timestep/3);io=json.loads((folder/'control-io.json').read_text());commands=[[],[]]
 for row in io:
  for update in row['native_updates']:
   if update['time']>=float(states[-1,0])-3-1e-8:commands[update['side']].append(update['command'][0]>.05)
 ratio=np.max(np.abs(forces)/np.maximum(abs(m.actuator_forcerange).max(axis=1),1e-12));return dict(initial_qpos=d0.qpos.copy(),initial_qvel=d0.qvel.copy(),approach_m=approach,duty=float(duty),normal_n=normal,positive_command_fraction=[float(np.mean(c)) if c else 0. for c in commands],max_actuator_limit_ratio=float(ratio),scene_sha256=digest(folder/'raw/scene.mjb'),trace_manifest_sha256=digest(folder/'raw/chunks.jsonl'))
for key in ['a1','b2']:
 row=next(r for r in nav['robots'] if r['robot']==key);checkpoint=row['checkpoint'];folder=Path(row['folder']);data['current']=key;atomic_json(index,data)
 for zero in [True,False]:
  summary=folder/('evaluation-pressure-validation-zero' if zero else 'evaluation-pressure-validation')/'summary.json'
  if not summary.exists():
   args=[sys.executable,str(ROOT/'scripts/evaluate_contact.py'),checkpoint,'--seeds','200,201,202,203,204','--label','pressure-validation']+(['--zero-control'] if zero else [])
   run_recorded(args,root/'07_logs'/('pressure-validation-'+key+('-zero' if zero else '')+'.log'),cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},timeout=600)
 results=[]
 for seed in criteria['validation_seeds']:
  learned=folder/'evaluation-pressure-validation'/f'seed-{seed}';zero=folder/'evaluation-pressure-validation-zero'/f'seed-{seed}';a=source(learned);b=source(zero);lr=json.loads((learned/'result.json').read_text());zr=json.loads((zero/'result.json').read_text());np.testing.assert_array_equal(a.pop('initial_qpos'),b.pop('initial_qpos'));np.testing.assert_array_equal(a.pop('initial_qvel'),b.pop('initial_qvel'))
  for step in json.loads((zero/'control-io.json').read_text()):assert np.max(np.abs(step['high_level_raw_actions']))==0
  passed=bool(min(a['approach_m'])>=.05 and a['duty']>=.9 and min(a['positive_command_fraction'])>=.8 and a['normal_n']>=max(1.,b['normal_n']+1,5*b['normal_n']) and not any(lr['fallen']) and not any(zr['fallen']) and lr['duration_s']>=12-1e-8 and zr['duration_s']>=12-1e-8 and lr['numerical_warnings']==0 and zr['numerical_warnings']==0 and lr['external_root_force_peak']==0 and a['max_actuator_limit_ratio']<=1+1e-6)
  result=dict(robot=key,seed=seed,passed=passed,qualification='sustained_active_pushing',learned=a,zero_high_level_control=b,checkpoint_sha256=digest(checkpoint),source_learned_result_sha256=digest(learned/'result.json'),source_zero_result_sha256=digest(zero/'result.json'),original_dynamic_gate_unchanged=lr['continuous_engagement']);atomic_json(learned/'pressure-audit.json',result);results.append(result)
 data['robots'].append(dict(robot=key,checkpoint=checkpoint,folder=str(folder),passed=all(r['passed'] for r in results),qualification='sustained_active_pushing',seeds=results));atomic_json(index,data);event('sustained_pressure_validation_completed',robot=key,passed=data['robots'][-1]['passed'])
data.update(state='complete',finished_utc=now());data.pop('current',None);atomic_json(index,data)
