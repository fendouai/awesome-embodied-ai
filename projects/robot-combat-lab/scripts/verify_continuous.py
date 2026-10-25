from pathlib import Path
import json,subprocess,re
import numpy as np,mujoco
from combatlab.models import ROOT
from combatlab.storage import load_model
from combatlab.engine import STATE_SPEC,sha256,write_json
from combatlab.endurance import ContinuousSimulation,ENDURANCE_FINGERPRINT
P=ROOT/'results/continuous';summary=json.loads((P/'summary.json').read_text());assert summary['terminal']['reason']=='budget_pause' and summary['terminal']['winner'] is None and summary['damage_prediction']=='not_calibrated'
checks=[];boundary=None;total_impulse=0.;contact_times=[]
for part in summary['parts']:
 f=ROOT/part['path'];r=json.loads((f/'result.json').read_text())
 for name,h in json.loads((f/'manifest.json').read_text()).items():assert sha256(f/name)==h
 assert r['continuous']['endurance_fingerprint']==ENDURANCE_FINGERPRINT
 m=load_model(f);raw=np.load(f/'trajectory.npz');states=raw['integration_states'];controls=raw['controls'];assert np.isfinite(states).all();assert np.max(abs(np.diff(states[:,0])-.002))<1e-8
 if boundary is not None:assert np.array_equal(boundary,states[0]),'Segment boundary mismatch'
 boundary=states[-1].copy();hw=np.load(f/'hardware.npz');assert len(hw['temperature_c'])==len(states)==len(hw['actuator_forcerange']);assert np.isnan(hw['temperature_c']).all()
 for flag,width in [(int(mujoco.mjtState.mjSTATE_XFRC_APPLIED),m.nbody*6),(int(mujoco.mjtState.mjSTATE_QFRC_APPLIED),m.nv)]:
  offset=mujoco.mj_stateSize(m,int(STATE_SPEC)&(flag-1));assert np.max(abs(states[:,offset:offset+width]))==0
 error=0.
 for ix in [0,1000,len(states)//2,len(states)-2]:
  actual=mujoco.MjData(m);expected=mujoco.MjData(m);mujoco.mj_setState(m,actual,states[ix],STATE_SPEC);actual.ctrl[:]=controls[ix+1];m.actuator_forcerange[:]=hw['actuator_forcerange'][ix];mujoco.mj_step(m,actual);mujoco.mj_setState(m,expected,states[ix+1],STATE_SPEC);error=max(error,float(np.max(abs(actual.qpos-expected.qpos))),float(np.max(abs(actual.qvel-expected.qvel))))
 assert error<1e-8
 contact=np.load(f/'contacts.npz')['samples'];total_impulse+=float(np.linalg.norm(contact[:,6:9],axis=1).sum()*.002);contact_times.extend(contact[:,0].tolist())
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames:format=duration','-of','json',str(f/'battle.mp4')]))
 assert abs(float(probe['format']['duration'])-(part['end']-part['start']))<.1 and probe['streams'][0]['width']==1280
 assert r['external_root_force_peak']==0 and r['numerical_warnings']==0 and r['max_torque_limit_violation_nm']<1e-8
 runtime=json.loads((f/'runtime.json').read_text());assert not runtime['hardware']['parameters'] and not any(runtime['hardware']['burned'])
 checks.append(dict(part=part['path'],start=part['start'],end=part['end'],physics_frames=len(states),max_replay_error=error,full_video=True,damage_unknown=True));raw.close()
assert abs(total_impulse-r['contact_impulse_ns'])<1e-6
contact_duration=len(np.unique(contact_times))*.002;assert abs(contact_duration-r['contact_duration_s'])<1e-8
restored=ContinuousSimulation.restore(ROOT/summary['parts'][-1]['path']);assert abs(restored.d.time-summary['recorded_until_s'])<1e-8;restored.save_on_chunks=False;restored.resume(1.);assert restored.advance(.1) and restored.terminal is None
browser=json.loads((ROOT/'evidence/continuous-browser.json').read_text());assert not browser['errors'] and browser['resume_kept_time'] and browser['hardware_unknown_visible']
first=re.search(r'results/live-sessions/[a-zA-Z0-9_-]+',browser['first_archive']).group();second=re.search(r'results/live-sessions/[a-zA-Z0-9_-]+',browser['resumed_archive']).group()
a=np.load(ROOT/first/'trajectory.npz')['integration_states'];b=np.load(ROOT/second/'trajectory.npz')['integration_states'];assert np.array_equal(a[-1],b[0]);budget=json.loads((ROOT/browser['budget_archive']/'result.json').read_text());assert budget['result']['reason']=='budget_pause' and budget['result']['winner'] is None
for source in json.loads((ROOT/'research/hardware-failure/source-manifest.json').read_text()):assert sha256(ROOT/source['local'])==source['sha256']
for source in json.loads((ROOT/'research/hardware-failure/official-manual-manifest.json').read_text()):assert sha256(ROOT/source['local'])==source['sha256']
write_json(P/'verification.json',dict(passed=True,recorded_until_s=summary['recorded_until_s'],contact_duration_s=contact_duration,contact_impulse_ns=total_impulse,parts=checks,full_state_boundaries_exact=True,no_undeclared_external_forces=True,checkpoint_restored_and_advanced=True,interactive_resume_exact_boundary=True,budget_pause_has_no_winner=True,source_fingerprint=ENDURANCE_FINGERPRINT,hardware_prediction='NOT VALIDATED: thermal/fracture/gearbox data unavailable; no damage winner claimed',official_manual_pages_verified=31))
print('CONTINUOUS VERIFIED',summary['recorded_until_s'],'s; actual contact',contact_duration,'s; exact chunk/resume boundaries; no hardware-failure claim.')
