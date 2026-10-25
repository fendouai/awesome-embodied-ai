"""Re-adjudicate from a full deterministic physics replay, not from saved score labels."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json
import numpy as np,mujoco
from combatlab.duel import Referee,DuelRules,DuelSimulation
from combatlab.engine import STATE_SPEC,write_json,sha256
from combatlab.storage import load_model

def verify(folder):
 folder=Path(folder);report=json.loads((folder/'duel-result.json').read_text());rules=DuelRules(**report['rules']);m=load_model(folder);d=mujoco.MjData(m);raw=np.load(folder/'trajectory.npz');states=raw['integration_states'];controls=raw['controls'];dt=float(raw['timestep']);cfg=json.loads((folder/'config.json').read_text());cadence=round(float(cfg['control_dt'])/dt);mujoco.mj_setState(m,d,states[0],STATE_SPEC);roots=[]
 for side in range(2):roots.append(next(j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and m.body(m.jnt_bodyid[j]).name.startswith(f'p{side}/')))
 qadr=[int(m.jnt_qposadr[j]) for j in roots];r=Referee(rules,[float(d.qpos[q+2]) for q in qadr]);view=SimpleNamespace(m=m,d=d);maxerror=0.;expected=mujoco.MjData(m)
 for i in range(len(states)-1):
  pre=d.qvel.copy();d.ctrl[:]=controls[i+1];assert not d.xfrc_applied.any() and not d.qfrc_applied.any();mujoco.mj_step(m,d)
  if (i+1)%cadence==0:mujoco.mj_forward(m,d)
  poses=[]
  for q in qadr:
   v=d.qpos[q:q+7];poses.append(dict(tilt_deg=float(np.degrees(np.arccos(np.clip(2*(v[3]**2+v[6]**2)-1,-1,1)))),height_m=float(v[2])))
  r.observe(d.time,poses,DuelSimulation.scoring_contacts(view,pre))
  if i%50==0 or i==len(states)-2:
   mujoco.mj_setState(m,expected,states[i+1],STATE_SPEC);maxerror=max(maxerror,float(np.max(abs(d.qpos-expected.qpos))));np.testing.assert_allclose(d.qpos,expected.qpos,atol=2e-7,rtol=1e-7)
 outcome=report['outcome'];assert r.outcome is not None and outcome is not None;assert r.outcome['winner']==outcome['winner'] and r.outcome['reason']==outcome['reason'] and r.scores==outcome['clean_upper_contacts'];np.testing.assert_allclose(r.scoring_impulse,outcome['short_window_normal_impulse_ns'],atol=1e-5,rtol=1e-7)
 data=dict(verified=True,physics_steps=len(states)-1,max_qpos_replay_error=maxerror,winner=r.outcome['winner'],score=r.scores,normal_impulse_ns=r.scoring_impulse,trajectory_sha256=sha256(folder/'trajectory.npz'),rules_sha256=sha256(folder/'duel-rules.json'),method='Replay every native physics step with archived controls; independently recompute body poses, pre-contact approach speeds, force-window impulses and referee outcome')
 write_json(folder/'duel-verification.json',data);manifest=json.loads((folder/'manifest.json').read_text());manifest['duel-verification.json']=sha256(folder/'duel-verification.json');write_json(folder/'manifest.json',manifest);print('DUEL VERIFIED',folder,data,flush=True);return data
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder');args=p.parse_args();verify(Path(args.folder))
