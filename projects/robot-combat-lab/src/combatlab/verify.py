from __future__ import annotations
from pathlib import Path
import json,hashlib,subprocess
import numpy as np,mujoco,onnxruntime as ort
from .models import ROOT,make_actors
from .engine import STATE_SPEC,write_json,SOFTWARE_FINGERPRINT
from .storage import load_model,load_binary
from .benchmark import rank
from .policies import gravity_body


def verify(require_videos=True):
    package=json.loads((ROOT/'results/leaderboards.json').read_text());results=package['matches'];assert len(results)==26
    preflight=json.loads((ROOT/'results/preflight.json').read_text());assert preflight['passed'] and preflight['software_fingerprint']==SOFTWARE_FINGERPRINT
    assert json.loads((ROOT/'results/reach-control-reference.json').read_text())['passed']
    checks=[];cache_hashes={}
    for m in json.loads((ROOT/'assets/provenance.json').read_text()):
        assert hashlib.sha256((ROOT/m['local']).read_bytes()).hexdigest()==m['sha256'],m['local']
    for number,r in enumerate(results):
        folder=ROOT/'results/matches'/r['match_id'];assert r['software_fingerprint']==SOFTWARE_FINGERPRINT
        manifest=json.loads((folder/'manifest.json').read_text())
        for name,expected in manifest.items():
            path=folder/name;assert path.exists(),str(path)
            actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==expected,str(path)
        model=load_model(folder);actors=make_actors(model,tuple(r['robots']));raw=np.load(folder/'trajectory.npz');states=raw['integration_states'];controls=raw['controls'];dt=float(raw['timestep']);q=states[:,1:1+model.nq];v=states[:,1+model.nq:1+model.nq+model.nv];times=states[:,0]
        assert np.isfinite(states).all() and len(states)>1000
        assert np.max(np.abs(np.diff(times)-dt))<1e-9
        assert len(states)==len(controls)
        assert r['external_root_force_peak']==0 and r['max_torque_limit_violation_nm']<1e-8 and r['numerical_warnings']==0
        assert np.allclose(model.actuator_gear[:,0],1.)
        assert np.all(model.jnt_type[[a.free_jid for a in actors]]==mujoco.mjtJoint.mjJNT_FREE)
        # Independently reconstruct recorded mechanical work from PD law and joint velocities.
        energy_error=[]
        for side,a in enumerate(actors):
            ids=a.actuator_ids;tau=controls[1:,ids]*model.actuator_gainprm[ids,0]+q[:-1,a.qadr]*model.actuator_biasprm[ids,1]+v[:-1,a.vadr]*model.actuator_biasprm[ids,2]
            tau=np.clip(tau,model.actuator_forcerange[ids,0],model.actuator_forcerange[ids,1]);energy=float(np.maximum(tau*v[1:,a.vadr],0).sum()*dt);error=abs(energy-r['positive_mechanical_work_j'][side]);assert error<1e-6;energy_error.append(error)
        # Re-execute individual physics steps directly from archived integration states/controls.
        replay_error=0.
        for ix in sorted(set([0,10,min(255,len(states)-2),len(states)//2,len(states)-2])):
            actual=mujoco.MjData(model);expected=mujoco.MjData(model);mujoco.mj_setState(model,actual,states[ix],STATE_SPEC);actual.ctrl[:]=controls[ix+1];mujoco.mj_step(model,actual);mujoco.mj_setState(model,expected,states[ix+1],STATE_SPEC)
            error=max(float(np.max(abs(actual.qpos-expected.qpos))),float(np.max(abs(actual.qvel-expected.qvel))));replay_error=max(replay_error,error);assert error<1e-8,(r['match_id'],ix,error)
        # Verify physical result from the complete trace, not from result flags alone.
        cfg=json.loads((folder/'config.json').read_text());timers=np.zeros(2);first=None
        for ix in range(1,len(times)):
            lost=[]
            for side,a in enumerate(actors):
                qq=q[ix,a.free_qadr:a.free_qadr+7];tilt=float(np.degrees(np.arccos(np.clip(-gravity_body(qq[3:7])[2],-1,1))));low=qq[2]<r['actors'][side]['nominal_root_height']*cfg['fall_height_ratio'];timers[side]=timers[side]+dt if low or tilt>cfg['fall_tilt_deg'] else 0.
                lost.append(timers[side]>=cfg['fall_hold'] or max(abs(qq[:2]))>cfg['ring_half_width'])
            if times[ix]>=cfg['settle'] and (any(lost) or times[ix]>=cfg['duration']):
                winner=None if all(lost) or not any(lost) else 1 if lost[0] else 0;reason='double_loss' if all(lost) else 'fall' if any(lost) and any(timers>=cfg['fall_hold']) else 'boundary_exit' if any(lost) else 'time_limit';first=dict(winner=winner,reason=reason,time=float(times[ix]));break
        assert first and first['winner']==r['result']['winner'] and first['reason']==r['result']['reason'] and abs(first['time']-r['result']['time'])<1e-8,(r['match_id'],first,r['result'])
        contact=np.load(folder/'contacts.npz')['samples'];assert len(contact)>0
        norms=np.linalg.norm(contact[:,6:9],axis=1);impulse=float(norms.sum()*dt);assert abs(impulse-r['contact_impulse_ns'])<1e-6
        unique,idx=np.unique(contact[:,0],return_inverse=True);forces=np.bincount(idx,weights=norms);assert abs(float(forces.max())-r['peak_contact_force_n'])<1e-6
        assert unique[0]<=r['result']['time']
        telemetry=json.loads((folder/'telemetry.json').read_text())
        for f in telemetry:
            ix=f['physics_index'];assert abs(times[ix]-f['time'])<1e-8
            for side,a in enumerate(actors):
                assert abs(q[ix,a.free_qadr+2]-f['actors'][side]['height_m'])<1e-8
                tilt=np.degrees(np.arccos(np.clip(-gravity_body(q[ix,a.free_qadr+3:a.free_qadr+7])[2],-1,1)));assert abs(tilt-f['actors'][side]['tilt_deg'])<1e-6
        video_info=None
        if require_videos:
            path=folder/'battle.mp4';assert path.exists(),str(path)
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height,nb_frames,avg_frame_rate','-of','json',str(path)]));stream=next(s for s in probe['streams'] if 'width' in s);duration=float(probe['format']['duration']);assert stream['width']==1280 and stream['height']==720;assert abs(duration-r['duration_s'])<.10;assert int(stream['nb_frames'])>=int(r['duration_s']*23.9);video_info=dict(duration=duration,width=stream['width'],height=stream['height'],frames=int(stream['nb_frames']))
        checks.append(dict(match_id=r['match_id'],physics_frames=len(states),contact_samples=len(contact),max_one_step_replay_error=replay_error,mechanical_work_error_j=max(energy_error),result_recomputed=first,telemetry_frames=len(telemetry),video=video_info));raw.close()
        print('VERIFY',number+1,'/',len(results),r['match_id'],'PASS',flush=True)
    # Run the original downloaded neural weights again on archived observations.
    sample=ROOT/'results/matches/ai-f2-m0';r=json.loads((sample/'result.json').read_text());io=np.load(sample/'policy_io.npz');neural=[]
    for side,a in enumerate(r['actors']):
        path=ROOT/a['policy_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==a['policy_sha256'];opts=ort.SessionOptions();opts.intra_op_num_threads=1;session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider']);name=session.get_inputs()[0].name;obs=io[f'p{side}_observations'];act=io[f'p{side}_actions'];error=0.
        for ix in [0,len(obs)//3,len(obs)//2,len(obs)-1]:error=max(error,float(np.max(abs(session.run(None,{name:obs[ix:ix+1]})[0].reshape(-1)-act[ix]))))
        assert error<1e-6;neural.append(dict(policy=a['policy'],weight_sha256=a['policy_sha256'],samples=4,max_action_error=error))
    # Reaggregate rankings from physical per-bout outcomes.
    regenerated=rank(results,persist=False)
    for track in ['robot','ai']:
        for before,after in zip(package['leaderboards'][track]['rows'],regenerated['leaderboards'][track]['rows']):
            assert before['key']==after['key'] and before['wins']==after['wins'] and before['draws']==after['draws'] and before['losses']==after['losses'] and abs(before['elo']-after['elo'])<1e-10
    report=dict(passed=True,require_videos=require_videos,source_assets_verified=True,preflight=True,no_contact_movement_control=True,matches=checks,neural_weights_reexecuted=neural,rankings_recomputed=True,scope='All published bouts: hashes, full-trace physical result, impulse/work and UI telemetry; five one-step replays per bout. Four neural IO samples per model.',software_fingerprint=SOFTWARE_FINGERPRINT)
    write_json(ROOT/'results/verification.json',report);return report
