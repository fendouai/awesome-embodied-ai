"""Check real checkpoint inference and archive physics, not cosmetic motion."""
import ast,copy,json
import numpy as np
import torch,mujoco
from torch import nn
from combatlab.models import ROOT
from combatlab.engine import STATE_SPEC
from combatlab.storage import load_model
from combatlab.robostriker_cpu import Stage3CPU


def test_robostriker_matches_published_exporter_on_recorded_inputs():
    base=ROOT/'research/adversarial-options/robostriker-source/source/whole_body_tracking/whole_body_tracking'
    tree=ast.parse((base/'utils/exporter.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OnnxStage3PolicyExporter')
    namespace=dict(torch=torch,nn=nn,copy=copy)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-source-exporter>','exec'),namespace)
    norm_ns={};exec(compile((base/'rsl_rl/rsl_rl/modules/normalizer.py').read_text(),'<pinned-source-normalizer>','exec'),norm_ns)
    net=Stage3CPU();latest=json.loads((ROOT/'results/robostriker-cpu/latest.json').read_text());raw=np.load(ROOT/latest['folder']/'policy_io.npz')
    for side in (0,1):
        norms=[]
        for key,width in [('taskpolicy_obs_normalizer_ego' if side==0 else 'taskpolicy_obs_normalizer_opponent',192),('prior_obs_normalizer',900),('decoder_obs_normalizer',90)]:
            norm=norm_ns['EmpiricalNormalization'](width);norm.load_state_dict(net.raw[key+'_state_dict']);norm.eval();norms.append(norm)
        exporter=namespace['OnnxStage3PolicyExporter']((net.ego if side==0 else net.opponent).layers,net.prior.layers,net.decoder.layers,32,True,*norms)
        obs=raw[f'p{side}_observations'];actions=raw[f'p{side}_actions']
        for idx in (0,len(obs)//2,len(obs)-1):
            task,hist,decoder=np.split(obs[idx],[192,1092]);out,z=net.infer(side,task,hist,decoder)
            with torch.inference_mode():expected=exporter(*[torch.as_tensor(x,dtype=torch.float32)[None] for x in (task,hist,decoder)]).numpy()[0]
            np.testing.assert_allclose(out,expected,atol=1e-6,rtol=1e-6)
            # Archive action is the decoded policy vector, not a fabricated strike.
            np.testing.assert_allclose(out,actions[idx],atol=1e-6,rtol=1e-6)


def test_ncp_saved_codes_and_noise_reproduce_checkpoint_actions():
    from combatlab.ncp_cpu import NCPPolicy
    latest=json.loads((ROOT/'results/ncp-cpu/latest.json').read_text());io=np.load(ROOT/latest['folder']/'policy_io.npz')
    for side,checkpoint in enumerate(('boxing_106.pth','boxing_100.pth')):
        p=NCPPolicy(side,checkpoint);obs=io[f'p{side}_observations'];codes=io[f'p{side}_codes'];noise=io[f'p{side}_noise']
        for i in (0,len(obs)//2,len(obs)-1):
            prop=p.norm(obs[i,:323],'prop');embedding=p.embedding[[int(codes[i])]]
            with torch.inference_mode():action=p.low(torch.cat([prop,embedding],-1)).numpy()[0]+noise[i]
            np.testing.assert_allclose(action,io[f'p{side}_actions'][i],rtol=1e-6,atol=1e-6)


def test_ports_archive_complete_physics_without_external_wrenches():
    for backend in ('robostriker-cpu','ncp-cpu'):
        latest=json.loads((ROOT/f'results/{backend}/latest.json').read_text());folder=ROOT/latest['folder'];m=load_model(folder);d=mujoco.MjData(m);raw=np.load(folder/'trajectory.npz');states=raw['integration_states'];controls=raw['controls'];result=json.loads((folder/'result.json').read_text())
        assert len(states)>1000 and result['numerical_warnings']==0
        for idx in (20,len(states)//3,len(states)-2):
            mujoco.mj_setState(m,d,states[idx],STATE_SPEC)
            assert not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
            # Controls are stored after the step which used them.
            d.ctrl[:]=controls[idx+1];mujoco.mj_step(m,d)
            expected=mujoco.MjData(m);mujoco.mj_setState(m,expected,states[idx+1],STATE_SPEC)
            np.testing.assert_allclose(d.qpos,expected.qpos,atol=2e-8,rtol=1e-7)
            np.testing.assert_allclose(d.qvel,expected.qvel,atol=2e-7,rtol=1e-6)
        assert result['result']['winner'] is None and not result['rated']


def test_robostriker_checkpoint_continues_same_policy_history(monkeypatch):
    # This is an in-memory test with no writes; do not trip the disk-write guard.
    import combatlab.endurance as endurance
    from types import SimpleNamespace
    monkeypatch.setattr(endurance.shutil,"disk_usage",lambda _:SimpleNamespace(free=2*1024**3))
    from combatlab.robostriker_cpu import RoboStrikerSimulation
    from combatlab.endurance import EnduranceConfig
    latest=json.loads((ROOT/'results/robostriker-cpu/latest.json').read_text());folder=ROOT/latest['folder']
    resumed=RoboStrikerSimulation.restore(folder);resumed.save_on_chunks=False
    config=EnduranceConfig(**json.loads((folder/'continuous-config.json').read_text()));direct=RoboStrikerSimulation(config);direct.save_on_chunks=False
    while direct.d.time<19.999:
        if not direct.advance(.1):break
    np.testing.assert_allclose(resumed.d.qpos,direct.d.qpos,atol=1e-10,rtol=0)
    resumed.resume(.2);direct.resume(.2)
    for _ in range(5):
        resumed.advance(.02);direct.advance(.02)
        np.testing.assert_allclose(resumed.d.qpos,direct.d.qpos,atol=1e-8,rtol=0)
        np.testing.assert_allclose(resumed.policies[0].last_action,direct.policies[0].last_action,atol=1e-6,rtol=0)
