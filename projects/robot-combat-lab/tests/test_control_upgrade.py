import numpy as np
import mujoco
import pytest
from combatlab.models import scene_xml,make_actors,initialize,configure_servos
from combatlab.control_upgrade import Holosoma,AMPRecovery,NvidiaAdaptive,UnitreeFullBodyGym

@pytest.fixture(scope='module')
def g1():
    model=mujoco.MjModel.from_xml_string(scene_xml(('g1_full','g1_full')))
    return model

def actor(model,robot,cls):
    a=make_actors(model,(robot,robot))[0];p=cls(a);configure_servos(model,[a]);d=mujoco.MjData(model);initialize(model,d,[a],distance=6);return a,p,d

def test_holosoma_sorted_observations_and_standing_phase(g1):
    a,p,d=actor(g1,'g1_full',Holosoma);p.update(g1,d,np.zeros(3),0)
    obs=p.last_observation
    assert len(obs)==100
    assert np.allclose(obs[:29],0)  # alphabetically first term = previous actions
    assert np.allclose(obs[35:37],[-1,-1])  # two standing cos phases
    assert np.allclose(obs[95:98],[0,0,-1])
    assert np.max(abs(obs[98:]))<1e-6
    assert np.allclose(a.kp[p.mp],__import__('json').loads(p.net.get_modelmeta().custom_metadata_map['kp']))
    assert not d.xfrc_applied.any()

def test_nvidia_native_switch_uses_walk_above_source_threshold(g1):
    a,p,d=actor(g1,'g1_full',NvidiaAdaptive)
    p.update(g1,d,np.zeros(3),0);assert p.path==p.balance_path
    p.update(g1,d,np.array([.08,0,0]),.02);assert p.path==p.walk_path
    assert len(p.last_observation)==516
    assert not d.xfrc_applied.any()

@pytest.mark.parametrize('robot',['h1_full','h1_2_full'])
def test_full_body_transfer_maps_policy_joint_names(robot):
    model=mujoco.MjModel.from_xml_string(scene_xml((robot,robot)))
    a,p,d=actor(model,robot,UnitreeFullBodyGym)
    assert len(p.mp)==p.cfg['num_actions']
    assert all('shoulder' not in a.names[i] and 'elbow' not in a.names[i] for i in p.mp)
    p.update(model,d,np.zeros(3),0)
    assert np.isfinite(p.target).all()
    upper=np.setdiff1d(np.arange(len(a.names)),p.mp)
    assert np.array_equal(p.target[upper],a.default[upper])
