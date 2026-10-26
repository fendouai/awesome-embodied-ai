import json,tempfile
import numpy as np
import mujoco
import pytest
from combatlab.models import ROOT
from combatlab.endurance import ContinuousSimulation,EnduranceConfig
from combatlab.hardware import HardwareMonitor,ThermalParameters


def test_fall_does_not_end_continuous_run():
    s=ContinuousSimulation(EnduranceConfig('test-fall',policies=('amp_recovery','amp_recovery'),max_sim_time=10))
    s.save_on_chunks=False
    for a in s.actors:
        s.d.qpos[a.free_qadr+2]=.25;s.d.qpos[a.free_qadr+3:a.free_qadr+7]=[np.sqrt(.5),0,np.sqrt(.5),0]
    mujoco.mj_forward(s.m,s.d)
    assert s.advance(1.)
    assert s.d.time>=.99 and s.terminal is None
    assert any(x['state'] in ['UNSTABLE','FALLEN'] for x in s.frames[-1]['actors'])
    assert all(x['mode']=='not_calibrated' and x['max_temperature_c'] is None for x in s.frames[-1]['hardware'])
    assert not s.hardware.loss_of_all_actuation()[0]


def test_time_budget_is_pause_without_winner():
    s=ContinuousSimulation(EnduranceConfig('test-budget',policies=('amp_recovery','amp_recovery'),max_sim_time=.2))
    assert not s.advance(.3)
    assert s.terminal['reason']=='budget_pause' and s.terminal['winner'] is None
    s.resume(.2)
    assert s.advance(.1) and s.terminal is None

@pytest.mark.parametrize('robot,policy',[('g1_full','nvidia_wbc'),('h1_full','unitree_full_gym')])
def test_checkpoint_restores_physics_and_policy_memory(robot,policy):
    s=ContinuousSimulation(EnduranceConfig('test-restore',robots=(robot,robot),policies=(policy,policy),max_sim_time=10))
    s.save_on_chunks=False;s.advance(2.32)
    with tempfile.TemporaryDirectory(prefix='continuous-test-',dir=ROOT/'results') as tmp:
        folder,_=s.save(tmp);r=ContinuousSimulation.restore(folder);r.save_on_chunks=False
        np.testing.assert_allclose(s.d.qpos,r.d.qpos,rtol=0,atol=1e-12)
        for _ in range(15):
            s.advance(.02);r.advance(.02)
            np.testing.assert_allclose(s.d.qpos,r.d.qpos,rtol=0,atol=1e-8)
            np.testing.assert_allclose(s.d.qvel,r.d.qvel,rtol=0,atol=1e-8)
            np.testing.assert_allclose(s.policies[0].last_action,r.policies[0].last_action,rtol=0,atol=1e-6)


def test_synthetic_heat_model_distinguishes_protection_from_damage():
    # Synthetic scalar fixture only, explicitly not a Unitree parameter set.
    from types import SimpleNamespace
    class Model:
        nu=1
        actuator_forcerange=np.array([[-1.,1.]])
        def actuator(self,i):return SimpleNamespace(name='toy')
    p=ThermalParameters(loss_w_per_nm2=1.,capacity_j_per_k=1.,conductance_w_per_k=0.,ambient_c=20.,initial_c=20.,trip_c=21.,reset_c=20.5,irreversible_c=30.,irreversible_hold_s=.1,provenance='synthetic unit-test fixture, NOT robot data',evidence_status='hypothetical')
    h=HardwareMonitor(Model(),[SimpleNamespace(actuator_ids=np.array([0]))],{'toy':p});h.step(np.array([2.]),.5,.5)
    assert h.temperature[0]==22. and h.tripped[0] and not h.burned[0]
    assert not h.loss_of_all_actuation()[0]
    assert np.all(h.model.actuator_forcerange==0)
    h.restore(h.state());assert h.base_limits[0,1]==1.
