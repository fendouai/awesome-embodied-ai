"""Competition rules are unit-tested separately from learned physics/controller quality."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np,mujoco
from combatlab.duel import Referee,DuelRules,DuelSimulation
UP=dict(tilt_deg=0.,height_m=.76)
DOWN=dict(tilt_deg=90.,height_m=.15)

def test_no_forced_winner_when_no_valid_difference():
 r=Referee(DuelRules(limit_s=1));r.observe(.1,[UP,UP]);o=r.observe(1.,[UP,UP]);assert o['winner'] is None and o['kind']=='draw' and not o['hardware_damage'];assert r.events[-1]['type']=='match_ended'

def test_clean_contacts_decide_and_slow_pressure_does_not_score():
 for speed,expected in [(1.,0),(.01,None)]:
  r=Referee(DuelRules(limit_s=1));r.observe(.01,[UP,UP],[dict(side=0,normal_force_n=100.,approach_m_s=speed)]);r.observe(.2,[UP,UP]);out=r.observe(1.,[UP,UP]);assert out['winner']==expected

def test_sustained_contact_is_one_episode_and_only_first80ms_impulse():
 r=Referee(DuelRules(limit_s=2))
 for t in np.arange(.01,1.01,.01):r.observe(t,[UP,UP],[dict(side=0,normal_force_n=100.,approach_m_s=1.)])
 o=r.observe(2.,[UP,UP]);assert o['clean_upper_contacts']==[1,0];assert o['short_window_normal_impulse_ns'][0]<10

def test_recovery_timeout_has_real_technical_stop_not_damage():
 rules=DuelRules(limit_s=3,down_confirm_s=.05,recovery_deadline_s=.5,recovery_hold_s=.1);r=Referee(rules)
 for t in np.arange(.01,.7,.01):
  if r.observe(t,[DOWN,UP]):break
 assert r.outcome['winner']==1 and r.outcome['kind']=='technical_stoppage' and not r.outcome['hardware_damage'];assert r.knockdowns==[1,0]

def test_genuine_recovery_pauses_count_until_confirmation():
 r=Referee(DuelRules(limit_s=3,down_confirm_s=.05,recovery_deadline_s=.5,recovery_hold_s=.2))
 for t in np.arange(.01,.8,.01):r.observe(t,[DOWN if t<.4 else UP,UP])
 assert not r.failed and r.down_since[0] is None and r.outcome is None;assert any(e['type']=='recovered' for e in r.events)

def test_two_unrecovered_fighters_are_not_assigned_a_fake_winner():
 r=Referee(DuelRules(limit_s=3,down_confirm_s=.05,recovery_deadline_s=.5,recovery_hold_s=.1))
 for t in np.arange(.01,.7,.01):
  if r.observe(t,[DOWN,DOWN]):break
 assert r.outcome['winner'] is None and r.outcome['reason']=='double_recovery_timeout'

def test_actual_contact_normal_direction_and_guard_classifier():
 for hand,expected in [('left_wrist_yaw_link',1),('left_elbow_link',0)]:
  xml=f'<mujoco><option gravity="0 0 0"/><worldbody><body name="p0/{hand}"><freejoint/><geom type="sphere" size=".1" mass="1"/></body><body name="p1/torso_link" pos=".19 0 0"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>';m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d);pre=np.zeros(m.nv);pre[0]=1.;rows=DuelSimulation.scoring_contacts(SimpleNamespace(m=m,d=d),pre);assert len(rows)==expected
  if expected:assert rows[0]['approach_m_s']>.99 and rows[0]['side']==0


def test_count_started_before_time_limit_is_allowed_to_finish():
 r=Referee(DuelRules(limit_s=1.,down_confirm_s=.05,recovery_deadline_s=.5,recovery_hold_s=.1))
 for t in np.arange(.01,1.5,.01):
  outcome=r.observe(t,[DOWN if t>=.9 else UP,UP])
  if outcome:break
 assert outcome['winner']==1 and outcome['kind']=='technical_stoppage' and outcome['time']>1.


def test_late_recovery_does_not_allow_extra_scoring_after_buzzer():
 r=Referee(DuelRules(limit_s=1.,down_confirm_s=.05,recovery_deadline_s=.5,recovery_hold_s=.1))
 for t in np.arange(.01,1.4,.01):
  poses=[DOWN if .9<=t<1.1 else UP,UP];hits=[dict(side=0,normal_force_n=100.,approach_m_s=1.)] if t>1 else []
  if r.observe(t,poses,hits):break
 assert r.outcome['time']>1 and r.scores==[0,0] and not r.failed
