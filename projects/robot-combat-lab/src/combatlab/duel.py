"""Rule-based physical contest adjudication. Never interprets contacts as hardware damage."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json,hashlib
import numpy as np,mujoco
from .robostriker_cpu import RoboStrikerSimulation
from .engine import write_json,sha256
from .models import ROOT

@dataclass(frozen=True)
class DuelRules:
    limit_s:float=180.
    down_confirm_s:float=.25
    recovery_deadline_s:float=10.
    recovery_hold_s:float=1.
    down_tilt_deg:float=65.
    down_height_ratio:float=.55
    upright_tilt_deg:float=25.
    upright_height_ratio:float=.75
    contact_gap_s:float=.10
    impact_window_s:float=.08
    minimum_approach_m_s:float=.25
    minimum_normal_impulse_ns:float=.10
    decision_impulse_margin_ns:float=.5

class Referee:
    def __init__(self,rules=None,initial_heights=(.76,.76)):
        self.rules=rules or DuelRules();self.heights=np.array(initial_heights);self.last_t=0.;self.bad_since=[None,None];self.down_since=[None,None];self.good_since=[None,None];self.knockdowns=[0,0];self.failed=set();self.scores=[0,0];self.scoring_impulse=[0.,0.];self.upright_time=[0.,0.];self.episodes=[None,None];self.events=[];self.outcome=None;self.scoring_closed=False
    def event(self,event_type,t,**fields):self.events.append(dict(type=event_type,time=float(t),**fields))
    def finish_episode(self,side,t):
        episode=self.episodes[side]
        if episode is None:return
        valid=episode['approach_m_s']>=self.rules.minimum_approach_m_s and episode['impulse_ns']>=self.rules.minimum_normal_impulse_ns
        self.event('scoring_contact' if valid else 'non_scoring_upper_contact',t,side=side,**episode)
        if valid:self.scores[side]+=1;self.scoring_impulse[side]+=episode['impulse_ns']
        self.episodes[side]=None
    def contacts(self,t,dt,contacts):
        # Each row: attacker hand versus opponent upper-body composite, Fn and
        # pre-step approach speed. Forearm/hand guards are deliberately excluded.
        for side in [0,1]:
            hits=[c for c in contacts if c['side']==side and c['normal_force_n']>0]
            e=self.episodes[side]
            if e is not None and t-e['last_time']>self.rules.contact_gap_s:self.finish_episode(side,t);e=None
            if not hits or any(d is not None for d in self.down_since):continue
            if e is None:
                e=dict(start_time=float(t),last_time=float(t),approach_m_s=max(c['approach_m_s'] for c in hits),impulse_ns=0.,peak_normal_force_n=0.,target='opponent_upper_body_composite');self.episodes[side]=e
            e['last_time']=float(t);force=sum(c['normal_force_n'] for c in hits);e['peak_normal_force_n']=max(e['peak_normal_force_n'],force)
            if t-e['start_time']<=self.rules.impact_window_s+1e-9:e['impulse_ns']+=force*dt
    def observe(self,t,poses,contacts=()):
        if self.outcome:return self.outcome
        t=float(t);dt=max(0.,t-self.last_t);self.last_t=t;good=[]
        for side,p in enumerate(poses):
            bad=p['tilt_deg']>self.rules.down_tilt_deg or p['height_m']<self.rules.down_height_ratio*self.heights[side]
            up=p['tilt_deg']<self.rules.upright_tilt_deg and p['height_m']>self.rules.upright_height_ratio*self.heights[side];good.append(up)
            if up:
                self.upright_time[side]+=dt
                if self.good_since[side] is None:self.good_since[side]=t
            else:self.good_since[side]=None
            if bad:
                if self.bad_since[side] is None:self.bad_since[side]=t
                if self.down_since[side] is None and t-self.bad_since[side]>=self.rules.down_confirm_s-1e-9:
                    self.down_since[side]=self.bad_since[side];self.knockdowns[side]+=1;self.event('down_confirmed',t,side=side,down_start=self.down_since[side])
            else:self.bad_since[side]=None
            down=self.down_since[side]
            if down is not None:
                # Starting a genuine upright recovery within the ten seconds
                # pauses the count for its1s stability confirmation.
                if self.good_since[side] is not None and self.good_since[side]-down<=self.rules.recovery_deadline_s+1e-9:
                    if t-self.good_since[side]>=self.rules.recovery_hold_s-1e-9:
                        self.event('recovered',t,side=side,down_duration=t-down);self.down_since[side]=None;self.bad_since[side]=None
                elif t-down>=self.rules.recovery_deadline_s-1e-9 and side not in self.failed:
                    self.failed.add(side);self.event('recovery_timeout',t,side=side,down_duration=t-down)
        if t<=self.rules.limit_s+1e-9:self.contacts(t,dt,contacts)
        if t>=self.rules.limit_s-1e-9 and not self.scoring_closed:
            for side in [0,1]:self.finish_episode(side,t)
            self.scoring_closed=True
        if len(self.failed)==2:self.conclude(t,None,'double_recovery_timeout','technical_draw')
        elif self.failed:
            loser=next(iter(self.failed));winner=1-loser
            if self.down_since[winner] is None and good[winner]:self.conclude(t,winner,'failed_to_restore_upright_before_recovery_deadline','technical_stoppage')
        if self.outcome is None and t>=self.rules.limit_s-1e-9:
            if not any(x is not None for x in self.down_since):self.decision(t)
            elif t>=self.rules.limit_s+self.rules.recovery_deadline_s+self.rules.recovery_hold_s-1e-9:self.conclude(t,None,'unresolved_recovery_at_extension_limit','technical_draw')
        return self.outcome
    def conclude(self,t,winner,reason,kind):
        for side in [0,1]:self.finish_episode(side,t)
        self.outcome=dict(winner=winner,reason=reason,kind=kind,time=float(t),hardware_damage=False,competition_rule_result=True,clean_upper_contacts=list(self.scores),short_window_normal_impulse_ns=list(self.scoring_impulse),knockdowns=list(self.knockdowns),upright_time_s=list(self.upright_time));self.event('match_ended',t,**{k:v for k,v in self.outcome.items() if k!='time'})
    def decision(self,t):
        for side in [0,1]:self.finish_episode(side,t)
        if sum(self.scores)==0:
            if self.knockdowns[0]!=self.knockdowns[1]:self.conclude(t,int(self.knockdowns[0]>self.knockdowns[1]),'fewer_down_episodes','stability_decision')
            else:self.conclude(t,None,'no_valid_scoring_difference','draw')
        elif self.scores[0]!=self.scores[1]:self.conclude(t,int(self.scores[1]>self.scores[0]),'more_valid_upper_body_contacts','contact_decision')
        elif abs(self.scoring_impulse[0]-self.scoring_impulse[1])>=self.rules.decision_impulse_margin_ns:self.conclude(t,int(self.scoring_impulse[1]>self.scoring_impulse[0]),'greater_short_window_normal_impulse','impulse_decision')
        else:self.conclude(t,None,'tied_under_published_rules','draw')
    def status(self,t):
        return dict(score=list(self.scores),impulse_ns=list(self.scoring_impulse),knockdowns=list(self.knockdowns),recovery_count_s=[None if x is None else max(0.,float(t)-x) for x in self.down_since],failed=sorted(self.failed),outcome=self.outcome)

class DuelSimulation(RoboStrikerSimulation):
    def __init__(self,cfg,rules=None):
        self.referee=None;super().__init__(cfg);self.referee=Referee(rules,[self.d.qpos[a.free_qadr+2] for a in self.actors]);self.save_on_chunks=False;self.record_frame()
    def record_frame(self):
        super().record_frame()
        if getattr(self,'referee',None) is not None:self.frames[-1]['duel']=self.referee.status(self.d.time)
        if hasattr(self,'policies'):self.frames[-1]['policy_role']=[p.key for p in self.policies]
    def scoring_contacts(self,pre_velocity):
        rows=[]
        for k,c in enumerate(self.d.contact[:self.d.ncon]):
            b1=int(self.m.geom_bodyid[c.geom1]);b2=int(self.m.geom_bodyid[c.geom2]);n1=self.m.body(b1).name or '';n2=self.m.body(b2).name or ''
            side=None
            for attacker in [0,1]:
                hands={f'p{attacker}/'+n for n in ['left_wrist_yaw_link','right_wrist_yaw_link','left_hand','right_hand']};targets={f'p{1-attacker}/'+n for n in ['torso_link','torso','head']}
                if (n1 in hands and n2 in targets) or (n2 in hands and n1 in targets):side=attacker;break
            if side is None:continue
            jac1=np.zeros((3,self.m.nv));jac2=np.zeros((3,self.m.nv));rot=np.zeros_like(jac1);mujoco.mj_jac(self.m,self.d,jac1,rot,c.pos,b1);mujoco.mj_jac(self.m,self.d,jac2,rot,c.pos,b2);approach=float(np.dot((jac1-jac2)@pre_velocity,c.frame[:3]));f=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,k,f);rows.append(dict(side=side,normal_force_n=max(0.,float(f[0])),approach_m_s=max(0.,approach)))
        return rows
    def advance(self,seconds,manual=None):
        if self.terminal is not None:return False
        for _ in range(round(seconds/self.cfg.timestep)):
            previous=self.d.qvel.copy();alive=super().advance(self.cfg.timestep,manual);poses=[]
            for a in self.actors:
                q=self.d.qpos[a.free_qadr:a.free_qadr+7];tilt=np.degrees(np.arccos(np.clip(2*(q[3]**2+q[6]**2)-1,-1,1)));poses.append(dict(tilt_deg=float(tilt),height_m=float(q[2])))
            if self.terminal and self.terminal['reason'] not in ['budget_pause']:
                self.referee.event('simulation_interruption',self.d.time,reason=self.terminal['reason']);return False
            outcome=self.referee.observe(self.d.time,poses,self.scoring_contacts(previous))
            if self.frames and abs(self.frames[-1]['time']-self.d.time)<1e-7:self.frames[-1]['duel']=self.referee.status(self.d.time)
            if outcome:
                self.terminal=dict(outcome);self.record_frame();return False
            if not alive:
                if self.terminal and self.terminal['reason']=='budget_pause' and any(x is not None for x in self.referee.down_since):
                    end=self.referee.rules.limit_s+self.referee.rules.recovery_deadline_s+self.referee.rules.recovery_hold_s
                    if self.d.time<end-1e-9:
                        self.resume(end-float(self.d.time));self.frames[-1]['terminal']=None;self.referee.event('recovery_count_extension',self.d.time,until_s=end);continue
                return False
        return True
    def save(self,base=None):
        folder,result=super().save(base or ROOT/'results/duels');rule=asdict(self.referee.rules);write_json(folder/'duel-rules.json',dict(**rule,version='physical-contest-v1',scoring='Terminal wrist versus opponent torso/head composite, approach gate, first80ms normal impulse; guards and down-state contacts do not score',interpretation='Competition judgement, not hardware damage or source-paper benchmark'))
        write_json(folder/'referee-events.json',self.referee.events);write_json(folder/'duel-result.json',dict(outcome=self.referee.outcome,status=self.referee.status(self.d.time),rules=rule,complete=self.referee.outcome is not None,simulation_interruption=None if self.referee.outcome else self.terminal))
        (folder/'duel.py').write_bytes(DUEL_SOURCE);result['duel']=dict(outcome=self.referee.outcome,rules=rule,complete=self.referee.outcome is not None);result['rated']=False;write_json(folder/'result.json',result);write_json(folder/'manifest.json',{p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'});return folder,result
DUEL_SOURCE=Path(__file__).read_bytes();DUEL_FINGERPRINT=hashlib.sha256(DUEL_SOURCE).hexdigest()
