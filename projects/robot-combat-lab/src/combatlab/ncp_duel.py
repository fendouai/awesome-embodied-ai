"""Same contest rules applied to NCP physical characters; not hardware robots."""
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .ncp_cpu import NCPEncounter
from .duel import Referee,DuelRules,DuelSimulation,DUEL_SOURCE
from .engine import write_json,sha256
from .models import ROOT

class NCPDuel(NCPEncounter):
    scoring_contacts=DuelSimulation.scoring_contacts
    def __init__(self,seed=0,rules=None,swap=False,checkpoints=None,policy_seeds=None):
        checkpoints=checkpoints or (("boxing_100.pth","boxing_106.pth") if swap else ("boxing_106.pth","boxing_100.pth"))
        policy_seeds=policy_seeds or ((seed+1,seed) if swap else (seed,seed+1))
        self.referee=None;super().__init__(seed=seed,checkpoints=checkpoints,policy_seeds=policy_seeds);self.referee=Referee(rules,[self.d.qpos[a['free_qadr']+2] for a in self.actors]);self.record_frame()
    def record_frame(self):
        super().record_frame()
        if self.referee is not None:self.frames[-1]['duel']=self.referee.status(self.d.time)
    def advance(self,seconds):
        if self.terminal:return False
        for _ in range(round(seconds/self.dt)):
            old=self.d.qvel.copy();alive=super().advance(self.dt)
            if not alive:return False
            poses=[]
            for a in self.actors:
                q=self.d.qpos[a['free_qadr']:a['free_qadr']+7];poses.append(dict(tilt_deg=float(np.degrees(np.arccos(np.clip(2*(q[3]**2+q[6]**2)-1,-1,1)))),height_m=float(q[2])))
            out=self.referee.observe(self.d.time,poses,self.scoring_contacts(old))
            if self.frames and abs(self.frames[-1]['time']-self.d.time)<1e-7:self.frames[-1]['duel']=self.referee.status(self.d.time)
            if out:self.terminal=out;self.record_frame();return False
        return True
    def save(self,base=None):
        folder,result=super().save(base or ROOT/'results/ncp-duels');rule=asdict(self.referee.rules);write_json(folder/'duel-rules.json',dict(**rule,version='physical-contest-v1',body_scope='NCP simulated character, not an identified robot',scoring='Hands versus opponent torso/head only; guards and down-state contacts excluded',interpretation='Competition judgement only, not tissue injury/hardware damage'))
        write_json(folder/'referee-events.json',self.referee.events);write_json(folder/'duel-result.json',dict(outcome=self.referee.outcome,status=self.referee.status(self.d.time),rules=rule,complete=self.referee.outcome is not None,simulation_interruption=None if self.referee.outcome else self.terminal));(folder/'duel.py').write_bytes(DUEL_SOURCE);(folder/'ncp_duel.py').write_bytes(NCP_DUEL_SOURCE);result['duel']=dict(outcome=self.referee.outcome,rules=rule,complete=self.referee.outcome is not None);write_json(folder/'result.json',result);write_json(folder/'manifest.json',{p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'});return folder,result
NCP_DUEL_SOURCE=Path(__file__).read_bytes()
