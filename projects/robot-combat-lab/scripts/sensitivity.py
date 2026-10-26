"""Paired half-timestep checks; never silently mix these results into rated bouts."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from combatlab.engine import *
rows=[]
for mid in ['robot-03-f2-m0','ai-f4-m0']:
 folder=ROOT/'results/matches'/mid;base=json.loads((folder/'result.json').read_text());cfg=json.loads((folder/'config.json').read_text());cfg['match_id']='halfdt-'+mid;cfg['timestep']=.001;cfg['rated']=False;cfg['note']='Validation only: same parameters, halve physics dt; keep 50Hz control.'
 sim=Simulation(MatchConfig(**cfg)).run();path,result=sim.save(ROOT/'results/validation')
 row=dict(reference=mid,validation=str(path.relative_to(ROOT)),reference_timestep=.002,validation_timestep=.001,reference_result=base['result'],validation_result=result['result'],winner_agrees=base['result']['winner']==result['result']['winner'],reason_agrees=base['result']['reason']==result['result']['reason'],reference_impulse_ns=base['contact_impulse_ns'],validation_impulse_ns=result['contact_impulse_ns'],relative_impulse_change=abs(result['contact_impulse_ns']-base['contact_impulse_ns'])/max(base['contact_impulse_ns'],1e-6),reference_peak_n=base['peak_contact_force_n'],validation_peak_n=result['peak_contact_force_n'])
 rows.append(row);print(mid,row['winner_agrees'],row['relative_impulse_change'],flush=True)
write_json(ROOT/'results/sensitivity.json',dict(checks=rows,scope='Two representative bouts only; not a full convergence proof. Large differences require cautious interpretation.'))
