import argparse,json,time
from combatlab.models import ROOT
from combatlab.endurance import ContinuousSimulation,EnduranceConfig
from combatlab.engine import write_json
from combatlab.hardware import ThermalParameters

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=60);parser.add_argument('--resume');parser.add_argument('--hardware-parameters');args=parser.parse_args()
 if args.resume:s=ContinuousSimulation.restore(args.resume);s.resume(args.seconds)
 else:
  params={k:ThermalParameters(**v) for k,v in json.load(open(args.hardware_parameters)).items()} if args.hardware_parameters else None
  s=ContinuousSimulation(EnduranceConfig('continuous-g1-'+str(time.time_ns()),policies=('nvidia_wbc','amp_recovery'),duration=args.seconds,max_sim_time=args.seconds,chunk_seconds=30),params)
 while s.advance(.2):
  if abs(s.d.time%10)<.01:print('CONTINUOUS',round(s.d.time,2),'s',len(s.parts),'saved parts','damage=UNKNOWN' if not s.hardware.parameters else 'parameterized model',flush=True)
 folder,result=s.save(ROOT/'results/continuous');write_json(ROOT/'results/continuous/summary.json',dict(status='paused',run_id=s.run_id,recorded_until_s=float(s.d.time),parts=s.parts,terminal=s.terminal,damage_prediction='not_calibrated' if not s.hardware.parameters else 'parameterized_unvalidated',hardware=[s.hardware.summary(i) for i in [0,1]],resumable=True));print('PAUSED WITHOUT WINNER',folder,s.terminal,flush=True)
