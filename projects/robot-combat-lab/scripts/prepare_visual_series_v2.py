"""Prepare a fresh native visual batch without replacing completed match records."""
import argparse,json,time
from pathlib import Path
from combatlab.models import ROOT
from combatlab.engine import write_json
p=argparse.ArgumentParser();p.add_argument('--round-id',default='round-04');p.add_argument('--start-match',type=int,default=1);p.add_argument('--match-count',type=int,default=4);p.add_argument('--period-seconds',type=int,default=180);a=p.parse_args()
if not 1<=a.start_match<=a.match_count or not 1<=a.period_seconds<=3600:raise SystemExit('Invalid match count, start or period')
base=ROOT/'results/tournaments'/a.round_id;capture=base/'vision-captures';start=capture/f'vision-{a.start_match:02d}'
if start.exists() and any(start.iterdir()):raise SystemExit(f'Existing capture preserved: {start}; select a new round or unused start number')
capture.mkdir(parents=True,exist_ok=True);config=dict(run_id=a.round_id+'-'+str(time.time_ns()),batch_enabled=True,batch_root=str(capture),start_match=a.start_match,period_seconds=a.period_seconds,warmup_seconds=20,match_count=a.match_count,capture_enabled=False,directory=str(capture/'pending'),start_ms=int(time.time()*1000),duration_ms=3600000)
request=ROOT/'integrations/bhuman/capture-request.json';write_json(request,config);write_json(base/'capture-request-initial.json',config)
protocol=json.loads((ROOT/'configs/competition/protocol-v2.json').read_text());protocol['local']['visual_soccer']['period_seconds']=a.period_seconds;protocol['local']['visual_soccer']['match_count']=a.match_count;write_json(base/'protocol.json',protocol)
print(json.dumps(dict(request=str(request),root=str(capture),environment_variable='BOTSTEST_CAPTURE_REQUEST',note='Open Competition3v3M with the native patched simulator. Each match is one local period, even when period_seconds=600; this alone does not constitute a full official two-half game.'),indent=2))
