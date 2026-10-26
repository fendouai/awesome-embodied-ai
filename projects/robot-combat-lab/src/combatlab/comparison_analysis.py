"""Derived readiness annotations; original result and physical trajectory stay intact."""
import json
from pathlib import Path
from .engine import write_json,sha256

def analyze(folder):
 folder=Path(folder);frames=json.loads((folder/'telemetry.json').read_text());result=json.loads((folder/'result.json').read_text());actors=[]
 for side in [0,1]:
  initial=frames[0]['actors'][side]['height_m'];pre=[f['actors'][side] for f in frames if 2<=f['time']<3];ready=bool(pre) and all(a['tilt_deg']<20 and a['height_m']>.7*initial for a in pre);timer=0.;fell_before=False;last=0.
  for f in frames:
   if f['time']>=3:break
   a=f['actors'][side];bad=a['tilt_deg']>65 or a['height_m']<.55*initial;timer=timer+max(0.,f['time']-last) if bad else 0.;fell_before|=timer>=.3;last=f['time']
  actors.append(dict(pre_push_ready=ready if result['task']=='push' else None,fell_before_push=fell_before if result['task']=='push' else None))
 data=dict(actors=actors,source_telemetry_sha256=sha256(folder/'telemetry.json'),criterion='Readiness during[2,3)s: tilt<20deg and height>70%initial. A pre-push fall cannot be attributed to disturbance.');write_json(folder/'analysis.json',data);m=folder/'manifest.json';manifest=json.loads(m.read_text());manifest['analysis.json']=sha256(folder/'analysis.json');write_json(m,manifest);return data
