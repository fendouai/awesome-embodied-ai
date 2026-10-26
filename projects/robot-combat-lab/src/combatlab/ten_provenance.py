"""Archive the extra deployment contracts without mutating controller inputs."""
from pathlib import Path
import json
from .models import ROOT
from .engine import write_json,sha256
P=ROOT/'assets/ten-platform'
def snapshot(folder,keys):
 folder=Path(folder);files=[]
 for key in keys:
  if key in ['b2_robot_lab','go2w_robot_lab','b2w_robot_lab']:
   robot=key.split('_')[0];files += [P/f'rl_sar/policy/{robot}/base.yaml',P/f'rl_sar/policy/{robot}/robot_lab/config.yaml']
  elif key=='go1_wtw':files += [ROOT/'research/ten-platform/wtw-parameters-readable.json']
  elif key=='g1_beyondmimic':files += [P/'rl_sar/policy/g1/base.yaml',P/'rl_sar/policy/g1/whole_body_tracking/dance_102/config.yaml']
 if not files:return
 (folder/'platform10.py').write_bytes((Path(__file__).parent/'platform10.py').read_bytes());copied=[]
 for i,path in enumerate(files):
  dest=folder/f'contract-{i:02d}-{path.name}';dest.write_bytes(path.read_bytes());copied.append(dict(original=str(path.relative_to(ROOT)),archived=dest.name,sha256=sha256(path)))
 write_json(folder/'deployment-contracts.json',dict(inputs=copied,scope='Pinned policy/config transfer; source Isaac/actuator-network dynamics parity and physical hardware performance are not asserted'))
 manifest=json.loads((folder/'manifest.json').read_text())
 for path in [folder/'platform10.py',folder/'deployment-contracts.json']+[folder/x['archived'] for x in copied]:manifest[path.name]=sha256(path)
 write_json(folder/'manifest.json',manifest)
