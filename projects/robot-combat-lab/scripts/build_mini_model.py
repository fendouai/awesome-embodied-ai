"""Rejected candidate diagnostic only. Never changes inertia or registers a robot."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np,json
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'assets/ten-platform/rapid/resources/robots/mini_cheetah/urdf/mini_cheetah.urdf';bad=[]
for link in ET.parse(p).getroot().findall('link'):
 e=link.find('inertial/inertia')
 if e is None:continue
 q={k:float(v) for k,v in e.attrib.items()};I=np.array([[q['ixx'],q['ixy'],q['ixz']],[q['ixy'],q['iyy'],q['iyz']],[q['ixz'],q['iyz'],q['izz']]]);v=np.linalg.eigvalsh(I)
 if v[0]+v[1]<v[2]-1e-9:bad.append(dict(link=link.get('name'),inertia=q,eigenvalues=v.tolist()))
report=dict(candidate='Rapid-Locomotion Mini Cheetah',status='excluded_not_registered',reason='Published URDF violates the physical principal-inertia triangle; no fabricated inertia used to meet the robot count',invalid_links=bad)
(ROOT/'research/ten-platform/excluded-candidates.json').write_text(json.dumps(report,indent=2));print('EXCLUDED',len(bad),'invalid inertia tensors; active Go1 uses separate matching WTW weights')
