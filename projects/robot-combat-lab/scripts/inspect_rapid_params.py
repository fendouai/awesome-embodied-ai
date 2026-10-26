"""Data-only parameter inspection. Tensor storage loaders are replaced by inert stubs."""
import pickle,io,json,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
class Safe(pickle.Unpickler):
 def find_class(self,m,n):
  if (m,n) in [('torch._utils','_rebuild_tensor_v2'),('torch.storage','_load_from_bytes')]:return lambda *args:None
  allowed={('collections','OrderedDict'):collections.OrderedDict,('numpy.core.multiarray','scalar'):np._core.multiarray.scalar,('numpy.core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,('numpy','dtype'):np.dtype,('numpy','ndarray'):np.ndarray}
  if (m,n) not in allowed:raise RuntimeError((m,n))
  return allowed[m,n]
p=ROOT/'assets/ten-platform/rapid/runs/rapid-locomotion/example/train/201852.132488/parameters.pkl';o=Safe(io.BytesIO(p.read_bytes())).load()
def data(x):return x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(type(x))
s=json.dumps(o,default=data,indent=2);(ROOT/'research/ten-platform/rapid-parameters-readable.json').write_text(s)
print(s[:500]);print('\n'.join(line for line in s.splitlines() if any(k in line for k in ['asset','file','num_obs','observe_','default_joint','action_scale','hip_scale','stiffness','damping','decimation','robot_name'])))
