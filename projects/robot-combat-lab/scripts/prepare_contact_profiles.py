import gc,json
from combatlab.comparison import Comparison
from combatlab.research_archive import workspace,atomic_json,event,digest
from combatlab.engine import jsonable
BASE={'g1_full':'nvidia_wbc','h1_full':'unitree_full_gym','h1_2_full':'unitree_full_gym','go2':'go2_robot_lab','a1':'a1_vq','go1':'go1_wtw','b2':'b2_robot_lab','go2w':'go2w_robot_lab','b2w':'b2w_robot_lab'}
for robot,key in BASE.items():
 s=Comparison(robot,key,'pd_stand');a=s.actors[0];p=s.policies[0];out=dict(key=robot,base_policy=key,xml=str(a.spec.xml),xml_sha256=digest(a.spec.xml),names=a.names,default=a.default,kp=a.kp,kd=a.kd,limits=a.torque_limits,mass_kg=float(s.m.body_mass[a.body_ids].sum()),native_motor_parameters=None,policy_files=[dict(path=str(f),sha256=digest(f)) for f in getattr(p,'paths',[p.path] if hasattr(p,'path') else [])]);atomic_json(workspace()/'09_models/profiles'/(robot+'.json'),json.loads(json.dumps(out,default=jsonable)));event('contact_body_profile_prepared',robot=robot,base_policy=key,joints=len(a.names),mass_kg=out['mass_kg']);print(robot,len(a.names),out['mass_kg'],flush=True);del s,a,p;gc.collect()
