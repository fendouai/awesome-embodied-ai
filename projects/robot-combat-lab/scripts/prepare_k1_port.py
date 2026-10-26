import csv,copy,json,xml.etree.ElementTree as ET
import numpy as np,mujoco
from combatlab.research_archive import workspace,event,atomic_json,digest
root=workspace();folder=root/'09_models/k1';source=ET.parse(folder/'native-scene.xml').getroot();body=copy.deepcopy(source.find('worldbody/body'));params={r['joint']:r for r in csv.DictReader((folder/'native-joints.tsv').open(),delimiter='\t')};mapping={j.get('name'):params[j.get('name')]['semantic'] for j in body.findall('.//joint') if j.get('type')!='free'}
body.set('name','pelvis');body.set('pos','0 0 .6');body.set('quat','1 0 0 0')
for j in body.findall('.//joint'):
 if j.get('type')=='free':j.set('name','root');continue
 old=j.get('name');j.set('name',mapping[old]);j.set('armature',params[old]['armature']);j.set('damping','0');j.set('frictionloss','0')
for i,b in enumerate(body.iter('body')):
 if i:b.set('name',f'link_{i}')
r=ET.Element('mujoco',model='Booster K1 native SimRobot collision/inertial export');ET.SubElement(r,'compiler',angle='radian',autolimits='true');ET.SubElement(r,'asset');ET.SubElement(r,'worldbody').append(body);act=ET.SubElement(r,'actuator')
for a in source.find('actuator'):
 old=a.get('joint')
 if old in mapping:
  a=copy.deepcopy(a);a.set('joint',mapping[old]);a.set('name','motor_'+mapping[old]);act.append(a)
path=folder/'k1-native.xml';ET.ElementTree(r).write(path,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(path));names=[m.joint(i).name for i in range(m.njnt) if m.jnt_type[i]!=mujoco.mjtJoint.mjJNT_FREE];by_semantic={v['semantic']:v for k,v in params.items() if k in mapping};assert len(names)==22
out=dict(key='k1',base_policy='bhuman_k1_walk',xml=str(path),names=names,default=[-.2 if 'HipPitch' in n else .4 if 'KneePitch' in n else -.25 if 'AnklePitch' in n else 0 for n in names],kp=[float(by_semantic[n]['kp']) for n in names],kd=[float(by_semantic[n]['kd']) for n in names],limits=[float(by_semantic[n]['max_torque']) for n in names],native_motor_parameters=[by_semantic[n] for n in names],source_native_mujoco_version='3.3.5',target_mujoco_version=mujoco.__version__,mass_kg=float(m.body_mass.sum()),scope='Original native collision and inertial model; graphics-only appearance not yet ported; native motor PD, armatures and velocity filters exported; motion policy port requires gait validation')
profiles=root/'09_models/profiles';profiles.mkdir(exist_ok=True);atomic_json(profiles/'k1.json',out);atomic_json(folder/'port-provenance.json',dict(source_xml_sha256=digest(folder/'native-scene.xml'),source_motor_sha256=digest(folder/'native-joints.tsv'),extracted_xml_sha256=digest(path),changed=['Selected one original robot; excluded field/ball/other robots','Renamed joints with native exported semantic mapping','Reset world placement/yaw for standalone body','Applied armatures set by native ServoMotor::act on first tick'],geometry_and_inertial_values_preserved=True,mass_kg=out['mass_kg'],joint_count=len(names),validated_gait=False));event('k1_collision_model_port_prepared',path=str(path),mass_kg=out['mass_kg'],joints=len(names));print(path,'mass',out['mass_kg'])
