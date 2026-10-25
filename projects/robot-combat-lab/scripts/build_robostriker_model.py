"""Convert pinned RoboStriker URDF to MJCF using MuJoCo's own URDF importer."""
from pathlib import Path
import xml.etree.ElementTree as ET,json
import mujoco
from combatlab.models import ROOT,GYM
from combatlab.engine import write_json,sha256
P=ROOT/'assets/adversarial-candidates/RoboStriker-Preview/description'
if __name__=='__main__':
 tree=ET.parse(P/'main.urdf');root=tree.getroot();mj=ET.SubElement(root,'mujoco');ET.SubElement(mj,'compiler',discardvisual='false',fusestatic='true');missing=[]
 for link in root.findall('link'):
  for visual in list(link.findall('visual')):
   mesh=visual.find('geometry/mesh')
   if mesh is not None:
    f=GYM/'resources/robots/g1_description/meshes'/Path(mesh.get('filename')).name
    if f.exists():mesh.set('filename',str(f.resolve()))
    else:missing.append(mesh.get('filename'));link.remove(visual)
 # Add a genuine floating root. No mocap/pose support.
 ET.SubElement(root,'link',name='world');j=ET.SubElement(root,'joint',name='floating_base',type='floating');ET.SubElement(j,'parent',link='world');ET.SubElement(j,'child',link='pelvis')
 temporary=P/'native-import.urdf';tree.write(temporary,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(temporary));mujoco.mj_saveLastXML(str(P/'native-g1.xml'),m)
 xml=ET.parse(P/'native-g1.xml');r=xml.getroot();r.set('model','RoboStriker source-URDF G1, MuJoCo CPU port');compiler=r.find('compiler');compiler.set('angle','radian');compiler.set('meshdir',str((GYM/'resources/robots/g1_description/meshes').resolve()))
 # Match the source URDF importer's replace_cylinders_with_capsules setting.
 count=0
 for g in r.findall('.//worldbody//geom'):
  if g.get('type')=='cylinder' and g.get('contype','1')!='0':g.set('type','capsule');count+=1
 # Canonical XML has explicit imported inertias; adding motors does not create inertia.
 acts=r.find('actuator')
 if acts is None:acts=ET.SubElement(r,'actuator')
 urdf_joints={j.get('name'):j for j in root.findall('joint')}
 for j in r.findall('.//worldbody//joint'):
  if j.get('type')=='free':continue
  name=j.get('name');limit=urdf_joints[name].find('limit');effort=float(limit.get('effort'));ET.SubElement(acts,'motor',name=name,joint=name,forcerange=f'{-effort} {effort}',forcelimited='true')
 xml.write(P/'native-g1.xml',encoding='unicode');native=mujoco.MjModel.from_xml_path(str(P/'native-g1.xml'))
 write_json(P/'native-model-audit.json',dict(urdf_sha256=sha256(P/'main.urdf'),mjcf_sha256=sha256(P/'native-g1.xml'),nq=native.nq,nv=native.nv,actuators=native.nu,bodies=native.nbody-1,mass_kg=float(native.body_mass.sum()),missing_visuals=missing,collision_cylinders_converted_to_capsules=count,notes='Source URDF physical description, MuJoCo import/fixed-body merge; visuals reuse local Unitree meshes; motor limits/gains/armatures are applied by the source-profile adapter. PhysX and MuJoCo contact/solver are not claimed identical.'))
 print('NATIVE G1 MODEL',native.nq,native.nu,native.nbody-1,float(native.body_mass.sum()),'missing visual',missing)
