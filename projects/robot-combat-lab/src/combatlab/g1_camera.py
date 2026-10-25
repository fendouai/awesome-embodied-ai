"""Original G1 Rev1.0 D435 mount, nominal D435i optics; no tracking/gimbal camera."""
import json,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .research_archive import workspace,atomic_json,digest

RGB_SIZE=(640,360);DEPTH_SIZE=(848,480)
def specification():
 root=workspace();folder=root/'02_sources/G1-native-camera';urdf=folder/'g1_29dof_rev_1_0.urdf';joint=ET.parse(urdf).getroot().find("joint[@name='d435_joint']");xyz=np.fromstring(joint.find('origin').get('xyz'),sep=' ');rpy=np.fromstring(joint.find('origin').get('rpy'),sep=' ')
 return dict(robot='g1_full',hardware='RealSense D435i',parent_link=joint.find('parent').get('link'),mount_xyz_m=xyz.tolist(),mount_rpy_rad=rpy.tolist(),mount_source='unitreerobotics/unitree_ros@15ebe53f4cd77e7d745ae83b9d7ff90e685e8998/robots/g1_description/g1_29dof_rev_1_0.urdf',mount_source_sha256=digest(urdf),nominal_color_offset_from_depth_link_m=[0,.015,0],color_offset_source='realsenseai/realsense-ros@7a87097c499e3b0b69518d54a8e7a9ac1924e87c/realsense2_description/urdf/_d435.urdf.xacro; use_nominal_extrinsics',rgb_resolution=list(RGB_SIZE),rgb_fovy_deg=42.5,depth_resolution=list(DEPTH_SIZE),depth_fovy_deg=58.,depth_min_m=.28,depth_max_m=10.,depth_quantization_m=.001,nominal_intrinsics_not_per_device_calibration=True,optical_convention='RealSense optical x right,y down,z forward; MuJoCo x right,y up,-z forward',no_mass_or_inertia_added=True,original_pelvis_and_torso_imu_preserved=True,limitations=['Nominal pinhole intrinsics, no per-serial calibration/distortion','Ideal rendered depth then range-gated/mm-quantized; IR stereo failure, glass and material-specific holes not fully calibrated','RGB rolling shutter and motion blur not calibrated in this first simulation prototype','LiDAR not used as hidden opponent-state fallback'])

def add_cameras(xml,keys):
 if list(keys)!=['g1_full','g1_full']:raise ValueError('Original camera profile currently audited for G1vsG1 only')
 s=specification();r=ET.fromstring(xml);R=Rotation.from_euler('xyz',s['mount_rpy_rad']).as_matrix();camx=-R[:,1];camy=R[:,2];axes=' '.join(map(str,np.r_[camx,camy]));pos=np.array(s['mount_xyz_m'])
 for side in range(2):
  body=r.find(f".//body[@name='p{side}/torso_link']");assert body is not None
  for kind,size,fov,offset in [('rgb',RGB_SIZE,s['rgb_fovy_deg'],[0,.015,0]),('depth',DEPTH_SIZE,s['depth_fovy_deg'],[0,0,0])]:
   p=pos+R@np.array(offset);ET.SubElement(body,'camera',name=f'p{side}/d435i_{kind}',pos=' '.join(map(str,p)),xyaxes=axes,resolution=f'{size[0]} {size[1]}',fovy=str(fov))
 return ET.tostring(r,encoding='unicode')

def restore_vendor_colors(xml):
 """Camera images use original identical hardware colors, not role-color shortcuts."""
 import mujoco
 from .models import ROBOTS
 source=mujoco.MjModel.from_xml_path(str(ROBOTS['g1_full'].xml));colors={}
 for g in range(source.ngeom):
  body=source.body(int(source.geom_bodyid[g])).name
  if source.geom_bodyid[g]:colors.setdefault(body,[]).append(source.geom_rgba[g].copy())
 r=ET.fromstring(xml)
 for b in r.findall('.//worldbody//body'):
  name=b.get('name','');base=name.split('/',1)[-1]
  for i,g in enumerate(b.findall('geom')):
   if i<len(colors.get(base,[])):g.set('rgba',' '.join(map(str,colors[base][i])))
 return ET.tostring(r,encoding='unicode')

def original_camera_scene(xml,keys):return add_cameras(restore_vendor_colors(xml),keys)

def pinhole(size,fovy):
 w,h=size;f=h/(2*np.tan(np.radians(fovy)/2));return np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1]],dtype=float)

class DepthToColor:
 """Nominal RealSense calibration; no object IDs, poses or simulator state inputs."""
 def __init__(self):
  s=specification();self.kc=pinhole(RGB_SIZE,s['rgb_fovy_deg']);self.kd=pinhole(DEPTH_SIZE,s['depth_fovy_deg']);u,v=np.meshgrid(np.arange(DEPTH_SIZE[0]),np.arange(DEPTH_SIZE[1]));self.rx=(u-self.kd[0,2])/self.kd[0,0];self.ry=(v-self.kd[1,2])/self.kd[1,1]
 def sensor_depth(self,ideal):
  s=specification();valid=np.isfinite(ideal)&(ideal>=s['depth_min_m'])&(ideal<=s['depth_max_m']);return np.where(valid,np.clip(np.round(ideal*1000),0,65535),0).astype(np.uint16)
 def align(self,z16):
  z=z16.astype(np.float32)*.001;valid=z>0;u=np.rint(self.kc[0,0]*(self.rx+.015/np.maximum(z,.0001))+self.kc[0,2]).astype(int);v=np.rint(self.kc[1,1]*self.ry+self.kc[1,2]).astype(int);valid&=(u>=0)&(u<RGB_SIZE[0])&(v>=0)&(v<RGB_SIZE[1]);out=np.full(RGB_SIZE[0]*RGB_SIZE[1],65535,dtype=np.uint16);np.minimum.at(out,v[valid]*RGB_SIZE[0]+u[valid],z16[valid]);out[out==65535]=0;return out.reshape(RGB_SIZE[1],RGB_SIZE[0])
