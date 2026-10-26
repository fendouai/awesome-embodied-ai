from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import copy, xml.etree.ElementTree as ET
import numpy as np
import mujoco, yaml

ROOT=Path(__file__).resolve().parents[2]
GYM=ROOT/'assets/vendor/gym'
LAB=ROOT/'assets/vendor/lab'
@dataclass
class RobotSpec:
    key:str
    label:str
    xml:Path
    config:Path
    scope:str

ROBOTS={
 'g1':RobotSpec('g1','Unitree G1',GYM/'resources/robots/g1_description/g1_12dof.xml',GYM/'deploy/deploy_mujoco/configs/g1.yaml','Official 12-DoF locomotion model; arms rigidly aggregated'),
 'h1':RobotSpec('h1','Unitree H1',GYM/'resources/robots/h1/h1.xml',GYM/'deploy/deploy_mujoco/configs/h1.yaml','Official 10-DoF locomotion model; arms rigidly aggregated'),
 'h1_2':RobotSpec('h1_2','Unitree H1-2',GYM/'resources/robots/h1_2/h1_2_12dof.xml',GYM/'deploy/deploy_mujoco/configs/h1_2.yaml','Official 12-DoF locomotion model; arms rigidly aggregated'),
 'g1_full':RobotSpec('g1_full','Unitree G1 / 29 DoF',GYM/'resources/robots/g1_description/g1_29dof_rev_1_0.xml',GYM/'deploy/deploy_mujoco/configs/g1.yaml','Official G1 rev1.0 full 29-DoF articulated model'),
}

ROBOTS['h1_full']=RobotSpec('h1_full','Unitree H1 / full body',ROOT/'assets/vendor/mujoco/unitree_robots/h1/h1.xml',GYM/'deploy/deploy_mujoco/configs/h1.yaml','Official articulated H1 model; public rigid-body/actuator parameters')
ROBOTS['h1_2_full']=RobotSpec('h1_2_full','Unitree H1-2 / full body',ROOT/'assets/vendor/mujoco/unitree_robots/h1_2/h1_2_handless.xml',GYM/'deploy/deploy_mujoco/configs/h1_2.yaml','Official articulated handless H1-2 model; no dexterous finger dynamics')

@dataclass
class Actor:
    prefix:str
    spec:RobotSpec
    root_id:int
    free_jid:int
    free_qadr:int
    free_vadr:int
    body_ids:np.ndarray
    joint_ids:np.ndarray
    qadr:np.ndarray
    vadr:np.ndarray
    actuator_ids:np.ndarray
    torque_limits:np.ndarray
    default:np.ndarray
    kp:np.ndarray
    kd:np.ndarray
    names:list[str]
    nominal_z:float=0.


def _namespace(el:ET.Element,prefix:str):
    for e in el.iter():
        for key in ['name','joint','joint1','joint2','body','body1','body2','site','site1','site2','mesh','material','texture','class','childclass','tendon','tendon1','tendon2','objname']:
            if e.get(key):e.set(key,prefix+e.get(key))


def scene_xml(robot_keys:tuple[str,str], timestep=.002, friction=.8)->str:
    r=ET.Element('mujoco',model='BotsTest contact arena')
    ET.SubElement(r,'compiler',angle='radian',autolimits='true')
    ET.SubElement(r,'option',timestep=str(timestep),gravity='0 0 -9.81',integrator='implicitfast',solver='Newton',iterations='80',tolerance='1e-9',cone='elliptic')
    v=ET.SubElement(r,'visual');ET.SubElement(v,'global',offwidth='1280',offheight='720');ET.SubElement(v,'quality',shadowsize='2048');ET.SubElement(v,'headlight',ambient='.4 .4 .4',diffuse='.6 .6 .6',specular='.1 .1 .1');ET.SubElement(v,'rgba',haze='.02 .03 .05 1')
    defaults=ET.SubElement(r,'default')
    asset=ET.SubElement(r,'asset')
    ET.SubElement(asset,'texture',name='floor_tex',type='2d',builtin='checker',rgb1='.09 .12 .16',rgb2='.13 .17 .22',width='512',height='512')
    ET.SubElement(asset,'material',name='floor_mat',texture='floor_tex',texrepeat='8 8',reflectance='.1')
    world=ET.SubElement(r,'worldbody')
    ET.SubElement(world,'light',name='key',pos='0 -3 6',dir='0 .4 -1',diffuse='.85 .85 .85',castshadow='true')
    ET.SubElement(world,'light',name='fill',pos='-3 3 4',dir='.5 -.5 -1',diffuse='.6 .7 .8',castshadow='false')
    ET.SubElement(world,'geom',name='ground',type='plane',size='8 8 .1',material='floor_mat',friction=f'{friction} .005 .0001',condim='3',contype='1',conaffinity='31')
    # Non-colliding markings have no mechanical effect. Boundary is a measured rule, not a hidden wall.
    for x in [-2.5,2.5]:ET.SubElement(world,'geom',name=f'line_x_{x}',type='box',pos=f'{x} 0 .001',size='.018 2.5 .001',rgba='.25 .65 .8 1',contype='0',conaffinity='0')
    for y in [-2.5,2.5]:ET.SubElement(world,'geom',name=f'line_y_{y}',type='box',pos=f'0 {y} .001',size='2.5 .018 .001',rgba='.25 .65 .8 1',contype='0',conaffinity='0')
    for j in range(-2,3):ET.SubElement(world,'geom',name=f'grid_{j}',type='box',pos=f'{j} 0 .001',size='.003 2.5 .0005',rgba='.2 .27 .34 .6',contype='0',conaffinity='0')
    actuator=ET.SubElement(r,'actuator');sensors=ET.SubElement(r,'sensor')
    for side,key in enumerate(robot_keys):
        spec=ROBOTS[key];src=ET.parse(spec.xml).getroot();prefix=f'p{side}/'
        # The upstream H1 includes a nonphysical 'not_use_joint' placeholder. Weld it at zero and omit its unused motor.
        for parent in src.iter():
            for child in list(parent):
                if child.get('name')=='not_use_joint' or child.get('joint')=='not_use_joint':parent.remove(child)
        meshdir=src.find('compiler').get('meshdir','')
        orig_asset=copy.deepcopy(src.find('asset'))
        for e in orig_asset:
            if e.tag=='mesh' and not e.get('name'):e.set('name',Path(e.get('file')).stem)
            if e.get('file'):
                candidates=[spec.xml.parent/meshdir/e.get('file'),spec.xml.parent/e.get('file')]
                resolved=next((p.resolve() for p in candidates if p.is_file()),None)
                if resolved is None:raise FileNotFoundError(f'Model asset not found: {e.get("file")} from {spec.xml}')
                e.set('file',str(resolved))
        _namespace(orig_asset,prefix)
        for e in orig_asset:asset.append(e)
        default=src.find('default')
        wrapper=ET.SubElement(defaults,'default',{'class':prefix+'defaults'})
        if default is not None:
            default=copy.deepcopy(default);_namespace(default,prefix)
            for e in default:wrapper.append(e)
        for e in src.find('worldbody'):
            e=copy.deepcopy(e);_namespace(e,prefix)
            if not e.get('childclass'):e.set('childclass',prefix+'defaults')
            for idx,g in enumerate(e.iter('geom')):
                if not g.get('name'):g.set('name',prefix+f'geom_{idx}')
                if g.get('contype','1')!='0':g.set('friction',f'{friction} .005 .0001');g.set('condim','3')
                col=np.fromstring(g.get('rgba','.6 .6 .6 1'),sep=' ')
                # Appearance tint only; geometry/inertia unchanged.
                if len(col)==4 and max(col[:3])>.3:
                    tint=np.array([.55,.85,1.]) if side==0 else np.array([1.,.65,.36]);col[:3]=.4*col[:3]+.6*tint;g.set('rgba',' '.join(map(str,col)))
            # Source locomotion assets sometimes omit rubber-hand collisions.
            # Add massless convex-mesh proxies only for cross-robot/ground contact;
            # keep the source self-collision behavior and all explicit inertias.
            body_bit=2 if side==0 else 4
            other_hand_bit=16 if side==0 else 8
            for g in e.iter('geom'):
                visual=g.get('contype')=='0' or g.get('class','').split('/')[-1]=='visual'
                if not visual:
                    g.set('contype',str(body_bit));g.set('conaffinity',str(1|2|4|other_hand_bit))
            for parent in e.iter('body'):
                for g in list(parent):
                    mesh=g.get('mesh','')
                    if g.tag!='geom' or 'rubber_hand' not in mesh:continue
                    visual=g.get('contype')=='0' or g.get('class','').split('/')[-1]=='visual'
                    if not visual:continue
                    if any(h.tag=='geom' and h.get('mesh')==mesh and h.get('contype')!='0' and h.get('class','').split('/')[-1]!='visual' for h in parent):continue
                    proxy=copy.deepcopy(g);proxy.attrib.pop('class',None);proxy.attrib.pop('material',None);proxy.set('type','mesh');proxy.set('name',g.get('name')+'_contact_proxy');proxy.set('density','0');proxy.set('mass','0');proxy.set('rgba','0 0 0 0');proxy.set('group','3');proxy.set('contype','8' if side==0 else '16');proxy.set('conaffinity','21' if side==0 else '11');proxy.set('friction',f'{friction} .005 .0001');parent.append(proxy)
            world.append(e)
        for tag,dest in [('actuator',actuator),('sensor',sensors)]:
            src_el=src.find(tag)
            if src_el is None:continue
            for e in src_el:
                e=copy.deepcopy(e);_namespace(e,prefix);dest.append(e)
        for tag in ['contact','equality']:
            el=src.find(tag)
            if el is not None:
                el=copy.deepcopy(el);_namespace(el,prefix)
                found=r.find(tag)
                if found is None:r.append(el)
                else:
                    for e in el:found.append(e)
    return ET.tostring(r,encoding='unicode')


def make_actors(m:mujoco.MjModel,keys:tuple[str,str])->list[Actor]:
    out=[]
    for side,key in enumerate(keys):
        pre=f'p{side}/';spec=ROBOTS[key]
        bodies=np.array([i for i in range(1,m.nbody) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) or '').startswith(pre)],dtype=int)
        jids=np.array([i for i in range(m.njnt) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) or '').startswith(pre) and m.jnt_type[i]!=mujoco.mjtJoint.mjJNT_FREE],dtype=int)
        aids=np.array([i for i in range(m.nu) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) or '').startswith(pre)],dtype=int)
        free=next(i for i in range(m.njnt) if m.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[i] in bodies)
        aids=np.array([next(u for u in aids if m.actuator_trnid[u,0]==j) for j in jids],dtype=int)
        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i)[len(pre):] for i in jids]
        cfg=yaml.safe_load(spec.config.read_text());n=len(jids);default=np.zeros(n);kp=np.full(n,40.);kd=np.full(n,5.)
        nd=len(cfg['default_angles'])
        legacy='h1_2' if key.startswith('h1_2') else 'h1' if key.startswith('h1') else 'g1'
        legacy_path=ROBOTS[legacy].xml
        legacy_names=[j.get('name') for j in ET.parse(legacy_path).getroot().findall('.//worldbody//joint') if j.get('type')!='free'][:nd]
        for z,name in enumerate(legacy_names):
            j=names.index(name);default[j]=cfg['default_angles'][z];kp[j]=cfg['kps'][z];kd[j]=cfg['kds'][z]
        for j,name in enumerate(names):
            if 'shoulder_pitch' in name:default[j]=.15
            if 'shoulder_roll' in name:default[j]=.18 if 'left' in name else -.18
            if 'elbow' in name:
                lo,hi=m.jnt_range[jids[j]];default[j]=-.5 if hi<=.05 else .5
        if n==29 and key.startswith('g1'):
            lab=yaml.safe_load((LAB/'deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml').read_text());mp=np.array(lab['joint_ids_map']);default[mp]=lab['default_joint_pos'];kp[:]=lab['stiffness'];kd[:]=lab['damping']
        limits=[]
        for jid,u in zip(jids,aids):
            if m.jnt_actfrclimited[jid] and max(abs(m.jnt_actfrcrange[jid]))>0:limit=max(abs(m.jnt_actfrcrange[jid]))
            elif m.actuator_forcelimited[u] and max(abs(m.actuator_forcerange[u]))>0:limit=max(abs(m.actuator_forcerange[u]))
            elif m.actuator_ctrllimited[u] and max(abs(m.actuator_ctrlrange[u]))>0:limit=max(abs(m.actuator_ctrlrange[u]))
            else:raise ValueError(f'No declared actuator torque limit for {names[len(limits)]}')
            limits.append(limit)
        out.append(Actor(pre,spec,int(bodies[0]),free,int(m.jnt_qposadr[free]),int(m.jnt_dofadr[free]),bodies,jids,m.jnt_qposadr[jids].copy(),m.jnt_dofadr[jids].copy(),aids,np.array(limits),default,kp,kd,names))
    return out


def initialize(m,d,actors,distance=2.,seed=0,noise=.0):
    rng=np.random.default_rng(seed);mujoco.mj_resetData(m,d)
    for i,a in enumerate(actors):
        q=a.free_qadr;d.qpos[q:q+3]=[(-.5 if i==0 else .5)*distance,0,1.3]
        d.qpos[q+3:q+7]=[1,0,0,0] if i==0 else [0,0,0,1]
        d.qpos[a.qadr]=a.default+rng.normal(0,noise,len(a.default))
    mujoco.mj_forward(m,d)
    # Place lowest collision geometry above ground, using actual compiled mesh bounds.
    for a in actors:
        lowest=100.
        for g in range(m.ngeom):
            if m.geom_bodyid[g] not in a.body_ids or not m.geom_contype[g]:continue
            if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                mid=m.geom_dataid[g];start=m.mesh_vertadr[mid];count=m.mesh_vertnum[mid];verts=m.mesh_vert[start:start+count]
                z=(verts@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()
            else:z=d.geom_xpos[g,2]-m.geom_rbound[g]
            lowest=min(lowest,float(z))
        d.qpos[a.free_qadr+2]-=lowest-.002
        a.nominal_z=float(d.qpos[a.free_qadr+2])
    mujoco.mj_forward(m,d)

def configure_servos(m,actors):
    """PD law tau=kp*(target-q)-kd*qdot, integrated implicitly by MuJoCo.

    Using affine actuator bias instead of an explicit outer-loop damping force is
    essential for small wrist inertias at a 2ms physics step. The PD gains and
    joint torque limits are unchanged; no free-root forces are introduced.
    """
    for a in actors:
        for i,u in enumerate(a.actuator_ids):
            m.actuator_gaintype[u]=mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_biastype[u]=mujoco.mjtBias.mjBIAS_AFFINE
            m.actuator_gainprm[u,:]=0;m.actuator_gainprm[u,0]=a.kp[i]
            m.actuator_biasprm[u,:]=0;m.actuator_biasprm[u,1]=-a.kp[i];m.actuator_biasprm[u,2]=-a.kd[i]
            m.actuator_forcelimited[u]=True;m.actuator_forcerange[u]=[-a.torque_limits[i],a.torque_limits[i]]
            m.actuator_ctrllimited[u]=False

def calibrate_stance(m,d,actors):
    """Common geometric calibration: align neutral COM above the ankle line.

    Adjust bilateral hip pitch and compensate ankle pitch to keep foot pitch.
    This changes controller reference angles only, never masses/geometry or root forces.
    """
    for a in actors:
        original=a.default.copy();R=d.xmat[a.root_id].reshape(3,3).copy()
        feet=[i for i in a.body_ids if 'ankle' in m.body(i).name]
        def evaluate(delta):
            q=original.copy()
            for j,name in enumerate(a.names):
                if 'hip_pitch' in name:q[j]+=delta
                if 'ankle_pitch' in name or name in ['left_ankle_joint','right_ankle_joint']:q[j]-=delta
            d.qpos[a.qadr]=q;mujoco.mj_forward(m,d)
            error=(R.T@(d.subtree_com[a.root_id]-np.mean(d.xpos[feet],axis=0)))[0]
            return error,q
        lo,hi=-.15,.15
        for _ in range(20):
            mid=(lo+hi)/2;err,q=evaluate(mid)
            if err>0:hi=mid
            else:lo=mid
        err,q=evaluate((lo+hi)/2);a.default=q
