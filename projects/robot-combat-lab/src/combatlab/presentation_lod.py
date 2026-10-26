"""Display-only mesh LOD; archived dynamics and replay verification use the exact model."""
from pathlib import Path
import hashlib,xml.etree.ElementTree as ET
import numpy as np,mujoco
from .models import ROOT
from .storage import load_model
from .engine import STATE_SPEC,write_json

def load_display_model(folder):
    folder=Path(folder);exact=load_model(folder)
    if not (folder/'scene.xml').exists():return exact
    import trimesh
    root=ET.fromstring((folder/'scene.xml').read_text());cache=ROOT/'results/render-lod-cache';cache.mkdir(parents=True,exist_ok=True);changes=[]
    for element in root.findall('.//asset/mesh'):
        filename=element.get('file')
        if not filename or not Path(filename).is_file():continue
        source=Path(filename);key=hashlib.sha256(source.read_bytes()).hexdigest();target=cache/(key+'-3000.stl')
        if not target.exists():
            mesh=trimesh.load(source,force='mesh',process=False)
            if len(mesh.faces)<=6000:continue
            simplified=mesh.simplify_quadric_decimation(face_count=3000);simplified.export(target)
        element.set('file',str(target));changes.append(dict(source=source.name,display_mesh=target.name,target_faces=3000))
    if not changes:return exact
    view=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    for attr in ['nq','nv','nu','na','nmocap','nuserdata','nbody','njnt']:
        if getattr(view,attr)!=getattr(exact,attr):raise ValueError('Display model state contract mismatch: '+attr)
    raw=np.load(folder/'trajectory.npz');states=raw['integration_states'];ed=mujoco.MjData(exact);vd=mujoco.MjData(view);error=0.
    for index in [0,len(states)//2,len(states)-1]:
        mujoco.mj_setState(exact,ed,states[index],STATE_SPEC);mujoco.mj_forward(exact,ed);mujoco.mj_setState(view,vd,states[index],STATE_SPEC);mujoco.mj_forward(view,vd)
        error=max(error,float(np.max(np.abs(ed.xpos-vd.xpos))),float(np.max(np.abs(ed.xmat-vd.xmat))))
    if error>1e-8:raise ValueError(f'Display body-pose mismatch: {error}')
    write_json(folder/'display-lod.json',dict(display_only=True,source_physics_model_modified=False,archived_states_modified=False,body_pose_max_error=error,meshes=changes))
    return view
