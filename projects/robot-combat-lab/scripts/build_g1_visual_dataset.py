"""Offline RGB-D labels from recorded physics. Object IDs are annotation-only."""
import json,time
from pathlib import Path
import numpy as np,mujoco,cv2
from PIL import Image
from scipy.spatial.transform import Rotation
from combatlab.contact_arena import ContactArena
from combatlab.g1_camera import original_camera_scene,RGB_SIZE,DEPTH_SIZE,DepthToColor,specification
from combatlab.engine import STATE_SPEC
from combatlab.research_archive import workspace,atomic_json,event,snapshot_sources,digest
root=workspace();out=root/'02_sources/G1-visual-dataset-v1';out.mkdir(exist_ok=False);(out/'samples').mkdir();snapshot_sources(out);atomic_json(out/'camera-specification.json',specification());env=ContactArena('g1_full',scene_transform=original_camera_scene);m=env.m;d=env.d;rgb=mujoco.Renderer(m,RGB_SIZE[1],RGB_SIZE[0]);depth=mujoco.Renderer(m,DEPTH_SIZE[1],DEPTH_SIZE[0]);depth.enable_depth_rendering();opt=mujoco.MjvOption();visual={int(m.geom_bodyid[i]) for i in range(m.ngeom) if m.geom_contype[i]==0 and m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH}
for i in range(m.ngeom):
 if int(m.geom_bodyid[i]) in visual and m.geom_contype[i]:m.geom_group[i]=5
opt.geomgroup[5]=0;alignment=DepthToColor();atomic_json(out/'intrinsics.json',dict(rgb=alignment.kc.tolist(),depth=alignment.kd.tolist()));rng=np.random.default_rng(20261025);rows=[];start=time.time();source_run=root/'03_training/ppo-g1_full-20261025T123750-21ffd7'
def save_sample(side,source_id,split,original_state,augmentation=None):
 global rows
 mujoco.mj_setState(m,d,original_state,STATE_SPEC)
 if augmentation:
  own=env.actors[side];other=env.actors[1-side];q=other.free_qadr;R=Rotation.from_quat(d.qpos[own.free_qadr+3:own.free_qadr+7][[1,2,3,0]]).as_matrix();delta=R@np.array([augmentation['distance_m']*np.cos(augmentation['angle_rad']),augmentation['distance_m']*np.sin(augmentation['angle_rad']),0]);d.qpos[q:q+2]=d.qpos[own.free_qadr:own.free_qadr+2]+delta[:2];rotation=Rotation.from_euler('z',augmentation['opponent_yaw_delta_rad'])*Rotation.from_quat(d.qpos[q+3:q+7][[1,2,3,0]]);d.qpos[q+3:q+7]=rotation.as_quat()[[3,0,1,2]]
 mujoco.mj_forward(m,d);rgb.update_scene(d,camera=f'p{side}/d435i_rgb',scene_option=opt);image=rgb.render().copy();rgb.enable_segmentation_rendering();seg=rgb.render().copy();rgb.disable_segmentation_rendering();labels=np.zeros(RGB_SIZE[::-1],dtype=np.uint8);geom_to_class=np.zeros(m.ngeom,dtype=np.uint8)
 for g in range(m.ngeom):
  name=m.body(int(m.geom_bodyid[g])).name or '';geom_to_class[g]=1 if name.startswith(f'p{side}/') else 2 if name.startswith(f'p{1-side}/') else 0
 valid=(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&(seg[:,:,0]>=0)&(seg[:,:,0]<m.ngeom);labels[valid]=geom_to_class[seg[:,:,0][valid]];depth.update_scene(d,camera=f'p{side}/d435i_depth',scene_option=opt);ideal=depth.render().copy();z16=alignment.sensor_depth(ideal);aligned=alignment.align(z16);small=cv2.resize(image,(320,180),interpolation=cv2.INTER_AREA);small_z=cv2.resize(aligned,(320,180),interpolation=cv2.INTER_NEAREST);small_labels=cv2.resize(labels,(320,180),interpolation=cv2.INTER_NEAREST);x=np.zeros((5,192,320),dtype=np.float32);x[:3,:180]=small.transpose(2,0,1)/255.;x[3,:180]=np.clip(small_z*.001/3,0,1);x[4,:180]=small_z>0;y=np.zeros((192,320),dtype=np.uint8);y[:180]=small_labels;sid=f'{len(rows):06}';folder=out/'samples'/sid;folder.mkdir();Image.fromarray(image).save(folder/'rgb.png');Image.fromarray(labels).save(folder/'annotation-only-segmentation.png');np.savez_compressed(folder/'depth.npz',ideal_depth_m=ideal,sensor_depth_mm=z16,aligned_sensor_depth_mm=aligned);np.savez_compressed(folder/'training-input.npz',input=x,label=y,source_state=original_state,rendered_qpos=d.qpos.copy());row=dict(id=sid,split=split,side=side,source=source_id,kind='recorded-physics-render' if augmentation is None else 'STATIC_RENDER_AUGMENTATION_NOT_PHYSICAL_MATCH',augmentation=augmentation,opponent_visible_pixels=int((labels==2).sum()),rgb_sha256=digest(folder/'rgb.png'),training_npz_sha256=digest(folder/'training-input.npz'));rows.append(row)
 with (out/'samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 if len(rows)%50==0:print('DATASET',len(rows),'elapsed',round(time.time()-start,1),flush=True)
try:
 for seed,split in [(100,'train'),(101,'validation'),(102,'test')]:
  raw=source_run/'evaluation'/f'seed-{seed}'/'raw';states=[]
  for p in sorted(raw.glob('physics-*.npz')):
   with np.load(p) as a:states.append(a['integration_states'])
  states=np.concatenate(states)
  for idx in range(0,len(states),60):
   for side in [0,1]:
    save_sample(side,dict(folder=str(raw),physics_index=idx,chunks_manifest_sha256=digest(raw/'chunks.jsonl')),split,states[idx])
    if idx%180==0:
     aug=dict(distance_m=float(rng.uniform(.7,3.)),angle_rad=float(rng.uniform(-np.pi,np.pi)),opponent_yaw_delta_rad=float(rng.uniform(-np.pi,np.pi)));save_sample(side,dict(folder=str(raw),physics_index=idx,chunks_manifest_sha256=digest(raw/'chunks.jsonl')),split,states[idx],aug)
 atomic_json(out/'manifest.json',dict(state='complete',samples=len(rows),split_counts={s:sum(r['split']==s for r in rows) for s in ['train','validation','test']},annotation_usage='Simulator segmentation IDs are OFFLINE LABELS ONLY; runtime model receives only RGB+depth+validity, never IDs or opponent simulator pose',same_original_robot_materials_both_sides=True,camera_mount_unchanged=True,source_run=str(source_run),sample_index_sha256=digest(out/'samples.jsonl')));event('g1_visual_dataset_built',folder=str(out),samples=len(rows));print(out,flush=True)
finally:rgb.close();depth.close();env.close()
