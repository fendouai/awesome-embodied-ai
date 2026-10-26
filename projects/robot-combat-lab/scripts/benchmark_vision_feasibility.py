"""Local feasibility timing only. No claim of trained10-robot visual recognition accuracy."""
import os,json,time,platform,resource,subprocess,io
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np,mujoco,onnxruntime as ort
from PIL import Image
from combatlab.contact_arena import ContactArena
from combatlab.models import ROOT
from combatlab.research_archive import workspace,atomic_json,event,digest,snapshot_sources
root=workspace();out=root/'06_reports/vision-feasibility-20261025';out.mkdir(exist_ok=True);snapshot_sources(out)
hw=json.loads(subprocess.check_output(['system_profiler','SPHardwareDataType','-json']))['SPHardwareDataType'][0];hardware={k:hw[k] for k in ['machine_name','machine_model','chip_type','physical_memory','number_processors'] if k in hw};hardware.update(memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'])),logical_cpus=int(subprocess.check_output(['sysctl','-n','hw.logicalcpu'])),python=platform.python_version(),mujoco=mujoco.__version__,onnxruntime=ort.__version__,providers=ort.get_available_providers())
env=ContactArena('g1_full');xml=ET.fromstring(env.xml)
for side in range(2):
 body=next(b for b in xml.findall('worldbody/body') if b.get('name','').startswith(f'p{side}/'));ET.SubElement(body,'camera',name='vision'+str(side),pos='.15 0 .55',xyaxes='0 -1 0 0 0 1',fovy='75')
model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));data=mujoco.MjData(model);data.qpos[:]=env.d.qpos;data.qvel[:]=env.d.qvel;mujoco.mj_forward(model,data);(out/'camera-scene.xml').write_text(ET.tostring(xml,encoding='unicode'));mujoco.mj_saveModel(model,str(out/'camera-scene.mjb'),None);render_rows=[];rgb_sample=None
for width,height in [(320,240),(640,480),(960,540)]:
 renderer=mujoco.Renderer(model,height,width)
 try:
  times=[]
  for i in range(37):
   start=time.perf_counter()
   for side in range(2):
    renderer.update_scene(data,camera='vision'+str(side));rgb=renderer.render().copy();renderer.enable_depth_rendering();depth=renderer.render().copy();renderer.disable_depth_rendering()
    if width==640 and i==5:
     Image.fromarray(rgb).save(out/f'camera-{side}-rgb.png');np.savez_compressed(out/f'camera-{side}-depth.npz',depth_m=depth);rgb_sample=rgb.copy();b=io.BytesIO();Image.fromarray(rgb).save(b,format='PNG');z=io.BytesIO();np.savez_compressed(z,depth_m=depth);atomic_json(out/f'camera-{side}-storage-sample.json',dict(width=width,height=height,rgb_raw_bytes=rgb.nbytes,depth_raw_bytes=depth.nbytes,rgb_lossless_png_bytes=len(b.getvalue()),depth_lossless_npz_bytes=len(z.getvalue()),depth_min_m=float(depth.min()),depth_max_m=float(depth.max())))
   ms=(time.perf_counter()-start)*1000
   if i>=5:times.append(ms)
  row=dict(width=width,height=height,agents=2,cameras_per_agent=1,modes=['RGB','float32_depth_m'],samples=len(times),pair_rgbd_p50_ms=float(np.median(times)),pair_rgbd_p95_ms=float(np.percentile(times,95)),pair_cycles_per_second=1000/np.median(times),measurement_scope='Two sequential RGB+depth render/readback cycles, classic OpenGL CGL, static two-G1 scene, no detector/encoding/physics rollout included',samples_ms=times);render_rows.append(row);print('RENDER',row,flush=True)
 finally:renderer.close()
path=ROOT/'integrations/bhuman/source/Config/NeuralNets/RobotBallPerceptor/ball_robot_640.onnx';Image.fromarray(rgb_sample).resize((640,640)).save(out/'detector-input.png');x=np.asarray(Image.fromarray(rgb_sample).resize((640,640)),dtype=np.float32).transpose(2,0,1)[None]/255;inference=[]
for label,threads,providers in [('cpu1',1,['CPUExecutionProvider']),('cpu4',4,['CPUExecutionProvider']),('coreml',1,[('CoreMLExecutionProvider',{'ModelFormat':'MLProgram','MLComputeUnits':'ALL','RequireStaticInputShapes':'1'}),'CPUExecutionProvider'])]:
 so=ort.SessionOptions();so.intra_op_num_threads=threads;so.inter_op_num_threads=1;so.enable_profiling=True;so.profile_file_prefix=str(out/('ort-'+label));so.log_severity_level=3
 try:
  start=time.perf_counter();session=ort.InferenceSession(str(path),sess_options=so,providers=providers);load_s=time.perf_counter()-start;inputname=session.get_inputs()[0].name;timings=[]
  for i in range(24):
   start=time.perf_counter();y=session.run(None,{inputname:x});elapsed=(time.perf_counter()-start)*1000
   if i>=5:timings.append(elapsed)
  profile=Path(session.end_profiling());events=json.loads(profile.read_text());counts={}
  for e in events:
   provider=e.get('args',{}).get('provider')
   if provider:counts[provider]=counts.get(provider,0)+1
  row=dict(backend=label,threads=threads,available_session_providers=session.get_providers(),session_load_s=load_s,per_image_p50_ms=float(np.median(timings)),per_image_p95_ms=float(np.percentile(timings,95)),samples_ms=timings,executed_provider_event_counts=counts,model=str(path),model_sha256=digest(path),input_shape=list(x.shape),output_shapes=[list(a.shape) for a in y],scope='Existing B-Human soccer ball/robot detector used ONLYas inference-load proxy;10robot recognition accuracy not evaluated, no end-to-end visual controller tested');inference.append(row);print('INFERENCE',row,flush=True);del session
 except Exception as error:inference.append(dict(backend=label,error=repr(error)));print(label,repr(error),flush=True)
summary=dict(hardware=hardware,rendering=render_rows,inference=inference,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,camera_placement_scope='Temporary body-mounted head-height cameras for load test, not calibrated real-robot extrinsics',not_an_end_to_end_visual_control_validation=True,raw_storage_calculation=dict(two_cameras_640x480_30hz_rgb8_depth_float32_bytes_per_second=2*640*480*30*7,per90s_match_bytes=2*640*480*30*7*90,for90_matches_bytes=2*640*480*30*7*90*90));atomic_json(out/'benchmark.json',summary);event('vision_feasibility_benchmarked',path=str(out/'benchmark.json'));env.close();print('BENCHMARK SAVED',out,flush=True)
