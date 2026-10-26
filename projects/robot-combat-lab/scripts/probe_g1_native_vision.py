import json,numpy as np,mujoco,onnxruntime as ort,cv2
from PIL import Image,ImageDraw
from combatlab.contact_arena import ContactArena
from combatlab.g1_camera import specification,add_cameras,RGB_SIZE,DEPTH_SIZE
from combatlab.models import ROOT
from combatlab.research_archive import workspace,atomic_json,event,digest
out=workspace()/'06_reports/g1-native-camera-probe';out.mkdir(exist_ok=True);atomic_json(out/'camera-specification.json',specification());env=ContactArena('g1_full',scene_transform=add_cameras);m=env.m;d=env.d;rgb=mujoco.Renderer(m,RGB_SIZE[1],RGB_SIZE[0]);depth=mujoco.Renderer(m,DEPTH_SIZE[1],DEPTH_SIZE[0]);depth.enable_depth_rendering();option=mujoco.MjvOption();visual={int(m.geom_bodyid[i]) for i in range(m.ngeom) if m.geom_contype[i]==0 and m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH}
for i in range(m.ngeom):
 if int(m.geom_bodyid[i]) in visual and m.geom_contype[i]:m.geom_group[i]=5
option.geomgroup[5]=0
model=ROOT/'integrations/bhuman/source/Config/NeuralNets/RobotBallPerceptor/ball_robot_640.onnx';opts=ort.SessionOptions();opts.intra_op_num_threads=1;session=ort.InferenceSession(str(model),sess_options=opts,providers=[('CoreMLExecutionProvider',{'ModelFormat':'MLProgram','MLComputeUnits':'ALL','RequireStaticInputShapes':'1'}),'CPUExecutionProvider']);grid=np.concatenate([np.array([(x,y,stride) for y in range(640//stride) for x in range(640//stride)]) for stride in [8,16,32]]);reports=[]
try:
 for side in range(2):
  rgb.update_scene(d,camera=f'p{side}/d435i_rgb',scene_option=option);image=rgb.render().copy();depth.update_scene(d,camera=f'p{side}/d435i_depth',scene_option=option);z=depth.render().copy();Image.fromarray(image).save(out/f'rgb-{side}.png');np.savez_compressed(out/f'depth-{side}.npz',depth_m=z)
  inp=np.full((1,3,640,640),114,dtype=np.float32);inp[0,:,:image.shape[0],:image.shape[1]]=image[:,:,::-1].transpose(2,0,1);pred=session.run(None,{session.get_inputs()[0].name:inp})[0][0];score=pred[:,4]*pred[:,6];xy=(pred[:,:2]+grid[:,:2])*grid[:,2:];wh=np.exp(np.clip(pred[:,2:4],-20,20))*grid[:,2:];boxes=np.c_[xy-wh/2,wh];ids=np.flatnonzero(score>.1);nms=cv2.dnn.NMSBoxes(boxes[ids].tolist(),score[ids].tolist(),.1,.45);selected=[int(ids[int(i)]) for i in np.array(nms).reshape(-1)];data=[];im=Image.fromarray(image);draw=ImageDraw.Draw(im)
  for i in selected:
   x,y,w,h=boxes[i];draw.rectangle((x,y,x+w,y+h),outline='lime',width=2);draw.text((x,max(0,y-14)),f'robot {score[i]:.2f}',fill='lime');data.append(dict(box_xywh=boxes[i].tolist(),confidence=float(score[i])))
  im.save(out/f'detector-{side}.png');reports.append(dict(side=side,detections=data,max_robot_score=float(score.max()),source_preprocessing='BGR float0..255, top-left640x360 in640square114padding, matching original B-Human code',source_decoder='YOLOX grid8/16/32, objectness*robot-class score, xywh pixel decoding and NMS',model_sha256=digest(model)))
 atomic_json(out/'detections.json',reports);event('g1_native_camera_probed',path=str(out),detections=reports);print(reports,flush=True)
finally:rgb.close();depth.close();env.close()
