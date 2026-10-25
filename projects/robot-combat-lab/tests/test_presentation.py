"""Presentation must preserve physics and display measured, dimensioned signals."""
import json,wave
import numpy as np
from combatlab.models import ROOT
from combatlab.presentation import ContactTrack,BroadcastRenderer
from combatlab.storage import load_model
from combatlab.engine import sha256


def test_contact_impulse_reconstructs_saved_physics():
 for backend in ['robostriker-cpu','ncp-cpu']:
  j=json.loads((ROOT/'results'/backend/'latest.json').read_text());folder=ROOT/j['folder'];track=ContactTrack(folder)
  np.testing.assert_allclose(track.impulse[-1],j['result']['contact_impulse_ns'],rtol=1e-10)
  assert all(e['peak_n']>3 and e['impulse_ns']>0 for e in track.events)
  info=json.loads((folder/'presentation.json').read_text());assert info['source_trajectory_sha256']==sha256(folder/'trajectory.npz');assert info['source_policy_io_sha256']==sha256(folder/'policy_io.npz')


def test_studio_materials_do_not_change_dynamics():
 j=json.loads((ROOT/'results/robostriker-cpu/latest.json').read_text());folder=ROOT/j['folder'];original=load_model(folder);r=BroadcastRenderer(folder)
 try:
  r.frame(4.)
  for field in ['body_mass','body_inertia','geom_size','geom_pos','geom_friction','geom_contype','geom_conaffinity','dof_armature','dof_damping','jnt_stiffness','actuator_gainprm','actuator_biasprm','actuator_forcerange']:
   np.testing.assert_array_equal(getattr(original,field),getattr(r.m,field))
  assert r.m.opt.timestep==original.opt.timestep
 finally:r.close()


def test_audio_cues_and_soccer_data_are_real_nonempty_artifacts():
 for folder in [ROOT/json.loads((ROOT/'results/robostriker-cpu/latest.json').read_text())['folder'],ROOT/json.loads((ROOT/'results/ncp-cpu/latest.json').read_text())['folder']]:
  with wave.open(str(folder/'contact-sonification.wav')) as f:audio=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2')
  assert np.max(abs(audio.astype(float)))>100 and np.max(abs(audio.astype(float)))<32767
 folder=ROOT/'results/bhuman-native/studio-match-clock';rows=np.loadtxt(folder/'match.csv',delimiter=',');assert len(rows)>10*(rows[-1,0]-rows[0,0])/1000 and rows.shape[1]==20
 assert np.all(np.diff(rows[:,0])>0) and np.all(np.diff(rows[:,1])>=0)
 assert set(rows[:,3]).issubset({0,1,2,3,4})
 with wave.open(str(folder/'referee-sonification.wav')) as f:audio=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2')
 assert np.any(audio)


def test_native_soccer_frames_are_not_black_and_have_real_cadence():
 from PIL import Image
 folder=ROOT/'results/bhuman-native/studio-match-clock';frames=[line.split(',') for line in (folder/'frame-times.csv').read_text().splitlines()]
 assert len(frames)>400
 stamps=np.array([int(f[0]) for f in frames]);assert np.all(np.diff(stamps)>0)
 for index in (0,len(frames)//2,len(frames)-1):
  pixels=np.asarray(Image.open(folder/'frames'/frames[index][1]));assert pixels[:,:,:3].std()>10
 info=json.loads((folder/'presentation.json').read_text());assert info['source_fps']>10 and info['temporal_interpolation'] is False


def test_soccer_events_use_the_same_physics_clock_as_video():
 folder=ROOT/'results/bhuman-native/studio-match-clock';info=json.loads((folder/'presentation.json').read_text());rows=np.loadtxt(folder/'match.csv',delimiter=',')
 assert np.max(np.ptp(rows[:,6:8],axis=0))>1
 assert all(0<=e['time_s']<=info['duration_s'] for e in info['events'])
 frames_raw=[line.split(',') for line in (folder/'frame-times.csv').read_text().splitlines()];wall=np.array([int(f[0]) for f in frames_raw]);aligned=rows[np.maximum(0,np.searchsorted(rows[:,0],wall,side='right')-1)]
 changes=np.any(np.diff(aligned[:,3:6],axis=0)!=0,axis=1)
 assert len(info['events'])==int(changes.sum())
 frames=np.array([float(line.split(',')[2]) for line in (folder/'frame-times.csv').read_text().splitlines()]);assert np.all(np.diff(frames)>=31.9)
