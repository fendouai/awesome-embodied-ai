from pathlib import Path
import sys,os
os.environ.setdefault('MUJOCO_GL','cgl');sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from combatlab.render import render_video
print(render_video(Path(sys.argv[1])),flush=True)
