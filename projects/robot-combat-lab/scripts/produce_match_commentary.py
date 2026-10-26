from pathlib import Path
import argparse
from combatlab.commentary import narrate
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('video',type=Path);p.add_argument('--kind',choices=['boxing','vision'],default='boxing');a=p.parse_args();print(narrate(a.folder,a.video,a.kind),flush=True)
