import argparse,os
os.environ.setdefault('MUJOCO_GL','cgl')
from .models import ROOT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preflight','tournament','rank','render','serve','report','verify','all']);parser.add_argument('--port',type=int,default=8767);parser.add_argument('--match');parser.add_argument('--fps',type=int,default=24);args=parser.parse_args()
    if args.command in ['preflight','all']:
        from .benchmark import preflight
        preflight()
    if args.command in ['tournament','all']:
        from .benchmark import tournament
        tournament()
    if args.command=='rank':
        from .benchmark import rank
        rank()
    if args.command in ['render','all']:
        from .render import render_video
        import json
        matches=json.loads((ROOT/'results/matches.json').read_text())
        for i,r in enumerate(matches):
            if args.match and r['match_id']!=args.match:continue
            folder=ROOT/'results/matches'/r['match_id']
            if (folder/'battle.mp4').exists():print('VIDEO EXISTS',r['match_id'],flush=True);continue
            print('VIDEO',i+1,'/',len(matches),r['match_id'],flush=True);render_video(folder,fps=args.fps)
    if args.command in ['report','all']:
        from .report import build_report
        build_report()
    if args.command in ['verify','all']:
        from .verify import verify
        verify()
    if args.command=='serve':
        import uvicorn
        from .server import app
        uvicorn.run(app,host='127.0.0.1',port=args.port)
if __name__=='__main__':main()
