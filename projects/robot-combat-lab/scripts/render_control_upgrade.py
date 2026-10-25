from combatlab.models import ROOT
from combatlab.render import render_video
import json
if __name__=='__main__':
 p=ROOT/'results/control-upgrade';rows=json.loads((p/'validation-results.json').read_text());chosen=[]
 for robot,j in [('g1_full',40),('h1_full',30),('h1_2_full',30)]:chosen.append(next(r for r in rows if r['robot']==robot and r['direction']=='forward' and r['impulse_ns']==j and r['seed']==1))
 chosen.append(next(r for r in rows if r['impulse_ns']==60))
 folders=[p/'archives'/r['id'] for r in chosen]+[p/'contact'/r['match_id'] for r in json.loads((p/'contact-results.json').read_text())]
 for i,folder in enumerate(folders):print('RENDER UPGRADE',i+1,'/',len(folders),folder.name,flush=True);render_video(folder)
