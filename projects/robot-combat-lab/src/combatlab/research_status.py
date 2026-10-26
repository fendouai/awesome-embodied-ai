"""Read-only view over the external, versioned research archive."""
from pathlib import Path
import json,shutil
from .research_archive import workspace,now

def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default

def status():
 try:root=workspace()
 except Exception as e:return dict(available=False,error=str(e))
 first=read(root/'00_index/training-pass01.json',{});second=read(root/'00_index/training-pass02.json',{});rows=[]
 for original in first.get('robots',[]):
  key=original['robot'];row=dict(original);new=next((x for x in second.get('robots',[]) if x['robot']==key),None)
  if new:row.update(new)
  if second.get('current')==key:
   candidates=[]
   for folder in (root/'03_training').glob('ppo-'+key+'-*'):
    c=read(folder/'config.json',{})
    if c.get('reward_version')==2:candidates.append(folder)
   if candidates:row.update(folder=str(sorted(candidates)[-1]),state='training')
  folder=Path(row['folder']) if row.get('folder') else None
  if folder:
   row['artifact_url']='/research/'+str(folder.relative_to(root));row['metrics']=read(folder/'metrics.json',{});row['checkpoint']=read(folder/'latest.json',{});row['evaluation']=read(folder/'evaluation/summary.json');row['config']=read(folder/'config.json',{});row.pop('latest',None)
   if 'k1-20261025T123859-0a6a26' in str(folder):row['validation_status']='quarantined_collision_filter_mismatch'
   else:row['validation_status']='continuous_contact_gate_passed' if row['evaluation'] and row['evaluation'].get('all_seeds_continuous_engagement') else 'not_validated'
  rows.append(row)
 matches=read(root/'04_matches/index.json',dict(planned=90,completed=0,state='not_started'))
 return dict(available=True,root=str(root),utc=now(),free_bytes=shutil.disk_usage(root).free,baseline=read(root/'08_manifests/baseline-summary.json'),robots=rows,training_first_pass=first.get('state'),training_first_pass_completed=sum(r['state']=='training_pass_complete' for r in first.get('robots',[])),training_second_pass=second.get('state'),current=second.get('current') or first.get('current'),contact_gate_passed=sum(r.get('validation_status')=='continuous_contact_gate_passed' for r in rows),planned_robot_pairs=45,planned_matches=90,matches=matches,archive_files=dict(dictionary='/research/00_index/data-dictionary.json',baseline_manifest='/research/08_manifests/baseline-files.jsonl',baseline_summary='/research/08_manifests/baseline-summary.json',events='/research/07_logs/events.jsonl',quarantined='/research/00_index/quarantined-experiments.json'))
