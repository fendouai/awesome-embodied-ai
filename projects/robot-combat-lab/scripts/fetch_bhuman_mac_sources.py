"""Fetch pinned sources and macOS assets only, using safe tar extraction."""
from pathlib import Path
import urllib.request,tarfile,json,os
from combatlab.models import ROOT
from combatlab.engine import write_json
P=ROOT/'integrations/bhuman/source'

def fetch(repo,sha,out):
 out.mkdir(parents=True,exist_ok=True);url=f'https://codeload.github.com/{repo}/tar.gz/{sha}';count=0;size=0;skipped=0
 with urllib.request.urlopen(url,timeout=120) as response,tarfile.open(fileobj=response,mode='r|gz') as archive:
  for member in archive:
   parts=Path(member.name).parts[1:]
   if not parts:continue
   rel=Path(*parts)
   if any(p in ['Windows','WindowsArm64','Linux','LinuxArm64'] for p in parts) or ('BoosterRoboticsSDK' in parts and 'lib' in parts):skipped+=1;continue
   if '..' in parts or rel.is_absolute():raise ValueError('Unsafe archive path')
   member.name=str(rel)
   archive.extract(member,path=out,filter='data')
   if member.isfile():size+=member.size;count+=1
 print('FETCHED MAC SOURCE',repo,count,'files',round(size/1e6,1),'MB; skipped foreign-platform entries',skipped,flush=True)
 return dict(repository=repo,commit=sha,local=str(out.relative_to(ROOT)),files=count,bytes=size,foreign_platform_entries_skipped=skipped)

if __name__=='__main__':
 specs=[('bhuman/BHumanCodeRelease','2e3374ba896da8288607f9a1881b876642fe18b8',P),('bhuman/SimRobot','9deb6ae6b2a553211612affb757490736f96bfb3',P/'Util/SimRobot'),('bhuman/CompiledNN','8332c8759f3911e08ded6ac7e088f85620b4c4b5',P/'Util/CompiledNN'),('pybind/pybind11','0bd8896a4010f2d91b2340570c24fa08606ec406',P/'Util/pybind11')];records=[]
 for repo,sha,path in specs:records.append(fetch(repo,sha,path));write_json(ROOT/'integrations/bhuman/source-manifest.json',records)
