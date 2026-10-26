"""Complete Qt framework layout omitted by the upstream Xcode-only staging path."""
from pathlib import Path
import os,json
from combatlab.models import ROOT
if __name__=='__main__':
 source=ROOT/'integrations/bhuman/source';app=source/'Build/macOS/SimRobot/Release/SimRobot.app';dest=app/'Contents/Frameworks';qt=source/'Util/SimRobot/Util/qt/macOS/lib';links=[]
 for f in qt.glob('*.framework'):
  target=dest/f.name
  if target.is_dir() and not target.is_symlink() and not any(target.iterdir()):target.rmdir()
  if not target.exists():target.symlink_to(os.path.relpath(f,target.parent),target_is_directory=True)
  assert (target/'Versions/A'/f.stem).exists();links.append(dict(framework=f.name,relative_link=os.readlink(target) if target.is_symlink() else None))
 (ROOT/'integrations/bhuman/bundle-staging-audit.json').write_text(json.dumps(dict(app=str(app.relative_to(ROOT)),frameworks=links,scope='NativeMacproject bundle; relative links require keeping the vendored source subtree'),indent=2));print('Native Qt frameworks linked into app bundle',len(links))
