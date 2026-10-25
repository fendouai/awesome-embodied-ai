# B-Human 2026 local Mac build

Pinned BHumanCodeRelease and submodules are recorded in source-manifest.json. The native ARM64 app was built with Command Line Tools, CMake 3.31.6 and Ninja 1.11.1.4. No full Xcode installation or Linux robot cross-build was needed for this desktop acceptance.

Build changes:

- Restrict upstream forced Booster and macOS cross-build targets to the Xcode generator; leave BUILD_BOOSTER=OFF for Ninja.
- Disable precompiled headers to avoid repeated 32 MB caches. `forced-includes/` contains the same generated PCH header text; `force-includes.cmake` passes it as normal compiler includes. This retains upstream implicit header dependencies.
- Add app bundle rpaths for `../Frameworks` and `../lib`; Ninja does not apply XCODE_ATTRIBUTE rpath settings.
- Reuse installed native ONNX Runtime 1.24.4 through the expected versioned path. Included C API headers are backward compatible. Retain source Qt 6.12 and MuJoCo 3.3.5 libraries.
- `scripts/stage_bhuman_mac_bundle.py` creates relative links to pinned Qt framework directories. The app is colocated with source dependencies and is not a standalone distributable.
- Ad-hoc sign the locally staged MuJoCo dylib as part of this local build. No OS security setting was disabled.
- `ConsoleRoboCupCtrl.cpp` opens field and Console views for the authored MacAcceptance3v3M scene. The .con script uses normal referee commands; no soccer decision, physics or actuator code was changed.

Rebuild in the project directory (paths in this local CMake cache refer to this checkout):

```sh
.venv/bin/cmake -S integrations/bhuman/source/Make/CMake -B integrations/bhuman/build-mac -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_BOOSTER=OFF -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja"
.venv/bin/cmake --build integrations/bhuman/build-mac --target SimRobot --parallel 2
PYTHONPATH=src .venv/bin/python scripts/stage_bhuman_mac_bundle.py
codesign --force --sign - integrations/bhuman/source/Build/macOS/SimRobot/Release/SimRobot.app/Contents/lib/libmujoco.3.3.5.dylib
```

First scene loading can stall in hostname DNS resolution. A captured process sample documents it. Startup Cognition→Motion warnings were observed, followed by normal scene stepping. Fast mode uses simulator truth. Captured evidence shows 2:0 and PLAYING at about 500 simulation steps/s. This is not a full match archive or camera-perception benchmark.

This product includes software developed by B-Human (http://www.b-human.de).
