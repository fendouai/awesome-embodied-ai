"""Build the public index and checksums for four completed B-Human exhibitions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from combatlab.engine import sha256, write_json
from combatlab.models import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-dir", type=Path, default=ROOT / "results/tournaments/round-03")
    parser.add_argument("--expected-matches", type=int, default=4)
    args = parser.parse_args()
    round_dir = args.round_dir.resolve()
    matches = []
    videos = []
    for number in range(1, args.expected_matches + 1):
        match_id = f"vision-k1-match-{number:02d}"
        result_path = round_dir / "matches" / match_id / "result.json"
        if not result_path.exists():
            raise SystemExit(f"missing match result: {result_path}")
        result = json.loads(result_path.read_text())
        if result.get("state") != "FINISHED" or not result.get("completed_by_native_referee"):
            raise SystemExit(f"match {number} did not finish under the native referee")
        video = round_dir / "videos" / Path(result["video"]).name
        poster = round_dir / "videos" / Path(result["poster"]).name
        telemetry = round_dir / "matches" / match_id / "match.csv"
        for asset in (video, poster, telemetry):
            if not asset.is_file() or asset.stat().st_size == 0:
                raise SystemExit(f"missing/empty match asset: {asset}")
        entry = {
            "number": number,
            "id": match_id,
            "period_seconds": result["period_seconds"],
            "completed_by_native_referee": True,
            "score": result["score"],
            "winner": result["winner"],
            "teams": result["teams"],
            "robots": result["robots"],
            "perception": result["perception"],
            "source_frames": result["source_frames"],
            "source_simulation_duration_seconds": result["source_simulation_duration_seconds"],
            "source_wall_duration_seconds": result["source_wall_duration_seconds"],
            "video": result["video"],
            "poster": result["poster"],
            "result": result["result"],
            "native_telemetry": result["native_telemetry"],
            "video_sha256": sha256(video),
            "video_bytes": video.stat().st_size,
            "native_telemetry_sha256": sha256(telemetry),
        }
        matches.append(entry)
        videos.append({"match": match_id, "video": result["video"], "poster": result["poster"], "sha256": entry["video_sha256"], "bytes": entry["video_bytes"]})
        write_json(round_dir / "matches" / match_id / "manifest.json", {
            "method": "SHA-256 and byte-count manifest for the published match artifacts",
            "files": {
                asset.name: {"sha256": sha256(asset), "bytes": asset.stat().st_size}
                for asset in (result_path, telemetry, video, poster)
            },
        })

    index = {
        "id": "round-03",
        "state": "complete",
        "track": "B-Human K1 3v3 normal RGB-camera perception soccer",
        "format": "Four native-refereed 30-second local exhibition matches; not official ten-minute halves",
        "engine": "B-Human SimRobot Core3 / MuJoCo",
        "sensing": "SimRobot CameraColor RGB virtual cameras through the normal CameraImage/ECImage/RobotBallPerceptor pipeline",
        "robots": {"model": "Booster K1", "count": 6, "sides": ["B-Human", "B-Team"]},
        "score_source": "Native B-Human GameController telemetry",
        "official_regulation_completed": False,
        "matches": matches,
    }
    write_json(round_dir / "index.json", index)
    write_json(round_dir / "video-archive.json", {"method": "Native SimRobot RGB framebuffer video; native referee telemetry archived per match", "videos": videos})
    with (round_dir / "vision-matches.csv").open("w", newline="") as f:
        fields = ["match", "period_seconds", "red", "white", "winner", "referee_state", "source_frames", "source_simulation_duration_seconds", "source_wall_duration_seconds", "video", "sha256", "native_telemetry_sha256"]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for m in matches:
            writer.writerow({"match": m["id"], "period_seconds": m["period_seconds"], "red": m["score"]["red"], "white": m["score"]["white"], "winner": m["winner"], "referee_state": "FINISHED", "source_frames": m["source_frames"], "source_simulation_duration_seconds": m["source_simulation_duration_seconds"], "source_wall_duration_seconds": m["source_wall_duration_seconds"], "video": m["video"], "sha256": m["video_sha256"], "native_telemetry_sha256": m["native_telemetry_sha256"]})
    print(f"packaged {len(matches)} native-refereed matches; all states FINISHED")


if __name__ == "__main__":
    main()
