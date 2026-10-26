"""Render one native B-Human RGB-camera exhibition run and its referee result."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from combatlab.engine import sha256, write_json
from combatlab.models import ROOT
from combatlab.render import font


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--period-seconds", type=int, default=30)
    args = parser.parse_args()
    capture = args.capture.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    frames = []
    for row in csv.reader((capture / "frame-times.csv").open()):
        if len(row) >= 3:
            frames.append((int(row[0]), row[1], float(row[2])))
    tele = np.loadtxt(capture / "match.csv", delimiter=",")
    if tele.ndim == 1:
        tele = tele[None, :]
    if not frames or tele.size == 0:
        raise SystemExit("capture has no native framebuffer or referee telemetry")

    # A new scene can leave a few old-state samples in these append-only files.
    # Start at the native referee's first PLAYING row for this 30-second period.
    playing = np.flatnonzero((tele[:, 3].astype(int) == 3) & (tele[:, 2].astype(int) == args.period_seconds))
    if len(playing):
        tele = tele[int(playing[0]):]
        match_start_wall = int(tele[0, 0])
        frames = [r for r in frames if r[0] >= match_start_wall]
    if not frames or tele.size == 0:
        raise SystemExit("capture contains no native camera frames for the referee PLAYING period")

    # Columns: wall ms, SimRobot clock ms, remaining seconds, game state,
    # red score, white score, ball xyz, then robot xy pairs.
    frame_wall = np.asarray([r[0] for r in frames], dtype=np.int64)
    frame_sim = np.asarray([r[2] for r in frames], dtype=float)
    tele_wall = tele[:, 0].astype(np.int64)
    # SimRobot may reset the local physics clock at referee transitions. Stitch
    # those native-clock segments end-to-end; this preserves frame timing while
    # omitting wall-clock startup delays between scene initialization steps.
    elapsed = np.zeros(len(frame_sim), dtype=float)
    offset = 0.0
    segment_start = frame_sim[0]
    for i in range(1, len(frame_sim)):
        if frame_sim[i] < frame_sim[i - 1] - 1000.:
            offset = elapsed[i - 1]
            segment_start = frame_sim[i]
        elapsed[i] = offset + max(0.0, (frame_sim[i] - segment_start) / 1000.0)
    duration_s = max(0.0, float(elapsed[-1]))
    state_names = {0: "INITIAL", 1: "READY", 2: "SET", 3: "PLAYING", 4: "FINISHED"}

    video = output / f"{args.match_id}.mp4"
    silent = output / f"{args.match_id}.silent.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "1280x720", "-r", "30", "-i", "-", "-an", "-c:v", "libx264",
           "-crf", "19", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(silent)]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    last_frame = -1
    final_canvas = None
    frame_count = max(1, round(duration_s * 30) + 1)
    for i in range(frame_count):
        t = min(duration_s, i / 30)
        frame_idx = max(0, bisect.bisect_right(elapsed, t) - 1)
        tele_idx = max(0, bisect.bisect_right(tele_wall, frame_wall[frame_idx]) - 1)
        row = tele[tele_idx]
        if frame_idx != last_frame:
            field = Image.open(capture / "frames" / frames[frame_idx][1]).convert("RGB")
            field = field.resize((1280, 672), Image.Resampling.LANCZOS)
            last_frame = frame_idx
        canvas = Image.new("RGB", (1280, 720), "#080f18")
        canvas.paste(field, (0, 48))
        draw = ImageDraw.Draw(canvas)

        def label(x: int, y: int, text: str, size: int = 16, color: str = "#dce7f3", bold: bool = False) -> None:
            draw.text((x, y), text, font=font(size, bold), fill=color)

        state = state_names.get(int(row[3]), f"STATE {int(row[3])}")
        label(20, 14, "BOTSTEST  /  B-HUMAN K1 3v3  /  NORMAL RGB CAMERA PIPELINE", 16, "#88d9ea", True)
        label(575, 10, f"RED  {int(row[4])} : {int(row[5])}  WHITE", 24, "#ffd28e", True)
        label(1110, 15, f"{max(0, int(row[2])):02d}s", 18, "#edf4fa", True)
        draw.rectangle((0, 704, 1280, 720), fill="#080f18")
        label(20, 705, f"LOCAL {args.period_seconds}s EXHIBITION  ·  NATIVE REFEREE {state}  ·  SIMROBOT CAMERA COLOR RGB", 10, "#aec0cf")
        process.stdin.write(np.asarray(canvas).tobytes())
        final_canvas = canvas

    if final_canvas is None:
        raise SystemExit("video has no renderable frames")
    if int(tele[-1, 3]) == 4:
        draw = ImageDraw.Draw(final_canvas)
        draw.rounded_rectangle((355, 250, 925, 440), radius=15, fill="#071321", outline="#e3ba79", width=3)
        draw.text((405, 272), "B-HUMAN REFEREE / MATCH FINISHED", font=font(22, True), fill="#a8c5d7")
        draw.text((405, 322), f"FINAL  RED {int(tele[-1, 4])} : {int(tele[-1, 5])} WHITE", font=font(34, True), fill="#ffd18e")
        draw.text((405, 380), f"CUSTOM LOCAL EXHIBITION · {args.period_seconds}s PERIOD", font=font(16), fill="#b8c8d6")
        for _ in range(60):
            process.stdin.write(np.asarray(final_canvas).tobytes())
    poster = output / f"{args.match_id}-poster.jpg"
    final_canvas.save(poster, quality=94)
    process.stdin.close()
    if process.wait() != 0:
        raise SystemExit("ffmpeg failed while rendering the camera frames")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-c:v", "copy", "-movflags", "+faststart", str(video)], check=True)
    silent.unlink()

    final = tele[-1]
    state = int(final[3])
    result = {
        "match_id": args.match_id,
        "competition": "B-Human K1 3v3 visual soccer exhibition",
        "state": state_names.get(state, str(state)),
        "completed_by_native_referee": state == 4,
        "period_seconds": args.period_seconds,
        "score": {"red": int(final[4]), "white": int(final[5]), "source": "native B-Human GameController telemetry"},
        "winner": "red" if state == 4 and final[4] > final[5] else "white" if state == 4 and final[5] > final[4] else "draw" if state == 4 else None,
        "teams": {"red": "B-Human", "white": "B-Team"},
        "robots": {"model": "Booster K1", "count": 6, "format": "3v3"},
        "perception": {"camera": "SimRobot CameraColor virtual RGB", "mode": "B-Human normal CameraImage/ECImage/RobotBallPerceptor pipeline", "fast_truth_mode": False},
        "native_simulator": "B-Human SimRobot Core3 / MuJoCo",
        "source_frames": len(frames),
        "unique_source_frames": len({sha256(capture / "frames" / r[1]) for r in frames}),
        "source_simulation_duration_seconds": round(duration_s, 3),
        "source_wall_duration_seconds": round((frame_wall[-1] - frame_wall[0]) / 1000, 3),
        "source_physics_clock_start_ms": round(float(frame_sim[0]), 3),
        "source_physics_clock_end_ms": round(float(frame_sim[-1]), 3),
        "temporal_interpolation": False,
        "video": video.name,
        "poster": poster.name,
        "video_sha256": sha256(video),
        "limitations": ["30-second local exhibition period; not the official ten-minute B-Human half", "K1 visual soccer score is not comparable to MuJoCo boxing scores"],
    }
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,width,height,avg_frame_rate,nb_frames", "-of", "json", str(video)]))
    result["video_probe"] = probe
    write_json(output / "result.json", result)
    write_json(output / "manifest.json", {"video": {"sha256": sha256(video), "bytes": video.stat().st_size}, "source_frames": len(frames), "telemetry_rows": int(len(tele))})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
