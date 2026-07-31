#!/usr/bin/env python3
"""Real-time phrase replacement source, independent from the HushMic chain."""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_phrase_trigger import CLIP_SAMPLES, TinyPhraseNet  # noqa: E402
from train_bc_phrase import BCResPhraseNet  # noqa: E402

RATE = 16_000
BLOCK = 320  # 20 ms
INPUT = "hushmic_ai_source"
SINK = "hushmic_phrase_sink"
SOURCE = "hushmic_phrase_source"


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def ensure_source() -> None:
    sources = run(["pactl", "list", "short", "sources"])
    if SOURCE in sources and SINK in run(["pactl", "list", "short", "sinks"]):
        return
    for line in run(["pactl", "list", "modules", "short"]).splitlines():
        if SINK in line or SOURCE in line:
            try:
                run(["pactl", "unload-module", line.split()[0]])
            except subprocess.CalledProcessError:
                pass
    run(["pactl", "load-module", "module-null-sink", f"sink_name={SINK}",
         "rate=16000", "channels=1", "channel_map=mono",
         "sink_properties=device.description=Phrase_Trigger"])
    run(["pactl", "load-module", "module-remap-source", f"master={SINK}.monitor",
         f"source_name={SOURCE}", "channels=1", "channel_map=mono", "remix=no",
         "source_properties=device.description=Мем"])


def load_mp3(path: Path) -> np.ndarray:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
               "-f", "f32le", "-ac", "1", "-ar", str(RATE), "pipe:1"]
    data = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(data, dtype=np.float32).copy()


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, float, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_format = checkpoint.get("format")
    if model_format == "hushmic-bc-phrase-net-v1":
        model = BCResPhraseNet()
    elif model_format == "hushmic-tiny-phrase-net-v1":
        model = TinyPhraseNet()
    else:
        raise RuntimeError("unsupported phrase detector checkpoint")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    # The held-out negative dialogue reached 0.873 once; require a margin above it.
    clip_samples = int(checkpoint.get("clip_samples", CLIP_SAMPLES))
    return model, max(0.98, float(checkpoint.get("threshold", 0.9))), clip_samples


def detect(model: torch.nn.Module, audio: np.ndarray, device: torch.device) -> float:
    with torch.inference_mode():
        tensor = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0).to(device)
        return float(model(tensor).softmax(-1)[0, 1].item())


def write_status(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"updated_at": time.time(), **values}, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a trained phrase-detector checkpoint")
    parser.add_argument("--sound", required=True,
                        help="Path to the replacement audio file")
    parser.add_argument("--input-source", default=INPUT)
    parser.add_argument("--status-file", default=f"/run/user/{os.getuid()}/hushmic-phrase/status.json")
    parser.add_argument("--delay-ms", type=int, default=1200)
    parser.add_argument("--cooldown-ms", type=int, default=1800)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_source()
    model, threshold, clip_samples = load_model(Path(args.checkpoint), device)
    replacement = load_mp3(Path(args.sound))
    capture = subprocess.Popen([
        "pw-cat", "--record", "--target", args.input_source, "--raw", "--rate", str(RATE),
        "--channels", "1", "--channel-map", "MONO", "--format", "f32", "--latency", "20ms", "-",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    playback = subprocess.Popen([
        "pw-cat", "--playback", "--target", SINK, "--raw", "--rate", str(RATE),
        "--channels", "1", "--channel-map", "MONO", "--format", "f32", "--latency", "20ms", "-",
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert capture.stdout is not None and playback.stdin is not None
    delay_blocks = max(1, args.delay_ms // 20)
    delayed: collections.deque[np.ndarray] = collections.deque(maxlen=delay_blocks + 4)
    detector_audio = np.zeros(clip_samples, dtype=np.float32)
    mute_blocks = 0
    cooldown_blocks = 0
    confident_hits = 0
    blocks = 0
    triggers = 0
    write_status(Path(args.status_file), state="running", input=args.input_source,
                 output=SOURCE, checkpoint=args.checkpoint, threshold=threshold,
                 delay_ms=args.delay_ms, device=str(device), triggers=0)
    try:
        while True:
            raw = capture.stdout.read(BLOCK * 4)
            if len(raw) != BLOCK * 4:
                raise RuntimeError("phrase capture stream ended")
            block = np.frombuffer(raw, dtype=np.float32).copy()
            delayed.append(block)
            detector_audio = np.roll(detector_audio, -BLOCK)
            detector_audio[-BLOCK:] = block
            score = 0.0
            if blocks % 4 == 0:
                score = detect(model, detector_audio, device)
                confident_hits = confident_hits + 1 if score >= threshold else 0
            if cooldown_blocks == 0 and mute_blocks == 0 and confident_hits >= 3:
                triggers += 1
                confident_hits = 0
                delayed.clear()
                playback.stdin.write(replacement.tobytes())
                playback.stdin.flush()
                mute_blocks = max(1, int((len(replacement) / BLOCK) + 10))
                cooldown_blocks = max(1, args.cooldown_ms // 20)
            elif mute_blocks > 0:
                mute_blocks -= 1
            elif len(delayed) > delay_blocks:
                playback.stdin.write(delayed.popleft().tobytes())
                playback.stdin.flush()
            if cooldown_blocks > 0:
                cooldown_blocks -= 1
            blocks += 1
            if blocks % 50 == 0:
                write_status(Path(args.status_file), state="running", input=args.input_source,
                             output=SOURCE, checkpoint=args.checkpoint, threshold=threshold,
                             delay_ms=args.delay_ms, device=str(device), triggers=triggers,
                             last_score=score)
    finally:
        write_status(Path(args.status_file), state="stopped", triggers=triggers)
        capture.terminate(); playback.terminate()


if __name__ == "__main__":
    main()
