#!/usr/bin/env python3
"""Live personalized GPU voice extraction for HushMic.

Audio graph:
  selected input source -> this worker -> HushMic AI virtual source

The real-time PipeWire streams run in separate threads. A causal separator
keeps its recurrent and overlap-add state while emitting one 80 ms chunk at
a time.
"""

from __future__ import annotations

import argparse
import atexit
from collections import deque
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
import yaml

# Keep third-party research checkouts outside this repository. Set these
# variables to local paths before launching the optional worker.
for _root_var in ("WESEP_ROOT", "WESPEAKER_ROOT"):
    _root = os.environ.get(_root_var)
    if _root:
        sys.path.insert(0, _root)

from wesep.models import get_model  # noqa: E402
from wesep.models.bsrnn_streaming import StreamingBSRNN  # noqa: E402
from wesep.models.bsrnn_multilevel_streaming import (  # noqa: E402
    StreamingBSRNNMultiLevel,
)
from wesep.models.bsrnn_streaming_v2 import (  # noqa: E402
    StreamingBSRNNTimbreV2,
)
from wesep.models.bsrnn_streaming_v3 import (  # noqa: E402
    StreamingBSRNNCausalNormV3,
)
from wesep.models.tfmlp_streaming import (  # noqa: E402
    StreamingSpeakerTFMLPNet,
)
from wesep.models.personal_grid_streaming import (  # noqa: E402
    StreamingPersonalGridNet,
)
from wesep.models.hushmic_output_verifier import (  # noqa: E402
    CausalOutputSpeakerVerifier,
)


STREAM_RATE = 16000
DEFAULT_CHUNK_MS = 80
STATUS_INTERVAL_BLOCKS = 125
SINK_NAME = "hushmic_ai_sink"
SOURCE_NAME = "hushmic_ai_source"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-source", default="hushmic_source")
    parser.add_argument(
        "--config",
        required=True,
        help="WeSep model configuration",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Personal separator checkpoint",
    )
    parser.add_argument(
        "--status-file",
        default=f"/run/user/{os.getuid()}/hushmic-personal/status.json",
    )
    parser.add_argument("--set-default", action="store_true")
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=DEFAULT_CHUNK_MS,
        help="Capture/inference block duration; must align to the STFT hop",
    )
    parser.add_argument(
        "--remove-virtual-source-on-exit",
        action="store_true",
        help="Remove the PipeWire source instead of preserving its device ID",
    )
    parser.add_argument(
        "--speaker-gate",
        action="store_true",
        help=(
            "Enable the hybrid fast/ECAPA target-speaker verifier. "
            "It is opt-in until the live latency test passes."
        ),
    )
    parser.add_argument(
        "--fast-verifier-checkpoint",
        help="Optional fast speaker-verifier checkpoint",
    )
    parser.add_argument(
        "--fast-verifier-rms-cap",
        type=float,
        default=0.20,
        help=(
            "Down-normalize unusually loud verifier spectrograms only; "
            "zero disables the cap"
        ),
    )
    parser.add_argument(
        "--speaker-base-checkpoint",
        help="Optional ECAPA/WeSpeaker base checkpoint",
    )
    parser.add_argument(
        "--speaker-gate-enrollment",
        help=(
            "Optional robust ECAPA enrollment payload. The separator and "
            "fast verifier keep using their original embedding."
        ),
    )
    parser.add_argument(
        "--verification-window-ms",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--verification-hop-ms",
        type=int,
        default=560,
        help="ECAPA cadence; 560 ms is seven 80 ms capture chunks",
    )
    parser.add_argument(
        "--ecapa-threshold",
        type=float,
        default=0.0,
        help="Minimum ECAPA similarity before the combined identity score",
    )
    parser.add_argument(
        "--identity-fast-weight",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--identity-score-threshold",
        type=float,
        default=0.45,
        help="Threshold for ECAPA similarity + weight * fast clip score",
    )
    parser.add_argument(
        "--identity-close-fast-weight",
        type=float,
        default=3.0,
        help=(
            "Fast-verifier weight used only when deciding whether an open "
            "gate is confidently hearing a foreign speaker"
        ),
    )
    parser.add_argument(
        "--identity-close-score-threshold",
        type=float,
        default=1.10,
        help=(
            "An open gate closes only after consecutive scores below this "
            "threshold; reopening still uses identity-score-threshold"
        ),
    )
    parser.add_argument(
        "--identity-reject-checks",
        type=int,
        default=3,
        help="Consecutive full-window rejections required to close the gate",
    )
    parser.add_argument(
        "--short-accept-window-ms",
        type=int,
        default=1000,
        help="Short ECAPA window used only to reopen a rejected gate",
    )
    parser.add_argument(
        "--short-accept-threshold",
        type=float,
        default=0.23,
    )
    parser.add_argument(
        "--short-accept-fast-threshold",
        type=float,
        default=0.60,
        help=(
            "Fast clip score that may rescue a weak short ECAPA match"
        ),
    )
    parser.add_argument(
        "--verification-min-rms-dbfs",
        type=float,
        default=-55.0,
    )
    parser.add_argument(
        "--gate-quiet-reset-ms",
        type=int,
        default=320,
        help="Silence needed to clear a rejected identity decision",
    )
    parser.add_argument("--gate-attack-ms", type=float, default=8.0)
    parser.add_argument("--gate-release-ms", type=float, default=40.0)
    return parser.parse_args()


def run_checked(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def matching_modules() -> list[int]:
    output = run_checked(["pactl", "list", "modules", "short"])
    matches = []
    for line in output.splitlines():
        fields = line.split("\t", 2)
        if len(fields) < 3:
            continue
        if SINK_NAME in fields[2] or SOURCE_NAME in fields[2]:
            try:
                matches.append(int(fields[0]))
            except ValueError:
                pass
    return matches


def unload_modules(module_ids: list[int]) -> None:
    for module_id in reversed(module_ids):
        subprocess.run(
            ["pactl", "unload-module", str(module_id)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def pulse_object_exists(kind: str, name: str) -> bool:
    output = run_checked(["pactl", "list", "short", kind])
    return any(
        len(fields := line.split("\t")) > 1 and fields[1] == name
        for line in output.splitlines()
    )


def create_virtual_source() -> tuple[list[int], bool]:
    sink_exists = pulse_object_exists("sinks", SINK_NAME)
    source_exists = pulse_object_exists("sources", SOURCE_NAME)
    if sink_exists and source_exists:
        run_checked(["pactl", "set-source-volume", SOURCE_NAME, "100%"])
        run_checked(["pactl", "set-source-mute", SOURCE_NAME, "0"])
        return [], True

    # A partial graph is stale and cannot be repaired reliably in place.
    unload_modules(matching_modules())
    sink_id = int(
        run_checked(
            [
                "pactl",
                "load-module",
                "module-null-sink",
                f"sink_name={SINK_NAME}",
                "rate=48000",
                "channels=1",
                "channel_map=mono",
                "sink_properties=device.description=HushMic_AI",
            ]
        )
    )
    try:
        source_id = int(
            run_checked(
                [
                    "pactl",
                    "load-module",
                    "module-remap-source",
                    f"master={SINK_NAME}.monitor",
                    f"source_name={SOURCE_NAME}",
                    "channels=1",
                    "channel_map=mono",
                    "remix=no",
                    "source_properties=device.description=1",
                ]
            )
        )
    except Exception:
        unload_modules([sink_id])
        raise
    run_checked(["pactl", "set-source-volume", SOURCE_NAME, "100%"])
    run_checked(["pactl", "set-source-mute", SOURCE_NAME, "0"])
    return [sink_id, source_id], False


def load_model(
    args: argparse.Namespace,
) -> tuple[
    StreamingBSRNN,
    torch.Tensor,
    torch.Tensor,
    dict[str, object],
]:
    device = torch.device("cuda")
    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    model_args = dict(config["model_args"]["tse_model"])
    model_args["joint_training"] = False
    model_args["spk_model_init"] = None
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_format = checkpoint.get("format")
    model_types = {
        "hushmic-streaming-bsrnn-v1": StreamingBSRNN,
        "hushmic-streaming-bsrnn-multilevel-v1":
            StreamingBSRNNMultiLevel,
        "hushmic-streaming-bsrnn-multilevel-bounded-v1":
            StreamingBSRNNMultiLevel,
        "hushmic-streaming-bsrnn-timbre-v2": StreamingBSRNNTimbreV2,
        "hushmic-streaming-bsrnn-causal-norm-v3":
            StreamingBSRNNCausalNormV3,
        "hushmic-streaming-speaker-tfmlp-v1":
            StreamingSpeakerTFMLPNet,
        "hushmic-streaming-personal-grid-v2":
            StreamingPersonalGridNet,
        "hushmic-streaming-personal-grid-v3":
            StreamingPersonalGridNet,
    }
    model_type = model_types.get(checkpoint_format)
    if model_type is None:
        raise RuntimeError(
            f"Unsupported checkpoint format: {checkpoint_format}"
        )
    if model_type is StreamingBSRNNMultiLevel:
        architecture = checkpoint.get("architecture", {})
        model = model_type(
            **model_args,
            enrollment_tf_frames=int(
                architecture.get("enrollment_tf_frames", 64)
            ),
            enrollment_context_tokens=int(
                architecture.get("enrollment_context_tokens", 32)
            ),
            enrollment_context_dim=int(
                architecture.get("enrollment_context_dim", 512)
            ),
            context_heads=int(architecture.get("context_heads", 4)),
        )
    else:
        model = model_type(**model_args)
    model.load_state_dict(checkpoint["model"])
    embedding = checkpoint["speaker_embedding"].to(device)
    output_gain = checkpoint["output_gain"].to(device)
    model.to(device).eval()
    return model, embedding, output_gain, config


def load_speaker_gate(
    args: argparse.Namespace,
    config: dict[str, object],
    frequency_bins: int,
    separator_embedding: torch.Tensor,
) -> tuple[
    CausalOutputSpeakerVerifier,
    torch.nn.Module,
    torch.Tensor,
    dict[str, object],
]:
    """Load the cheap causal verifier and the slower independent ECAPA."""

    device = separator_embedding.device
    checkpoint = torch.load(
        args.fast_verifier_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("format") != (
        CausalOutputSpeakerVerifier.architecture_version
    ):
        raise RuntimeError(
            "Unsupported fast verifier checkpoint: "
            f"{checkpoint.get('format')}"
        )
    architecture = dict(checkpoint["architecture"])
    if architecture["frequency_bins"] != frequency_bins:
        raise RuntimeError(
            "Fast verifier and separator use different STFT sizes"
        )
    verifier = CausalOutputSpeakerVerifier(
        **architecture,
    )
    verifier.load_state_dict(checkpoint["model"], strict=True)
    verifier.to(device).eval()

    checkpoint_embedding = checkpoint["speaker_embedding"].to(device)
    embedding_similarity = float(
        F.cosine_similarity(
            checkpoint_embedding,
            separator_embedding,
        ).mean()
    )
    if embedding_similarity < 0.999:
        raise RuntimeError(
            "Fast verifier was trained for another enrolled speaker "
            f"(cosine={embedding_similarity:.4f})"
        )

    speaker_model_args = dict(config["model_args"]["tse_model"])
    speaker_model_args["spk_model_init"] = None
    speaker_container = get_model(config["model"]["tse_model"])(
        **speaker_model_args
    )
    speaker_checkpoint = torch.load(
        args.speaker_base_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    speaker_container.load_state_dict(
        speaker_checkpoint["models"][0],
        strict=True,
    )
    del speaker_checkpoint
    speaker_model = speaker_container.spk_model
    speaker_model.to(device).eval()
    for parameter in speaker_model.parameters():
        parameter.requires_grad = False
    del speaker_container
    target_embedding = F.normalize(separator_embedding, dim=-1)
    enrollment_path = getattr(args, "speaker_gate_enrollment", None)
    if enrollment_path:
        enrollment = torch.load(
            enrollment_path,
            map_location="cpu",
            weights_only=False,
        )
        if enrollment.get("format") != "hushmic-speaker-enrollment-v1":
            raise RuntimeError(
                "Unsupported speaker-gate enrollment: "
                f"{enrollment.get('format')}"
            )
        candidate = enrollment["speaker_embedding"].to(device).float()
        if candidate.shape != separator_embedding.shape:
            raise RuntimeError(
                "Speaker-gate enrollment has the wrong shape: "
                f"{tuple(candidate.shape)} != "
                f"{tuple(separator_embedding.shape)}"
            )
        if not torch.isfinite(candidate).all():
            raise RuntimeError("Speaker-gate enrollment is not finite")
        target_embedding = F.normalize(candidate, dim=-1)

    return (
        verifier,
        speaker_model,
        target_embedding,
        checkpoint.get("metrics", {}),
    )


def make_verification_fbank(audio: np.ndarray) -> torch.Tensor:
    waveform = torch.from_numpy(audio).unsqueeze(0) * (1 << 15)
    features = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=STREAM_RATE,
        window_type="hamming",
        use_energy=False,
    )
    return features - features.mean(dim=0, keepdim=True)


def normalize_fast_verifier_input(
    spectrogram: torch.Tensor,
    rms_cap: float,
) -> tuple[torch.Tensor, float, float]:
    """Limit recognition loudness without changing the audible signal."""

    rms_tensor = spectrogram.abs().square().mean().sqrt()
    if rms_cap > 0.0:
        gain_tensor = (
            torch.as_tensor(
                rms_cap,
                device=spectrogram.device,
                dtype=rms_tensor.dtype,
            )
            / rms_tensor.clamp_min(1e-8)
        ).clamp(max=1.0)
    else:
        gain_tensor = torch.ones_like(rms_tensor)
    return (
        spectrogram * gain_tensor,
        float(rms_tensor),
        float(gain_tensor),
    )


class HybridSpeakerGate:
    """Independent identity supervisor for the causal separator output."""

    def __init__(self, args: argparse.Namespace):
        self.ecapa_threshold = args.ecapa_threshold
        self.identity_fast_weight = args.identity_fast_weight
        self.identity_score_threshold = args.identity_score_threshold
        self.identity_close_fast_weight = getattr(
            args,
            "identity_close_fast_weight",
            self.identity_fast_weight,
        )
        self.identity_close_score_threshold = getattr(
            args,
            "identity_close_score_threshold",
            self.identity_score_threshold,
        )
        self.identity_reject_checks = args.identity_reject_checks
        self.fast_window_frames = max(
            1,
            round(args.verification_window_ms / 8.0),
        )
        self.reset()

    def reset(self) -> None:
        # Start open: the cheap separator already suppresses most non-target
        # audio, and this supervisor must not add two seconds to user onset.
        self.is_open = True
        self.ecapa_accepted: bool | None = None
        self.last_fast_probability = 0.0
        self.last_fast_clip_probability = 0.0
        self.last_ecapa_similarity: float | None = None
        self.last_identity_score: float | None = None
        self.last_identity_close_score: float | None = None
        self.reject_count = 0
        self.fast_logits: deque[float] = deque(
            maxlen=self.fast_window_frames
        )

    def update_fast(self, logits: torch.Tensor) -> bool:
        values = logits.detach().float().flatten().cpu().tolist()
        for logit in values:
            self.fast_logits.append(logit)
            probability = 1.0 / (1.0 + math.exp(-logit))
            self.last_fast_probability = probability
        mean_logit = sum(self.fast_logits) / len(self.fast_logits)
        self.last_fast_clip_probability = (
            1.0 / (1.0 + math.exp(-mean_logit))
        )
        return self.is_open

    def update_ecapa(self, similarity: float) -> bool:
        self.last_ecapa_similarity = similarity
        self.last_identity_score = (
            similarity
            + self.identity_fast_weight
            * self.last_fast_clip_probability
        )
        self.last_identity_close_score = (
            similarity
            + self.identity_close_fast_weight
            * self.last_fast_clip_probability
        )
        accepted = (
            similarity >= self.ecapa_threshold
            and self.last_identity_score
            >= self.identity_score_threshold
        )
        # While open, absence of strong target evidence is not enough to mute
        # the user. Close only on repeated, confidently foreign scores. Once
        # closed, retain the stricter target acceptance rule for reopening.
        if self.is_open:
            confidently_foreign = (
                self.last_identity_close_score
                < self.identity_close_score_threshold
            )
            if not confidently_foreign:
                self.reject_count = 0
                self.ecapa_accepted = True
            else:
                self.reject_count += 1
                self.ecapa_accepted = False
                if self.reject_count >= self.identity_reject_checks:
                    self.is_open = False
        elif accepted:
            self.reject_count = 0
            self.ecapa_accepted = True
            self.is_open = True
        else:
            self.reject_count += 1
            if self.reject_count >= self.identity_reject_checks:
                self.ecapa_accepted = False
                self.is_open = False
        return self.is_open

    def accept_short_window(self, similarity: float) -> None:
        self.last_ecapa_similarity = similarity
        self.reject_count = 0
        self.ecapa_accepted = True
        self.is_open = True

    def reset_after_quiet(self, *, preserve_closed: bool = True) -> None:
        was_open = self.is_open
        self.reset()
        if preserve_closed and not was_open:
            # Silence clears stale recurrent evidence but must not undo a
            # verified foreign-speaker rejection. The next target can reopen
            # through the short/full identity checks.
            self.is_open = False
            self.ecapa_accepted = False


class AudioWindow:
    def __init__(self, samples: int):
        if samples <= 0:
            raise ValueError("Verification window must be positive")
        self.samples = samples
        self.blocks: deque[np.ndarray] = deque()
        self.total = 0

    @property
    def full(self) -> bool:
        return self.total >= self.samples

    def clear(self) -> None:
        self.blocks.clear()
        self.total = 0

    def append(self, block: np.ndarray) -> None:
        self.blocks.append(block.copy())
        self.total += block.size
        while (
            len(self.blocks) > 1
            and self.total - self.blocks[0].size >= self.samples
        ):
            self.total -= self.blocks.popleft().size

    def latest(self) -> np.ndarray:
        if not self.full:
            raise RuntimeError("Audio verification window is not full")
        return np.concatenate(tuple(self.blocks))[-self.samples:]


class SmoothedGain:
    def __init__(
        self,
        sample_rate: int,
        attack_ms: float,
        release_ms: float,
        device: torch.device,
    ):
        self.sample_rate = sample_rate
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.device = device
        self.value = 0.0

    def reset(self) -> None:
        self.value = 0.0

    def apply(self, audio: torch.Tensor, opened: bool) -> torch.Tensor:
        target = 1.0 if opened else 0.0
        duration_ms = self.attack_ms if opened else self.release_ms
        if duration_ms <= 0.0:
            self.value = target
            return audio * target
        coefficient = np.exp(
            -1000.0 / (duration_ms * self.sample_rate)
        )
        indexes = torch.arange(
            1,
            audio.numel() + 1,
            device=self.device,
            dtype=audio.dtype,
        )
        curve = target + (self.value - target) * torch.pow(
            torch.as_tensor(
                coefficient,
                device=self.device,
                dtype=audio.dtype,
            ),
            indexes,
        )
        self.value = target + (self.value - target) * (
            coefficient ** audio.numel()
        )
        return audio * curve


class StreamingIstft:
    def __init__(self, window: torch.Tensor, stride: int):
        self.window = window
        self.window_square = window.square()
        self.stride = stride
        self.audio = torch.zeros(
            1, window.numel(), device=window.device
        )
        self.weight = torch.zeros_like(self.audio)

    def push(self, spec: torch.Tensor) -> torch.Tensor:
        frames = torch.fft.irfft(
            spec.transpose(1, 2),
            n=self.window.numel(),
            dim=-1,
        )
        pieces = []
        for frame_index in range(frames.shape[1]):
            self.audio += frames[:, frame_index] * self.window
            self.weight += self.window_square
            pieces.append(
                self.audio[:, :self.stride]
                / self.weight[:, :self.stride].clamp_min(1e-8)
            )
            zeros = torch.zeros(
                1, self.stride, device=self.audio.device
            )
            self.audio = torch.cat(
                (self.audio[:, self.stride:], zeros),
                dim=-1,
            )
            self.weight = torch.cat(
                (self.weight[:, self.stride:], zeros.clone()),
                dim=-1,
            )
        return torch.cat(pieces, dim=-1)


def write_status(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"updated_at": time.time(), **values}, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_exact(stream, size: int) -> bytes:
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        count = stream.readinto(view[offset:])
        if not count:
            raise EOFError(f"PipeWire capture ended after {offset}/{size} bytes")
        offset += count
    return bytes(data)


def queue_latest(target: queue.Queue, item: object) -> int:
    dropped = 0
    while True:
        try:
            target.put(item, timeout=0.1)
            return dropped
        except queue.Full:
            # Inference is much faster than real time. If the graph was
            # suspended long enough to fill the queue, discard stale audio
            # instead of allowing latency to grow without bound.
            try:
                target.get_nowait()
                dropped += 1
            except queue.Empty:
                pass


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    status_path = Path(args.status_file)
    write_status(status_path, state="loading", input=args.input_source)
    started = time.time()
    model, embedding, output_gain, config = load_model(args)
    chunk_samples = STREAM_RATE * args.chunk_ms // 1000
    if chunk_samples <= 0 or chunk_samples % model.stride:
        raise ValueError(
            "--chunk-ms must produce a positive whole number of STFT hops"
        )
    device = torch.device("cuda")
    fast_verifier = None
    speaker_model = None
    target_embedding = None
    verifier_metrics: dict[str, object] = {}
    if args.speaker_gate:
        if args.verification_window_ms < 250:
            raise ValueError("--verification-window-ms must be at least 250")
        if args.verification_hop_ms <= 0:
            raise ValueError("--verification-hop-ms must be positive")
        if not (
            250
            <= args.short_accept_window_ms
            <= args.verification_window_ms
        ):
            raise ValueError(
                "--short-accept-window-ms must be between 250 and "
                "--verification-window-ms"
            )
        if args.gate_quiet_reset_ms <= 0:
            raise ValueError("--gate-quiet-reset-ms must be positive")
        if args.identity_reject_checks <= 0:
            raise ValueError("--identity-reject-checks must be positive")
        (
            fast_verifier,
            speaker_model,
            target_embedding,
            verifier_metrics,
        ) = load_speaker_gate(
            args,
            config,
            model.win // 2 + 1,
            embedding,
        )
    window = torch.hamming_window(
        model.win,
        device=device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        warmup = torch.zeros(
            1,
            model.win - model.stride + chunk_samples,
            device=device,
        )
        warmup_spec = torch.stft(
            warmup,
            n_fft=model.win,
            hop_length=model.stride,
            window=window,
            center=False,
            return_complex=True,
        )
        warmup_estimated, _warmup_state = model.forward_spectrogram(
            warmup_spec,
            embedding,
            None,
        )
        if fast_verifier is not None:
            warmup_verifier_input, _warmup_rms, _warmup_gain = (
                normalize_fast_verifier_input(
                    warmup_estimated * output_gain,
                    args.fast_verifier_rms_cap,
                )
            )
            fast_verifier(
                warmup_verifier_input,
                embedding,
                None,
            )
        torch.cuda.synchronize()

    module_ids, virtual_source_reused = create_virtual_source()
    previous_default = None
    if args.set_default:
        previous_default = run_checked(["pactl", "get-default-source"])
        run_checked(["pactl", "set-default-source", SOURCE_NAME])

    stop = threading.Event()
    errors: queue.Queue[BaseException] = queue.Queue()
    captured: queue.Queue[bytes] = queue.Queue(maxsize=4)
    processed: queue.Queue[bytes] = queue.Queue(maxsize=4)
    processes: list[subprocess.Popen] = []
    capture_drops = 0
    playback_drops = 0
    state_resets = 0

    def cleanup() -> None:
        stop.set()
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        if args.set_default and previous_default:
            current = subprocess.run(
                ["pactl", "get-default-source"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout.strip()
            if current == SOURCE_NAME:
                subprocess.run(
                    ["pactl", "set-default-source", previous_default],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        if args.remove_virtual_source_on_exit:
            unload_modules(
                module_ids if module_ids else matching_modules()
            )
        write_status(status_path, state="stopped")

    atexit.register(cleanup)

    def request_stop(_signum, _frame) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    common = [
        "--raw",
        "--rate",
        str(STREAM_RATE),
        "--channels",
        "1",
        "--channel-map",
        "MONO",
        "--format",
        "f32",
        "--latency",
        "20ms",
    ]
    capture_process = subprocess.Popen(
        [
            "pw-cat",
            "--record",
            "--target",
            args.input_source,
            *common,
            "-",
        ],
        stdout=subprocess.PIPE,
        bufsize=0,
    )
    playback_process = subprocess.Popen(
        [
            "pw-cat",
            "--playback",
            "--target",
            SINK_NAME,
            *common,
            "-",
        ],
        stdin=subprocess.PIPE,
        bufsize=0,
    )
    processes.extend([capture_process, playback_process])

    def capture_worker() -> None:
        nonlocal capture_drops
        try:
            assert capture_process.stdout is not None
            while not stop.is_set():
                capture_drops += queue_latest(
                    captured,
                    read_exact(
                        capture_process.stdout,
                        chunk_samples * np.dtype(np.float32).itemsize,
                    ),
                )
        except BaseException as error:
            if not stop.is_set():
                errors.put(error)
                stop.set()

    def playback_worker() -> None:
        try:
            assert playback_process.stdin is not None
            while not stop.is_set():
                try:
                    block = processed.get(timeout=0.25)
                except queue.Empty:
                    continue
                playback_process.stdin.write(block)
                playback_process.stdin.flush()
        except BaseException as error:
            if not stop.is_set():
                errors.put(error)
                stop.set()

    threading.Thread(target=capture_worker, name="capture", daemon=True).start()
    threading.Thread(target=playback_worker, name="playback", daemon=True).start()

    static_status = {
        "input": args.input_source,
        "output": SOURCE_NAME,
        "checkpoint": args.checkpoint,
        "chunk_ms": args.chunk_ms,
        "stft_delay_ms": (
            (model.win - model.stride) * 1000 / STREAM_RATE
        ),
        "model_load_seconds": time.time() - started,
        "output_gain": float(output_gain),
        "virtual_source_reused": virtual_source_reused,
        "speaker_gate_enabled": args.speaker_gate,
    }
    if args.speaker_gate:
        static_status.update(
            {
                "fast_verifier_checkpoint":
                    args.fast_verifier_checkpoint,
                "speaker_gate_enrollment":
                    args.speaker_gate_enrollment,
                "fast_verifier_rms_cap":
                    args.fast_verifier_rms_cap,
                "ecapa_threshold": args.ecapa_threshold,
                "identity_fast_weight": args.identity_fast_weight,
                "identity_score_threshold":
                    args.identity_score_threshold,
                "identity_close_fast_weight":
                    args.identity_close_fast_weight,
                "identity_close_score_threshold":
                    args.identity_close_score_threshold,
                "identity_reject_checks":
                    args.identity_reject_checks,
                "verification_window_ms":
                    args.verification_window_ms,
                "verification_hop_ms": args.verification_hop_ms,
                "short_accept_window_ms":
                    args.short_accept_window_ms,
                "short_accept_threshold":
                    args.short_accept_threshold,
                "short_accept_fast_threshold":
                    args.short_accept_fast_threshold,
                "gate_quiet_reset_ms": args.gate_quiet_reset_ms,
                "verifier_validation_balanced_accuracy":
                    verifier_metrics.get(
                        "balanced_accuracy",
                        verifier_metrics.get("combined", {}).get(
                            "balanced_accuracy"
                        ),
                    ),
            }
        )
    write_status(
        status_path,
        state="running",
        **static_status,
    )
    print(
        f"[hushmic-personal] ready: {args.input_source} -> {SOURCE_NAME}; "
        f"chunk={args.chunk_ms}ms gain={float(output_gain):.6f}",
        flush=True,
    )

    block_count = 0
    total_inference = 0.0
    inference_window: deque[float] = deque(
        maxlen=STATUS_INTERVAL_BLOCKS
    )
    input_rms_ema = 0.0
    output_rms_ema = 0.0
    input_peak_window = 0.0
    output_peak_window = 0.0
    gate_controller = HybridSpeakerGate(args) if args.speaker_gate else None
    gate_gain = (
        SmoothedGain(
            STREAM_RATE,
            args.gate_attack_ms,
            args.gate_release_ms,
            device,
        )
        if args.speaker_gate
        else None
    )
    verification_audio = (
        AudioWindow(
            round(args.verification_window_ms * STREAM_RATE / 1000)
        )
        if args.speaker_gate
        else None
    )
    verification_input_audio = (
        AudioWindow(
            round(args.verification_window_ms * STREAM_RATE / 1000)
        )
        if args.speaker_gate
        else None
    )
    verification_hop_samples = round(
        args.verification_hop_ms * STREAM_RATE / 1000
    )
    verification_pending_samples = 0
    verification_window_ready = False
    verification_first_check_pending = True
    verification_checks = 0
    verification_short_checks = 0
    verification_quiet_skips = 0
    verification_target_absent_rejections = 0
    quiet_output_samples = 0
    closed_short_next = True
    last_verification_kind: str | None = None
    ecapa_compute_window: deque[float] = deque(maxlen=32)
    gate_open_chunks = 0
    fast_verifier_input_rms = 0.0
    fast_verifier_input_gain = 1.0
    analysis_history = torch.zeros(
        1,
        model.win - model.stride,
        device=device,
    )
    synthesis = StreamingIstft(window, model.stride)
    states = None
    fast_state = None
    while not stop.is_set():
        if not errors.empty():
            raise errors.get()
        if capture_process.poll() is not None:
            raise RuntimeError(
                f"pw-cat capture exited with {capture_process.returncode}"
            )
        if playback_process.poll() is not None:
            raise RuntimeError(
                f"pw-cat playback exited with {playback_process.returncode}"
            )
        try:
            raw = captured.get(timeout=0.25)
        except queue.Empty:
            continue
        input_array = np.frombuffer(raw, dtype=np.float32).copy()
        input_rms = float(np.sqrt(np.mean(np.square(input_array))))
        input_peak = float(np.max(np.abs(input_array)))
        new_audio = torch.from_numpy(
            input_array
        ).unsqueeze(0).to(device, non_blocking=True)
        analysis = torch.cat(
            (analysis_history, new_audio),
            dim=-1,
        )
        analysis_history = analysis[:, -(
            model.win - model.stride
        ):]
        spec = torch.stft(
            analysis,
            n_fft=model.win,
            hop_length=model.stride,
            window=window,
            center=False,
            return_complex=True,
        )
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            estimated_spec, states = model.forward_spectrogram(
                spec,
                embedding,
                states,
            )
            estimated_spec *= output_gain
            if fast_verifier is not None:
                (
                    verifier_input,
                    fast_verifier_input_rms,
                    fast_verifier_input_gain,
                ) = normalize_fast_verifier_input(
                    estimated_spec,
                    args.fast_verifier_rms_cap,
                )
                fast_logits, fast_state = fast_verifier(
                    verifier_input,
                    embedding,
                    fast_state,
                )
                assert gate_controller is not None
                gate_controller.update_fast(fast_logits)
            raw_output = synthesis.push(estimated_spec)[0]

        if gate_controller is not None:
            assert speaker_model is not None
            assert target_embedding is not None
            assert verification_audio is not None
            assert verification_input_audio is not None
            assert gate_gain is not None
            raw_output_array = (
                raw_output.detach().float().cpu().numpy()
            )
            raw_block_rms = float(
                np.sqrt(np.mean(np.square(raw_output_array)))
            )
            raw_block_dbfs = 20.0 * np.log10(
                max(raw_block_rms, 1e-9)
            )
            input_block_dbfs = 20.0 * np.log10(max(input_rms, 1e-9))
            if (
                raw_block_dbfs < args.verification_min_rms_dbfs
                and input_block_dbfs < args.verification_min_rms_dbfs
            ):
                quiet_output_samples += raw_output_array.size
            else:
                quiet_output_samples = 0
            if (
                not gate_controller.is_open
                and quiet_output_samples
                >= round(args.gate_quiet_reset_ms * STREAM_RATE / 1000)
            ):
                gate_controller.reset_after_quiet(preserve_closed=True)
                # Reset recurrent identity evidence at the same boundary as
                # the gate/logit history so a previous speaker cannot leak
                # into the next utterance.
                fast_state = None
                closed_short_next = True
            verification_audio.append(raw_output_array)
            verification_input_audio.append(input_array)
            if verification_window_ready:
                verification_pending_samples += raw_output_array.size
            elif verification_audio.full:
                verification_window_ready = True
            if (
                verification_audio.full
                and (
                    verification_first_check_pending
                    or verification_pending_samples
                    >= verification_hop_samples
                )
            ):
                verification_pending_samples = 0
                verification_first_check_pending = False
                verification_block = verification_audio.latest()
                verification_input_block = verification_input_audio.latest()
                use_short_window = (
                    not gate_controller.is_open
                    and closed_short_next
                )
                if use_short_window:
                    short_samples = round(
                        args.short_accept_window_ms
                        * STREAM_RATE
                        / 1000
                    )
                    verification_block = verification_block[
                        -short_samples:
                    ]
                    verification_input_block = verification_input_block[
                        -short_samples:
                    ]
                verification_rms = float(
                    np.sqrt(np.mean(np.square(verification_block)))
                )
                verification_dbfs = 20.0 * np.log10(
                    max(verification_rms, 1e-9)
                )
                verification_input_rms = float(
                    np.sqrt(np.mean(np.square(verification_input_block)))
                )
                verification_input_dbfs = 20.0 * np.log10(
                    max(verification_input_rms, 1e-9)
                )
                if verification_dbfs >= args.verification_min_rms_dbfs:
                    ecapa_started = time.perf_counter()
                    fbank = make_verification_fbank(
                        verification_block
                    )
                    with torch.inference_mode():
                        result = speaker_model(
                            fbank.unsqueeze(0).to(
                                device,
                                non_blocking=True,
                            )
                        )
                        output_embedding = (
                            result[-1]
                            if isinstance(result, tuple)
                            else result
                        )
                        similarity = float(
                            (
                                F.normalize(output_embedding, dim=-1)
                                * target_embedding
                            ).sum()
                        )
                    torch.cuda.synchronize()
                    ecapa_compute_window.append(
                        (time.perf_counter() - ecapa_started) * 1000.0
                    )
                    verification_checks += 1
                    if use_short_window:
                        verification_short_checks += 1
                        last_verification_kind = "short_accept"
                        if (
                            similarity >= args.short_accept_threshold
                            or (
                                similarity >= args.ecapa_threshold
                                and gate_controller
                                .last_fast_clip_probability
                                >= args.short_accept_fast_threshold
                            )
                        ):
                            gate_controller.accept_short_window(
                                similarity
                            )
                            closed_short_next = True
                        else:
                            closed_short_next = False
                    else:
                        last_verification_kind = "full_identity"
                        gate_controller.update_ecapa(similarity)
                        closed_short_next = (
                            not gate_controller.is_open
                        )
                elif (
                    verification_input_dbfs
                    < args.verification_min_rms_dbfs
                ):
                    # Reset only on real input silence. A foreign speaker can
                    # be quiet at the separator output and must not reopen it.
                    gate_controller.reset_after_quiet(preserve_closed=True)
                    fast_state = None
                    closed_short_next = True
                    verification_quiet_skips += 1
                else:
                    # Input is active but the separator output is already
                    # inaudible. There is nothing for the gate to leak, so do
                    # not latch an open gate closed between target phrases.
                    # A gate that was already closed stays closed and keeps
                    # alternating short/full checks until target evidence
                    # actually appears.
                    last_verification_kind = "input_active_target_absent"
                    verification_target_absent_rejections += 1
                    if gate_controller.is_open:
                        gate_controller.reject_count = 0
                        gate_controller.ecapa_accepted = None
                        gate_controller.last_identity_score = None
                        closed_short_next = True
                    else:
                        closed_short_next = not use_short_window
            output = gate_gain.apply(
                raw_output,
                gate_controller.is_open,
            )
            gate_open_chunks += int(gate_controller.is_open)
        else:
            output = raw_output
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        if not bool(torch.isfinite(output).all()):
            states = None
            analysis_history.zero_()
            synthesis = StreamingIstft(window, model.stride)
            output = torch.zeros_like(output)
            fast_state = None
            if gate_controller is not None:
                gate_controller.reset()
            if gate_gain is not None:
                gate_gain.reset()
            if verification_audio is not None:
                verification_audio.clear()
            verification_pending_samples = 0
            verification_window_ready = False
            verification_first_check_pending = True
            quiet_output_samples = 0
            closed_short_next = True
            state_resets += 1
        output_array = (
            output.cpu()
            .clamp(-0.999, 0.999)
            .numpy()
            .astype(np.float32)
        )
        output_rms = float(np.sqrt(np.mean(np.square(output_array))))
        output_peak = float(np.max(np.abs(output_array)))
        playback_drops += queue_latest(processed, output_array.tobytes())
        block_count += 1
        total_inference += inference_seconds
        inference_window.append(inference_seconds * 1000.0)
        smoothing = 0.04
        input_rms_ema += smoothing * (input_rms - input_rms_ema)
        output_rms_ema += smoothing * (output_rms - output_rms_ema)
        input_peak_window = max(input_peak_window, input_peak)
        output_peak_window = max(output_peak_window, output_peak)
        if block_count % STATUS_INTERVAL_BLOCKS == 0:
            mean_ms = total_inference * 1000.0 / block_count
            p95_ms = float(np.percentile(inference_window, 95))
            max_ms = max(inference_window)
            db_floor = 1e-9
            write_status(
                status_path,
                state="running",
                **static_status,
                chunks=block_count,
                mean_inference_ms=mean_ms,
                recent_p95_inference_ms=p95_ms,
                recent_max_inference_ms=max_ms,
                realtime_headroom_ms=args.chunk_ms - p95_ms,
                input_rms_dbfs=20.0 * np.log10(
                    max(input_rms_ema, db_floor)
                ),
                output_rms_dbfs=20.0 * np.log10(
                    max(output_rms_ema, db_floor)
                ),
                input_peak_dbfs=20.0 * np.log10(
                    max(input_peak_window, db_floor)
                ),
                output_peak_dbfs=20.0 * np.log10(
                    max(output_peak_window, db_floor)
                ),
                capture_queue_drops=capture_drops,
                playback_queue_drops=playback_drops,
                state_resets=state_resets,
                capture_queue_depth=captured.qsize(),
                playback_queue_depth=processed.qsize(),
                gpu_memory_mb=torch.cuda.memory_allocated() / (1024 * 1024),
                gate_open=(
                    gate_controller.is_open
                    if gate_controller is not None
                    else None
                ),
                gate_open_fraction=(
                    gate_open_chunks / block_count
                    if gate_controller is not None
                    else None
                ),
                fast_probability=(
                    gate_controller.last_fast_probability
                    if gate_controller is not None
                    else None
                ),
                fast_verifier_input_rms=(
                    fast_verifier_input_rms
                    if gate_controller is not None
                    else None
                ),
                fast_verifier_input_gain=(
                    fast_verifier_input_gain
                    if gate_controller is not None
                    else None
                ),
                fast_clip_probability=(
                    gate_controller.last_fast_clip_probability
                    if gate_controller is not None
                    else None
                ),
                ecapa_similarity=(
                    gate_controller.last_ecapa_similarity
                    if gate_controller is not None
                    else None
                ),
                ecapa_accepted=(
                    gate_controller.ecapa_accepted
                    if gate_controller is not None
                    else None
                ),
                identity_score=(
                    gate_controller.last_identity_score
                    if gate_controller is not None
                    else None
                ),
                identity_close_score=(
                    gate_controller.last_identity_close_score
                    if gate_controller is not None
                    else None
                ),
                identity_reject_count=(
                    gate_controller.reject_count
                    if gate_controller is not None
                    else None
                ),
                ecapa_checks=verification_checks,
                ecapa_short_checks=verification_short_checks,
                ecapa_quiet_skips=verification_quiet_skips,
                target_absent_rejections=(
                    verification_target_absent_rejections
                ),
                last_verification_kind=last_verification_kind,
                recent_mean_ecapa_ms=(
                    float(np.mean(ecapa_compute_window))
                    if ecapa_compute_window
                    else None
                ),
                recent_max_ecapa_ms=(
                    max(ecapa_compute_window)
                    if ecapa_compute_window
                    else None
                ),
            )
            print(
                f"[hushmic-personal] blocks={block_count} "
                f"mean_cuda={mean_ms:.1f}ms p95={p95_ms:.1f}ms "
                f"drops={capture_drops}/{playback_drops}",
                flush=True,
            )
            input_peak_window = 0.0
            output_peak_window = 0.0

    if not errors.empty():
        raise errors.get()


if __name__ == "__main__":
    main()
