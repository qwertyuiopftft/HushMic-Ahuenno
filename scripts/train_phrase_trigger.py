#!/usr/bin/env python3
"""Train a speaker-specific neural keyword detector without speech-to-text."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 16_000
CLIP_SAMPLES = SAMPLE_RATE


def read_audio(path: Path) -> torch.Tensor:
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE:
        raise RuntimeError(f"{path}: expected {SAMPLE_RATE} Hz, got {rate}")
    return torch.as_tensor(audio).flatten().contiguous()


def speech_segments(audio: torch.Tensor) -> list[tuple[int, int]]:
    """Energy VAD for the deliberately separated trigger repetitions."""
    frame, hop = 320, 160
    if audio.numel() < frame:
        return []
    energy = audio.unfold(0, frame, hop).square().mean(-1).sqrt()
    active = energy > 0.006
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            left = max(0, start * hop - 960)
            right = min(audio.numel(), index * hop + 960)
            duration = (right - left) / SAMPLE_RATE
            if 0.18 <= duration <= 1.5:
                spans.append((left, right))
            start = None
    return spans


def crop(audio: torch.Tensor, left: int, length: int = CLIP_SAMPLES) -> torch.Tensor:
    right = left + length
    output = torch.zeros(length)
    source_left = max(0, left)
    source_right = min(audio.numel(), right)
    if source_right > source_left:
        output[source_left - left:source_right - left] = audio[source_left:source_right]
    return output


def positive_crop(audio: torch.Tensor, segment: tuple[int, int], rng: random.Random) -> torch.Tensor:
    left, right = segment
    center = (left + right) // 2
    jitter = rng.randint(-2800, 2800)
    return crop(audio, center - CLIP_SAMPLES // 2 + jitter)


def augment(audio: torch.Tensor, rng: random.Random) -> torch.Tensor:
    shift = rng.randint(-1200, 1200)
    audio = torch.roll(audio, shift)
    audio = audio * rng.uniform(0.70, 1.35)
    if rng.random() < 0.8:
        noise = torch.randn_like(audio) * rng.uniform(0.0003, 0.004)
        audio = audio + noise
    return audio.clamp(-1.0, 1.0)


class PhraseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        positive: torch.Tensor,
        positive_segments: list[tuple[int, int]],
        negative: torch.Tensor,
        length: int,
        seed: int,
        augment_enabled: bool,
    ) -> None:
        self.positive = positive
        self.positive_segments = positive_segments
        self.negative = negative
        self.length = length
        self.seed = seed
        self.augment_enabled = augment_enabled

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rng = random.Random(self.seed + index * 10_007)
        positive = index % 2 == 0
        if positive:
            audio = positive_crop(self.positive, rng.choice(self.positive_segments), rng)
            label = 1
        else:
            start = rng.randrange(max(1, self.negative.numel() - CLIP_SAMPLES))
            audio = crop(self.negative, start)
            label = 0
        if self.augment_enabled:
            audio = augment(audio, rng)
        return audio, label


class TinyPhraseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mel = torch.nn.Sequential()
        self.register_buffer("window", torch.hann_window(400), persistent=False)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(48, 2),
        )

    def features(self, audio: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(
            audio,
            n_fft=400,
            hop_length=160,
            win_length=400,
            window=self.window.to(audio.device),
            return_complex=True,
        ).abs().square().clamp_min(1e-7).log()
        # 201 FFT bins -> 40 robust frequency bands without external dependencies.
        bands = F.adaptive_avg_pool2d(spectrum.unsqueeze(1), (40, spectrum.shape[-1]))
        return bands

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(audio))


@torch.no_grad()
def probabilities(model: TinyPhraseNet, dataset: PhraseDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    values, labels = [], []
    model.eval()
    for audio, label in loader:
        values.append(model(audio.to(device)).softmax(-1)[:, 1].cpu())
        labels.append(label)
    return torch.cat(values), torch.cat(labels)


def choose_threshold(probability: torch.Tensor, label: torch.Tensor) -> tuple[float, dict[str, float]]:
    best: tuple[float, dict[str, float]] | None = None
    for threshold in torch.linspace(0.50, 0.995, 100):
        predicted = probability >= threshold
        tp = int((predicted & (label == 1)).sum())
        fp = int((predicted & (label == 0)).sum())
        fn = int((~predicted & (label == 1)).sum())
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        score = f1 - 0.25 * (fp / max(1, int((label == 0).sum())))
        metrics = {"threshold": float(threshold), "recall": recall, "precision": precision, "f1": f1, "false_positives": fp}
        if best is None or score > best[0]:
            best = (score, metrics)
    assert best is not None
    return best[1]["threshold"], best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-train", required=True)
    parser.add_argument("--positive-validation", required=True)
    parser.add_argument("--negative-train", required=True)
    parser.add_argument("--negative-validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_train = read_audio(Path(args.positive_train)); pos_val = read_audio(Path(args.positive_validation))
    neg_train = read_audio(Path(args.negative_train)); neg_val = read_audio(Path(args.negative_validation))
    train_segments = speech_segments(pos_train); val_segments = speech_segments(pos_val)
    if len(train_segments) < 10 or len(val_segments) < 10:
        raise RuntimeError(f"not enough positive repetitions: train={len(train_segments)}, validation={len(val_segments)}")
    train_set = PhraseDataset(pos_train, train_segments, neg_train, 4096, args.seed, True)
    val_set = PhraseDataset(pos_val, val_segments, neg_val, 2048, args.seed + 1, False)
    loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    model = TinyPhraseNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_threshold = 0.9
    best_metrics: dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for audio, label in loader:
            logits = model(audio.to(device)); loss = F.cross_entropy(logits, label.to(device))
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); optimizer.step()
            losses.append(loss.item())
        probability, label = probabilities(model, val_set, device)
        threshold, metrics = choose_threshold(probability, label)
        validation_score = metrics["f1"] - 0.25 * metrics["false_positives"] / 1024
        if validation_score > best_score:
            best_score = validation_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_threshold = threshold
            best_metrics = metrics
        print(json.dumps({"epoch": epoch, "loss": sum(losses) / len(losses), **metrics}), flush=True)
    assert best_state is not None
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "hushmic-tiny-phrase-net-v1", "sample_rate": SAMPLE_RATE,
        "clip_samples": CLIP_SAMPLES, "phrase": "иди нахуй", "threshold": best_threshold,
        "model": best_state,
        "validation": best_metrics,
        "positive_train_repetitions": len(train_segments), "positive_validation_repetitions": len(val_segments),
    }, output)
    print(json.dumps({"event": "saved", "path": str(output), "device": str(device), "validation": metrics}), flush=True)


if __name__ == "__main__":
    main()
