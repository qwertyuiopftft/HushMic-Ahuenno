# Training and evaluation notes

[Русская версия](TRAINING.ru.md)

HushMic-Ahuenno has three distinct model layers:

1. **DPDFNet denoiser** — a causal speech-enhancement model adapted for the
   48 kHz streaming path.
2. **Speaker personalization** — an optional separator/verifier branch trained
   with enrollment material and evaluated with target/foreign/noise windows.
3. **Phrase trigger** — a small, deliberately unserious detector trained with
   positive and negative examples.

The fine-tuned DPDFNet checkpoint is documented in
[`MODEL_CARD.md`](MODEL_CARD.md). Its private training recordings are not
redistributed. Aggregate results are in [`BENCHMARKS.md`](BENCHMARKS.md).

When creating a new personal model:

- keep raw recordings outside the repository;
- record consent for every additional speaker;
- separate training and validation clips;
- test quiet starts, loud speech, long pauses and overlapping speakers;
- report false positives, missed target speech and p95 processing time;
- keep the default HushMic path usable if the personal worker is unavailable.

Do not treat a high score on one room or one microphone as evidence of general
speaker recognition quality.
