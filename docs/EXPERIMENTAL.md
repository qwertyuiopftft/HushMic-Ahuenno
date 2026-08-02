# Experimental extensions

[Русская версия](EXPERIMENTAL.ru.md)

The main HushMic runtime is the Rust DPDFNet denoiser described in the main
README. This repository also contains optional research scripts used to test
personal-speaker extraction.

These scripts are not enabled by the normal installer and are intentionally
kept separate from the default audio path:

- `scripts/hushmic_personal_gpu.py`: a streaming PyTorch/WeSep speaker
  separator. It needs a local WeSep checkout, a compatible checkpoint and a
  CUDA-capable PyTorch installation. Set `WESEP_ROOT` and
  `WESPEAKER_ROOT`; all model and enrollment paths are explicit CLI options.
- `scripts/hushmic_calibrate`: a local-only recorder for synchronized raw,
  DPDFNet and personalized-source samples. Set `HUSHMIC_RAW_SOURCE` and,
  when needed, `HUSHMIC_PERSONAL_SOURCE`; recordings default to the user's
  state directory and are never written to the repository.

Personal recordings, speaker embeddings, checkpoints, evaluation renders and
machine-specific service files are deliberately excluded from Git. Pass all
paths explicitly when running the experimental scripts; do not commit private
audio or model weights without checking their licence and privacy first.

The speaker gate is opt-in. It combines a short-window verifier with an
enrollment embedding and may reduce non-target speech while keeping uncertain
or quiet target speech open. Complete removal is not guaranteed. Thresholds are
deliberately configurable because microphones and rooms vary substantially.
