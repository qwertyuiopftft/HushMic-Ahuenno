# Development benchmarks

These numbers summarize private, offline development tests for the optional
speaker-personalization path. They are included to show the engineering trade-
offs, not to present a universal speech benchmark. The recordings,
checkpoints, manifests and machine-specific paths are not published.

## Speaker-gate validation

| Test | Target recall | False-positive rate | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Calibrated conjunction, 160 windows | 92.86% | 0% | 96.43% |
| Independent clean window validation, 112 target / 48 inactive windows | 91.07% | 0% | 95.54% |

The zero-false-positive operating point is intentionally conservative: a lower
false-positive rate can reject more short or unusual target-speaker speech.
These are private-dataset results and should not be read as a guarantee for a
different microphone, room or speaker.

## Streaming soak test

One offline 10-minute run processed 12,118 80-ms chunks without a realtime
deadline miss, NaN/Inf event or exception. Measured separator + verifier time:

| Metric | Result |
| --- | ---: |
| Mean per-chunk compute | 48.07 ms |
| p95 per-chunk compute | 49.35 ms |
| Realtime deadline | 80 ms |
| Processed audio | 969.44 s |

The soak test validates state handling and timing on one CUDA setup; it is not
a promise of the same latency on every GPU or CPU.

## Known limitations

- The tests use private recordings and cannot be reproduced from this repo
  alone.
- Audio quality still depends on microphone placement, gain, room acoustics and
  the amount of overlap between speakers.
