# Contributing

[Русская версия](CONTRIBUTING.ru.md)

HushMic-Ahuenno is a Linux audio project. Small fixes, reproducible bug
reports, PipeWire compatibility improvements and latency measurements are
welcome.

Before opening a pull request:

1. Run `git diff --check`.
2. Run the relevant Rust tests when a Rust toolchain is available.
3. Do not add private recordings, speaker embeddings, credentials or local
   absolute paths.
4. For audio changes, describe the sample rate, block size, PipeWire version
   and whether the test used CPU or CUDA.

Experimental speaker and phrase-trigger code must remain opt-in and must not
silently replace the default microphone path.
