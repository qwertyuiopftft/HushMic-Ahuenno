# hushmic-denoiser

[English version](README.md)

Переиспользуемая Rust-библиотека потокового шумоподавления DPDFNet: на вход
подаётся mono 48 кГц, на выходе — очищенный поток.

```rust
let denoiser = Denoiser::from_file("dpdfnet8_48khz_hr.onnx")?;
let cleaned = denoiser.process(&input_frame)?;
```

Модели DPDFNet работают с 48 кГц mono. `dpdfnet8` даёт лучшее качество,
`dpdfnet2` легче для слабого CPU. ONNX Runtime загружается динамически; путь
можно задать через `ORT_DYLIB_PATH`.

Подробности ручного filter-chain: [русские заметки](../dpdfnet-ladspa/examples/run-filter-chain.ru.md).

Лицензия: MIT или Apache-2.0, на выбор.
