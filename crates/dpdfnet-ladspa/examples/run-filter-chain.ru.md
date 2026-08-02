# Запуск DPDFNet filter-chain

[English version](run-filter-chain.md)

Это заметки для ручного запуска LADSPA-фильтра через PipeWire. Обычному
пользователю достаточно `hushmic`; этот документ нужен для отладки.

Нужны `libdpdfnet_ladspa.so`, ONNX Runtime, модель `dpdfnet8` или более лёгкая
`dpdfnet2` и фрагмент `live/dpdfnet-mono.conf`.

```bash
export HUSHMIC_MODEL_PATH="$HOME/hushmic-rt/models/dpdfnet8_48khz_hr.onnx"
export ORT_DYLIB_PATH="$HOME/hushmic-rt/lib/libonnxruntime.so"
```

Путь к `.so` в конфигурации PipeWire должен совпадать с фактическим путём.
Модель можно заменить без пересборки через `HUSHMIC_MODEL_PATH`. Если CPU не
успевает за realtime, попробуй `dpdfnet2`.
