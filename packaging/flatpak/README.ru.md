# HushMic как Flatpak

[English version](README.md)

Манифест, PipeWire-compat patch, AppStream-метаданные и список crate находятся
в каталоге `packaging/flatpak`. Старый app-id сохранён ради совместимости с уже
установленным приложением.

```bash
flatpak-builder --user --install --force-clean build \
  packaging/flatpak/io.github.fovty.HushMic.yml
flatpak run io.github.fovty.HushMic
```
