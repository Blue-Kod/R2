# Orbbec SDK v1.10.37 (runtime)

Библиотеки Orbbec SDK v1 для камер семейства Astra (OpenNI-протокол):
`libOrbbecSDK.so`, `libob_usb.so`, `liblive555.so`.

Источник: https://github.com/orbbec/OrbbecSDK/releases/tag/v1.10.37
(архивы `OrbbecSDK_C_C++_..._linux_x64_release.zip` и `..._linux_arm64_release.zip`).
Лицензия — Apache 2.0, см. `LICENSE.txt`.

`robov_core/orbbec_camera.py` загружает их через ctypes, поэтому сборка
pybind11 и привязка к версии Python не нужны. Каталог выбирается по
`platform.machine()`: `x64` для x86_64, `arm64` для aarch64 (Orange Pi).

Права на USB и `usbfs_memory_mb=128`:

```bash
sudo tools/setup_orbbec.sh
```

Проверка без запуска панели:

```bash
.venv/bin/python tools/test_orbbec_camera.py
```
