#!/bin/sh
# Один раз после подключения Orbbec Astra: udev-права и буфер USB.
# Нужен root (sudo).
set -e
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cp "$ROOT/robov_core/vendor/orbbec/99-obsensor-libusb.rules" \
   /etc/udev/rules.d/99-obsensor-libusb.rules
udevadm control --reload-rules
udevadm trigger
echo 128 > /sys/module/usbcore/parameters/usbfs_memory_mb
echo "Готово. Если камера уже была подключена — переподключите её."
