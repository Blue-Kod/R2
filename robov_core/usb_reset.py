"""Best-effort software re-enumeration of the USB camera.

Когда камера «отваливается» (обычно из-за просадки питания при движении
серво), помогает логический «реплаг» — заставить хост заново проинициализировать
устройство, не выдёргивая кабель. Пробуем по порядку, первый успех выигрывает:

  1. ``uhubctl -a cycle`` — настоящий power-cycle порта (нужен хаб с PPPS);
  2. ``authorized 0→1``   — логическое переподключение устройства;
  3. ``unbind``/``bind``  — перепривязка драйвера к устройству;
  4. ``USBDEVFS_RESET``   — сброс устройства через ioctl.

Требуются права root (launcher/systemd их дают). Если устройство «залипло»
по питанию и хаб без PPPS — софт не спасёт, нужен питаемый хаб/ёмкости.
"""

from __future__ import annotations

import fcntl
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

USBDEVFS_RESET = 0x5514
_SYS_VIDEO = Path("/sys/class/video4linux")
_SYS_USB = Path("/sys/bus/usb/drivers/usb")


def _video_node(source) -> str:
    if isinstance(source, str) and source:
        return Path(source).resolve().name or "video0"
    return f"video{int(source)}"


def _usb_device(source) -> Optional[Path]:
    """sysfs-каталог USB-устройства камеры (тот, где есть idVendor)."""
    base = _SYS_VIDEO / _video_node(source) / "device"
    if not base.exists():
        return None
    try:
        p = base.resolve()
    except OSError:
        return None
    while p != p.parent:
        if (p / "idVendor").exists():
            return p
        p = p.parent
    return None


def _try_uhubctl(busid: str) -> bool:
    if shutil.which("uhubctl") is None:
        return False
    if "." in busid:
        parent, port = busid.rsplit(".", 1)
    else:
        bus, _, port = busid.partition("-")
        parent = f"{bus}-1"  # корневой хаб
    if not port:
        return False
    try:
        r = subprocess.run(
            ["uhubctl", "-l", parent, "-p", port, "-a", "cycle"],
            capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and "error" not in (r.stderr or "").lower():
            print(f"[USB] uhubctl cycle {parent} port {port}: OK", flush=True)
            return True
        print(f"[USB] uhubctl failed: {r.stderr.strip() or r.stdout.strip()}", flush=True)
    except Exception as exc:
        print(f"[USB] uhubctl error: {exc}", flush=True)
    return False


def _try_authorized(dev: Path) -> bool:
    auth = dev / "authorized"
    if not auth.exists():
        return False
    try:
        auth.write_text("0", encoding="ascii")
        time.sleep(0.5)
        auth.write_text("1", encoding="ascii")
        print(f"[USB] re-authorized {dev.name}", flush=True)
        return True
    except OSError as exc:
        print(f"[USB] authorized toggle failed: {exc}", flush=True)
        return False


def _try_bind(dev: Path) -> bool:
    try:
        (_SYS_USB / "unbind").write_text(dev.name, encoding="ascii")
        time.sleep(0.5)
        (_SYS_USB / "bind").write_text(dev.name, encoding="ascii")
        print(f"[USB] rebound {dev.name}", flush=True)
        return True
    except OSError as exc:
        print(f"[USB] bind toggle failed: {exc}", flush=True)
        return False


def _try_ioctl(dev: Path) -> bool:
    try:
        busnum = int((dev / "busnum").read_text().strip())
        devnum = int((dev / "devnum").read_text().strip())
    except (OSError, ValueError):
        return False
    node = Path(f"/dev/bus/usb/{busnum:03d}/{devnum:03d}")
    try:
        fd = node.open("rb", buffering=0)
    except OSError as exc:
        print(f"[USB] open {node} failed: {exc}", flush=True)
        return False
    try:
        fcntl.ioctl(fd, USBDEVFS_RESET, 0)
        print(f"[USB] USBDEVFS_RESET {node}: OK", flush=True)
        return True
    except OSError as exc:
        print(f"[USB] USBDEVFS_RESET {node} failed: {exc}", flush=True)
        return False
    finally:
        fd.close()


def reset(source=0) -> bool:
    """Попытаться переподключить USB-камеру. True, если что-то сработало."""
    dev = _usb_device(source)
    if dev is None:
        print(f"[USB] camera device not found for source={source!r}", flush=True)
        return False
    print(f"[USB] resetting camera {dev.name}...", flush=True)
    return (_try_uhubctl(dev.name)
            or _try_authorized(dev)
            or _try_bind(dev)
            or _try_ioctl(dev))
