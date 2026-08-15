"""Self-signed TLS certificate management for the WebXR HTTPS server.

Quest Browser (Meta Quest) cannot use Chromium's insecure-as-secure flag, so
the R2 serves the same Flask app over HTTPS with a self-signed certificate.
The user taps through the browser warning once; afterwards the origin is a
secure context and ``navigator.xr`` becomes available.

The certificate is regenerated only when the detected LAN IP changes, so a
headset moving between networks still resolves the hostname SAN.
"""

import os
import socket
import subprocess
from pathlib import Path

CERT_DIR = Path(__file__).resolve().parent.parent / "certs"
CERT_FILE = CERT_DIR / "cert.pem"
KEY_FILE = CERT_DIR / "key.pem"
_MARKER = CERT_DIR / "san.txt"
CERT_DAYS = 3650
_IP_CACHE = {}


def _detect_ips() -> tuple:
    ips = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ("127.0.0.1", "127.0.1.1") and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        if ip not in ips:
            ips.append(ip)
    except OSError:
        pass
    finally:
        s.close()
    return tuple(ips)


def _san_entries(ips: tuple) -> str:
    entries = ["DNS:localhost", "DNS:r2"]
    entries += [f"IP:{ip}" for ip in ips]
    return ",".join(entries)


def _ensure_cert() -> tuple:
    ips = _detect_ips()
    old_san = ""
    if _MARKER.exists():
        try:
            old_san = _MARKER.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    new_san = _san_entries(ips)

    if CERT_FILE.exists() and KEY_FILE.exists() and old_san == new_san:
        return CERT_FILE, KEY_FILE

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(KEY_FILE), "-out", str(CERT_FILE),
            "-days", str(CERT_DAYS),
            "-subj", "/CN=R2 Robot",
            "-addext", f"subjectAltName={new_san}",
        ],
        check=True, capture_output=True,
    )
    _MARKER.write_text(new_san, encoding="utf-8")
    return CERT_FILE, KEY_FILE


def get_cert_paths() -> tuple:
    """Return (cert, key) paths for Flask's ``ssl_context``."""
    return _ensure_cert()
