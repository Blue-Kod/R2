from __future__ import annotations

import re
import socket
import subprocess
import time

import cv2


def check_internet(host="8.8.8.8", port=53, timeout=3) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except OSError:
        pass
    try:
        import requests
        requests.get("https://github.com", timeout=timeout)
        return True
    except Exception:
        return False


def parse_wifi_qr(data: str) -> dict | None:
    m = re.match(
        r"WIFI:T:(?P<auth>[^;]*);S:(?P<ssid>[^;]*);P:(?P<password>[^;]*);;",
        data.strip(),
    )
    if not m:
        return None
    return {"auth": m.group("auth"), "ssid": m.group("ssid"), "password": m.group("password")}


def connect_via_nmcli(ssid: str, password: str) -> bool:
    try:
        result = subprocess.run(
            ["nmcli", "dev", "wifi", "connect", ssid, "password", password, "hidden", "yes"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def connect_via_wpasupplicant(ssid: str, password: str) -> bool:
    try:
        wpa = subprocess.run(
            ["wpa_passphrase", ssid, password],
            capture_output=True, text=True, timeout=10,
        )
        if wpa.returncode != 0:
            return False
        config = wpa.stdout.replace(
            "\tssid=\"" + ssid + "\"",
            "\tssid=\"" + ssid + "\"\n\tscan_ssid=1",
        )
        with open("/etc/wpa_supplicant/wpa_supplicant.conf", "a") as f:
            f.write(config)
        subprocess.run(
            ["wpa_cli", "-i", "wlan0", "reconfigure"],
            capture_output=True, timeout=10,
        )
        time.sleep(5)
        return check_internet()
    except Exception:
        return False


def connect_to_wifi(ssid: str, password: str) -> bool:
    if connect_via_nmcli(ssid, password):
        time.sleep(3)
        if check_internet():
            return True
    return connect_via_wpasupplicant(ssid, password)


def scan_qr_frame(frame) -> str | None:
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(frame)
    if data:
        return data.strip()
    return None


def start_wifi_setup(speak_func, log_func) -> None:
    log_func("[WiFi] No internet — starting QR setup")

    from robov_core.high_level import get_stereo_camera

    camera = get_stereo_camera()
    if camera is None:
        log_func("[WiFi] Camera not available, skipping QR scan")
        return

    speak_func("Я не подключён к интернету. Пожалуйста, покажите QR-код с настройками Wi-Fi перед камерой.")
    time.sleep(1)

    last_speak_time = 0
    speak_interval = 10

    while not check_internet():
        frame = camera.get_latest_frame()
        if frame is None or frame.size == 0:
            time.sleep(0.5)
            continue

        data = scan_qr_frame(frame)
        if data:
            info = parse_wifi_qr(data)
            if info:
                ssid = info["ssid"]
                password = info["password"]
                log_func(f"[WiFi] QR detected: SSID={ssid}")
                speak_func(f"Найден QR-код. Подключаюсь к сети {ssid}.")

                if connect_to_wifi(ssid, password):
                    speak_func("Подключение к Wi-Fi выполнено. Я готов к работе.")
                    log_func(f"[WiFi] Connected to {ssid}")
                    return
                else:
                    speak_func("Не удалось подключиться. Попробуйте другой QR-код.")
                    log_func(f"[WiFi] Failed to connect to {ssid}")

        now = time.time()
        if now - last_speak_time > speak_interval:
            speak_func("Пожалуйста, разместите QR-код с настройками Wi-Fi перед камерой.")
            last_speak_time = now

        time.sleep(0.3)

    log_func("[WiFi] Internet available, setup not needed")
