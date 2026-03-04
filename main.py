#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Мониторинг Orange Pi RK3399 с веб‑интерфейсом.
Показывает температуру CPU, загрузку, использование RAM и реальные системные логи.
При запуске автоматически открывает браузер в полноэкранном режиме (kiosk) после того,
как веб‑сервер станет доступен.
Все сообщения дублируются в консоль и в файл logs.txt.
Если запущено от root, переключается на обычного пользователя для запуска браузера,
чтобы избежать ошибок с сессией X11 (SESSION_MANAGER, DBUS). Добавляет флаг --no-sandbox
для Chromium при запуске от root.
"""

import os
import sys
import subprocess
import shutil
import webbrowser
import datetime
import pwd
import time
import socket
import threading
from collections import deque
from pathlib import Path

import psutil
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Лог-файл для сообщений
LOG_FILE = "logs.txt"

def log_message(*args):
    """Выводит сообщение в консоль и дописывает в лог-файл с временной меткой."""
    msg = " ".join(str(arg) for arg in args)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[!] Не удалось записать в лог-файл {LOG_FILE}: {e}")

def get_cpu_temp():
    """Чтение температуры процессора Orange Pi (RK3399)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read()) / 1000
            return f"{temp:.1f}°C"
    except Exception as e:
        log_message(f"Не удалось прочитать температуру: {e}")
        return "N/A"

def get_recent_logs(n=50):
    """
    Возвращает последние n строк из системного лога.
    Пытается читать /var/log/syslog, /var/log/messages или journalctl.
    """
    log_files = ['/var/log/syslog', '/var/log/messages']

    for log_file in log_files:
        if os.path.exists(log_file) and os.access(log_file, os.R_OK):
            try:
                with open(log_file, 'rb') as f:
                    f.seek(0, os.SEEK_END)
                    file_size = f.tell()
                    block_size = 1024
                    lines = deque()
                    pos = file_size

                    while len(lines) < n and pos > 0:
                        read_size = min(block_size, pos)
                        pos -= read_size
                        f.seek(pos, os.SEEK_SET)
                        chunk = f.read(read_size).decode('utf-8', errors='ignore')
                        chunk_lines = chunk.splitlines()
                        lines.extendleft(reversed(chunk_lines))

                    return list(lines)[-n:]
            except Exception as e:
                log_message(f"Не удалось прочитать {log_file}: {e}")

    try:
        output = subprocess.check_output(
            ['journalctl', '-n', str(n), '--no-pager'],
            stderr=subprocess.DEVNULL,
            universal_newlines=True
        )
        return output.splitlines()
    except Exception as e:
        log_message(f"journalctl не удался: {e}")

    return ["Нет доступа к системным логам"]

@app.route('/')
def index():
    return render_template('screen.html')

@app.route('/get_system_data')
def get_system_data():
    now = datetime.datetime.now().strftime("%H:%M:%S")
    cpu_load = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    temp = get_cpu_temp()
    logs = get_recent_logs(50)
    return jsonify(logs)

def get_display_user():
    """Возвращает имя обычного пользователя для запуска браузера (если запущено от root)."""
    if os.geteuid() != 0:
        return None
    user = os.environ.get('SUDO_USER')
    if user and user != 'root':
        return user
    for u in pwd.getpwall():
        if 1000 <= u.pw_uid < 65534:
            return u.pw_name
    return None

def run_browser_as_user(command):
    """
    Запускает браузер от имени обычного пользователя, если мы root.
    Использует fork + setuid для смены пользователя.
    """
    user = get_display_user()
    if not user:
        log_message(f"Запуск браузера от текущего пользователя: {' '.join(command)}")
        subprocess.Popen(command)
        return

    try:
        pw = pwd.getpwnam(user)
        uid = pw.pw_uid
        gid = pw.pw_gid

        pid = os.fork()
        if pid == 0:
            os.setgid(gid)
            os.setuid(uid)
            os.environ['HOME'] = pw.pw_dir
            os.environ['USER'] = user
            os.environ['LOGNAME'] = user
            os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':0')
            xauth = os.path.join(pw.pw_dir, '.Xauthority')
            if os.path.exists(xauth):
                os.environ['XAUTHORITY'] = xauth

            log_message(f"Запуск браузера от пользователя {user}: {' '.join(command)}")
            try:
                subprocess.Popen(command)
            except Exception as e:
                log_message(f"Ошибка запуска браузера: {e}")
            finally:
                os._exit(0)
        else:
            pass
    except Exception as e:
        log_message(f"Не удалось переключиться на пользователя {user}: {e}")
        log_message(f"Запуск браузера от root: {' '.join(command)}")
        subprocess.Popen(command)

def open_browser_kiosk():
    """Запускает браузер в полноэкранном режиме (kiosk) с указанным URL."""
    url = "http://127.0.0.1:5000"
    is_root = (os.geteuid() == 0)

    if shutil.which("chromium-browser"):
        cmd = ["chromium-browser", "--kiosk", url]
        if is_root:
            cmd.insert(1, "--no-sandbox")
        log_message("Используется chromium-browser")
        run_browser_as_user(cmd)
    elif shutil.which("chromium"):
        cmd = ["chromium", "--kiosk", url]
        if is_root:
            cmd.insert(1, "--no-sandbox")
        log_message("Используется chromium")
        run_browser_as_user(cmd)
    elif shutil.which("firefox"):
        cmd = ["firefox", "--kiosk", url]
        log_message("Используется firefox")
        run_browser_as_user(cmd)
    else:
        webbrowser.open(url)
        log_message("Не удалось найти браузер с поддержкой kiosk. Открыто в обычном режиме.")

def wait_for_server(host='127.0.0.1', port=5000, timeout=10):
    """Ожидает, пока сервер не станет доступен по указанному адресу."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                log_message(f"Сервер доступен на {host}:{port}")
                return True
        except (socket.timeout, ConnectionRefusedError):
            time.sleep(0.5)
    log_message(f"Сервер не стал доступен за {timeout} секунд")
    return False

def open_browser_when_ready():
    """Ожидает готовности сервера и запускает браузер."""
    if wait_for_server(timeout=15):
        open_browser_kiosk()
    else:
        log_message("Не удалось дождаться сервера, браузер не запущен.")

if __name__ == '__main__':
    # Запускаем поток, который подождёт сервер и откроет браузер
    threading.Thread(target=open_browser_when_ready, daemon=True).start()

    # Запуск Flask-сервера
    app.run(host='0.0.0.0', port=5000, debug=False)
