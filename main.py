#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Мониторинг Orange Pi RK3399 с веб‑интерфейсом.
Показывает температуру CPU, загрузку, использование RAM и реальные системные логи.
При запуске автоматически открывает браузер в полноэкранном режиме (kiosk).
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
from collections import deque
from pathlib import Path

import psutil
from flask import Flask, render_template, jsonify

app = Flask(__name__)


def get_cpu_temp():
    """Чтение температуры процессора Orange Pi (RK3399)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read()) / 1000
            return f"{temp:.1f}°C"
    except Exception:
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
                print(f"Не удалось прочитать {log_file}: {e}")

    try:
        output = subprocess.check_output(
            ['journalctl', '-n', str(n), '--no-pager'],
            stderr=subprocess.DEVNULL,
            universal_newlines=True
        )
        return output.splitlines()
    except Exception:
        pass

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
    # Если запущено через sudo, берём SUDO_USER
    user = os.environ.get('SUDO_USER')
    if user and user != 'root':
        return user
    # Иначе ищем первого пользователя с UID >= 1000
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
        # Запускаем от текущего пользователя (не root или не нашли)
        subprocess.Popen(command)
        return

    try:
        pw = pwd.getpwnam(user)
        uid = pw.pw_uid
        gid = pw.pw_gid

        pid = os.fork()
        if pid == 0:
            # Дочерний процесс – меняем пользователя
            os.setgid(gid)
            os.setuid(uid)
            # Устанавливаем правильное окружение
            os.environ['HOME'] = pw.pw_dir
            os.environ['USER'] = user
            os.environ['LOGNAME'] = user
            os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':0')
            xauth = os.path.join(pw.pw_dir, '.Xauthority')
            if os.path.exists(xauth):
                os.environ['XAUTHORITY'] = xauth

            # Запускаем браузер
            try:
                subprocess.Popen(command)
            except Exception as e:
                print(f"Ошибка запуска браузера: {e}")
            finally:
                os._exit(0)
        else:
            # Родитель ничего не ждёт
            pass
    except Exception as e:
        print(f"Не удалось переключиться на пользователя {user}: {e}")
        subprocess.Popen(command)


def open_browser_kiosk():
    """Запускает браузер в полноэкранном режиме (kiosk) с указанным URL."""
    url = "http://127.0.0.1:5000"
    is_root = (os.geteuid() == 0)

    if shutil.which("chromium-browser"):
        cmd = ["chromium-browser", "--kiosk", url]
        if is_root:
            cmd.insert(1, "--no-sandbox")
        run_browser_as_user(cmd)
    elif shutil.which("chromium"):
        cmd = ["chromium", "--kiosk", url]
        if is_root:
            cmd.insert(1, "--no-sandbox")
        run_browser_as_user(cmd)
    elif shutil.which("firefox"):
        cmd = ["firefox", "--kiosk", url]
        run_browser_as_user(cmd)
    else:
        webbrowser.open(url)
        print("Не удалось найти браузер с поддержкой kiosk. Открыто в обычном режиме.")


if __name__ == '__main__':
    open_browser_kiosk()
    app.run(host='0.0.0.0', port=5000, debug=False)
