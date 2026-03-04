#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Мониторинг Orange Pi RK3399 с веб‑интерфейсом.
Показывает температуру CPU, загрузку, использование RAM и реальные системные логи.
При запуске автоматически открывает браузер в полноэкранном режиме (kiosk).
"""

import os
import sys
import subprocess
import shutil
import webbrowser
import datetime
from collections import deque
from pathlib import Path

import psutil
from flask import Flask, render_template, jsonify

app = Flask(__name__)


def get_cpu_temp():
    """Чтение температуры процессора Orange Pi (RK3399)."""
    try:
        # Стандартный путь для большинства Rockchip плат
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
                # Читаем файл с конца, чтобы получить последние n строк
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
                        # Разбиваем на строки и добавляем спереди
                        chunk_lines = chunk.splitlines()
                        lines.extendleft(reversed(chunk_lines))

                    # Обрезаем до нужного количества
                    return list(lines)[-n:]
            except Exception as e:
                print(f"Не удалось прочитать {log_file}: {e}")

    # Пробуем journalctl (может потребоваться группа adm или sudo)
    try:
        output = subprocess.check_output(
            ['journalctl', '-n', str(n), '--no-pager'],
            stderr=subprocess.DEVNULL,
            universal_newlines=True
        )
        return output.splitlines()
    except Exception:
        pass

    # Если ничего не сработало
    return ["Нет доступа к системным логам"]


@app.route('/')
def index():
    """Главная страница (screen.html)."""
    return render_template('screen.html')


@app.route('/get_system_data')
def get_system_data():
    """
    Возвращает JSON с метриками системы и последними логами.
    Логи теперь реальные, не генерируются искусственно.
    """
    now = datetime.datetime.now().strftime("%H:%M:%S")
    cpu_load = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    temp = get_cpu_temp()
    logs = get_recent_logs(50)

    # Добавляем текущие метрики как одну из строк лога (опционально)
    # Можно раскомментировать, чтобы видеть их в общем потоке:
    # logs.insert(0, f"[{now}] CPU_TEMP={temp} CPU_LOAD={cpu_load}% RAM={ram}%")

    return jsonify(logs)


def open_browser_kiosk():
    """Запускает браузер в полноэкранном режиме (kiosk) с указанным URL."""
    url = "http://127.0.0.1:5000"
    # Пробуем Chromium
    if shutil.which("chromium-browser"):
        subprocess.Popen(["chromium-browser", "--kiosk", url])
    elif shutil.which("chromium"):
        subprocess.Popen(["chromium", "--kiosk", url])
    # Пробуем Firefox
    elif shutil.which("firefox"):
        subprocess.Popen(["firefox", "--kiosk", url])
    else:
        # Если нет подходящего браузера, открываем в обычном режиме
        webbrowser.open(url)
        print("Не удалось найти браузер с поддержкой kiosk. Открыто в обычном режиме.")


if __name__ == '__main__':
    # Открываем браузер (процесс не блокирует выполнение Flask)
    open_browser_kiosk()

    # Запуск Flask-сервера
    app.run(host='0.0.0.0', port=5000, debug=False)
