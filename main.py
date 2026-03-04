#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Мониторинг Orange Pi RK3399 с GUI на Tkinter.
Показывает температуру CPU, загрузку, использование RAM и реальные системные логи.
Обновление раз в 2 секунды.
Если передан аргумент --web, запускается веб-версия (Flask) для удалённого доступа.
"""

import tkinter as tk
from tkinter import scrolledtext
import sys
import datetime
import os
import subprocess
from collections import deque

import psutil

# Лог-файл для сообщений (совместимость с лаунчером)
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


class MonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Orange Pi Monitor")
        self.root.geometry("800x600")
        # Полноэкранный режим (можно отключить клавишей Escape)
        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', self.exit_fullscreen)

        # Верхняя панель с метриками
        self.frame_metrics = tk.Frame(root, bg='black')
        self.frame_metrics.pack(fill=tk.X, padx=10, pady=10)

        self.label_temp = tk.Label(self.frame_metrics, text="CPU Temp: --", font=("Courier", 16),
                                    fg='cyan', bg='black')
        self.label_temp.pack(side=tk.LEFT, padx=20)

        self.label_cpu = tk.Label(self.frame_metrics, text="CPU Load: --", font=("Courier", 16),
                                   fg='lightgreen', bg='black')
        self.label_cpu.pack(side=tk.LEFT, padx=20)

        self.label_ram = tk.Label(self.frame_metrics, text="RAM Used: --", font=("Courier", 16),
                                   fg='yellow', bg='black')
        self.label_ram.pack(side=tk.LEFT, padx=20)

        # Область логов
        self.text_log = scrolledtext.ScrolledText(root, bg='black', fg='lightgray',
                                                   font=("Courier", 10), wrap=tk.WORD)
        self.text_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Кнопка выхода
        self.btn_quit = tk.Button(root, text="Выход (ESC)", command=self.quit_app,
                                   bg='red', fg='white', font=("Arial", 12))
        self.btn_quit.pack(pady=5)

        self.update_data()

    def exit_fullscreen(self, event=None):
        self.root.attributes('-fullscreen', False)

    def quit_app(self):
        self.root.quit()
        self.root.destroy()

    def update_data(self):
        # Обновляем метрики
        temp = get_cpu_temp()
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        self.label_temp.config(text=f"CPU Temp: {temp}")
        self.label_cpu.config(text=f"CPU Load: {cpu}%")
        self.label_ram.config(text=f"RAM Used: {ram}%")

        # Обновляем логи
        logs = get_recent_logs(50)
        self.text_log.delete(1.0, tk.END)
        for line in logs:
            self.text_log.insert(tk.END, line + "\n")
        self.text_log.see(tk.END)  # прокрутка вниз

        # Планируем следующее обновление через 2000 мс
        self.root.after(2000, self.update_data)


def run_web_server():
    """Запуск веб-версии (Flask) для удалённого доступа."""
    log_message("Запуск веб-версии (Flask)")
    from flask import Flask, render_template, jsonify
    app = Flask(__name__)

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

    app.run(host='0.0.0.0', port=5000, debug=False)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        run_web_server()
    else:
        log_message("Запуск Tkinter GUI")
        root = tk.Tk()
        app = MonitorApp(root)
        root.mainloop()


if __name__ == "__main__":
    main()
