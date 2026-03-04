#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Мониторинг Orange Pi RK3399.
Полноэкранное Tkinter-окно с температурой CPU, загрузкой, использованием RAM и системными логами.
Обновление каждые 2 секунды.
Запускается без веб-сервера, только локальный GUI.
"""

import tkinter as tk
from tkinter import scrolledtext
import datetime
import os
import subprocess
import time
from collections import deque

import psutil

LOG_FILE = "logs.txt"

def log_message(*args):
    msg = " ".join(str(arg) for arg in args)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read()) / 1000
            return f"{temp:.1f}°C"
    except Exception as e:
        log_message(f"Не удалось прочитать температуру: {e}")
        return "N/A"

def get_recent_logs(n=50):
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
        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', self.exit_fullscreen)

        # Верхняя панель
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
        temp = get_cpu_temp()
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        self.label_temp.config(text=f"CPU Temp: {temp}")
        self.label_cpu.config(text=f"CPU Load: {cpu}%")
        self.label_ram.config(text=f"RAM Used: {ram}%")

        logs = get_recent_logs(50)
        self.text_log.delete(1.0, tk.END)
        for line in logs:
            self.text_log.insert(tk.END, line + "\n")
        self.text_log.see(tk.END)

        self.root.after(2000, self.update_data)

def main():
    log_message("Запуск Tkinter GUI...")
    root = tk.Tk()
    app = MonitorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
