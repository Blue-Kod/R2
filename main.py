#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chill Monitor с глазами, системными логами и кнопкой обновления.
Глаза вытянуты вверх, без зрачков, подрагивают раз в секунду, моргают случайно.
Логи (последние 500 строк) отображаются внизу с прокруткой.
Кнопка "Обновление" запускает лаунчер для обновления кода.
"""

import tkinter as tk
from tkinter import scrolledtext
import random
import datetime
import os
import subprocess
import time
from collections import deque

import psutil

# Лог-файл для отладки самого скрипта
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

def get_recent_logs(n=500):
    """Возвращает последние n строк системного лога."""
    log_files = ['/var/log/syslog', '/var/log/messages']
    for log_file in log_files:
        if os.path.exists(log_file) and os.access(log_file, os.R_OK):
            try:
                with open(log_file, 'rb') as f:
                    f.seek(0, os.SEEK_END)
                    file_size = f.tell()
                    block_size = 4096
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

class ChillEyes:
    def __init__(self, root):
        self.root = root
        self.root.title("Chill Monitor")
        self.root.configure(bg='black')
        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', self.quit_fullscreen)

        # Верхняя часть – холст для глаз
        self.canvas = tk.Canvas(root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Нижняя часть – фрейм с логами и кнопками
        bottom_frame = tk.Frame(root, bg='black')
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Кнопки
        btn_frame = tk.Frame(bottom_frame, bg='black')
        btn_frame.pack(fill=tk.X, pady=5)

        self.btn_minimize = tk.Button(btn_frame, text="Свернуть", command=self.minimize_window,
                                      bg='#333', fg='white', font=('Arial', 12),
                                      relief=tk.FLAT, activebackground='#555')
        self.btn_minimize.pack(side=tk.LEFT, padx=5)

        self.btn_update = tk.Button(btn_frame, text="Обновление", command=self.run_update,
                                    bg='#333', fg='white', font=('Arial', 12),
                                    relief=tk.FLAT, activebackground='#555')
        self.btn_update.pack(side=tk.LEFT, padx=5)

        # Область логов с прокруткой
        self.log_text = scrolledtext.ScrolledText(bottom_frame, bg='#111', fg='#888',
                                                   font=('Courier', 9), wrap=tk.WORD,
                                                   height=15, relief=tk.FLAT)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Параметры глаз
        self.eye_width = 60           # ширина (стала меньше, так как вертикальные)
        self.eye_height = 140          # высота (вытянутые вверх)
        self.eye_spacing = 30          # расстояние между глазами
        self.eye_color = '#aaaaaa'     # светло-серый
        self.blink_color = '#222222'   # цвет закрытого глаза
        self.blinking = False
        self.blink_after_id = None
        self.eyes_open = True

        # Смещение для дрожания
        self.shake_offset = 0
        self.shake_after_id = None

        # Координаты глаз (будут установлены после отображения окна)
        self.eye_y = 0
        self.left_eye_x = 0
        self.right_eye_x = 0

        # Обновим позицию после того, как окно отобразится
        self.root.after(100, self.initialize_position)

        # Запускаем моргание и дрожание
        self.schedule_blink()
        self.schedule_shake()

        # Запускаем обновление логов и метрик
        self.update_logs()

    def quit_fullscreen(self, event=None):
        self.root.attributes('-fullscreen', False)

    def minimize_window(self):
        self.root.iconify()

    def initialize_position(self):
        """Вычисляет позицию глаз после того, как холст получил размеры."""
        self.update_position()

    def update_position(self):
        """Пересчитывает координаты глаз на основе текущего размера холста."""
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w > 10 and h > 10:
            self.eye_y = h // 2 - self.eye_height // 2
            self.left_eye_x = w // 2 - self.eye_width - self.eye_spacing // 2
            self.right_eye_x = w // 2 + self.eye_spacing // 2
            self.draw_eyes()

    def draw_eyes(self):
        """Рисует глаза с учётом состояния и смещения."""
        self.canvas.delete('eye')

        # Радиус скругления (зависит от ширины/высоты)
        r = 20

        # Левый глаз
        x1 = self.left_eye_x + self.shake_offset
        y1 = self.eye_y
        x2 = x1 + self.eye_width
        y2 = y1 + self.eye_height
        color = self.eye_color if self.eyes_open else self.blink_color

        # Прямоугольник с заливкой
        self.canvas.create_rectangle(x1, y1, x2, y2,
                                     fill=color, outline='', tags='eye')
        # Угловые закругления (круги)
        self.canvas.create_oval(x1, y1, x1 + 2*r, y1 + 2*r,
                                fill=color, outline='', tags='eye')
        self.canvas.create_oval(x2 - 2*r, y1, x2, y1 + 2*r,
                                fill=color, outline='', tags='eye')
        self.canvas.create_oval(x1, y2 - 2*r, x1 + 2*r, y2,
                                fill=color, outline='', tags='eye')
        self.canvas.create_oval(x2 - 2*r, y2 - 2*r, x2, y2,
                                fill=color, outline='', tags='eye')

        # Правый глаз
        x1 = self.right_eye_x + self.shake_offset
        y1 = self.eye_y
        x2 = x1 + self.eye_width
        y2 = y1 + self.eye_height
        self.canvas.create_rectangle(x1, y1, x2, y2,
                                     fill=color, outline='', tags='eye')
        self.canvas.create_oval(x1, y1, x1 + 2*r, y1 + 2*r,
                                fill=color, outline='', tags='eye')
        self.canvas.create_oval(x2 - 2*r, y1, x2, y1 + 2*r,
                                fill=color, outline='', tags='eye')
        self.canvas.create_oval(x1, y2 - 2*r, x1 + 2*r, y2,
                                fill=color, outline='', tags='eye')
        self.canvas.create_oval(x2 - 2*r, y2 - 2*r, x2, y2,
                                fill=color, outline='', tags='eye')

    def blink(self):
        """Закрыть глаза, потом открыть через 150 мс."""
        if self.blinking:
            return
        self.blinking = True
        self.eyes_open = False
        self.draw_eyes()
        self.root.after(150, self.open_eyes)

    def open_eyes(self):
        self.eyes_open = True
        self.draw_eyes()
        self.blinking = False

    def schedule_blink(self):
        """Планирует следующее моргание через 5-15 секунд."""
        interval = random.randint(5000, 15000)
        self.blink_after_id = self.root.after(interval, self.do_blink)

    def do_blink(self):
        self.blink()
        self.schedule_blink()

    def schedule_shake(self):
        """Запускает цикл дрожания раз в секунду."""
        self.shake_loop()

    def shake_loop(self):
        """Меняет смещение глаз случайным образом и перерисовывает."""
        # Новое случайное смещение от -3 до 3
        self.shake_offset = random.randint(-3, 3)
        self.draw_eyes()
        # Повтор через 1 секунду
        self.shake_after_id = self.root.after(1000, self.shake_loop)

    def update_logs(self):
        """Обновляет текст логов и метрики вверху."""
        # Получаем метрики
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        temp = get_cpu_temp()
        metrics = f"CPU:{cpu}%  RAM:{ram}%  TEMP:{temp}"

        # Получаем последние логи
        logs = get_recent_logs(500)

        # Обновляем текстовое поле
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, metrics + "\n\n")
        for line in logs:
            self.log_text.insert(tk.END, line + "\n")
        self.log_text.config(state=tk.DISABLED)
        # Прокрутка вниз, чтобы видеть новые строки
        self.log_text.see(tk.END)

        # Планируем следующее обновление через 2 секунды
        self.root.after(2000, self.update_logs)

    def run_update(self):
        """Запускает лаунчер для обновления кода."""
        # Блокируем кнопку, чтобы не запустить повторно
        self.btn_update.config(state=tk.DISABLED, text="Обновляется...")
        # Запускаем лаунчер в фоне
        try:
            # Путь к лаунчеру (рядом с main.py)
            launcher_path = os.path.join(os.path.dirname(__file__), "launcher.py")
            if os.path.exists(launcher_path):
                subprocess.Popen([sys.executable, launcher_path])
                log_message("Запущен процесс обновления")
            else:
                log_message("launcher.py не найден")
                self.btn_update.config(state=tk.NORMAL, text="Обновление")
                return
        except Exception as e:
            log_message(f"Ошибка запуска лаунчера: {e}")
        # Через некоторое время вернём кнопку в активное состояние
        self.root.after(10000, self.enable_update_button)  # 10 секунд, чтобы процесс успел завершиться

    def enable_update_button(self):
        self.btn_update.config(state=tk.NORMAL, text="Обновление")

def main():
    log_message("Запуск Chill Monitor")
    root = tk.Tk()
    app = ChillEyes(root)
    root.bind('<Configure>', lambda e: app.update_position())
    root.mainloop()

if __name__ == "__main__":
    main()
