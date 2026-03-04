#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Мониторинг Orange Pi RK3399 с чилловым интерфейсом.
Чёрный фон, по центру два скруглённых прямоугольных глаза, которые моргают и подрагивают.
Внизу — серые логи системы и кнопка "Свернуть".
"""

import tkinter as tk
import random
import datetime
import os
import subprocess
import time
from collections import deque

import psutil

# Лог-файл (на всякий случай, для отладки)
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

def get_recent_logs(n=20):
    """Возвращает последние n строк системного лога."""
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

class ChillEyes:
    def __init__(self, root):
        self.root = root
        self.root.title("Chill Monitor")
        self.root.configure(bg='black')
        # Полноэкранный режим
        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', self.quit_fullscreen)

        # Основной холст для рисования
        self.canvas = tk.Canvas(root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Параметры глаз
        self.eye_width = 150          # ширина глаза
        self.eye_height = 80           # высота глаза
        self.eye_spacing = 40          # расстояние между глазами
        self.eye_color = '#aaaaaa'     # светло-серый
        self.pupil_color = '#333333'   # тёмно-серый зрачок
        self.blink_color = '#222222'   # цвет закрытого глаза
        self.blinking = False
        self.blink_after_id = None

        # Состояние глаз: открыты/закрыты (True = открыты)
        self.eyes_open = True
        # Текущее смещение для дрожания
        self.shake_offset = 0
        self.shake_direction = 1
        self.shake_after_id = None

        # Начальные координаты глаз (центр экрана)
        self.canvas.update()  # чтобы получить размеры
        w = self.canvas.winfo_width() or 800
        h = self.canvas.winfo_height() or 600
        self.eye_y = h // 2 - 40
        self.left_eye_x = w // 2 - self.eye_width - self.eye_spacing//2
        self.right_eye_x = w // 2 + self.eye_spacing//2

        # Текстовые метрики (температура, CPU, RAM) будем рисовать отдельно?
        # Покажем их маленьким шрифтом вверху или внизу? Решим: вверху мелкими цифрами.
        self.metrics_text = None

        # Логи – будем хранить список строк и обновлять каждые 2 секунды
        self.log_lines = []
        self.log_texts = []   # список ID текстовых объектов на canvas
        self.update_logs()

        # Запускаем анимацию глаз
        self.schedule_blink()
        self.schedule_shake()

        # Кнопка "Свернуть"
        self.btn_minimize = tk.Button(root, text="Свернуть", command=self.minimize_window,
                                      bg='#333', fg='white', font=('Arial', 12),
                                      relief=tk.FLAT, activebackground='#555')
        self.btn_minimize.place(relx=0.5, rely=0.95, anchor='center')

    def quit_fullscreen(self, event=None):
        self.root.attributes('-fullscreen', False)

    def minimize_window(self):
        self.root.iconify()

    def draw_eyes(self):
        """Рисует глаза с учётом текущего состояния (открыты/закрыты, смещение)."""
        self.canvas.delete('eye')  # удаляем только элементы с тегом 'eye'

        x1 = self.left_eye_x + self.shake_offset
        y1 = self.eye_y
        x2 = x1 + self.eye_width
        y2 = y1 + self.eye_height

        # Левый глаз
        self.canvas.create_rectangle(x1, y1, x2, y2,
                                     fill=self.eye_color if self.eyes_open else self.blink_color,
                                     outline='', tags='eye', width=0)
        # Скругление углов: рисуем кружки в углах тем же цветом
        r = 20  # радиус скругления
        self.canvas.create_oval(x1, y1, x1 + 2*r, y1 + 2*r,
                                fill=self.eye_color if self.eyes_open else self.blink_color,
                                outline='', tags='eye')
        self.canvas.create_oval(x2 - 2*r, y1, x2, y1 + 2*r,
                                fill=self.eye_color if self.eyes_open else self.blink_color,
                                outline='', tags='eye')
        self.canvas.create_oval(x1, y2 - 2*r, x1 + 2*r, y2,
                                fill=self.eye_color if self.eyes_open else self.blink_color,
                                outline='', tags='eye')
        self.canvas.create_oval(x2 - 2*r, y2 - 2*r, x2, y2,
                                fill=self.eye_color if self.eyes_open else self.blink_color,
                                outline='', tags='eye')

        # Если глаз открыт, рисуем зрачок
        if self.eyes_open:
            pupil_w = self.eye_width // 3
            pupil_h = self.eye_height // 2
            pupil_x = x1 + (self.eye_width - pupil_w) // 2
            pupil_y = y1 + (self.eye_height - pupil_h) // 2
            self.canvas.create_oval(pupil_x, pupil_y,
                                    pupil_x + pupil_w, pupil_y + pupil_h,
                                    fill=self.pupil_color, outline='', tags='eye')

        # Правый глаз (аналогично)
        x1 = self.right_eye_x + self.shake_offset
        y1 = self.eye_y
        x2 = x1 + self.eye_width
        y2 = y1 + self.eye_height

        self.canvas.create_rectangle(x1, y1, x2, y2,
                                     fill=self.eye_color if self.eyes_open else self.blink_color,
                                     outline='', tags='eye', width=0)
        self.canvas.create_oval(x1, y1, x1 + 2*r, y1 + 2*r,
                                fill=self.eye_color if self.eyes_open else self.blink_color,
                                outline='', tags='eye')
        self.canvas.create_oval(x2 - 2*r, y1, x2, y1 + 2*r,
                                fill=self.eye_color if self.eyes_open else self.blink_color,
                                outline='', tags='eye')
        self.canvas.create_oval(x1, y2 - 2*r, x1 + 2*r, y2,
                                fill=self.eye_color if self.eyes_open else self.blink_color,
                                outline='', tags='eye')
        self.canvas.create_oval(x2 - 2*r, y2 - 2*r, x2, y2,
                                fill=self.eye_color if self.eyes_open else self.blink_color,
                                outline='', tags='eye')

        if self.eyes_open:
            pupil_w = self.eye_width // 3
            pupil_h = self.eye_height // 2
            pupil_x = x1 + (self.eye_width - pupil_w) // 2
            pupil_y = y1 + (self.eye_height - pupil_h) // 2
            self.canvas.create_oval(pupil_x, pupil_y,
                                    pupil_x + pupil_w, pupil_y + pupil_h,
                                    fill=self.pupil_color, outline='', tags='eye')

    def blink(self):
        """Один цикл моргания: закрыть, открыть через короткое время."""
        if self.blinking:
            return
        self.blinking = True
        self.eyes_open = False
        self.draw_eyes()
        # Через 150 мс открыть
        self.root.after(150, self.open_eyes)

    def open_eyes(self):
        self.eyes_open = True
        self.draw_eyes()
        self.blinking = False

    def schedule_blink(self):
        """Планирует следующее моргание через случайный интервал."""
        interval = random.randint(5000, 15000)  # миллисекунды
        self.blink_after_id = self.root.after(interval, self.do_blink)

    def do_blink(self):
        self.blink()
        self.schedule_blink()

    def schedule_shake(self):
        """Планирует небольшое дрожание глаз."""
        # Дрожание состоит из серии мелких движений
        self.shake()

    def shake(self):
        """Один шаг дрожания (сдвиг на 1-2 пикселя, смена направления)."""
        # Если глаза не моргают, немного двигаем их
        max_offset = 3
        self.shake_offset += self.shake_direction * random.randint(1, 2)
        if abs(self.shake_offset) > max_offset:
            self.shake_direction *= -1
            self.shake_offset = self.shake_direction * max_offset
        self.draw_eyes()
        # Повторяем дрожание с интервалом 50-100 мс, пока не отменят
        self.shake_after_id = self.root.after(random.randint(50, 100), self.shake)

    def update_logs(self):
        """Обновляет отображение логов и метрик внизу экрана."""
        # Получаем новые логи
        logs = get_recent_logs(15)
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        temp = get_cpu_temp()

        # Удаляем старые текстовые элементы с тегом 'log'
        self.canvas.delete('log')

        # Рисуем метрики мелким шрифтом вверху
        metrics = f"CPU:{cpu}%  RAM:{ram}%  TEMP:{temp}"
        self.canvas.create_text(10, 10, text=metrics, anchor='nw',
                                fill='#666666', font=('Courier', 12), tags='log')

        # Рисуем логи внизу (с отступом от кнопки)
        y = self.canvas.winfo_height() - 250
        if y < 100:  # если окно маленькое, поднимем
            y = 150
        for i, line in enumerate(logs[-10:]):  # последние 10 строк
            self.canvas.create_text(10, y + i*18, text=line, anchor='nw',
                                    fill='#666666', font=('Courier', 10), tags='log')

        # Запланировать следующее обновление через 2 секунды
        self.root.after(2000, self.update_logs)

    def on_resize(self, event):
        """Пересчитывает позиции глаз при изменении размера окна."""
        if event.width > 10 and event.height > 10:
            self.eye_y = event.height // 2 - 40
            self.left_eye_x = event.width // 2 - self.eye_width - self.eye_spacing//2
            self.right_eye_x = event.width // 2 + self.eye_spacing//2
            self.draw_eyes()

def main():
    log_message("Запуск Chill Monitor")
    root = tk.Tk()
    app = ChillEyes(root)
    root.bind('<Configure>', app.on_resize)  # отслеживаем изменение размера
    root.mainloop()

if __name__ == "__main__":
    main()
