#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chill Monitor на PySide6 с глазами, системными логами и кнопкой обновления.
Глаза вытянуты вверх, без зрачков, подрагивают раз в секунду, моргают случайно.
Логи (последние 500 строк) отображаются внизу с прокруткой.
Кнопка "Обновление" запускает лаунчер для обновления кода.
Все элементы на чёрном фоне, без лишних обводок.
"""

import sys
import os
import random
import datetime
import subprocess
from collections import deque

import psutil
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QTextEdit, QGraphicsView,
                               QGraphicsScene, QGraphicsRectItem)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QPointF, QEasingCurve
from PySide6.QtGui import QColor, QBrush, QPen, QFont, QPainter, QTextCursor

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


class RoundedRectItem(QGraphicsRectItem):
    """Прямоугольник со скруглёнными углами."""
    def __init__(self, x, y, w, h, radius, color, parent=None):
        super().__init__(x, y, w, h, parent)
        self.radius = radius
        self.color = color
        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.NoPen))

    def paint(self, painter, option, widget=None):
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
        painter.drawRoundedRect(self.rect(), self.radius, self.radius)


class EyesWidget(QGraphicsView):
    """Виджет с глазами, поддерживающий анимацию и моргание."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background: transparent; border: none;")
        self.setRenderHint(QPainter.Antialiasing)

        # Параметры глаз
        self.eye_width = 240
        self.eye_height = 560
        self.eye_spacing = 30
        self.eye_radius = 60
        self.eye_color_open = QColor(170, 170, 170)   # #aaaaaa
        self.eye_color_closed = QColor(34, 34, 34)    # #222222

        self.left_eye = None
        self.right_eye = None
        self.eyes_open = True
        self.blinking = False
        self.shake_offset = 0

        self.setup_eyes()

        # Таймеры и анимации
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.start_blink)
        self.schedule_next_blink()

        self.shake_timer = QTimer(self)
        self.shake_timer.timeout.connect(self.shake)
        self.shake_timer.start(1000)  # каждую секунду

    def resizeEvent(self, event):
        """Центрируем глаза при изменении размера."""
        super().resizeEvent(event)
        self.center_eyes()

    def center_eyes(self):
        if not self.left_eye or not self.right_eye:
            return
        w = self.width()
        h = self.height()
        # Вычисляем позицию
        eye_y = max(0, (h - self.eye_height) // 2)
        left_x = (w - 2*self.eye_width - self.eye_spacing) // 2
        right_x = left_x + self.eye_width + self.eye_spacing

        self.left_eye.setPos(left_x, eye_y)
        self.right_eye.setPos(right_x, eye_y)

    def setup_eyes(self):
        self.scene().clear()
        # Левый глаз
        self.left_eye = RoundedRectItem(0, 0, self.eye_width, self.eye_height,
                                        self.eye_radius, self.eye_color_open)
        self.scene().addItem(self.left_eye)
        # Правый глаз
        self.right_eye = RoundedRectItem(0, 0, self.eye_width, self.eye_height,
                                         self.eye_radius, self.eye_color_open)
        self.scene().addItem(self.right_eye)
        self.center_eyes()

    def set_eyes_color(self, color):
        self.left_eye.setBrush(QBrush(color))
        self.right_eye.setBrush(QBrush(color))

    def start_blink(self):
        if self.blinking or not self.eyes_open:
            return
        self.blinking = True
        self.eyes_open = False
        self.set_eyes_color(self.eye_color_closed)
        QTimer.singleShot(150, self.end_blink)

    def end_blink(self):
        self.eyes_open = True
        self.set_eyes_color(self.eye_color_open)
        self.blinking = False
        self.schedule_next_blink()

    def schedule_next_blink(self):
        interval = random.randint(5000, 15000)
        self.blink_timer.start(interval)

    def shake(self):
        """Короткая анимация дрожания."""
        # Создаём анимацию для смещения по X
        anim = QPropertyAnimation(self, b"shake_offset")
        anim.setDuration(300)
        anim.setStartValue(-3)
        anim.setKeyValueAt(0.25, 2)
        anim.setKeyValueAt(0.5, -2)
        anim.setKeyValueAt(0.75, 1)
        anim.setEndValue(0)
        anim.setEasingCurve(QEasingCurve.OutSine)
        anim.start()

    def get_shake_offset(self):
        return self._shake_offset if hasattr(self, '_shake_offset') else 0

    def set_shake_offset(self, offset):
        self._shake_offset = offset
        if self.left_eye and self.right_eye:
            # Сохраняем базовые позиции и добавляем смещение
            self.center_eyes()
            self.left_eye.moveBy(offset, 0)
            self.right_eye.moveBy(offset, 0)

    shake_offset = property(get_shake_offset, set_shake_offset)


class ChillMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chill Monitor")
        self.setStyleSheet("background-color: black; color: #888;")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.showFullScreen()

        # Центральный виджет
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Виджет глаз
        self.eyes = EyesWidget()
        layout.addWidget(self.eyes, stretch=1)

        # Нижняя панель
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(5)

        # Кнопки
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(5)

        self.btn_minimize = QPushButton("Свернуть")
        self.btn_minimize.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #777;
            }
        """)
        self.btn_minimize.clicked.connect(self.showMinimized)
        btn_layout.addWidget(self.btn_minimize)

        self.btn_update = QPushButton("Обновление")
        self.btn_update.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #777;
            }
            QPushButton:disabled {
                background-color: #222;
                color: #666;
            }
        """)
        self.btn_update.clicked.connect(self.run_update)
        btn_layout.addWidget(self.btn_update)
        btn_layout.addStretch()

        bottom_layout.addLayout(btn_layout)

        # Текстовое поле для логов
        self.log_text = QTextEdit()
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #111;
                color: #888;
                border: none;
                font-family: Courier;
                font-size: 9pt;
            }
        """)
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        bottom_layout.addWidget(self.log_text)

        layout.addWidget(bottom_widget)

        # Таймер обновления логов
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_logs)
        self.update_timer.start(2000)
        self.update_logs()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.showNormal()
        super().keyPressEvent(event)

    def update_logs(self):
        metrics = f"CPU: {psutil.cpu_percent()}%  RAM: {psutil.virtual_memory().percent}%  TEMP: {get_cpu_temp()}"
        logs = get_recent_logs(500)
        text = metrics + "\n\n" + "\n".join(logs)
        self.log_text.setText(text)
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    def run_update(self):
        self.btn_update.setEnabled(False)
        self.btn_update.setText("Обновляется...")
        try:
            launcher_path = os.path.join(os.path.dirname(__file__), "launcher.py")
            if os.path.exists(launcher_path):
                subprocess.Popen([sys.executable, launcher_path])
                log_message("Запущен процесс обновления")
            else:
                log_message("launcher.py не найден")
        except Exception as e:
            log_message(f"Ошибка запуска лаунчера: {e}")
        QTimer.singleShot(10000, self.enable_update_button)

    def enable_update_button(self):
        self.btn_update.setEnabled(True)
        self.btn_update.setText("Обновление")


def main():
    app = QApplication(sys.argv)
    window = ChillMonitor()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
