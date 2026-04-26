import webview
import sys
import os
import threading
import time

# ФИКС 1: Явное указание использования GTK для Linux/ARM
# Это помогает избежать ошибок сегментации, которые были у тебя в Terminator
os.environ['PYWEBVIEW_GUI'] = 'gtk'

def logic_thread(window):
    """Фоновый поток для управления роботом (Astra)"""
    print("[R2] Логика запущена.")
    # Имитация получения данных с датчиков/аккумулятора
    time.sleep(3)
    # Пример передачи данных в JS (изменение интерфейса)
    window.evaluate_js("document.body.style.backgroundColor = '#001a00';")
    print("[R2] Цвет фона изменен через JS")

def start_webview():
    # ФИКС 2: HTML с метатегами для корректного отображения на маленьких экранах
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <style>
            body { 
                background-color: #000; color: #0f0; 
                font-family: monospace; display: flex; 
                flex-direction: column; justify-content: center; 
                align-items: center; height: 100vh; margin: 0; 
            }
            .status { border: 1px solid #0f0; padding: 10px; border-radius: 5px; }
            h1 { text-shadow: 0 0 10px #0f0; }
        </style>
    </head>
    <body>
        <h1>R2: ACTIVE</h1>
        <div class="status" id="volt">Ожидание данных...</div>
    </body>
    </html>
    """

    # ФИКС 3: Настройки окна для робота
    # fullscreen=True уберет рамки и закроет рабочий стол
    window = webview.create_window(
        'R2 Control Panel',
        html=html_content,
        width=800,
        height=480,
        fullscreen=False # Поставь True, когда будешь готов к киоск-режиму
    )

    # Запускаем логику в отдельном потоке, чтобы окно не висло
    t = threading.Thread(target=logic_thread, args=(window,))
    t.daemon = True
    t.start()

    # ФИКС 4: Запуск с принудительным указанием бэкенда
    # debug=True поможет увидеть консоль браузера по правому клику
    webview.start(gui='gtk', debug=True)

if __name__ == '__main__':
    try:
        start_webview()
    except Exception as e:
        print(f"[!] Критическая ошибка GUI: {e}")