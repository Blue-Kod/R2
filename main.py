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


# ... (в начало файла добавить import pwd)

def get_display_user():
    """Возвращает имя обычного пользователя для запуска графических приложений (если запущено от root)."""
    if os.geteuid() != 0:
        return None
    user = os.environ.get('SUDO_USER')
    if user and user != 'root':
        return user
    for u in pwd.getpwall():
        if 1000 <= u.pw_uid < 65534:
            return u.pw_name
    return None

def run_terminal_as_user(terminal_cmd):
    """Запускает терминал от имени обычного пользователя, если мы root."""
    user = get_display_user()
    if not user:
        # Не root или не нашли пользователя – запускаем как есть
        subprocess.Popen(terminal_cmd)
        return

    try:
        pw = pwd.getpwnam(user)
        uid = pw.pw_uid
        gid = pw.pw_gid

        pid = os.fork()
        if pid == 0:
            # Дочерний процесс
            os.setgid(gid)
            os.setuid(uid)
            os.environ['HOME'] = pw.pw_dir
            os.environ['USER'] = user
            os.environ['LOGNAME'] = user
            os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':0')
            xauth = os.path.join(pw.pw_dir, '.Xauthority')
            if os.path.exists(xauth):
                os.environ['XAUTHORITY'] = xauth

            try:
                subprocess.Popen(terminal_cmd)
            except Exception as e:
                print(f"Ошибка запуска терминала: {e}")
            finally:
                os._exit(0)
        else:
            # Родитель ничего не ждёт
            pass
    except Exception as e:
        print(f"Не удалось переключиться на пользователя {user}: {e}")
        subprocess.Popen(terminal_cmd)

def run_main_in_terminal():
    """Запускает main.py в новом окне терминала (с поддержкой запуска от обычного пользователя, если мы root)."""
    main_path = Path(MAIN_SCRIPT)
    if not main_path.exists():
        print(f"[!] Ошибка: файл {MAIN_SCRIPT} не найден в текущей директории.")
        return False

    cmd_str = f"{sys.executable} {MAIN_SCRIPT}"
    log_file = Path(f"{MAIN_SCRIPT}.log")
    cmd_with_log = f"{cmd_str} >> {log_file} 2>&1"  # можно использовать для перенаправления

    has_gui = os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    if not has_gui:
        print("[L] Графическая среда не обнаружена. Запускаю в текущем терминале...")
        return run_main_current_terminal(cmd_str)

    terminals = [
        ('xterm', ['xterm', '-hold', '-e'], True),
        ('xfce4-terminal', ['xfce4-terminal', '--hold', '-e'], True),
        ('gnome-terminal', ['gnome-terminal', '--'], True),
        ('konsole', ['konsole', '-e'], True),
        ('lxterminal', ['lxterminal', '-e'], True),
        ('terminator', ['terminator', '-e'], True),
        ('urxvt', ['urxvt', '-e'], True),
        ('rxvt', ['rxvt', '-e'], True),
    ]

    for term_name, base_args, use_string in terminals:
        if shutil.which(term_name):
            if use_string:
                full_args = base_args + [cmd_str]
            else:
                full_args = base_args + [sys.executable, MAIN_SCRIPT]
            print(f"[L] Запускаю {MAIN_SCRIPT} в терминале {term_name}...")
            try:
                # Если мы root, запускаем терминал от обычного пользователя
                if os.geteuid() == 0:
                    run_terminal_as_user(full_args)
                else:
                    subprocess.Popen(full_args)
                return True
            except Exception as e:
                print(f"[!] Не удалось запустить {term_name}: {e}")
                continue

    print("[L] Не найден подходящий эмулятор терминала. Запускаю в текущем терминале...")
    return run_main_current_terminal(cmd_str)

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
