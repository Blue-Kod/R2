#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import threading
import time
import socket
import datetime
import pwd
import shutil
from collections import deque

import psutil
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import ptyprocess

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

LOG_FILE = "logs.txt"
active_sessions = {}  # {sid: {'proc': ptyprocess.PtyProcess, 'thread': Thread}}

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

# Маршруты страниц
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/screen')
def screen():
    return render_template('screen.html')

@app.route('/terminal')
def terminal():
    return render_template('terminal.html')

@app.route('/logs')
def logs():
    return render_template('logs.html')

# API данных
@app.route('/api/data')
def api_data():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    temp = get_cpu_temp()
    logs = get_recent_logs(500)
    return jsonify({
        'cpu': cpu,
        'ram': ram,
        'temp': temp,
        'logs': logs
    })

# API управления
@app.route('/api/update', methods=['POST'])
def api_update():
    try:
        launcher_path = os.path.join(os.path.dirname(__file__), "launcher.py")
        if os.path.exists(launcher_path):
            subprocess.Popen([sys.executable, launcher_path])
            log_message("Запущен процесс обновления")
            def shutdown():
                time.sleep(1)
                os._exit(0)
            threading.Thread(target=shutdown, daemon=True).start()
            return jsonify({'status': 'ok', 'message': 'Обновление запущено'})
        else:
            return jsonify({'status': 'error', 'message': 'launcher.py не найден'}), 404
    except Exception as e:
        log_message(f"Ошибка запуска лаунчера: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/shutdown', methods=['POST'])
def api_shutdown():
    log_message("Получена команда на завершение")
    def shutdown():
        time.sleep(1)
        os._exit(0)
    threading.Thread(target=shutdown, daemon=True).start()
    return jsonify({'status': 'ok', 'message': 'Завершение работы'})

# SocketIO обработчики для терминала
@socketio.on('connect')
def handle_connect():
    sid = request.sid
    log_message(f"Терминал подключен: {sid}")
    try:
        proc = ptyprocess.PtyProcess.spawn(['/bin/bash'])
        active_sessions[sid] = {'proc': proc}

        def read_output():
            try:
                while True:
                    data = proc.read(1024)
                    if not data:
                        break
                    socketio.emit('output', data.decode('utf-8', errors='replace'), room=sid)
            except Exception as e:
                log_message(f"Ошибка чтения из pty для {sid}: {e}")
            finally:
                socketio.emit('exit', room=sid)
                active_sessions.pop(sid, None)

        thread = threading.Thread(target=read_output, daemon=True)
        thread.start()
        active_sessions[sid]['thread'] = thread
    except Exception as e:
        log_message(f"Не удалось создать pty для {sid}: {e}")
        socketio.emit('output', f"Ошибка запуска терминала: {e}\r\n", room=sid)
        socketio.emit('exit', room=sid)

@socketio.on('input')
def handle_input(data):
    sid = request.sid
    proc_info = active_sessions.get(sid)
    if proc_info:
        try:
            proc_info['proc'].write(data.encode('utf-8'))
        except Exception as e:
            log_message(f"Ошибка записи в pty для {sid}: {e}")

@socketio.on('resize')
def handle_resize(data):
    sid = request.sid
    proc_info = active_sessions.get(sid)
    if proc_info:
        try:
            rows = data.get('rows')
            cols = data.get('cols')
            if rows and cols:
                proc_info['proc'].setwinsize(rows, cols)
        except Exception as e:
            log_message(f"Ошибка изменения размера pty для {sid}: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    log_message(f"Терминал отключен: {sid}")
    proc_info = active_sessions.pop(sid, None)
    if proc_info:
        try:
            proc_info['proc'].terminate()
        except:
            pass

# Функции для запуска браузера (без изменений)
def get_display_user():
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
    user = get_display_user()
    if not user:
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
            try:
                subprocess.Popen(command)
            except Exception as e:
                log_message(f"Ошибка запуска браузера: {e}")
            finally:
                os._exit(0)
    except Exception as e:
        log_message(f"Не удалось переключиться на пользователя {user}: {e}")
        subprocess.Popen(command)

def open_browser_kiosk():
    url = "http://127.0.0.1:5000/screen"
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
        run_browser_as_user(["firefox", "--kiosk", url])
    else:
        log_message("Не найден браузер с поддержкой kiosk. Открываем обычный.")
        subprocess.Popen(["xdg-open", url])

def wait_for_server(host='127.0.0.1', port=5000, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (socket.timeout, ConnectionRefusedError):
            time.sleep(0.5)
    return False

def start_browser_when_ready():
    if wait_for_server(timeout=15):
        time.sleep(5)
        open_browser_kiosk()
    else:
        log_message("Сервер не запустился вовремя, браузер не открыт.")

def main():
    log_message("Запуск веб-сервера R2")
    threading.Thread(target=start_browser_when_ready, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()
