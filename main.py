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
from flask import Flask, render_template, jsonify, request, Response
import cv2
import numpy as np

from camera import StereoCamera
from servo import ServoController   # импортируем класс

HTTP_PORT = 80
# Убрали LOG_FILE – логируем только в консоль

# Глобальные объекты
shell_manager = None
camera = None
servo_controller = None

# ---------- Вспомогательные функции (без записи в файл) ----------
def log_message(*args):
    """Вывод в консоль с временной меткой (без сохранения в файл)."""
    msg = " ".join(str(arg) for arg in args)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read()) / 1000
            return f"{temp:.1f}°C"
    except Exception as e:
        log_message(f"Не удалось прочитать температуру: {e}")
        return "N/A"

def get_recent_logs(n=500):
    # (без изменений, оставляем как есть – читает системные логи)
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

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

# ---------- Flask приложение ----------
app = Flask(__name__)

# Маршруты страниц (без изменений)
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

# API для данных и управления (без изменений)
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

@app.route('/api/ip')
def api_ip():
    return jsonify({'ip': get_ip_address()})

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

# ---------- Терминал (shell) ----------
class ShellManager:
    # (без изменений, только log_message теперь без файла)
    def __init__(self):
        self.proc = None
        self.output_buffer = deque(maxlen=2000)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        try:
            import ptyprocess
            self.proc = ptyprocess.PtyProcess.spawn(['/bin/bash', '-i'])
            self.proc.setwinsize(24, 80)
        except Exception as e:
            log_message(f"Не удалось запустить shell: {e}")
            self.running = False
            return

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        log_message("Shell процесс запущен")

    def _reader(self):
        try:
            while self.running:
                try:
                    data = self.proc.read(1024)
                    if not data:
                        break
                    text = data.decode('utf-8', errors='replace')
                    with self.lock:
                        self.output_buffer.append(text)
                except Exception as e:
                    log_message(f"Ошибка чтения из shell: {e}")
                    break
        finally:
            log_message("Поток чтения shell завершён")
            self.running = False

    def write(self, cmd):
        if not self.running or not self.proc:
            return False
        try:
            if not cmd.endswith('\n'):
                cmd += '\n'
            self.proc.write(cmd.encode('utf-8'))
            return True
        except Exception as e:
            log_message(f"Ошибка записи в shell: {e}")
            return False

    def get_output(self):
        with self.lock:
            return ''.join(self.output_buffer)

    def stop(self):
        self.running = False
        if self.proc:
            try:
                self.proc.terminate()
            except:
                pass

@app.route('/api/cmd/send', methods=['POST'])
def cmd_send():
    data = request.get_json()
    if not data or 'command' not in data:
        return jsonify({'error': 'No command provided'}), 400
    cmd = data['command'].strip()
    if not cmd:
        return jsonify({'error': 'Empty command'}), 400

    if not shell_manager.running:
        shell_manager.start()

    if shell_manager.write(cmd):
        return jsonify({'status': 'ok'})
    else:
        return jsonify({'error': 'Shell not available'}), 500

@app.route('/api/cmd/output', methods=['GET'])
def cmd_output():
    if not shell_manager.running:
        shell_manager.start()
        time.sleep(0.5)
    output = shell_manager.get_output()
    return jsonify({'output': output})

# ---------- Видеопоток и управление камерой (без изменений) ----------
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if camera:
                frame = camera.get_frame()
                if frame is not None:
                    _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                else:
                    black = np.zeros((720, 1280, 3), dtype=np.uint8)
                    _, jpeg = cv2.imencode('.jpg', black)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                black = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black, "No Camera", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                _, jpeg = cv2.imencode('.jpg', black)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update', methods=['POST'])
def update_camera():
    data = request.json
    if camera is not None:
        alpha = data.get('alpha_depth')
        if alpha is not None:
            alpha = float(alpha) / 100.0
        show_left = data.get('show_left')
        num_disp = data.get('num_disp')
        camera.update_params(alpha_depth=alpha, show_left=show_left, num_disp=num_disp)
        return jsonify(ok=True)
    return jsonify(error='Camera not initialized'), 500

@app.route('/get_fps')
def get_fps():
    if camera:
        return jsonify(fps=f"{camera.fps:.1f}")
    return jsonify(fps="0.0")

@app.route('/api/depth', methods=['POST'])
def depth_at():
    data = request.json
    x = data.get('x')
    y = data.get('y')
    if x is None or y is None:
        return jsonify({'error': 'Missing coordinates'}), 400
    if camera is None:
        return jsonify({'depth': None})
    depth = camera.get_depth_at(int(x), int(y))
    return jsonify({'depth': depth})

# ---------- API для сервоприводов (обновлено) ----------
@app.route('/api/servo/<int:channel>/<int:angle>', methods=['POST'])
def set_servo(channel, angle):
    """Установка угла сервопривода. Диапазон определяется конфигурацией канала."""
    if servo_controller is None:
        return jsonify({'error': 'Servo controller not initialized'}), 500
    if channel not in servo_controller.channel_configs:
        return jsonify({'error': f'Channel {channel} not configured'}), 400
    # Проверка, что угол в пределах настроек канала (на всякий случай)
    min_angle, max_angle, _, _ = servo_controller.channel_configs[channel]
    if angle < min_angle or angle > max_angle:
        return jsonify({'error': f'Angle must be {min_angle}-{max_angle} for channel {channel}'}), 400
    success = servo_controller.set_servo(channel, angle)
    if success:
        return jsonify({'status': 'ok', 'channel': channel, 'angle': angle})
    else:
        return jsonify({'error': 'Failed to set servo'}), 500

@app.route('/api/servo/test', methods=['POST'])
def servo_test():
    """Запуск тестового цикла для всех четырёх сервоприводов."""
    if servo_controller is None:
        return jsonify({'error': 'Servo controller not initialized'}), 500
    def run_test():
        servo_controller.test_cycle(channels=[0,1,2,3], delay=1)
    threading.Thread(target=run_test, daemon=True).start()
    return jsonify({'status': 'ok', 'message': 'Test cycle started'})

@app.route('/api/servo/calibrate', methods=['POST'])
def servo_calibrate():
    """Калибровка импульсов для конкретного канала."""
    data = request.json
    if servo_controller is None:
        return jsonify({'error': 'Servo controller not initialized'}), 500
    channel = data.get('channel')
    if channel is None:
        return jsonify({'error': 'Missing channel'}), 400
    min_pulse = data.get('min_pulse')
    max_pulse = data.get('max_pulse')
    min_new, max_new = servo_controller.calibrate_channel(channel, min_pulse, max_pulse)
    return jsonify({'status': 'ok', 'channel': channel, 'min': min_new, 'max': max_new})

# ---------- Запуск сервера (без изменений) ----------
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
    url = f"http://127.0.0.1:{HTTP_PORT}/screen"
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

def wait_for_server(host='127.0.0.1', port=HTTP_PORT, timeout=15):
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
    global shell_manager, camera, servo_controller
    log_message("Запуск веб-сервера R2")

    # Инициализация shell
    shell_manager = ShellManager()
    shell_manager.start()

    # Инициализация камеры
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "cam_params.json")

    if os.path.exists(config_path):
        try:
            camera = StereoCamera(config_path, source=0)
            log_message("Камера инициализирована")
        except Exception as e:
            log_message(f"Ошибка инициализации камеры: {e}")
            camera = None
    else:
        log_message(f"Файл калибровки {config_path} не найден")
        try:
            files = os.listdir(script_dir)
            log_message(f"Файлы в {script_dir}: {files}")
        except:
            pass
        camera = None

    # Инициализация сервоприводов с новой конфигурацией
    try:
        # Можно явно передать конфигурацию, но используем умолчания из класса
        servo_controller = ServoController(bus=0, address=0x40, freq=50)
        log_message("Сервоконтроллер инициализирован с конфигурацией:")
        for ch, (min_a, max_a, min_p, max_p) in servo_controller.channel_configs.items():
            log_message(f"  Канал {ch}: {min_a}–{max_a}°, импульсы {min_p}–{max_p}")
    except Exception as e:
        log_message(f"Ошибка инициализации сервоконтроллера: {e}")
        servo_controller = None

    # Запуск браузера (опционально)
    threading.Thread(target=start_browser_when_ready, daemon=True).start()

    app.run(host='0.0.0.0', port=HTTP_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()
