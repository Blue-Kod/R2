import sys
import os
from collections import deque

# ========== ПЕРЕХВАТ КОНСОЛЬНОГО ВЫВОДА (САМОЕ НАЧАЛО) ==========
console_log_buffer = deque(maxlen=500)

class Tee:
    def __init__(self, name, buffer, original):
        self.name = name
        self.buffer = buffer
        self.original = original

    def write(self, data):
        if not data:
            return
        # Преобразуем байты в строку, если нужно
        if isinstance(data, bytes):
            try:
                data = data.decode('utf-8', errors='replace')
            except:
                data = str(data)
        # Добавляем в буфер (всегда как есть, включая переносы строк)
        self.buffer.append(data)
        # Записываем в оригинальный вывод, если он не None
        if self.original and self.original is not self:
            try:
                self.original.write(data)
            except:
                pass

    def flush(self):
        if self.original and self.original is not self:
            try:
                self.original.flush()
            except:
                pass

    def isatty(self):
        return False

    def fileno(self):
        return -1

# Сохраняем оригинальные stdout/stderr (sys.__stdout__ уже содержит оригинал)
_original_stdout = sys.__stdout__
_original_stderr = sys.__stderr__

# Заменяем глобальные потоки
sys.stdout = Tee('stdout', console_log_buffer, _original_stdout)
sys.stderr = Tee('stderr', console_log_buffer, _original_stderr)

# Теперь можно безопасно импортировать остальные модули
import subprocess
import threading
import time
import socket
import datetime
import pwd
import shutil
import psutil
import numpy as np
from flask import Flask, render_template, jsonify, request, Response
import cv2

from camera import StereoCamera
from servo import ServoController

HTTP_PORT = 80

def log_message(*args):
    msg = " ".join(str(arg) for arg in args)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read()) / 1000
            return f"{temp:.1f}°C"
    except:
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
            except:
                pass
    try:
        output = subprocess.check_output(['journalctl', '-n', str(n), '--no-pager'], stderr=subprocess.DEVNULL, universal_newlines=True)
        return output.splitlines()
    except:
        return ["Нет доступа к системным логам"]

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

# ---------- Поток сервотрекинга (без изменений) ----------
def servo_tracking_loop():
    global servo_tracking_enabled, camera, servo_controller
    default_neck = 90
    default_head_tilt = 90
    max_neck_delta = 30
    max_tilt_delta = 15
    last_neck = default_neck
    last_tilt = default_head_tilt
    last_target_time = time.time()
    target_lost = True
    timeout_seconds = 10.0

    while True:
        if servo_tracking_enabled and camera is not None and servo_controller is not None:
            dx, dy = camera.get_eye_offsets()
            scale_x = camera.tracking_scale_x if camera.tracking_scale_x != 0 else 1.0
            scale_y = camera.tracking_scale_y if camera.tracking_scale_y != 0 else 1.0

            target_present = (dx != 0.0 or dy != 0.0)

            if target_present:
                last_target_time = time.time()
                target_lost = False

                neck_angle = default_neck + (dx / scale_x) * max_neck_delta
                neck_angle = max(default_neck - max_neck_delta, min(default_neck + max_neck_delta, neck_angle))
                tilt_angle = default_head_tilt + (dy / scale_y) * max_tilt_delta
                tilt_angle = max(default_head_tilt - max_tilt_delta, min(default_head_tilt + max_tilt_delta, tilt_angle))

                if abs(neck_angle - last_neck) > 1:
                    servo_controller.set_servo(0, int(round(neck_angle)), smooth=True, step_delay=0.01, step_angle=2)
                    last_neck = neck_angle
                if abs(tilt_angle - last_tilt) > 1:
                    servo_controller.set_servo(3, int(round(tilt_angle)), smooth=True, step_delay=0.01, step_angle=2)
                    last_tilt = tilt_angle
            else:
                if not target_lost:
                    target_lost = True
                    last_target_time = time.time()
                    log_message("Цель потеряна, ожидание {} секунд перед возвратом в центр".format(timeout_seconds))
                else:
                    elapsed = time.time() - last_target_time
                    if elapsed >= timeout_seconds:
                        if abs(last_neck - default_neck) > 1:
                            servo_controller.set_servo(0, default_neck, smooth=True, step_delay=0.01, step_angle=2)
                            last_neck = default_neck
                        if abs(last_tilt - default_head_tilt) > 1:
                            servo_controller.set_servo(3, default_head_tilt, smooth=True, step_delay=0.01, step_angle=2)
                            last_tilt = default_head_tilt
        else:
            target_lost = True
            last_target_time = time.time()
        time.sleep(0.05)

# ---------- Flask приложение ----------
app = Flask(__name__)

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

@app.route('/api/data')
def api_data():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    temp = get_cpu_temp()
    fps = camera.fps if camera else 0.0
    return jsonify({
        'cpu': cpu,
        'ram': ram,
        'temp': temp,
        'fps': round(fps, 1)
    })

@app.route('/api/console_logs')
def api_console_logs():
    # Возвращаем все строки буфера как единую строку
    logs = list(console_log_buffer)
    return jsonify({'logs': ''.join(logs)})

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
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/shutdown', methods=['POST'])
def api_shutdown():
    log_message("Получена команда на завершение")
    def shutdown():
        time.sleep(1)
        os._exit(0)
    threading.Thread(target=shutdown, daemon=True).start()
    return jsonify({'status': 'ok', 'message': 'Завершение работы'})

# ---------- Терминал (без изменений) ----------
class ShellManager:
    def __init__(self):
        self.proc = None
        self.output_buffer = deque(maxlen=2000)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
    def start(self):
        if self.running: return
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
    def _reader(self):
        try:
            while self.running:
                try:
                    data = self.proc.read(1024)
                    if not data: break
                    text = data.decode('utf-8', errors='replace')
                    with self.lock:
                        self.output_buffer.append(text)
                except: break
        finally:
            self.running = False
    def write(self, cmd):
        if not self.running or not self.proc: return False
        try:
            if not cmd.endswith('\n'): cmd += '\n'
            self.proc.write(cmd.encode('utf-8'))
            return True
        except: return False
    def get_output(self):
        with self.lock:
            return ''.join(self.output_buffer)
    def stop(self):
        self.running = False
        if self.proc:
            try: self.proc.terminate()
            except: pass

shell_manager = ShellManager()
shell_manager.start()

@app.route('/api/cmd/send', methods=['POST'])
def cmd_send():
    data = request.get_json()
    if not data or 'command' not in data: return jsonify({'error': 'No command'}), 400
    cmd = data['command'].strip()
    if not cmd: return jsonify({'error': 'Empty'}), 400
    if shell_manager.write(cmd): return jsonify({'status': 'ok'})
    else: return jsonify({'error': 'Shell not available'}), 500

@app.route('/api/cmd/output', methods=['GET'])
def cmd_output():
    if not shell_manager.running: shell_manager.start()
    time.sleep(0.1)
    return jsonify({'output': shell_manager.get_output()})

# ---------- Видеопоток ----------
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
    if camera:
        alpha = data.get('alpha_depth')
        if alpha is not None: alpha = float(alpha) / 100.0
        show_left = data.get('show_left')
        num_disp = data.get('num_disp')
        camera.update_params(alpha_depth=alpha, show_left=show_left, num_disp=num_disp)
        return jsonify(ok=True)
    return jsonify(error='Camera not initialized'), 500

@app.route('/api/depth', methods=['POST'])
def depth_at():
    data = request.json
    x, y = data.get('x'), data.get('y')
    if x is None or y is None: return jsonify({'error': 'Missing coordinates'}), 400
    if camera is None: return jsonify({'depth': None})
    depth = camera.get_depth_at(int(x), int(y))
    return jsonify({'depth': depth})

# ---------- API для облака точек ----------
@app.route('/api/pointcloud')
def api_pointcloud():
    if camera is None:
        return jsonify({'points': []})
    points = camera.get_pointcloud_subsampled(step=6)
    return jsonify({'points': points})

# ---------- API для параметров камеры и трекинга ----------
@app.route('/api/camera/params', methods=['GET', 'POST'])
def camera_params():
    if camera is None: return jsonify({'error': 'Camera not initialized'}), 500
    if request.method == 'GET':
        with camera.lock:
            params = {
                'depth_enabled': camera.depth_enabled,
                'face_tracking_enabled': camera.face_tracking_enabled,
                'tracking_mode': camera.tracking_mode,
                'tracking_scale_x': camera.tracking_scale_x,
                'tracking_scale_y': camera.tracking_scale_y,
                'tracking_offset_x': camera.tracking_offset_x,
                'tracking_offset_y': camera.tracking_offset_y,
                'alpha_depth': camera.alpha_depth,
                'show_left': camera.show_left,
                'num_disp': camera.num_disp,
                'wls_enabled': camera.wls_enabled,
            }
        return jsonify(params)
    else:
        data = request.json
        camera.update_params(
            depth_enabled=data.get('depth_enabled'),
            face_tracking_enabled=data.get('face_tracking_enabled'),
            tracking_mode=data.get('tracking_mode'),
            tracking_scale_x=data.get('tracking_scale_x'),
            tracking_scale_y=data.get('tracking_scale_y'),
            tracking_offset_x=data.get('tracking_offset_x'),
            tracking_offset_y=data.get('tracking_offset_y'),
            alpha_depth=data.get('alpha_depth'),
            show_left=data.get('show_left'),
            num_disp=data.get('num_disp'),
            wls_enabled=data.get('wls_enabled')
        )
        return jsonify({'status': 'ok'})

@app.route('/api/tracking/offsets')
def tracking_offsets():
    if camera is None: return jsonify({'dx': 0, 'dy': 0})
    dx, dy = camera.get_eye_offsets()
    return jsonify({'dx': dx, 'dy': dy})

# ---------- API для сервоприводов ----------
@app.route('/api/servo/<int:channel>/<int:angle>', methods=['POST'])
def set_servo(channel, angle):
    if servo_controller is None: return jsonify({'error': 'Servo controller not initialized'}), 500
    if channel not in servo_controller.channel_configs: return jsonify({'error': f'Channel {channel} not configured'}), 400
    min_angle, max_angle, _, _ = servo_controller.channel_configs[channel]
    if angle < min_angle or angle > max_angle: return jsonify({'error': f'Angle must be {min_angle}-{max_angle}'}), 400
    success = servo_controller.set_servo(channel, angle, smooth=True, step_delay=0.01, step_angle=2)
    if success: return jsonify({'status': 'ok', 'channel': channel, 'angle': angle})
    else: return jsonify({'error': 'Failed to set servo'}), 500

@app.route('/api/servo/tracking', methods=['GET', 'POST'])
def servo_tracking():
    global servo_tracking_enabled
    if request.method == 'GET':
        return jsonify({'enabled': servo_tracking_enabled})
    else:
        data = request.json
        servo_tracking_enabled = bool(data.get('enabled', False))
        log_message(f"Сервотрекинг {'включён' if servo_tracking_enabled else 'выключен'}")
        return jsonify({'status': 'ok', 'enabled': servo_tracking_enabled})

# ---------- Запуск сервера и браузера ----------
def get_display_user():
    if os.geteuid() != 0: return None
    user = os.environ.get('SUDO_USER')
    if user and user != 'root': return user
    for u in pwd.getpwall():
        if 1000 <= u.pw_uid < 65534: return u.pw_name
    return None

def run_browser_as_user(command):
    user = get_display_user()
    if not user:
        subprocess.Popen(command)
        return
    try:
        pw = pwd.getpwnam(user)
        uid, gid = pw.pw_uid, pw.pw_gid
        pid = os.fork()
        if pid == 0:
            os.setgid(gid)
            os.setuid(uid)
            os.environ['HOME'] = pw.pw_dir
            os.environ['USER'] = user
            os.environ['LOGNAME'] = user
            os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':0')
            xauth = os.path.join(pw.pw_dir, '.Xauthority')
            if os.path.exists(xauth): os.environ['XAUTHORITY'] = xauth
            subprocess.Popen(command)
            os._exit(0)
    except Exception as e:
        log_message(f"Не удалось переключиться на пользователя {user}: {e}")
        subprocess.Popen(command)

def open_browser_kiosk():
    url = f"http://127.0.0.1:{HTTP_PORT}/screen"
    is_root = (os.geteuid() == 0)
    if shutil.which("chromium-browser"):
        cmd = ["chromium-browser", "--kiosk", url]
        if is_root: cmd.insert(1, "--no-sandbox")
        run_browser_as_user(cmd)
    elif shutil.which("chromium"):
        cmd = ["chromium", "--kiosk", url]
        if is_root: cmd.insert(1, "--no-sandbox")
        run_browser_as_user(cmd)
    elif shutil.which("firefox"):
        run_browser_as_user(["firefox", "--kiosk", url])
    else:
        log_message("Не найден браузер, открываем обычный.")
        subprocess.Popen(["xdg-open", url])

def wait_for_server(host='127.0.0.1', port=HTTP_PORT, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except:
            time.sleep(0.5)
    return False

def start_browser_when_ready():
    if wait_for_server(timeout=15):
        time.sleep(5)
        open_browser_kiosk()
    else:
        log_message("Сервер не запустился вовремя.")

def main():
    global shell_manager, camera, servo_controller, servo_tracking_thread
    log_message("Запуск веб-сервера R2 (Liquid Glass Edition)")

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
        camera = None

    try:
        servo_controller = ServoController(bus=0, address=0x40, freq=50)
        log_message("Сервоконтроллер инициализирован")
        default_angles = {0: 90, 1: 135, 2: 135, 3: 90, 4: 135, 5: 135}
        for ch, angle in default_angles.items():
            if ch in servo_controller.channel_configs:
                servo_controller.set_servo(ch, angle, smooth=False)
                log_message(f"Серво {ch} установлено в {angle}°")
    except Exception as e:
        log_message(f"Ошибка инициализации сервоконтроллера: {e}")
        servo_controller = None

    servo_tracking_thread = threading.Thread(target=servo_tracking_loop, daemon=True)
    servo_tracking_thread.start()

    threading.Thread(target=start_browser_when_ready, daemon=True).start()
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()