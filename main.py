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
import json
import cv2
import numpy as np
from collections import deque

import psutil
from flask import Flask, render_template, jsonify, request, Response

# Попытка импорта AI-компонентов (если не установлены, AI будет отключён)
try:
    from ultralytics import YOLO
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("AI modules not found. Running without AI.")

# ---------- Класс стерео-движка (из camera_test.py) ----------
class StereoEngine:
    def __init__(self, config_path, source=0):
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.img_size = tuple(cfg['imSize'])  # 1280x720
        self.low_size = (self.img_size[0] // 2, self.img_size[1] // 2)  # 640x360

        # Матрицы калибровки
        self.Kl, self.Dl = np.array(cfg['Kl']), np.array(cfg['Dl'])
        self.Kr, self.Dr = np.array(cfg['Kr']), np.array(cfg['Dr'])
        self.R, self.T = np.array(cfg['R']), np.array(cfg['T'])

        # Ректификация
        self.R1, self.R2, self.P1, self.P2, self.Q = cv2.fisheye.stereoRectify(
            self.Kl, self.Dl, self.Kr, self.Dr, self.img_size, self.R, self.T, flags=0
        )
        self.mapL1, self.mapL2 = cv2.fisheye.initUndistortRectifyMap(self.Kl, self.Dl, self.R1, self.P1, self.img_size,
                                                                     cv2.CV_16SC2)
        self.mapR1, self.mapR2 = cv2.fisheye.initUndistortRectifyMap(self.Kr, self.Dr, self.R2, self.P2, self.img_size,
                                                                     cv2.CV_16SC2)

        self.Q_low = self.Q.copy()
        self.Q_low[:2, :3] *= 0.5

        # AI: YOLOv8-seg на CPU (если доступно)
        self.ai_enabled = AI_AVAILABLE
        if AI_AVAILABLE:
            print("Initializing YOLOv8-seg on CPU...")
            self.model = YOLO('yolov8n-seg.pt')
            self.model.to('cpu')
        else:
            self.model = None

        # Параметры управления
        self.num_disp = 7          # будет умножено на 16
        self.block_size = 11
        self.alpha_depth = 0.3
        self.alpha_seg = 0.5
        self.mainLeft = True

        self._init_matchers()

        self.raw_frame = None
        self.processed_view = None
        self.points_3d = None
        self.fps = 0
        self.running = True
        self.lock = threading.Lock()

        # Видеозахват с правильным бэкендом для Linux
        if os.name == 'nt':
            backend = cv2.CAP_DSHOW
        else:
            backend = cv2.CAP_V4L2

        self.cap = cv2.VideoCapture(source, backend)
        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть камеру {source}")

        # Пытаемся установить нужное разрешение
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Проверяем реальное разрешение
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Реальное разрешение камеры: {w} x {h}")

        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._processing_loop, daemon=True).start()

    def _init_matchers(self):
        max_d = self.num_disp * 16
        self.matcher_l = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=max_d, blockSize=self.block_size,
            P1=8 * 3 * self.block_size ** 2, P2=32 * 3 * self.block_size ** 2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        # Для WLS нужен правый матчер
        try:
            self.matcher_r = cv2.ximgproc.createRightMatcher(self.matcher_l)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.matcher_l)
            self.wls_filter.setLambda(8000)
            self.wls_filter.setSigmaColor(1.2)
            self.wls_available = True
        except AttributeError:
            print("WLS filter not available (install opencv-contrib-python). Disabling WLS.")
            self.wls_available = False
            self.matcher_r = None

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.raw_frame = frame
            else:
                time.sleep(0.001)

    def _processing_loop(self):
        last_time = time.time()
        H, W = self.img_size[1], self.img_size[0]

        while self.running:
            if self.raw_frame is None:
                time.sleep(0.005)
                continue

            frame = self.raw_frame
            if frame.shape[1] == 2560 and frame.shape[0] == 720:
                imgL = frame[:, :1280]
                imgR = frame[:, 1280:]
            else:
                mid = frame.shape[1] // 2
                imgL = frame[:, :mid]
                imgR = frame[:, mid:]
                imgL = cv2.resize(imgL, self.img_size)
                imgR = cv2.resize(imgR, self.img_size)

            rectL = cv2.remap(imgL, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
            rectR = cv2.remap(imgR, self.mapR1, self.mapR2, cv2.INTER_LINEAR)

            main_view = rectL if self.mainLeft else rectR

            # AI Инференс
            results = None
            if self.ai_enabled and self.model is not None:
                small_main = cv2.resize(main_view, (640, 360))
                results = self.model.predict(small_main, stream=False, verbose=False, imgsz=640)[0]

            # Стерео матчинг на половинном разрешении
            lowL = cv2.resize(rectL, self.low_size, interpolation=cv2.INTER_AREA)
            lowR = cv2.resize(rectR, self.low_size, interpolation=cv2.INTER_AREA)

            grayL = cv2.cvtColor(lowL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(lowR, cv2.COLOR_BGR2GRAY)

            dispL = self.matcher_l.compute(grayL, grayR).astype(np.float32) / 16.0

            if self.wls_available and self.matcher_r is not None:
                dispR = self.matcher_r.compute(grayR, grayL).astype(np.float32) / 16.0
                filtered = self.wls_filter.filter(dispL, lowL, disparity_map_right=dispR)
                d_float = filtered
            else:
                d_float = dispL

            with self.lock:
                self.points_3d = cv2.reprojectImageTo3D(d_float, self.Q_low)

                # Визуализация глубины
                disp_vis = np.clip((d_float / (self.num_disp * 16)) * 255, 0, 255).astype(np.uint8)
                disp_color = cv2.resize(cv2.applyColorMap(disp_vis, cv2.COLORMAP_MAGMA), self.img_size)

                canvas = cv2.addWeighted(main_view, 1.0 - self.alpha_depth, disp_color, self.alpha_depth, 0)

                # AI Маски (если есть)
                if self.ai_enabled and results is not None and results.masks is not None:
                    masks_np = results.masks.data.cpu().numpy()
                    for i, mask in enumerate(masks_np):
                        mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
                        mask_bool = mask_resized > 0.5
                        cls_id = int(results.boxes[i].cls[0])
                        color = [int(c) for c in np.random.RandomState(cls_id).randint(60, 255, 3)]

                        obj_layer = canvas.copy()
                        obj_layer[mask_bool] = color
                        cv2.addWeighted(obj_layer, self.alpha_seg, canvas, 1.0 - self.alpha_seg, 0, canvas)

                        # Расстояние до объекта
                        m_low = cv2.resize(mask_bool.astype(np.uint8), self.low_size,
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                        z_vals = self.points_3d[m_low, 2]
                        valid = z_vals[(z_vals > 50) & (z_vals < 15000)]
                        if len(valid) > 0:
                            mz = np.median(valid) / 10.0
                            coords = np.argwhere(mask_bool)
                            if len(coords) > 0:
                                cy, cx = coords.mean(axis=0).astype(int)
                                cv2.putText(canvas, f"{results.names[cls_id].upper()} {mz:.1f}cm", (cx - 50, cy),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                self.processed_view = canvas
                self.fps = 1.0 / (time.time() - last_time)
                last_time = time.time()

    def update_params(self, nD, a_depth, a_seg, ai_on, m_left):
        with self.lock:
            self.alpha_depth = a_depth / 100
            self.alpha_seg = a_seg / 100
            self.ai_enabled = ai_on and AI_AVAILABLE
            self.mainLeft = m_left
            if nD != self.num_disp:
                self.num_disp = nD
                self._init_matchers()

    def get_depth_at(self, x, y):
        """Возвращает расстояние в см для пикселя (x, y) на основном виде."""
        if self.points_3d is None:
            return None
        # Преобразуем координаты из full-resolution в low-resolution (points_3d хранится в low_size)
        scale_x = self.low_size[0] / self.img_size[0]
        scale_y = self.low_size[1] / self.img_size[1]
        lx = int(x * scale_x)
        ly = int(y * scale_y)
        if lx < 0 or lx >= self.low_size[0] or ly < 0 or ly >= self.low_size[1]:
            return None
        z = self.points_3d[ly, lx, 2]  # Z координата в миллиметрах (из Q матрицы)
        if z > 0 and z < 15000:
            return z / 10.0  # в см
        return None

# ---------- Глобальные объекты ----------
LOG_FILE = "logs.txt"
shell_manager = None  # будет инициализирован позже
engine = None         # будет инициализирован после создания app

# ---------- Вспомогательные функции ----------
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

def get_ip_address():
    """Возвращает локальный IP-адрес (первый не loopback)."""
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

# ---------- Маршруты страниц ----------
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

# ---------- API для данных и управления ----------
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

# ---------- API для терминала ----------
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

# ---------- API для камеры ----------
@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            if engine and engine.processed_view is not None:
                _, buf = cv2.imencode('.jpg', engine.processed_view, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.01)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update', methods=['POST'])
def update_camera():
    d = request.json
    engine.update_params(d['nD'], d['a_depth'], d['a_seg'], d['ai_on'], d['m_left'])
    return jsonify(ok=True)

@app.route('/get_fps')
def get_fps():
    return jsonify(fps=f"{engine.fps:.1f}" if engine else "0.0")

@app.route('/api/depth', methods=['POST'])
def depth_at():
    """Возвращает расстояние в см для точки (x, y) на изображении."""
    data = request.get_json()
    x = data.get('x')
    y = data.get('y')
    if x is None or y is None:
        return jsonify({'error': 'Missing coordinates'}), 400
    if engine is None:
        return jsonify({'depth': None})
    depth = engine.get_depth_at(int(x), int(y))
    return jsonify({'depth': depth})

# ---------- Запуск сервера ----------
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
    global shell_manager, engine
    log_message("Запуск веб-сервера R2")
    
    # Инициализация глобальных компонентов
    shell_manager = ShellManager()
    shell_manager.start()
    
    # Инициализация камеры (путь к файлу калибровки)
    config_path = "cam_params.json"  # Убедитесь, что файл существует
    if os.path.exists(config_path):
        try:
            engine = StereoEngine(config_path, source=0)
            log_message("Камера инициализирована")
        except Exception as e:
            log_message(f"Ошибка инициализации камеры: {e}")
            engine = None
    else:
        log_message(f"Файл калибровки {config_path} не найден, камера недоступна")
        engine = None

    # Запуск браузера в режиме киоска (если нужно)
    threading.Thread(target=start_browser_when_ready, daemon=True).start()

    # Запуск Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()
