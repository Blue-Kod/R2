import os
import subprocess
import sys
import threading
import time
import shutil
import mimetypes
import functools
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import (
    Flask, Response, jsonify, render_template, request,
    send_file, session, redirect, url_for, abort
)

from robov_core.high_level import (
    APP_PASSWORD, HTTP_HOST, HTTP_PORT, ROOT_DIR,
    get_stereo_camera, health_snapshot, ip_address,
    shell_output, shell_start, shell_write,
    set_emote, get_emote, supported_emotes,
    set_eyes_position, get_eyes_position, get_logs,
    get_servo_angles, get_servo_angles_physical, get_servo_offsets,
    set_servo_physical, log, cleanup,
    start_desktop, stop_desktop, get_desktop_frame,
    is_desktop_active, stop_desktop_safe,
)


_mjpeg_counter: int = 0
_mjpeg_fps: int = 0
_mjpeg_lock: threading.Lock = threading.Lock()

def _mjpeg_fps_worker() -> None:
    global _mjpeg_fps
    while True:
        time.sleep(1.0)
        with _mjpeg_lock:
            _mjpeg_fps = _mjpeg_counter
            _mjpeg_counter = 0

threading.Thread(target=_mjpeg_fps_worker, daemon=True, name="mjpeg-fps").start()


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(ROOT_DIR / "templates"))
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True
    app.secret_key = os.urandom(24).hex()

    # --- Auth helpers ---

    def require_auth(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not session.get("authenticated"):
                if request.path.startswith("/api/"):
                    return jsonify({"error": "Unauthorized"}), 401
                return redirect(url_for("login_page"))
            return f(*args, **kwargs)
        return wrapper

    # --- Login ---

    @app.route("/login", methods=["GET", "POST"])
    def login_page():
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            password = str(data.get("password", ""))
            if password == APP_PASSWORD:
                session["authenticated"] = True
                return jsonify({"status": "ok"})
            return jsonify({"error": "Wrong password"}), 403
        return render_template("login.html")

    @app.route("/api/login", methods=["POST"])
    def api_login():
        data = request.get_json(silent=True) or {}
        password = str(data.get("password", ""))
        if password == APP_PASSWORD:
            session["authenticated"] = True
            return jsonify({"status": "ok"})
        return jsonify({"error": "Wrong password"}), 403

    @app.route("/api/logout", methods=["POST"])
    def api_logout():
        session.clear()
        return jsonify({"status": "ok"})

    @app.route("/api/check-auth", methods=["GET"])
    def api_check_auth():
        return jsonify({"authenticated": session.get("authenticated", False)})

    # --- Main ---

    @app.route("/")
    @require_auth
    def index():
        camera = get_stereo_camera()
        camera_params = {}
        if camera:
            with camera.lock:
                camera_params = {
                    "depth_enabled": getattr(camera, 'depth_enabled', False),
                    "hud_enabled": getattr(camera, 'hud_enabled', False),
                    "wls_enabled": getattr(camera, 'wls_enabled', False),
                    "alpha_depth": getattr(camera, 'alpha_depth', 0.3),
                    "show_left": getattr(camera, 'show_left', True),
                    "num_disp": getattr(camera, 'num_disp', 128),
                    "fps": round(getattr(camera, 'fps', 0.0), 1),
                    "img_size": getattr(camera, 'img_size', [640, 360]),
                }

        servo_angles = get_servo_angles_physical()
        servo_offsets = get_servo_offsets()

        current_emote = get_emote()
        eyes_x, eyes_y = get_eyes_position()

        system_data = health_snapshot()
        system_data["ip"] = ip_address()

        return render_template("index.html",
                               camera_params=camera_params,
                               servo_angles=servo_angles,
                               servo_offsets=servo_offsets,
                               current_emote=current_emote,
                               eyes_position={"x": eyes_x, "y": eyes_y},
                               supported_emotes=supported_emotes(),
                               system_data=system_data)

    # --- API ---

    @app.route("/api/data")
    @require_auth
    def api_data():
        payload = health_snapshot()
        camera = get_stereo_camera()
        payload["fps"] = round(camera.fps, 1) if camera else 0.0
        payload["cam_w"] = camera.actual_width if camera else 0
        payload["cam_h"] = camera.actual_height if camera else 0
        with _mjpeg_lock:
            payload["stream_fps"] = _mjpeg_fps
        payload["logs"] = get_logs(500)
        return jsonify(payload)

    @app.route("/api/ip")
    @require_auth
    def api_ip():
        return jsonify({"ip": ip_address()})

    @app.route("/api/update", methods=["POST"])
    @require_auth
    def api_update():
        launcher_path = ROOT_DIR / "launcher.py"
        if not launcher_path.exists():
            return jsonify({"status": "error", "message": "launcher.py not found"}), 404
        try:
            subprocess.Popen([sys.executable, str(launcher_path)])
            log("Update process started")

            def shutdown():
                time.sleep(1)
                os._exit(0)

            threading.Thread(target=shutdown, daemon=True).start()
            return jsonify({"status": "ok", "message": "Update started"})
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/shutdown", methods=["POST"])
    @require_auth
    def api_shutdown():
        log("Shutdown command received")
        threading.Thread(target=lambda: (cleanup(), time.sleep(1), os._exit(0)), daemon=True).start()
        return jsonify({"status": "ok", "message": "Shutting down"})

    # --- Shell ---

    @app.route("/api/cmd/send", methods=["POST"])
    @require_auth
    def cmd_send():
        data = request.get_json(silent=True) or {}
        command = str(data.get("command", "")).strip()
        if not command:
            return jsonify({"error": "No command"}), 400
        if shell_write(command):
            return jsonify({"status": "ok"})
        return jsonify({"error": "Shell not available"}), 500

    @app.route("/api/cmd/output", methods=["GET"])
    @require_auth
    def cmd_output():
        if not shell_output():
            shell_start()
        time.sleep(0.1)
        return jsonify({"output": shell_output()})

    # --- Video ---

    @app.route("/video_feed")
    @require_auth
    def video_feed():
        def stream():
            global _mjpeg_counter
            while True:
                camera = get_stereo_camera()
                frame = camera.get_latest_frame() if camera else None
                if frame is None:
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "No Camera", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

                with _mjpeg_lock:
                    _mjpeg_counter += 1

                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"

        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # --- Camera ---

    @app.route("/update", methods=["POST"])
    @require_auth
    def update_camera():
        camera = get_stereo_camera()
        if not camera:
            return jsonify({"error": "Camera not initialized"}), 500
        data = request.get_json(silent=True) or {}
        alpha = data.get("alpha_depth")
        alpha = float(alpha) / 100.0 if alpha is not None else None
        camera.update_params(
            alpha_depth=alpha,
            show_left=data.get("show_left"),
            num_disp=data.get("num_disp"),
        )
        return jsonify({"ok": True})

    @app.route("/api/depth", methods=["POST"])
    @require_auth
    def depth_at():
        data = request.get_json(silent=True) or {}
        x, y = data.get("x"), data.get("y")
        if x is None or y is None:
            return jsonify({"error": "Missing coordinates"}), 400
        camera = get_stereo_camera()
        if not camera:
            return jsonify({"depth": None})
        return jsonify({"depth": camera.get_depth_at(int(x), int(y))})

    @app.route("/api/camera/params", methods=["GET", "POST"])
    @require_auth
    def camera_params():
        camera = get_stereo_camera()
        if not camera:
            return jsonify({"error": "Camera not initialized"}), 500
        if request.method == "GET":
            with camera.lock:
                return jsonify({
                    "depth_enabled": camera.depth_enabled,
                    "hud_enabled": camera.hud_enabled,
                    "wls_enabled": camera.wls_enabled,
                    "alpha_depth": camera.alpha_depth,
                    "show_left": camera.show_left,
                    "num_disp": camera.num_disp,
                })
        data = request.get_json(silent=True) or {}
        camera.update_params(
            depth_enabled=data.get("depth_enabled"),
            hud_enabled=data.get("hud_enabled"),
            wls_enabled=data.get("wls_enabled"),
            alpha_depth=data.get("alpha_depth"),
            show_left=data.get("show_left"),
            num_disp=data.get("num_disp"),
        )
        return jsonify({"status": "ok"})

    # --- Servo ---

    @app.route("/api/servo/<int:channel>/<int:angle>", methods=["POST"])
    @require_auth
    def set_servo(channel: int, angle: int):
        from robov_core.high_level import _servo as servo_ref
        if not servo_ref:
            return jsonify({"error": "Servo controller not initialized"}), 500
        if channel not in servo_ref.channel_configs:
            return jsonify({"error": f"Channel {channel} not configured"}), 400
        min_angle, max_angle, _, _ = servo_ref.channel_configs[channel]
        if angle < min_angle or angle > max_angle:
            return jsonify({"error": f"Angle must be {min_angle}-{max_angle}"}), 400
        if set_servo_physical(channel, angle):
            return jsonify({"status": "ok", "channel": channel, "angle": angle})
        return jsonify({"error": "Failed to set servo"}), 500

    # --- Emote / Eyes ---

    @app.route("/api/emote", methods=["GET", "POST"])
    @require_auth
    def api_emote():
        if request.method == "GET":
            return jsonify({"status": "ok", "emote": get_emote(), "supported": supported_emotes()})
        data = request.get_json(silent=True) or {}
        emotion_name = str(data.get("emotion_name") or "")
        if set_emote(emotion_name):
            return jsonify({"status": "ok", "emote": get_emote()})
        return jsonify({"status": "error", "message": "Unsupported emotion",
                        "supported": supported_emotes()}), 400

    @app.route("/api/eyes", methods=["GET", "POST"])
    @require_auth
    def api_eyes():
        if request.method == "GET":
            x, y = get_eyes_position()
            return jsonify({"status": "ok", "x": x, "y": y})
        data = request.get_json(silent=True) or {}
        try:
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))
        except (TypeError, ValueError):
            return jsonify({"status": "error", "message": "x and y must be numbers"}), 400
        set_eyes_position(x, y)
        x, y = get_eyes_position()
        return jsonify({"status": "ok", "x": x, "y": y})

    # --- Python exec ---

    @app.route("/api/python/exec", methods=["POST"])
    @require_auth
    def python_exec():
        import io
        import contextlib
        import traceback

        data = request.get_json(silent=True) or {}
        code = str(data.get("code", "")).strip()
        code = f"from robov_core.high_level import *\n{code}"

        if not code:
            return jsonify({"stdout": "", "stderr": "No code provided"}), 400

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec_globals = {"__name__": "__main__", "print": print}
                exec(code, exec_globals)
            return jsonify({"stdout": stdout_capture.getvalue(), "stderr": stderr_capture.getvalue()})
        except Exception as e:
            stderr_capture.write(traceback.format_exc())
            return jsonify({"stdout": stdout_capture.getvalue(), "stderr": stderr_capture.getvalue()}), 200

    # --- File manager ---

    @app.route("/file_manager")
    @require_auth
    def file_manager():
        return render_template("file_manager.html")

    def normalize_path(path_str: str) -> Path:
        if not path_str or path_str.strip() == "":
            return Path.cwd().resolve()
        path = Path(path_str).resolve()
        return path

    def get_file_info(file_path: Path) -> Optional[dict]:
        try:
            stat = file_path.stat()
            return {
                "name": file_path.name,
                "path": str(file_path).replace("\\", "/"),
                "type": "directory" if file_path.is_dir() else "file",
                "size": stat.st_size if file_path.is_file() else 0,
                "modified": stat.st_mtime,
                "permissions": oct(stat.st_mode)[-3:]
            }
        except Exception:
            return None

    @app.route("/api/files", methods=["GET"])
    @require_auth
    def api_files_list():
        path_str = request.args.get("path", "")
        path = normalize_path(path_str)
        if not path.exists():
            path = Path.cwd().resolve()
        items = []
        try:
            if path.is_dir():
                for item in path.iterdir():
                    info = get_file_info(item)
                    if info:
                        items.append(info)
            else:
                info = get_file_info(path)
                if info:
                    items.append(info)
        except PermissionError:
            return jsonify({"error": "Permission denied", "path": str(path).replace("\\", "/")}), 403
        except Exception as e:
            return jsonify({"error": str(e), "path": str(path).replace("\\", "/")}), 500
        items.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))
        return jsonify({"path": str(path).replace("\\", "/"), "items": items})

    @app.route("/api/files/read", methods=["GET"])
    @require_auth
    def api_files_read():
        path_str = request.args.get("path", "")
        path = normalize_path(path_str)
        if not path.exists() or not path.is_file():
            return jsonify({"error": "File not found"}), 404
        try:
            content = None
            encodings = ['utf-8', 'utf-8-sig', 'cp1251', 'latin1']
            for encoding in encodings:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            if content is None:
                with open(path, 'rb') as f:
                    content = f.read().decode('latin1', errors='replace')
            return jsonify({"content": content, "encoding": encoding if content else "binary",
                            "size": path.stat().st_size})
        except PermissionError:
            return jsonify({"error": "Permission denied"}), 403
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/files/write", methods=["POST"])
    @require_auth
    def api_files_write():
        data = request.get_json(silent=True) or {}
        path_str = data.get("path", "")
        content = data.get("content", "")
        path = normalize_path(path_str)
        if not path.parent.exists():
            return jsonify({"error": "Parent directory does not exist"}), 404
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return jsonify({"success": True, "message": "File saved successfully",
                            "size": len(content.encode('utf-8'))})
        except PermissionError:
            return jsonify({"error": "Permission denied"}), 403
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/files/create", methods=["POST"])
    @require_auth
    def api_files_create():
        data = request.get_json(silent=True) or {}
        path_str = data.get("path", "/")
        name = data.get("name", "")
        item_type = data.get("type", "file")
        if not name:
            return jsonify({"error": "Name is required"}), 400
        parent_path = normalize_path(path_str)
        item_path = parent_path / name
        if item_path.exists():
            return jsonify({"error": "Already exists"}), 409
        try:
            if item_type == "directory":
                item_path.mkdir(parents=True, exist_ok=True)
            else:
                item_path.parent.mkdir(parents=True, exist_ok=True)
                item_path.touch()
            return jsonify({"success": True, "message": f"{item_type.capitalize()} created successfully",
                            "path": str(item_path).replace("\\", "/")})
        except PermissionError:
            return jsonify({"error": "Permission denied"}), 403
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/files/delete", methods=["POST"])
    @require_auth
    def api_files_delete():
        data = request.get_json(silent=True) or {}
        path_str = data.get("path", "")
        path = normalize_path(path_str)
        if not path.exists():
            return jsonify({"error": "Path does not exist"}), 404
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return jsonify({"success": True, "message": "Deleted successfully"})
        except PermissionError:
            return jsonify({"error": "Permission denied"}), 403
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/files/rename", methods=["POST"])
    @require_auth
    def api_files_rename():
        data = request.get_json(silent=True) or {}
        old_path_str = data.get("old_path", "")
        new_name = data.get("new_name", "")
        if not new_name:
            return jsonify({"error": "New name is required"}), 400
        old_path = normalize_path(old_path_str)
        if not old_path.exists():
            return jsonify({"error": "Path does not exist"}), 404
        new_path = old_path.parent / new_name
        if new_path.exists():
            return jsonify({"error": "Target already exists"}), 409
        try:
            old_path.rename(new_path)
            return jsonify({"success": True, "message": "Renamed successfully",
                            "new_path": str(new_path).replace("\\", "/")})
        except PermissionError:
            return jsonify({"error": "Permission denied"}), 403
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/files/upload", methods=["POST"])
    @require_auth
    def api_files_upload():
        path_str = request.form.get("path", "/")
        parent_path = normalize_path(path_str)
        if not parent_path.exists():
            return jsonify({"error": "Target directory does not exist"}), 404
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No files provided"}), 400
        uploaded_count = 0
        errors = []
        for file in files:
            if file.filename == "":
                continue
            try:
                filename = file.filename
                file_path = parent_path / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file.save(str(file_path))
                uploaded_count += 1
            except Exception as e:
                errors.append(f"{file.filename}: {str(e)}")
        response_data = {"success": uploaded_count > 0,
                         "uploaded_count": uploaded_count,
                         "total_files": len([f for f in files if f.filename != ""])}
        if errors:
            response_data["errors"] = errors
        return jsonify(response_data)

    @app.route("/api/files/download", methods=["GET"])
    @require_auth
    def api_files_download():
        path_str = request.args.get("path", "")
        path = normalize_path(path_str)
        if not path.exists() or not path.is_file():
            return jsonify({"error": "File not found"}), 404
        try:
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type is None:
                mime_type = 'application/octet-stream'
            return send_file(str(path), as_attachment=True, download_name=path.name, mimetype=mime_type)
        except PermissionError:
            return jsonify({"error": "Permission denied"}), 403
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- Desktop remote ---

    @app.route("/desktop")
    @require_auth
    def desktop_page():
        return render_template("desktop.html")

    @app.route("/api/desktop/start", methods=["POST"])
    @require_auth
    def api_desktop_start():
        if start_desktop():
            return jsonify({"status": "ok"})
        return jsonify({"error": "Failed to start desktop capture"}), 500

    @app.route("/api/desktop/stop", methods=["POST"])
    @require_auth
    def api_desktop_stop():
        stop_desktop_safe()
        return jsonify({"status": "ok"})

    @app.route("/api/desktop/status", methods=["GET"])
    @require_auth
    def api_desktop_status():
        return jsonify({"active": is_desktop_active()})

    @app.route("/desktop_feed")
    @require_auth
    def desktop_feed():
        def stream():
            while True:
                frame = get_desktop_frame()
                if frame is None:
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Desktop inactive", (150, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                time.sleep(0.066)

        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app
