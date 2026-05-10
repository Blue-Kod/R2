import os
import subprocess
import sys
import threading
import time
import shutil
import mimetypes
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

from robov_core.config import AppConfig
from robov_core import high_level
from robov_core.high_level import (
    get_stereo_camera,
    health_snapshot,
    ip_address,
    shell_output,
    shell_start,
    shell_write,
    set_emote,
    get_emote,
    supported_emotes,
    set_eyes_position,
    get_eyes_position,
    get_logs,
    get_servo_angles,
    get_servo_angles_physical,
    get_servo_offsets,
    set_servo_physical,
)


def create_app() -> Flask:
    config = AppConfig()
    app = Flask(__name__, template_folder=str(config.root_dir / "templates"))
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

    @app.route("/")
    def index():
        # Get all dynamic data for page load
        camera = get_stereo_camera()
        camera_params = {}
        if camera:
            with camera.lock:
                camera_params = {
                    "depth_enabled": getattr(camera, 'depth_enabled', False),
                    "alpha_depth": getattr(camera, 'alpha_depth', 0.3),
                    "show_left": getattr(camera, 'show_left', True),
                    "num_disp": getattr(camera, 'num_disp', 128),
                    "fps": round(getattr(camera, 'fps', 0.0), 1),
                    "img_size": getattr(camera, 'img_size', [640, 360]),
                }
        
        # Get servo data (physical angles with inversion)
        servo_angles = get_servo_angles_physical()
        servo_offsets = get_servo_offsets()
        
        # Get current emote and eyes position
        current_emote = get_emote()
        eyes_x, eyes_y = get_eyes_position()
        
        # Get system data
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

    @app.route("/api/data")
    def api_data():
        payload = health_snapshot()
        camera = get_stereo_camera()
        payload["fps"] = round(camera.fps, 1) if camera else 0.0
        payload["logs"] = get_logs(500)
        return jsonify(payload)

    @app.route("/api/ip")
    def api_ip():
        return jsonify({"ip": ip_address()})

    @app.route("/api/update", methods=["POST"])
    def api_update():
        launcher_path = config.launcher_path
        if not launcher_path.exists():
            return jsonify({"status": "error", "message": "launcher.py not found"}), 404
        try:
            subprocess.Popen([sys.executable, str(launcher_path)])
            from robov_core.high_level import log
            log("Update process started")

            def shutdown():
                time.sleep(1)
                os._exit(0)

            threading.Thread(target=shutdown, daemon=True).start()
            return jsonify({"status": "ok", "message": "Update started"})
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/api/shutdown", methods=["POST"])
    def api_shutdown():
        from robov_core.high_level import log, cleanup
        log("Shutdown command received")

        def shutdown():
            cleanup()
            time.sleep(1)
            os._exit(0)

        threading.Thread(target=shutdown, daemon=True).start()
        return jsonify({"status": "ok", "message": "Shutting down"})

    @app.route("/api/cmd/send", methods=["POST"])
    def cmd_send():
        data = request.get_json(silent=True) or {}
        command = str(data.get("command", "")).strip()
        if not command:
            return jsonify({"error": "No command"}), 400
        if shell_write(command):
            return jsonify({"status": "ok"})
        return jsonify({"error": "Shell not available"}), 500

    @app.route("/api/cmd/output", methods=["GET"])
    def cmd_output():
        if not shell_output():
            shell_start()
        time.sleep(0.1)
        return jsonify({"output": shell_output()})

    @app.route("/video_feed")
    def video_feed():
        def stream():
            while True:
                camera = get_stereo_camera()
                if camera:
                    frame = camera.get_frame()
                    if frame is None:
                        frame = np.zeros((360, 640, 3), dtype=np.uint8)
                else:
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "No Camera", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Optimize JPEG encoding for performance
                _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])  # Reduced quality
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                time.sleep(0.066)  # Target ~15 FPS (1/15 ≈ 0.066)

        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/update", methods=["POST"])
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
    def camera_params():
        camera = get_stereo_camera()
        if not camera:
            return jsonify({"error": "Camera not initialized"}), 500
        if request.method == "GET":
            with camera.lock:
                return jsonify(
                    {
                        "depth_enabled": getattr(camera, 'depth_enabled', False),
                        "alpha_depth": getattr(camera, 'alpha_depth', 0.3),
                        "show_left": getattr(camera, 'show_left', True),
                        "num_disp": getattr(camera, 'num_disp', 7),
                    }
                )

        data = request.get_json(silent=True) or {}
        camera.update_params(
            depth_enabled=data.get("depth_enabled"),
            alpha_depth=data.get("alpha_depth"),
            show_left=data.get("show_left"),
            num_disp=data.get("num_disp"),
        )
        return jsonify({"status": "ok"})

    
    @app.route("/api/servo/<int:channel>/<int:angle>", methods=["POST"])
    def set_servo(channel, angle):
        servo = high_level._servo
        if not servo:
            return jsonify({"error": "Servo controller not initialized"}), 500
        if channel not in servo.channel_configs:
            return jsonify({"error": f"Channel {channel} not configured"}), 400
        min_angle, max_angle, _, _ = servo.channel_configs[channel]
        if angle < min_angle or angle > max_angle:
            return jsonify({"error": f"Angle must be {min_angle}-{max_angle}"}), 400
        if set_servo_physical(channel, angle):
            return jsonify({"status": "ok", "channel": channel, "angle": angle})
        return jsonify({"error": "Failed to set servo"}), 500

    @app.route("/api/ai/command", methods=["POST"])
    def ai_command():
        """Send command to AI and get response."""
        data = request.get_json(silent=True) or {}
        command_text = str(data.get("command", "")).strip()
        
        if not command_text:
            return jsonify({"error": "No command provided"}), 400
        
        try:
            from robov_core.ai import command as ai_command_func
            response = ai_command_func(command_text)
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/ai/get_current_response", methods=["GET"])
    def ai_get_current_response():
        """Возвращает текущий накопленный текст ответа AI."""
        try:
            from robov_core.ai import get_current_response
            current = get_current_response()
        except (ImportError, AttributeError):
            current = ""
        return jsonify({"response": current})

    @app.route("/api/ai/audio", methods=["POST"])
    def ai_audio():
        """Enable or disable AI audio output."""
        data = request.get_json(silent=True) or {}
        enabled = bool(data.get("enabled", True))
        
        try:
            from robov_core.ai import enable_ai_audio
            enable_ai_audio(enabled)
            return jsonify({"status": "ok", "audio_enabled": enabled})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/emote", methods=["GET", "POST"])
    def api_emote():
        if request.method == "GET":
            return jsonify(
                {
                    "status": "ok",
                    "emote": get_emote(),
                    "supported": supported_emotes(),
                }
            )
        data = request.get_json(silent=True) or {}
        emotion_name = str(data.get("emotion_name") or "")
        if set_emote(emotion_name):
            return jsonify({"status": "ok", "emote": get_emote()})
        return jsonify(
            {
                "status": "error",
                "message": "Unsupported emotion",
                "supported": supported_emotes(),
            }
        ), 400

    @app.route("/api/eyes", methods=["GET", "POST"])
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

    @app.route("/api/python/exec", methods=["POST"])
    def python_exec():
        """Execute Python code and return the result."""
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
                exec_globals = {
                    "__name__": "__main__",
                    "print": print,
                }
                exec(code, exec_globals)
            
            return jsonify({
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue()
            })
        except Exception as e:
            stderr_capture.write(traceback.format_exc())
            return jsonify({
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue()
            }), 200

    @app.route("/file_manager")
    def file_manager():
        """File manager page."""
        try:
            return render_template("file_manager.html")
        except Exception as e:
            # Log the error for debugging
            import traceback
            error_details = traceback.format_exc()
            print(f"File manager error: {error_details}")
            
            # Return a simple error page or redirect to main page
            return render_template("index.html", 
                               error=f"File manager temporarily unavailable: {str(e)}"), 500

    # ===================== FILE MANAGER (DEMO ONLY) =====================

    
    return app