import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from r2_app.config import AppConfig
from r2_app.high_level import (
    _camera,
    _servo,
    health_snapshot,
    ip_address,
    shell_output,
    shell_start,
    shell_write,
    gemini_test,
    set_emote,
    get_emote,
    supported_emotes,
    set_eyes_position,
    get_eyes_position,
    servo_tracking_enabled,
    set_servo_tracking,
    get_logs,
)


def create_app() -> Flask:
    config = AppConfig()
    app = Flask(__name__, template_folder=str(config.root_dir / "templates"))
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/data")
    def api_data():
        payload = health_snapshot()
        payload["fps"] = round(_camera.fps, 1) if _camera else 0.0
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
            from r2_app.high_level import log
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
        from r2_app.high_level import log, cleanup
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
                camera = _camera
                if camera:
                    frame = camera.get_frame()
                    if frame is None:
                        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "No Camera", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                time.sleep(0.03)

        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/update", methods=["POST"])
    def update_camera():
        camera = _camera
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
        camera = _camera
        if not camera:
            return jsonify({"depth": None})
        return jsonify({"depth": camera.get_depth_at(int(x), int(y))})

    @app.route("/api/camera/params", methods=["GET", "POST"])
    def camera_params():
        camera = _camera
        if not camera:
            return jsonify({"error": "Camera not initialized"}), 500
        if request.method == "GET":
            with camera.lock:
                return jsonify(
                    {
                        "depth_enabled": camera.depth_enabled,
                        "face_tracking_enabled": camera.face_tracking_enabled,
                        "tracking_mode": camera.tracking_mode,
                        "tracking_scale_x": camera.tracking_scale_x,
                        "tracking_scale_y": camera.tracking_scale_y,
                        "tracking_offset_x": camera.tracking_offset_x,
                        "tracking_offset_y": camera.tracking_offset_y,
                        "alpha_depth": camera.alpha_depth,
                        "show_left": camera.show_left,
                        "num_disp": camera.num_disp,
                        "wls_enabled": camera.wls_enabled,
                    }
                )

        data = request.get_json(silent=True) or {}
        camera.update_params(
            depth_enabled=data.get("depth_enabled"),
            face_tracking_enabled=data.get("face_tracking_enabled"),
            tracking_mode=data.get("tracking_mode"),
            tracking_scale_x=data.get("tracking_scale_x"),
            tracking_scale_y=data.get("tracking_scale_y"),
            tracking_offset_x=data.get("tracking_offset_x"),
            tracking_offset_y=data.get("tracking_offset_y"),
            alpha_depth=data.get("alpha_depth"),
            show_left=data.get("show_left"),
            num_disp=data.get("num_disp"),
            wls_enabled=data.get("wls_enabled"),
        )
        return jsonify({"status": "ok"})

    @app.route("/api/tracking/offsets")
    def tracking_offsets():
        camera = _camera
        if not camera:
            return jsonify({"dx": 0, "dy": 0})
        dx, dy = camera.get_eye_offsets()
        return jsonify({"dx": dx, "dy": dy})

    @app.route("/api/pointcloud")
    def api_pointcloud():
        camera = _camera
        if not camera or not camera.depth_enabled:
            return jsonify([])
        step = request.args.get("step", default=2, type=int)
        return jsonify(camera.get_point_cloud_sample(step=step))

    @app.route("/api/servo/<int:channel>/<int:angle>", methods=["POST"])
    def set_servo(channel, angle):
        servo = _servo
        if not servo:
            return jsonify({"error": "Servo controller not initialized"}), 500
        if channel not in servo.channel_configs:
            return jsonify({"error": f"Channel {channel} not configured"}), 400
        min_angle, max_angle, _, _ = servo.channel_configs[channel]
        if angle < min_angle or angle > max_angle:
            return jsonify({"error": f"Angle must be {min_angle}-{max_angle}"}), 400
        if servo.set_servo(channel, angle, smooth=True, step_delay=0.01, step_angle=2):
            return jsonify({"status": "ok", "channel": channel, "angle": angle})
        return jsonify({"error": "Failed to set servo"}), 500

    @app.route("/api/servo/tracking", methods=["GET", "POST"])
    def servo_tracking():
        if request.method == "GET":
            return jsonify({"enabled": servo_tracking_enabled})

        data = request.get_json(silent=True) or {}
        set_servo_tracking(bool(data.get("enabled", False)))
        from r2_app.high_level import log
        log(f"Servo tracking {'enabled' if servo_tracking_enabled else 'disabled'}")
        return jsonify({"status": "ok", "enabled": servo_tracking_enabled})

    @app.route("/api/ai/gemini/test", methods=["POST"])
    def gemini_test_route():
        data = request.get_json(silent=True) or {}
        prompt = str(data.get("prompt") or "Ping from R2")
        result = gemini_test(prompt=prompt)
        if result["ok"]:
            return jsonify({"status": "ok", "response": result["text"]})
        return jsonify({"status": "error", "message": result["error"]}), 400

    @app.route("/api/ai/command", methods=["POST"])
    def ai_command():
        """Send command to AI and get response."""
        data = request.get_json(silent=True) or {}
        command_text = str(data.get("command", "")).strip()
        
        if not command_text:
            return jsonify({"error": "No command provided"}), 400
        
        try:
            from ai import command as ai_command_func
            response = ai_command_func(command_text)
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/ai/get_current_response", methods=["GET"])
    def ai_get_current_response():
        """Возвращает текущий накопленный текст ответа AI."""
        try:
            from ai import get_current_response
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
            from ai import enable_ai_audio
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
        code = f"from r2_app.high_level import *\n{code}"
        
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

    return app