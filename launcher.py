#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launcher for R2 robotics project - First-Run Installer and Process Manager.
Handles automated dependency installation, GitHub updates, and main.py execution.

Requirements:
- Must be run with root privileges
- Performs first-run installation of system and Python dependencies
- Manages updates from GitHub repository
- Launches main.py as subprocess
"""

import os
import sys
import subprocess
import tempfile
import zipfile
import shutil
import requests
import argparse
import platform
import filecmp
import datetime
import time
import pwd
import socket
import json
import shlex
from pathlib import Path

# Constants
REPO_URL = "https://github.com/Blue-Kod/R2"
ARCHIVE_URL = "https://github.com/Blue-Kod/R2/archive/refs/heads/main.zip"
REQUIREMENTS_FILE = "requirements.txt"

# Complete APT dependencies for Orange Pi 4 Pro / Debian Bullseye
REQUIREMENTS_APT = [
    "python3-pip",
    "python3-pygame",
    "libsdl2-2.0-0",
    "libsdl2-image-2.0-0",
    "libsdl2-ttf-2.0-0",
    "libportaudio2",
    "unclutter-xfixes",
    "libopencv-dev",
    "python3-opencv",
    "i2c-tools",
    "espeak-ng"
]

MAIN_SCRIPT = "main.py"
AUTOSTART_DESKTOP_FILE = "r2-monitor.desktop"
INTERNET_CHECK_HOST = "8.8.8.8"
LAST_COMMIT_FILE = ".last_commit"
SETUP_COMPLETE_FLAG = ".setup_complete"  # Flag for first-run installation

# Piper TTS model (Russian, Ruslan, medium)
TTS_MODEL_DIR = "models"
TTS_MODEL_NAME = "ru_RU-ruslan-medium"
TTS_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium"
TTS_MODEL_URL = f"{TTS_HF_BASE}/{TTS_MODEL_NAME}.onnx"
TTS_JSON_URL = f"{TTS_HF_BASE}/{TTS_MODEL_NAME}.onnx.json"

# LAS2 stereo depth model
LAS2_DIR = "models/las2"
LAS2_WEIGHT_DIR = "models/las2/checkpoints"
LAS2_HF_REPO = "tomtomtommi/LiteAnyStereoV2"
LAS2_WEIGHT_FILENAME = "LAS2_S.pth"
LAS2_ONNX_FILENAME = "las2_s_640x384.onnx"
LAS2_ONNX_PATH = "models/las2_s_640x384.onnx"

def log_message(*args):
    msg = " ".join(str(arg) for arg in args)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def check_root():
    """Verify script is running with root privileges. Exit if not."""
    if platform.system() != "Linux":
        return
    if os.geteuid() != 0:
        log_message("[!] This script must be run with sudo!")
        log_message("[!] Run: sudo python3 launcher.py")
        sys.exit(1)

def is_internet_available(timeout=3):
    try:
        socket.create_connection((INTERNET_CHECK_HOST, 53), timeout=timeout)
        return True
    except OSError:
        pass
    try:
        requests.get("https://github.com", timeout=timeout)
        return True
    except requests.RequestException:
        return False

def wait_for_internet(max_wait=60):
    log_message(f"[L] Waiting for internet (up to {max_wait}s)...")
    start = time.time()
    while time.time() - start < max_wait:
        if is_internet_available(timeout=2):
            log_message("[L] Internet available.")
            return True
        log_message("[L] No internet, waiting 5s...")
        time.sleep(5)
    log_message("[L] Internet not available within timeout.")
    return False

def get_display_user():
    """Returns the username of the graphical session user."""
    user = os.environ.get('SUDO_USER')
    if user and user != 'root':
        return user
    try:
        for u in pwd.getpwall():
            if 1000 <= u.pw_uid < 65534:
                return u.pw_name
    except:
        pass
    return 'orangepi'  # fallback

def fix_permissions(path, user):
    """Recursively change file ownership to user."""
    try:
        pw = pwd.getpwnam(user)
        uid, gid = pw.pw_uid, pw.pw_gid
        log_message(f"[L] Changing ownership of {path} to {user} ({uid}:{gid})")
        for root, dirs, files in os.walk(path):
            for d in dirs:
                os.chown(os.path.join(root, d), uid, gid)
            for f in files:
                os.chown(os.path.join(root, f), uid, gid)
        os.chown(path, uid, gid)
    except Exception as e:
        log_message(f"[!] Failed to change ownership: {e}")

def is_apt_package_installed(package):
    """Check if an APT package is installed."""
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f=${Status}", package],
            capture_output=True, text=True, timeout=10
        )
        return "install ok installed" in result.stdout
    except Exception:
        return False

def is_setup_complete():
    """Check if first-run setup has been completed."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    flag_path = os.path.join(script_dir, SETUP_COMPLETE_FLAG)
    return os.path.exists(flag_path)

def mark_setup_complete():
    """Create the setup complete flag file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    flag_path = os.path.join(script_dir, SETUP_COMPLETE_FLAG)
    try:
        with open(flag_path, 'w') as f:
            f.write(datetime.datetime.now().isoformat())
        log_message("[L] Setup complete flag created.")
        return True
    except Exception as e:
        log_message(f"[!] Failed to create setup flag: {e}")
        return False

def _download_file(url, dest_path):
    """Download a file with progress logging. Returns True on success."""
    try:
        log_message(f"[TTS] Downloading {os.path.basename(dest_path)}...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
        log_message(f"[TTS] Saved {os.path.basename(dest_path)} ({downloaded // (1024*1024)} MB)")
        return True
    except Exception as e:
        log_message(f"[TTS] Failed to download {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def ensure_tts_model():
    """Download Piper TTS model files if not present locally."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, TTS_MODEL_DIR)
    onnx_path = os.path.join(model_dir, f"{TTS_MODEL_NAME}.onnx")
    json_path = os.path.join(model_dir, f"{TTS_MODEL_NAME}.onnx.json")

    if os.path.exists(onnx_path) and os.path.exists(json_path):
        log_message("[TTS] Piper model found locally.")
        return True

    if not is_internet_available(timeout=5):
        log_message("[TTS] No internet - cannot download model.")
        return False

    os.makedirs(model_dir, exist_ok=True)
    log_message("[TTS] Piper model not found, downloading from HuggingFace...")

    ok = True
    if not os.path.exists(onnx_path):
        ok = _download_file(TTS_MODEL_URL, onnx_path) and ok
    if not os.path.exists(json_path):
        ok = _download_file(TTS_JSON_URL, json_path) and ok

    if ok:
        log_message("[TTS] Piper model ready.")
    else:
        log_message("[TTS] Model download incomplete - TTS may not work.")
    return ok


def setup_las():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(script_dir, LAS2_ONNX_PATH)
    weight_dir = os.path.join(script_dir, LAS2_WEIGHT_DIR)
    weight_path = os.path.join(weight_dir, LAS2_WEIGHT_FILENAME)

    if os.path.isfile(onnx_path):
        log_message("[LAS2] ONNX model found, skipping setup.")
        return True

    log_message("[LAS2] Starting LAS2-S setup...")

    if not os.path.isfile(weight_path):
        if not is_internet_available(timeout=5):
            log_message("[LAS2] No internet — cannot download weights.")
            return False
        log_message("[LAS2] Downloading LAS2_S.pth from HuggingFace...")
        os.makedirs(weight_dir, exist_ok=True)
        weight_url = f"https://huggingface.co/{LAS2_HF_REPO}/resolve/main/{LAS2_WEIGHT_FILENAME}"
        if not _download_file(weight_url, weight_path):
            log_message("[LAS2] Weight download failed.")
            return False

    log_message("[LAS2] Installing torch for ONNX export...")
    env = os.environ.copy()
    env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "torch", "torchvision", "timm", "onnx",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            capture_output=True, text=True, timeout=600, env=env,
        )
        if result.returncode != 0:
            log_message(f"[LAS2] torch install failed: {result.stderr}")
            return False
    except Exception as e:
        log_message(f"[LAS2] torch install error: {e}")
        return False

    log_message("[LAS2] Exporting ONNX model (640x384)...")
    export_script = os.path.join(script_dir, LAS2_DIR, "export_onnx.py")
    try:
        result = subprocess.run(
            [sys.executable, export_script,
             "--version", "las2", "--model_size", "s",
             "--restore_ckpt", weight_path,
             "--width", "640", "--height", "384", "--max_disp", "192",
             "--output_name", onnx_path],
            capture_output=True, text=True, timeout=300,
            cwd=os.path.join(script_dir, LAS2_DIR), env=env,
        )
        if result.returncode != 0:
            log_message(f"[LAS2] ONNX export failed: {result.stderr}")
            return False
        if not os.path.isfile(onnx_path):
            log_message("[LAS2] ONNX export did not produce output file.")
            return False
    except Exception as e:
        log_message(f"[LAS2] ONNX export error: {e}")
        return False

    log_message("[LAS2] Removing torch (no longer needed)...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y",
             "torch", "torchvision", "timm", "onnx"],
            capture_output=True, timeout=60, env=env,
        )
    except Exception:
        pass

    onnx_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    log_message(f"[LAS2] Setup complete. ONNX model: {onnx_mb:.1f} MB")
    return True

def install_apt_dependencies():
    """
    Install all required APT packages for first-run setup.
    Returns True on success, False on failure.
    """
    if platform.system() != "Linux":
        log_message("[APT] Only available on Linux.")
        return True

    log_message("[APT] Installing system packages...")
    
    # Update package lists
    log_message("[APT] Updating package lists...")
    try:
        result = subprocess.run(
            ["apt-get", "update", "-y"],
            capture_output=True, text=True, timeout=180
        )
        if result.returncode != 0:
            log_message(f"[APT] apt-get update failed: {result.stderr}")
            # Continue anyway, packages might be cached
    except Exception as e:
        log_message(f"[APT] Error during apt-get update: {e}")
        # Continue anyway

    # Install all packages
    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"

    failed_packages = []
    for package in REQUIREMENTS_APT:
        log_message(f"[APT] Checking/installing: {package}")
        if is_apt_package_installed(package):
            log_message(f"[APT] {package} already installed.")
            continue

        log_message(f"[APT] Installing {package}...")
        try:
            result = subprocess.run(
                ["apt-get", "install", "-y", "-f", package],
                capture_output=True, text=True, timeout=300,
                env=env
            )
            if result.returncode == 0:
                log_message(f"[APT] [+] {package} installed successfully.")
            else:
                log_message(f"[APT] [-] Failed to install {package}: {result.stderr}")
                failed_packages.append(package)
        except Exception as e:
            log_message(f"[APT] [!] Exception installing {package}: {e}")
            failed_packages.append(package)

    if failed_packages:
        log_message(f"[APT] Failed packages: {', '.join(failed_packages)}")
        return False

    log_message("[APT] All system packages installed successfully.")
    return True

def install_pip_dependencies():
    """
    Install all PIP packages from requirements.txt.
    Returns True on success, False on failure.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    req_path = os.path.join(script_dir, REQUIREMENTS_FILE)

    if not os.path.exists(req_path):
        log_message(f"[PIP] {REQUIREMENTS_FILE} not found, skipping.")
        return True

    log_message("[PIP] Installing Python packages from requirements.txt...")

    env = os.environ.copy()
    env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"

    try:
        result = subprocess.run(
            ["python3", "-m", "pip", "install", "--no-cache-dir", "-r", req_path],
            capture_output=True, text=True, timeout=300,
            env=env
        )

        if result.returncode == 0:
            log_message("[PIP] All Python packages installed successfully.")
            return True
        else:
            log_message(f"[PIP] pip install failed (code {result.returncode})")
            log_message(f"[PIP] Error: {result.stderr}")
            return False
    except Exception as e:
        log_message(f"[PIP] Exception during pip install: {e}")
        return False

def perform_first_run_setup():
    """
    Perform complete first-run installation.
    Returns True on success, False on failure.
    """
    log_message("=" * 60)
    log_message("[SETUP] Starting first-run installation...")
    log_message("=" * 60)

    # Install APT dependencies
    if not install_apt_dependencies():
        log_message("[SETUP] APT installation failed.")
        return False

    # Install PIP dependencies
    if not install_pip_dependencies():
        log_message("[SETUP] PIP installation failed.")
        return False

    # Mark setup as complete
    if not mark_setup_complete():
        log_message("[SETUP] Warning: Could not create setup flag.")

    log_message("=" * 60)
    log_message("[SETUP] First-run installation completed successfully!")
    log_message("[SETUP] Restarting launcher to load new environment...")
    log_message("=" * 60)

    return True

def restart_script():
    """Restart the launcher script using os.execv()."""
    script_path = os.path.abspath(__file__)
    log_message(f"[L] Restarting: {sys.executable} {script_path}")
    os.execv(sys.executable, [sys.executable, script_path] + sys.argv[1:])

def apply_self_update(new_launcher_path):
    current_script = os.path.abspath(__file__)

    if not os.path.exists(new_launcher_path):
        log_message(f"[!] Critical: Temporary file {new_launcher_path} not found.")
        return False

    if filecmp.cmp(current_script, new_launcher_path, shallow=False):
        log_message("[L] Launcher is up-to-date.")
        return False

    log_message("[L] New launcher version detected. Updating...")
    try:
        os.chmod(new_launcher_path, 0o755)
        shutil.copy2(new_launcher_path, current_script)
        log_message("[L] Launcher replaced. Restarting...")
        restart_script()
    except Exception as e:
        log_message(f"[!] Error replacing launcher: {e}")
        return False

def get_remote_head_commit_info(owner, repo, branch='main'):
    api_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}"
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code != 200:
            log_message(f"[!] GitHub API returned {response.status_code}")
            return None, None
        data = response.json()
        sha = data.get('sha')
        date_str = data.get('commit', {}).get('committer', {}).get('date')
        commit_date = None
        if date_str:
            commit_date = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        return sha, commit_date
    except Exception as e:
        log_message(f"[!] Error getting commit info: {e}")
        return None, None

def is_repo_up_to_date(target_dir):
    last_commit_path = os.path.join(target_dir, LAST_COMMIT_FILE)
    if not os.path.exists(last_commit_path):
        return False
    try:
        with open(last_commit_path, 'r') as f:
            local_sha = f.read().strip()
    except:
        return False
    remote_sha, _ = get_remote_head_commit_info("Blue-Kod", "R2", "main")
    if remote_sha is None:
        return False
    return local_sha == remote_sha

def save_last_commit_info(target_dir, sha):
    last_commit_path = os.path.join(target_dir, LAST_COMMIT_FILE)
    try:
        with open(last_commit_path, 'w') as f:
            f.write(sha)
        log_message(f"[L] Saved commit SHA: {sha[:8]}")
    except Exception as e:
        log_message(f"[!] Error saving .last_commit: {e}")

def download_and_extract_repo(target_dir, script_name, target_user):
    if is_repo_up_to_date(target_dir):
        return True

    remote_sha, _ = get_remote_head_commit_info("Blue-Kod", "R2", "main")
    if remote_sha is None:
        log_message("[!] Could not get remote commit SHA, update aborted.")
        return False

    try:
        log_message("[L] Downloading repository...")
        response = requests.get(ARCHIVE_URL, stream=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_zip = tmp_file.name

        with tempfile.TemporaryDirectory() as tmp_extract_dir:
            with zipfile.ZipFile(tmp_zip, 'r') as zip_ref:
                zip_ref.extractall(tmp_extract_dir)

            extracted_items = os.listdir(tmp_extract_dir)
            if not extracted_items:
                raise Exception("[!] Archive is empty")
            repo_root = os.path.join(tmp_extract_dir, extracted_items[0])
            if not os.path.isdir(repo_root):
                for item in extracted_items:
                    if os.path.isdir(os.path.join(tmp_extract_dir, item)):
                        repo_root = os.path.join(tmp_extract_dir, item)
                        break
                else:
                    raise Exception("[!] Could not find repo root folder")

            new_launcher_tmp = None
            for root, dirs, files in os.walk(repo_root):
                rel_path = os.path.relpath(root, repo_root)
                dest_dir = target_dir if rel_path == "." else os.path.join(target_dir, rel_path)
                if rel_path != ".":
                    os.makedirs(dest_dir, exist_ok=True)

                for file in files:
                    src_file = os.path.join(root, file)
                    if '.git' in rel_path.split(os.sep):
                        continue

                    if file == script_name:
                        log_message(f"[L] Found new {script_name} version, checking...")
                        fd, new_launcher_tmp = tempfile.mkstemp(prefix="launcher_new_", suffix=".py")
                        os.close(fd)
                        shutil.copy2(src_file, new_launcher_tmp)
                        continue

                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy2(src_file, dest_file)
                    log_message(f"[L] Copied: {os.path.join(rel_path, file) if rel_path != '.' else file}")

            if new_launcher_tmp:
                apply_self_update(new_launcher_tmp)

        os.unlink(tmp_zip)
        fix_permissions(target_dir, target_user)
        save_last_commit_info(target_dir, remote_sha)

        return True

    except Exception as e:
        log_message(f"[!] Error downloading/extracting repo: {e}")
        return False

def get_terminal_command(script_path, user):
    launcher_cmd = f"sudo python3 {script_path}"
    hold_cmd = 'echo; echo "Press any key to close..."; read'
    full_cmd = f"{launcher_cmd}; {hold_cmd}"

    if shutil.which("terminator"):
        return ["terminator", "-e", f"bash -c '{full_cmd}'"]
    elif shutil.which("gnome-terminal"):
        return ["gnome-terminal", "--", "bash", "-c", full_cmd]
    elif shutil.which("x-terminal-emulator"):
        return ["x-terminal-emulator", "-e", f"bash -c '{full_cmd}'"]
    elif shutil.which("xterm"):
        return ["xterm", "-hold", "-e", f"bash -c '{full_cmd}'"]
    return None

def setup_autostart_linux(target_user):
    script_path = os.path.abspath(__file__)
    try:
        pw = pwd.getpwnam(target_user)
        user_home = pw.pw_dir
        uid, gid = pw.pw_uid, pw.pw_gid
    except KeyError:
        log_message(f"[!] User {target_user} not found.")
        return False

    autostart_dir = os.path.join(user_home, ".config", "autostart")
    os.makedirs(autostart_dir, exist_ok=True)
    desktop_file_path = os.path.join(autostart_dir, AUTOSTART_DESKTOP_FILE)

    terminal_cmd = get_terminal_command(script_path, target_user)
    if not terminal_cmd:
        log_message("[!] No suitable terminal found.")
        return False

    cmd_str = " ".join(shlex.quote(arg) for arg in terminal_cmd)

    display = os.environ.get('DISPLAY', ':0')
    xauth = os.environ.get('XAUTHORITY', f"{user_home}/.Xauthority")

    desktop_content = f"""[Desktop Entry]
Type=Application
Name=R2 Main Program
Exec={cmd_str}
Path={os.path.dirname(script_path)}
Environment="DISPLAY={display}" "XAUTHORITY={xauth}"
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Phase=Applications
"""
    try:
        with open(desktop_file_path, 'w') as f:
            f.write(desktop_content)
        os.chown(desktop_file_path, uid, gid)
        log_message(f"[L] Autostart installed: {desktop_file_path}")
        return True
    except Exception as e:
        log_message(f"[!] Error creating .desktop file: {e}")
        return False

def remove_autostart_linux(target_user):
    try:
        pw = pwd.getpwnam(target_user)
        user_home = pw.pw_dir
    except KeyError:
        log_message(f"[!] User {target_user} not found.")
        return
    desktop_file = os.path.join(user_home, ".config", "autostart", AUTOSTART_DESKTOP_FILE)
    if os.path.exists(desktop_file):
        try:
            os.remove(desktop_file)
            log_message(f"[L] .desktop file removed.")
        except Exception as e:
            log_message(f"[!] Error removing: {e}")

def is_autostart_installed(target_user):
    try:
        pw = pwd.getpwnam(target_user)
        user_home = pw.pw_dir
    except KeyError:
        return False
    desktop_file = os.path.join(user_home, ".config", "autostart", AUTOSTART_DESKTOP_FILE)
    return os.path.exists(desktop_file)

def start_main():
    """Launch main.py as a subprocess."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(script_dir, MAIN_SCRIPT)
    if not os.path.exists(main_path):
        log_message(f"[!] {MAIN_SCRIPT} not found.")
        return False
    try:
        log_message(f"[L] Starting {MAIN_SCRIPT}...")

        system_name = platform.system()
        if system_name == "Windows":
            subprocess.Popen(
                ["python3", main_path],
                cwd=script_dir,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            return True

        if system_name == "Linux":
            env = os.environ.copy()
            env["DISPLAY"] = ":0"
            # Get XAUTHORITY from environment or construct from user home
            xauth = os.environ.get('XAUTHORITY')
            if not xauth:
                try:
                    import pwd
                    user = get_display_user()
                    pw = pwd.getpwnam(user)
                    xauth = os.path.join(pw.pw_dir, '.Xauthority')
                except:
                    xauth = '/root/.Xauthority'
            env["XAUTHORITY"] = xauth
            env["PYTHONPATH"] = script_dir

            python_cmd = f"python3 {shlex.quote(main_path)}"
            hold_cmd = 'echo; echo "main.py finished. Press any key to close."; read'
            full_cmd = f"{python_cmd}; {hold_cmd}"

            if shutil.which("terminator"):
                subprocess.Popen(["terminator", "-e", f"bash -c '{full_cmd}'"], cwd=script_dir, env=env)
                return True
            if shutil.which("gnome-terminal"):
                subprocess.Popen(["gnome-terminal", "--", "bash", "-c", full_cmd], cwd=script_dir, env=env)
                return True
            if shutil.which("x-terminal-emulator"):
                subprocess.Popen(["x-terminal-emulator", "-e", f"bash -c '{full_cmd}'"], cwd=script_dir, env=env)
                return True
            if shutil.which("xterm"):
                subprocess.Popen(["xterm", "-hold", "-e", f"bash -c '{full_cmd}'"], cwd=script_dir, env=env)
                return True

            log_message("[!] No GUI terminal found, running in current process.")
            subprocess.Popen(["python3", main_path], cwd=script_dir)
            return True

        subprocess.Popen(["python3", main_path], cwd=script_dir)
        return True
    except Exception as e:
        log_message(f"[!] Error starting {MAIN_SCRIPT}: {e}")
        return False

def main():
    check_root()

    parser = argparse.ArgumentParser(description="Launcher for R2 project", add_help=False)
    parser.add_argument("--install-autostart", action="store_true")
    parser.add_argument("--remove-autostart", action="store_true")
    parser.add_argument("--no-start", action="store_true")
    parser.add_argument("--dont-install-autostart", action="store_true")
    args, unknown = parser.parse_known_args()

    target_user = get_display_user()
    log_message(f"[L] Target user: {target_user}")

    if args.install_autostart or args.remove_autostart:
        if platform.system() != "Linux":
            log_message("[!] Autostart only available on Linux.")
            sys.exit(1)
        if args.install_autostart:
            setup_autostart_linux(target_user)
        elif args.remove_autostart:
            remove_autostart_linux(target_user)
        sys.exit(0)

    log_message(r"""
   _____     ___  
   |  __ \  |__ \ 
   | |__) |    ) |
   |  _  /    / / 
   | | \ \   / /_ 
   |_|  \_\ |____|
  -----------------
  >  Launcher.py  <
  -----------------
  """)


    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.basename(__file__)
    os.chdir(script_dir)
    log_message(f"[L] Working directory: {script_dir}")

    # FIRST-RUN SETUP CHECK
    if not is_setup_complete():
        log_message("[SETUP] First run detected - performing installation...")
        if not perform_first_run_setup():
            log_message("[SETUP] Setup failed. Exiting.")
            sys.exit(1)
        # Restart script to ensure new environment is loaded
        restart_script()
        # restart_script() calls os.execv() which replaces the process,
        # so we should never reach this point
        return

    if platform.system() == "Linux" and shutil.which("unclutter"):
        subprocess.Popen(["unclutter", "--timeout", "5", "--fork"])

    log_message("[L] Setup already complete, proceeding...")

    # Autostart setup — always refresh to pick up any changes
    if platform.system() == "Linux" and not args.dont_install_autostart:
        if not is_autostart_installed(target_user):
            log_message("[L] Autostart not found. Installing...")
        else:
            log_message("[L] Updating autostart...")
        setup_autostart_linux(target_user)

    # Internet check and repo update
    internet_ok = wait_for_internet(max_wait=60)
    repo_updated = False

    if internet_ok:
        log_message("[L] Updating repository from GitHub...")
        repo_updated = download_and_extract_repo(script_dir, script_name, target_user)
        if not repo_updated:
            log_message("[*] Repository update failed or not needed.")

        log_message("[L] Ensuring TTS model is available...")
        ensure_tts_model()

        log_message("[L] Setting up LAS2 stereo depth model...")
        setup_las()
    else:
        log_message("[*] No internet, skipping update.")

    # Start main.py
    if not args.no_start:
        time.sleep(2)
        start_main()
    else:
        log_message("[L] main.py launch skipped.")

    log_message("[L] Launcher work completed.")

if __name__ == "__main__":
    main()