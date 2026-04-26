#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launcher for auto-updating and starting the application from GitHub repository.
WARNING: This script must be run with root privileges (via sudo).
If run as a regular user - it will report an error and terminate.
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
BASE_DIR = "/home/orangepi/R2/"
REPO_URL = "https://github.com/Blue-Kod/R2"
ARCHIVE_URL = "https://github.com/Blue-Kod/R2/archive/refs/heads/main.zip"
REQUIREMENTS_FILE = "requirements.txt"
MAIN_SCRIPT = "main.py"
AUTOSTART_DESKTOP_FILE = "r2-monitor.desktop"
INTERNET_CHECK_HOST = "8.8.8.8"
LAST_COMMIT_FILE = ".last_commit"
DEPS_UPDATED_FLAG = ".deps_updated"      # flag that dependencies already checked for current commit

def log_message(*args):
    msg = " ".join(str(arg) for arg in args)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def check_root():
    """Check if script is run with root privileges."""
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
    log_message(f"[L] Waiting for internet up to {max_wait} sec...")
    start = time.time()
    while time.time() - start < max_wait:
        if is_internet_available(timeout=2):
            log_message("[L] Internet available.")
            return True
        log_message("[L] Internet unavailable, waiting 5 sec...")
        time.sleep(5)
    log_message("[L] Internet did not appear in the allotted time.")
    return False

def get_display_user():
    """Returns the username of the graphical session (real person)."""
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
    """Recursively change owner of files in path to user."""
    try:
        pw = pwd.getpwnam(user)
        uid, gid = pw.pw_uid, pw.pw_gid
        log_message(f"[L] Changing owner of {path} to {user} ({uid}:{gid})")
        for root, dirs, files in os.walk(path):
            for d in dirs:
                os.chown(os.path.join(root, d), uid, gid)
            for f in files:
                os.chown(os.path.join(root, f), uid, gid)
        os.chown(path, uid, gid)
    except Exception as e:
        log_message(f"[!] Failed to change owner: {e}")

def apply_self_update(new_launcher_path):
    """Replace current launcher with new version and restart."""
    current_script = os.path.join(BASE_DIR, os.path.basename(__file__))
    if filecmp.cmp(current_script, new_launcher_path, shallow=False):
        log_message("[L] Current launcher version is up to date.")
        os.unlink(new_launcher_path)
        return False

    log_message("[L] New launcher version detected. Replacing and restarting...")
    try:
        shutil.move(new_launcher_path, current_script)
        st = os.stat(current_script)
        os.chmod(current_script, st.st_mode)
        log_message("[L] Launcher successfully updated. Restarting...")
        # Use sys.executable to get the actual Python interpreter path
        python_exe = sys.executable if sys.executable else "/usr/bin/python3"
        os.execv(python_exe, [python_exe, current_script] + sys.argv[1:])
    except Exception as e:
        log_message(f"[!] Error during self-update: {e}")
        try:
            os.unlink(new_launcher_path)
        except:
            pass
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
        log_message("[!] Failed to get remote commit SHA, update aborted.")
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
                    raise Exception("[!] Could not find repository root folder")

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
                        log_message(f"[L] Found new version of {script_name}, checking if update needed...")
                        fd, new_launcher_tmp = tempfile.mkstemp(prefix="launcher_new_", suffix=".py")
                        os.close(fd)
                        shutil.copy2(src_file, new_launcher_tmp)
                        continue

                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy2(src_file, dest_file)
                    log_message(f"[L] Copied: {os.path.join(rel_path, file) if rel_path != '.' else file}")

        os.unlink(tmp_zip)

        # After update, remove dependencies flag so they get rechecked on next run
        deps_flag = os.path.join(target_dir, DEPS_UPDATED_FLAG)
        if os.path.exists(deps_flag):
            os.remove(deps_flag)
            log_message("[L] Dependencies flag reset.")

        fix_permissions(target_dir, target_user)
        save_last_commit_info(target_dir, remote_sha)

        if new_launcher_tmp:
            apply_self_update(new_launcher_tmp)
        return True

    except Exception as e:
        log_message(f"[!] Error downloading/extracting repository: {e}")
        return False

def mark_dependencies_updated():
    """Creates a flag file indicating dependencies for current commit are updated."""
    flag_path = os.path.join(BASE_DIR, DEPS_UPDATED_FLAG)
    try:
        with open(flag_path, 'w') as f:
            f.write(datetime.datetime.now().isoformat())
        log_message("[L] Dependencies update flag set.")
    except Exception as e:
        log_message(f"[!] Failed to create flag: {e}")

def install_requirements(force=False):
    """
    Installs pip dependencies.
    force=True - force installation (e.g., after repo update).
    If force=False, checks flag and pip dependency matching against requirements.txt.
    """
    if not os.path.exists(REQUIREMENTS_FILE):
        return True

    flag_path = os.path.join(BASE_DIR, DEPS_UPDATED_FLAG)

    # If not forced, check if update is needed
    if not force:
        # Check flag: if exists, dependencies already checked for this version
        if os.path.exists(flag_path):
            log_message("[L] Dependencies already checked for this version, skipping.")
            return True

        # Check dependencies match requirements.txt via pip dry-run
        log_message("[L] Checking dependencies against requirements.txt...")
        try:
            env = os.environ.copy()
            env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
            result = subprocess.run(
                ["sudo", "python3", "-m", "pip", "install",
                 "-r", REQUIREMENTS_FILE,
                 "--dry-run", "--quiet"],
                capture_output=True, text=True, timeout=60,
                env=env
            )
            if result.returncode == 0:
                log_message("[L] All dependencies match requirements.txt.")
                mark_dependencies_updated()
                return True
            else:
                log_message("[L] Dependencies don't match requirements.txt, starting installation.")
        except Exception as e:
            log_message(f"[!] Error checking dependencies: {e}")
            # continue to installation

    # Installation
    try:
        log_message("[L] Installing/updating pip dependencies...")
        env = os.environ.copy()
        env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"

        cmd = [
            "sudo", "python3", "-m", "pip", "install",
            "--no-cache-dir",
            "--upgrade-strategy", "only-if-needed",
            "-r", REQUIREMENTS_FILE
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        if result.returncode != 0:
            log_message(f"[!] PIP Error (Code {result.returncode}):")
            log_message(result.stderr)
            return False

        log_message("[L] Dependencies successfully installed.")
        mark_dependencies_updated()
        return True
    except Exception as e:
        log_message(f"[!] Error installing dependencies: {e}")
        return False

def get_terminal_command(script_path, user):
    launcher_cmd = f"sudo python3 {script_path}"
    hold_cmd = 'echo; echo "Launcher finished. Press any key to close."; read'
    full_cmd = f"{launcher_cmd}; {hold_cmd}"

    if shutil.which("terminator"):
        return ["terminator", "--fullscreen", "-e", f"bash -c '{full_cmd}'"]
    elif shutil.which("gnome-terminal"):
        return ["gnome-terminal", "--full-screen", "--", "bash", "-c", full_cmd]
    elif shutil.which("x-terminal-emulator"):
        return ["x-terminal-emulator", "-e", f"bash -c '{full_cmd}'"]
    elif shutil.which("xterm"):
        return ["xterm", "-fullscreen", "-hold", "-e", f"bash -c '{full_cmd}'"]
    return None

def setup_autostart_linux(target_user):
    script_path = os.path.join(BASE_DIR, os.path.basename(__file__))
    try:
        pw = pwd.getpwnam(target_user)
        user_home = pw.pw_dir
        uid, gid = pw.pw_uid, pw.pw_gid
    except KeyError:
        log_message(f"[!] User {target_user} not found in system.")
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
Name=Orange Pi Monitor (Terminal with sudo)
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
        log_message(f"[L] Command: {cmd_str}")
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
            log_message(f"[L] .desktop file deleted.")
        except Exception as e:
            log_message(f"[!] Error deleting: {e}")

def is_autostart_installed(target_user):
    try:
        pw = pwd.getpwnam(target_user)
        user_home = pw.pw_dir
    except KeyError:
        return False
    desktop_file = os.path.join(user_home, ".config", "autostart", AUTOSTART_DESKTOP_FILE)
    return os.path.exists(desktop_file)

def start_main():
    # Use BASE_DIR to find main.py
    main_path = os.path.join(BASE_DIR, MAIN_SCRIPT)
    if not os.path.exists(main_path):
        log_message(f"[!] {MAIN_SCRIPT} not found in {BASE_DIR}.")
        return False
    try:
        log_message(f"[L] Starting {MAIN_SCRIPT}...")
        
        system_name = platform.system()
        if system_name == "Windows":
            # Explicitly create new console window
            subprocess.Popen(
                ["sudo", "python3", main_path],
                cwd=BASE_DIR,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            return True
        
        if system_name == "Linux":
            # Try to start main.py in new GUI terminal
            python_cmd = f"sudo python3 {shlex.quote(main_path)}"
            hold_cmd = 'echo; echo "main.py finished. Press any key to close."; read'
            full_cmd = f"{python_cmd}; {hold_cmd}"
            
            if shutil.which("terminator"):
                subprocess.Popen(["terminator", "--fullscreen", "-e", f"bash -c '{full_cmd}'"], cwd=BASE_DIR)
                return True
            if shutil.which("gnome-terminal"):
                subprocess.Popen(["gnome-terminal", "--full-screen", "--", "bash", "-c", full_cmd], cwd=BASE_DIR)
                return True
            if shutil.which("x-terminal-emulator"):
                subprocess.Popen(["x-terminal-emulator", "-e", f"bash -c '{full_cmd}'"], cwd=BASE_DIR)
                return True
            if shutil.which("xterm"):
                subprocess.Popen(["xterm", "-fullscreen", "-hold", "-e", f"bash -c '{full_cmd}'"], cwd=BASE_DIR)
                return True
            
            log_message("[!] GUI terminal not found, starting main.py in current process.")
            subprocess.Popen(["sudo", "python3", main_path], cwd=BASE_DIR)
            return True
        
        # For other OS use regular start as fallback
        subprocess.Popen(["sudo", "python3", main_path], cwd=BASE_DIR)
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
    log_message(f"[L] Target user for permissions: {target_user}")
    
    if args.install_autostart or args.remove_autostart:
        if platform.system() != "Linux":
            log_message("[!] Autostart only for Linux.")
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
    if platform.system() == "Linux" and shutil.which("unclutter"):
        subprocess.Popen(["unclutter", "--timeout", "5", "--fork"])
    
    # Use BASE_DIR as the working directory
    script_name = os.path.basename(__file__)
    os.chdir(BASE_DIR)
    log_message(f"[L] Working directory: {BASE_DIR}")
    
    if platform.system() == "Linux" and not args.dont_install_autostart:
        if not is_autostart_installed(target_user):
            log_message("[L] Autostart not found. Installing...")
            setup_autostart_linux(target_user)
        else:
            log_message("[L] Autostart already installed.")
    
    internet_ok = wait_for_internet(max_wait=60)
    repo_updated = False
    if internet_ok:
        log_message("[L] Trying to update repository...")
        repo_updated = download_and_extract_repo(BASE_DIR, script_name, target_user)
        if repo_updated:
            # Repository was either updated or already up to date
            # force=True only if actual download happened (SHA changed)
            # In download_and_extract_repo we reset the flag on download, so
            # install_requirements will see missing flag and perform check
            install_requirements(force=False)
        else:
            log_message("[*] Repository update failed.")
    else:
        log_message("[*] Internet unavailable, skipping update.")
        # Even without internet we can start main.py if dependencies are already installed
        install_requirements(force=False)
    
    if not args.no_start:
        time.sleep(2)
        start_main()
    else:
        log_message("[L] main.py start skipped.")
    
    log_message("[L] Launcher work completed. Window can be closed.")

if __name__ == "__main__":
    main()