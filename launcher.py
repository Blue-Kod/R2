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
from pathlib import Path

# Constants
REPO_URL = "https://github.com/Blue-Kod/R2"
ARCHIVE_URL = "https://github.com/Blue-Kod/R2/archive/refs/heads/main.zip"
REQUIREMENTS_FILE = "requirements.txt"

# Complete APT dependencies for Orange Pi 4 Pro / Debian Bullseye
REQUIREMENTS_APT = [
    "python3-pip",
    "libopencv-dev",
    "python3-opencv",
    "i2c-tools",
    "espeak-ng"
]

MAIN_SCRIPT = "main.py"
SERVICE_NAME = "r2-robot"
SERVICE_FILE = f"/etc/systemd/system/{SERVICE_NAME}.service"
INTERNET_CHECK_HOST = "8.8.8.8"
LAST_COMMIT_FILE = ".last_commit"
SETUP_COMPLETE_FLAG = ".setup_complete"  # Flag for first-run installation

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

def setup_sudoers(target_user: str) -> bool:
    """Create sudoers drop-in file so the robot can run without a password
    at boot (autostart) and execute privileged commands.

    Allows:
      - python3 (for launching launcher.py and main.py)
      - shutdown, amixer, aplay (system commands used by the robot)
    """
    if platform.system() != "Linux":
        return True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sudoers_path = "/etc/sudoers.d/r2"

    python_bin = shutil.which("python3") or "/usr/bin/python3"
    systemctl_bin = shutil.which("systemctl") or "/usr/bin/systemctl"
    journalctl_bin = shutil.which("journalctl") or "/usr/bin/journalctl"
    commands = [
        python_bin,
        systemctl_bin,
        journalctl_bin,
        "/usr/sbin/shutdown",
        "/usr/bin/amixer",
        "/usr/bin/aplay",
    ]

    lines = [
        "# R2 Robot - passwordless sudo for autostart and system commands",
        f"# Added by R2 launcher on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"{target_user} ALL=(ALL) NOPASSWD: " + ", ".join(commands),
        "",
    ]

    try:
        content = "\n".join(lines) + "\n"
        with open(sudoers_path, "w") as f:
            f.write(content)
        os.chmod(sudoers_path, 0o440)
        log_message(f"[SUDOERS] Created {sudoers_path} for user {target_user}")
        log_message(f"[SUDOERS] Allowed: {', '.join(commands)}")
        return True
    except Exception as e:
        log_message(f"[!] Failed to create sudoers file: {e}")
        return False


def perform_first_run_setup():
    """
    Perform complete first-run installation.
    Returns True on success, False on failure.
    """
    log_message("=" * 60)
    log_message("[SETUP] Starting first-run installation...")
    log_message("=" * 60)

    target_user = get_display_user()

    # Install APT dependencies
    if not install_apt_dependencies():
        log_message("[SETUP] APT installation failed.")
        return False

    # Install PIP dependencies
    if not install_pip_dependencies():
        log_message("[SETUP] PIP installation failed.")
        return False

    # Sudoers setup — allow passwordless shutdown, audio, etc.
    setup_sudoers(target_user)

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

SKIP_DIRS = {".git", "venv", "__pycache__", "models", "dev", "node_modules", ".idea", ".vscode"}
SKIP_FILES = {".last_commit", ".setup_complete"}

def get_changed_files(old_sha, new_sha):
    """Use GitHub Compare API to get files changed between two commits.
    Returns (to_download, to_delete) where to_download is [(path, raw_url)].
    """
    compare_url = f"https://api.github.com/repos/Blue-Kod/R2/compare/{old_sha}...{new_sha}"
    try:
        response = requests.get(compare_url, timeout=15)
        if response.status_code == 404:
            log_message("[L] Compare API: commits too far apart or not found.")
            return None, None
        response.raise_for_status()
        data = response.json()

        status = data.get("status")
        if status == "identical":
            log_message("[L] Commits identical, nothing to update.")
            return [], []
        if status == "diverged":
            log_message("[L] Branches diverged, using compare results anyway.")

        to_download = []
        to_delete = []

        for f in data.get("files", []):
            filename = f["filename"]
            file_status = f["status"]

            if any(part in SKIP_DIRS for part in Path(filename).parts):
                continue
            if filename in SKIP_FILES:
                continue

            if file_status in ("added", "modified"):
                raw_url = f"https://raw.githubusercontent.com/Blue-Kod/R2/main/{filename}"
                to_download.append((filename, raw_url))
            elif file_status == "removed":
                to_delete.append(filename)
            elif file_status == "renamed":
                old_name = f.get("previous_filename", "")
                if old_name:
                    to_delete.append(old_name)
                raw_url = f"https://raw.githubusercontent.com/Blue-Kod/R2/main/{filename}"
                to_download.append((filename, raw_url))

        log_message(f"[L] Compare: {len(to_download)} to download, {len(to_delete)} to delete.")
        return to_download, to_delete

    except Exception as e:
        log_message(f"[!] Compare API error: {e}")
        return None, None


def download_changed_files(target_dir, to_download, to_delete, script_name):
    """Download only changed files and delete removed ones.
    Returns (success, new_launcher_tmp_path).
    """
    new_launcher_tmp = None
    downloaded = 0
    failed = 0

    for filename, raw_url in to_download:
        dest_file = os.path.join(target_dir, filename)
        dest_dir = os.path.dirname(dest_file)

        if dest_dir and not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)

        if filename == script_name:
            log_message(f"[L] New {script_name} detected, downloading to temp...")
            fd, new_launcher_tmp = tempfile.mkstemp(prefix="launcher_new_", suffix=".py")
            os.close(fd)
            target = new_launcher_tmp
        else:
            target = dest_file

        try:
            resp = requests.get(raw_url, timeout=30)
            resp.raise_for_status()
            with open(target, 'wb') as f:
                f.write(resp.content)
            downloaded += 1
            log_message(f"[L] Updated: {filename}")
        except Exception as e:
            log_message(f"[!] Failed to download {filename}: {e}")
            failed += 1

    deleted = 0
    for filename in to_delete:
        filepath = os.path.join(target_dir, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                deleted += 1
                log_message(f"[L] Deleted: {filename}")
            except Exception as e:
                log_message(f"[!] Failed to delete {filename}: {e}")

    log_message(f"[L] Done: {downloaded} downloaded, {deleted} deleted, {failed} failed.")
    return failed == 0, new_launcher_tmp


def download_and_extract_repo(target_dir, script_name, target_user):
    """Incremental update via GitHub Compare API with ZIP fallback."""
    if is_repo_up_to_date(target_dir):
        return True

    last_commit_path = os.path.join(target_dir, LAST_COMMIT_FILE)
    local_sha = None
    if os.path.exists(last_commit_path):
        try:
            with open(last_commit_path, 'r') as f:
                local_sha = f.read().strip()
        except Exception:
            pass

    remote_sha, _ = get_remote_head_commit_info("Blue-Kod", "R2", "main")
    if remote_sha is None:
        log_message("[!] Could not get remote commit SHA, update aborted.")
        return False

    # --- Attempt 1: Incremental update via Compare API ---
    if local_sha:
        log_message(f"[L] Incremental update: {local_sha[:8]} -> {remote_sha[:8]}")
        to_download, to_delete = get_changed_files(local_sha, remote_sha)

        if to_download is not None:
            success, new_launcher_tmp = download_changed_files(
                target_dir, to_download, to_delete, script_name
            )
            if new_launcher_tmp:
                apply_self_update(new_launcher_tmp)
            if success:
                fix_permissions(target_dir, target_user)
                save_last_commit_info(target_dir, remote_sha)
                return True
            log_message("[!] Incremental update had failures, falling back to full download.")

    # --- Attempt 2: Full ZIP download (first run or Compare fallback) ---
    log_message("[L] Full repository download...")
    try:
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
                raise Exception("Archive is empty")
            repo_root = os.path.join(tmp_extract_dir, extracted_items[0])
            if not os.path.isdir(repo_root):
                for item in extracted_items:
                    if os.path.isdir(os.path.join(tmp_extract_dir, item)):
                        repo_root = os.path.join(tmp_extract_dir, item)
                        break
                else:
                    raise Exception("Could not find repo root folder")

            new_launcher_tmp = None
            for root, dirs, files in os.walk(repo_root):
                rel_path = os.path.relpath(root, repo_root)
                dest_dir = target_dir if rel_path == "." else os.path.join(target_dir, rel_path)
                if rel_path != ".":
                    os.makedirs(dest_dir, exist_ok=True)

                for file in files:
                    src_file = os.path.join(root, file)
                    parts = Path(rel_path).parts
                    if any(p in SKIP_DIRS for p in parts):
                        continue
                    if file in SKIP_FILES:
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

def setup_autostart_linux(target_user):
    """Create systemd service that runs launcher.py at boot."""
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    python_bin = shutil.which("python3") or "/usr/bin/python3"

    unit = f"""[Unit]
Description=R2 Robot
After=network.target

[Service]
Type=simple
User=root
ExecStart={python_bin} {script_path}
WorkingDirectory={script_dir}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
    try:
        with open(SERVICE_FILE, "w") as f:
            f.write(unit)
        subprocess.run(["systemctl", "daemon-reload"], capture_output=True, timeout=10)
        subprocess.run(["systemctl", "enable", SERVICE_NAME], capture_output=True, timeout=10)
        log_message(f"[L] Systemd service installed: {SERVICE_FILE}")
        return True
    except Exception as e:
        log_message(f"[!] Error installing systemd service: {e}")
        return False


def remove_autostart_linux(target_user):
    """Remove systemd service."""
    try:
        subprocess.run(["systemctl", "disable", SERVICE_NAME], capture_output=True, timeout=10)
    except Exception:
        pass
    if os.path.exists(SERVICE_FILE):
        try:
            os.remove(SERVICE_FILE)
            subprocess.run(["systemctl", "daemon-reload"], capture_output=True, timeout=10)
            log_message(f"[L] Systemd service removed: {SERVICE_FILE}")
        except Exception as e:
            log_message(f"[!] Error removing service: {e}")


def is_autostart_installed(target_user):
    return os.path.exists(SERVICE_FILE)

def start_main():
    """Launch main.py and wait for it to finish."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(script_dir, MAIN_SCRIPT)
    if not os.path.exists(main_path):
        log_message(f"[!] {MAIN_SCRIPT} not found.")
        return False
    try:
        log_message(f"[L] Running {MAIN_SCRIPT}...")
        system_name = platform.system()
        env = os.environ.copy()
        if system_name == "Linux":
            env["PYTHONPATH"] = script_dir
        result = subprocess.run(
            [sys.executable or "python3", main_path],
            cwd=script_dir, env=env,
        )
        return result.returncode == 0
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