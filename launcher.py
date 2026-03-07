#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launcher для автообновления и запуска приложения из GitHub репозитория.
ВНИМАНИЕ: Этот скрипт должен запускаться с правами root (через sudo).
Если запущен обычным пользователем – он сообщит об ошибке и завершится.
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
from pathlib import Path

# Константы
REPO_URL = "https://github.com/Blue-Kod/R2"
ARCHIVE_URL = "https://github.com/Blue-Kod/R2/archive/refs/heads/main.zip"
REQUIREMENTS_FILE = "requirements.txt"
MAIN_SCRIPT = "main.py"
AUTOSTART_DESKTOP_FILE = "r2-monitor.desktop"
INTERNET_CHECK_HOST = "8.8.8.8"

def log_message(*args):
    msg = " ".join(str(arg) for arg in args)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def check_root():
    """Проверяет, запущен ли скрипт с правами root."""
    if os.geteuid() != 0:
        log_message("[!] Этот скрипт должен запускаться с sudo!")
        log_message("[!] Запустите: sudo python3 launcher.py")
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
    log_message(f"[L] Ожидание интернета до {max_wait} сек...")
    start = time.time()
    while time.time() - start < max_wait:
        if is_internet_available(timeout=2):
            log_message("[L] Интернет доступен.")
            return True
        log_message("[L] Интернет недоступен, ждём 5 сек...")
        time.sleep(5)
    log_message("[L] Интернет не появился за отведённое время.")
    return False

def get_display_user():
    """Возвращает имя пользователя, от которого запущена графическая сессия (реальный человек)."""
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
    """Рекурсивно меняет владельца файлов в path на user."""
    try:
        pw = pwd.getpwnam(user)
        uid, gid = pw.pw_uid, pw.pw_gid
        log_message(f"[L] Меняем владельца {path} на {user} ({uid}:{gid})")
        for root, dirs, files in os.walk(path):
            for d in dirs:
                os.chown(os.path.join(root, d), uid, gid)
            for f in files:
                os.chown(os.path.join(root, f), uid, gid)
        os.chown(path, uid, gid)
    except Exception as e:
        log_message(f"[!] Не удалось изменить владельца: {e}")

def apply_self_update(new_launcher_path):
    current_script = os.path.abspath(__file__)
    if filecmp.cmp(current_script, new_launcher_path, shallow=False):
        log_message("[L] Текущая версия лаунчера актуальна.")
        os.unlink(new_launcher_path)
        return False

    log_message("[L] Обнаружена новая версия лаунчера. Выполняю замену и перезапуск...")
    try:
        shutil.move(new_launcher_path, current_script)
        st = os.stat(current_script)
        os.chmod(current_script, st.st_mode)
        log_message("[L] Лаунчер успешно обновлён. Перезапускаю...")
        os.execv(sys.executable, [sys.executable, current_script] + sys.argv[1:])
    except Exception as e:
        log_message(f"[!] Ошибка при самообновлении: {e}")
        try:
            os.unlink(new_launcher_path)
        except:
            pass
        return False

def download_and_extract_repo(target_dir, script_name, target_user):
    try:
        log_message("[L] Скачивание репозитория...")
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
                raise Exception("[!] Архив пуст")
            repo_root = os.path.join(tmp_extract_dir, extracted_items[0])
            if not os.path.isdir(repo_root):
                for item in extracted_items:
                    if os.path.isdir(os.path.join(tmp_extract_dir, item)):
                        repo_root = os.path.join(tmp_extract_dir, item)
                        break
                else:
                    raise Exception("[!] Не удалось найти корневую папку репозитория")

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
                        log_message(f"[L] Найдена новая версия {script_name}, проверяем необходимость обновления...")
                        fd, new_launcher_tmp = tempfile.mkstemp(prefix="launcher_new_", suffix=".py")
                        os.close(fd)
                        shutil.copy2(src_file, new_launcher_tmp)
                        continue

                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy2(src_file, dest_file)
                    log_message(f"[L] Скопирован: {os.path.join(rel_path, file) if rel_path != '.' else file}")

        os.unlink(tmp_zip)

        # После успешного обновления меняем владельца на target_user
        fix_permissions(target_dir, target_user)

        if new_launcher_tmp:
            apply_self_update(new_launcher_tmp)
        return True

    except Exception as e:
        log_message(f"[!] Ошибка при загрузке/распаковке репозитория: {e}")
        return False

def install_requirements():
    req_path = Path(REQUIREMENTS_FILE)
    if not req_path.exists():
        log_message("[*] Файл requirements.txt не найден, пропускаем установку зависимостей.")
        return True

    try:
        log_message("[L] Установка зависимостей (глобально)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
        log_message("[L] Зависимости успешно установлены/обновлены.")
        return True
    except subprocess.CalledProcessError as e:
        log_message(f"[!] Ошибка при установке зависимостей: {e}")
        return False

def get_terminal_command(script_path, user):
    launcher_cmd = f"sudo {sys.executable} {script_path}"
    hold_cmd = 'echo; echo "Launcher finished. Press any key to close this window."; read'
    full_cmd = f"{launcher_cmd}; {hold_cmd}"

    if shutil.which("terminator"):
        return ["terminator", "--fullscreen", "-e", f"bash -c '{full_cmd}'"]
    elif shutil.which("gnome-terminal"):
        return ["gnome-terminal", "--", "bash", "-c", full_cmd]
    elif shutil.which("x-terminal-emulator"):
        return ["x-terminal-emulator", "-e", f"bash -c '{full_cmd}'"]
    elif shutil.which("xterm"):
        return ["xterm", "-hold", "-e", f"bash -c '{full_cmd}'"]
    else:
        return None

def setup_autostart_linux(target_user):
    script_path = os.path.abspath(__file__)
    try:
        pw = pwd.getpwnam(target_user)
        user_home = pw.pw_dir
        uid, gid = pw.pw_uid, pw.pw_gid
    except KeyError:
        log_message(f"[!] Пользователь {target_user} не найден в системе.")
        return False

    autostart_dir = os.path.join(user_home, ".config", "autostart")
    os.makedirs(autostart_dir, exist_ok=True)
    desktop_file_path = os.path.join(autostart_dir, AUTOSTART_DESKTOP_FILE)

    terminal_cmd = get_terminal_command(script_path, target_user)
    if not terminal_cmd:
        log_message("[!] Не найден подходящий терминал.")
        return False

    import shlex
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
        # Файл должен принадлежать пользователю, чтобы система автозапуска его увидела
        os.chown(desktop_file_path, uid, gid)
        log_message(f"[L] Автозапуск установлен: {desktop_file_path}")
        log_message(f"[L] Команда: {cmd_str}")
        return True
    except Exception as e:
        log_message(f"[!] Ошибка создания .desktop файла: {e}")
        return False

def remove_autostart_linux(target_user):
    try:
        pw = pwd.getpwnam(target_user)
        user_home = pw.pw_dir
    except KeyError:
        log_message(f"[!] Пользователь {target_user} не найден.")
        return
    desktop_file = os.path.join(user_home, ".config", "autostart", AUTOSTART_DESKTOP_FILE)
    if os.path.exists(desktop_file):
        try:
            os.remove(desktop_file)
            log_message(f"[L] .desktop файл удалён.")
        except Exception as e:
            log_message(f"[!] Ошибка удаления: {e}")

def is_autostart_installed(target_user):
    try:
        pw = pwd.getpwnam(target_user)
        user_home = pw.pw_dir
    except KeyError:
        return False
    desktop_file = os.path.join(user_home, ".config", "autostart", AUTOSTART_DESKTOP_FILE)
    return os.path.exists(desktop_file)

def start_main():
    main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAIN_SCRIPT)
    if not os.path.exists(main_path):
        log_message(f"[!] {MAIN_SCRIPT} не найден.")
        return False
    try:
        log_message(f"[L] Запуск {MAIN_SCRIPT}...")
        subprocess.Popen([sys.executable, main_path])
        return True
    except Exception as e:
        log_message(f"[!] Ошибка запуска {MAIN_SCRIPT}: {e}")
        return False

def main():
    # Проверяем, что запущены с root
    check_root()

    parser = argparse.ArgumentParser(description="Launcher for R2 project", add_help=False)
    parser.add_argument("--install-autostart", action="store_true")
    parser.add_argument("--remove-autostart", action="store_true")
    parser.add_argument("--no-start", action="store_true")
    parser.add_argument("--dont-install-autostart", action="store_true")
    args, unknown = parser.parse_known_args()

    # Пользователь, которому будут принадлежать файлы после обновления (обычный юзер)
    target_user = get_display_user()
    log_message(f"[L] Целевой пользователь для прав: {target_user}")

    if args.install_autostart or args.remove_autostart:
        if platform.system() != "Linux":
            log_message("[!] Автозапуск только для Linux.")
            sys.exit(1)
        if args.install_autostart:
            setup_autostart_linux(target_user)
        elif args.remove_autostart:
            remove_autostart_linux(target_user)
        sys.exit(0)

    log_message("""
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
    subprocess.Popen(["unclutter", "--timeout", "5", "--fork"])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.basename(__file__)
    os.chdir(script_dir)
    log_message(f"[L] Рабочая директория: {script_dir}")

    # Автоматическая установка автозапуска (Linux, не запрещено)
    if platform.system() == "Linux" and not args.dont_install_autostart:
        if not is_autostart_installed(target_user):
            log_message("[L] Автозапуск не обнаружен. Устанавливаем...")
            setup_autostart_linux(target_user)
        else:
            log_message("[L] Автозапуск уже установлен.")

    # Ожидание интернета
    internet_ok = wait_for_internet(max_wait=60)
    if internet_ok:
        log_message("[L] Пробуем обновить репозиторий...")
        success = download_and_extract_repo(script_dir, script_name, target_user)
        if success:
            install_requirements()
        else:
            log_message("[*] Обновление не удалось.")
    else:
        log_message("[*] Интернет отсутствует, пропускаем обновление.")

    # Запуск main.py
    if not args.no_start:
        time.sleep(2)
        start_main()
    else:
        log_message("[L] Запуск main.py пропущен.")

    log_message("[L] Работа лаунчера завершена. Окно можно закрыть.")

if __name__ == "__main__":
    main()
