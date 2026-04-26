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
import json
import shlex
from pathlib import Path

# Константы
REPO_URL = "https://github.com/Blue-Kod/R2"
ARCHIVE_URL = "https://github.com/Blue-Kod/R2/archive/refs/heads/main.zip"
REQUIREMENTS_FILE = "requirements.txt"
REQUIREMENTS_APT = [  # System dependencies for Debian/ARM (pywebview GTK backend)
    "python3-gi",
    "python3-gi-cairo",
    "gir1.2-gtk-3.0",
    "libwebkit2gtk-4.0-dev",
    "libglib2.0-dev",
    "libgtk-3-dev",
    "unclutter-xfixes",  # Hides mouse cursor for kiosk mode
]
MAIN_SCRIPT = "main.py"
AUTOSTART_DESKTOP_FILE = "r2-monitor.desktop"
INTERNET_CHECK_HOST = "8.8.8.8"
LAST_COMMIT_FILE = ".last_commit"
DEPS_UPDATED_FLAG = ".deps_updated"      # флаг, что зависимости уже проверены для текущего коммита

def log_message(*args):
    msg = " ".join(str(arg) for arg in args)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def check_root():
    """Проверяет, запущен ли скрипт с правами root."""
    if platform.system() != "Linux":
        return
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

def is_apt_package_installed(package):
    """Check if an APT package is installed on Debian/Ubuntu systems."""
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f=${Status}", package],
            capture_output=True, text=True, timeout=10
        )
        return "install ok installed" in result.stdout
    except Exception:
        return False

def apply_self_update(new_launcher_path):
    current_script = os.path.abspath(__file__)

    # Проверка на случай, если файл всё же не создался
    if not os.path.exists(new_launcher_path):
        log_message(f"[!] Критическая ошибка: Временный файл {new_launcher_path} не найден.")
        return False

    if filecmp.cmp(current_script, new_launcher_path, shallow=False):
        log_message("[L] Текущая версия лаунчера актуальна.")
        return False

    log_message("[L] Обнаружена новая версия лаунчера. Обновляюсь...")
    try:
        # Устанавливаем права на исполнение для нового файла
        os.chmod(new_launcher_path, 0o755)

        # Копируем поверх текущего скрипта
        shutil.copy2(new_launcher_path, current_script)
        log_message("[L] Файл заменен. Перезапуск...")

        # Полная команда перезапуска
        os.execv(sys.executable, [sys.executable, current_script] + sys.argv[1:])
    except Exception as e:
        log_message(f"[!] Ошибка при замене файла лаунчера: {e}")
        return False

def get_remote_head_commit_info(owner, repo, branch='main'):
    api_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}"
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code != 200:
            log_message(f"[!] GitHub API вернул {response.status_code}")
            return None, None
        data = response.json()
        sha = data.get('sha')
        date_str = data.get('commit', {}).get('committer', {}).get('date')
        commit_date = None
        if date_str:
            commit_date = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        return sha, commit_date
    except Exception as e:
        log_message(f"[!] Ошибка получения информации о коммите: {e}")
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
        log_message(f"[L] Сохранён SHA коммита: {sha[:8]}")
    except Exception as e:
        log_message(f"[!] Ошибка сохранения .last_commit: {e}")

def download_and_extract_repo(target_dir, script_name, target_user):
    if is_repo_up_to_date(target_dir):
        return True

    remote_sha, _ = get_remote_head_commit_info("Blue-Kod", "R2", "main")
    if remote_sha is None:
        log_message("[!] Не удалось получить SHA удалённого коммита, обновление прервано.")
        return False

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
            if new_launcher_tmp:
                # Вызываем обновление, пока папка существует
                apply_self_update(new_launcher_tmp)

        os.unlink(tmp_zip)

        # После обновления удаляем флаг зависимостей, чтобы при следующей проверке они перепроверились
        deps_flag = os.path.join(target_dir, DEPS_UPDATED_FLAG)
        if os.path.exists(deps_flag):
            os.remove(deps_flag)
            log_message("[L] Флаг зависимостей сброшен.")
        fix_permissions(target_dir, target_user)
        save_last_commit_info(target_dir, remote_sha)

        return True

    except Exception as e:
        log_message(f"[!] Ошибка при загрузке/распаковке репозитория: {e}")
        return False

def mark_dependencies_updated():
    """Создаёт файл-флаг, указывающий что зависимости для текущего коммита обновлены."""
    flag_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEPS_UPDATED_FLAG)
    try:
        with open(flag_path, 'w') as f:
            f.write(datetime.datetime.now().isoformat())
        log_message("[L] Флаг обновления зависимостей установлен.")
    except Exception as e:
        log_message(f"[!] Не удалось создать флаг: {e}")

def install_requirements(force=False):
    """
    Install pip dependencies. Skips installation if all packages are already satisfied.
    Logs detailed information about the installation process.
    """
    if not os.path.exists(REQUIREMENTS_FILE):
        return True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    flag_path = os.path.join(script_dir, DEPS_UPDATED_FLAG)

    # If not forced, check if we need to update
    if not force:
        # Check flag first
        if os.path.exists(flag_path):
            log_message("[PIP] Зависимости уже установлены, пропускаю.")
            return True

        # Check if requirements are satisfied via pip dry-run
        log_message("[PIP] Проверка зависимостей requirements.txt...")
        try:
            env = os.environ.copy()
            env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
            # Remove --quiet to get detailed dry-run output
            result = subprocess.run(
                ["sudo", "python3", "-m", "pip", "install",
                 "-r", REQUIREMENTS_FILE,
                 "--dry-run"],
                capture_output=True, text=True, timeout=60,
                env=env
            )
            # Log dry-run output
            log_message("[PIP] Вывод dry-run:")
            if result.stdout.strip():
                log_message(f"[PIP] {result.stdout}")
            if result.stderr.strip():
                log_message(f"[PIP] Dry-run ошибка: {result.stderr}")

            if result.returncode == 0:
                log_message("[PIP] Все требования соблюдены.")
                mark_dependencies_updated()
                return True
            else:
                log_message("[PIP] Не все требования соблюдены, начинаю установку...")
        except Exception as e:
            log_message(f"[PIP] Ошибка при проверке зависимостей: {e}")
            # Proceed to installation

    # Install dependencies
    try:
        log_message("[PIP] Устанавливаю/обновляю зависимости...")
        env = os.environ.copy()
        env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"

        cmd = [
            "sudo", "python3", "-m", "pip", "install",
            "--no-cache-dir",
            "--upgrade-strategy", "only-if-needed",
            "-r", REQUIREMENTS_FILE
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        # Log installation output
        if result.stdout.strip():
            log_message(f"[PIP] Вывод установки: {result.stdout[:500]}...")
        if result.stderr.strip():
            log_message(f"[PIP] Ошибка установки: {result.stderr[:500]}...")

        if result.returncode != 0:
            log_message(f"[PIP] Ошибка при установке (Code {result.returncode})")
            return False

        log_message("[PIP] Зависимости успешно установлены.")
        mark_dependencies_updated()
        return True
    except Exception as e:
        log_message(f"[PIP] Ошибка при установке {e}")
        return False


def install_apt_requirements(force=False):
    """
    Поштучная установка системных зависимостей APT.
    """
    if platform.system() != "Linux":
        log_message("[APT] Доступно только на Linux.")
        return True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    flag_path = os.path.join(script_dir, ".apt_deps_updated")

    if os.geteuid() != 0:
        log_message("[APT] Ошибка: Требуются права root (sudo).")
        return False

    if not REQUIREMENTS_APT:
        return True

    # 1. Обновляем списки (один раз перед циклом)
    log_message("[APT] Обновление списков пакетов...")
    try:
        subprocess.run(["apt-get", "update", "-y"], check=True, timeout=120, capture_output=True)
    except Exception as e:
        log_message(f"[!] Ошибка при update (возможно нет интернета): {e}")
        # Продолжаем, вдруг пакеты в кэше

    # 2. Установка по одному
    any_failed = False
    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"  # Чтобы apt не спрашивал подтверждений

    for package in REQUIREMENTS_APT:
        if not force and is_apt_package_installed(package):
            log_message(f"[APT] {package} уже в системе.")
            continue

        log_message(f"[APT] Установка: {package}...")
        try:
            # -y: да, -f: fix-broken (исправить зависимости)
            cmd = ["apt-get", "install", "-y", "-f", package]
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                log_message(f"[APT] [+] {package} успешно установлен.")
            else:
                log_message(f"[APT] [-] Не удалось поставить {package}. Код: {result.returncode}")
                log_message(f"[APT] Ошибка: {result.stderr.strip()}")
                any_failed = True
        except Exception as e:
            log_message(f"[APT] [!] Исключение при установке {package}: {e}")
            any_failed = True

    if not any_failed:
        with open(flag_path, 'w') as f:
            f.write(datetime.datetime.now().isoformat())
        log_message("[APT] Все пакеты обработаны без ошибок.")
        return True

    return False

def get_terminal_command(script_path, user):
    launcher_cmd = f"sudo python3 {script_path}"
    hold_cmd = 'echo; echo "Press any key to close..."; read'
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.join(script_dir, MAIN_SCRIPT)
    if not os.path.exists(main_path):
        log_message(f"[!] {MAIN_SCRIPT} не найден.")
        return False
    try:
        log_message(f"[L] Запуск {MAIN_SCRIPT}...")

        system_name = platform.system()
        if system_name == "Windows":
            # Явно создаём новое окно консоли.
            subprocess.Popen(
                ["sudo", "python3", main_path],
                cwd=script_dir,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            return True

        if system_name == "Linux":
            # Пытаемся запустить main.py в новом терминале GUI.
            python_cmd = f"sudo python3 {shlex.quote(main_path)}"
            hold_cmd = 'echo; echo "main.py finished. Press any key to close."; read'
            full_cmd = f"{python_cmd}; {hold_cmd}"

            if shutil.which("terminator"):
                subprocess.Popen(["terminator", "--fullscreen", "-e", f"bash -c '{full_cmd}'"], cwd=script_dir)
                return True
            if shutil.which("gnome-terminal"):
                subprocess.Popen(["gnome-terminal", "--full-screen", "--", "bash", "-c", full_cmd], cwd=script_dir)
                return True
            if shutil.which("x-terminal-emulator"):
                subprocess.Popen(["x-terminal-emulator", "-e", f"bash -c '{full_cmd}'"], cwd=script_dir)
                return True
            if shutil.which("xterm"):
                subprocess.Popen(["xterm", "-fullscreen", "-hold", "-e", f"bash -c '{full_cmd}'"], cwd=script_dir)
                return True

            log_message("[!] Терминал GUI не найден, запускаю main.py в текущем процессе.")
            subprocess.Popen(["sudo", "python3", main_path], cwd=script_dir)
            return True

        # Для остальных ОС используем обычный запуск как fallback.
        subprocess.Popen(["sudo", "python3", main_path], cwd=script_dir)
        return True
    except Exception as e:
        log_message(f"[!] Ошибка запуска {MAIN_SCRIPT}: {e}")
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.basename(__file__)
    os.chdir(script_dir)
    log_message(f"[L] Рабочая директория: {script_dir}")

    if platform.system() == "Linux" and not args.dont_install_autostart:
        if not is_autostart_installed(target_user):
            log_message("[L] Автозапуск не обнаружен. Устанавливаем...")
            setup_autostart_linux(target_user)
        else:
            log_message("[L] Автозапуск уже установлен.")

    internet_ok = wait_for_internet(max_wait=60)
    repo_updated = False

    # Install APT dependencies first (system packages for pywebview GTK backend)
    if platform.system() == "Linux":
        log_message("[L] Проверка системных зависимостей (APT)...")
        install_apt_requirements(force=False)

    if internet_ok:
        log_message("[L] Пробуем обновить репозиторий...")
        repo_updated = download_and_extract_repo(script_dir, script_name, target_user)
        if repo_updated:
            # Репозиторий либо был обновлён, либо уже актуален.
            # force=True только если действительно было скачивание (SHA изменился).
            # В download_and_extract_repo мы сбрасываем флаг при скачивании, поэтому
            # install_requirements увидит отсутствие флага и выполнит проверку.
            install_requirements(force=False)
        else:
            log_message("[*] Обновление репозитория не удалось.")
    else:
        log_message("[*] Интернет отсутствует, пропускаем обновление.")
        # Даже без интернета можно запустить main.py, если зависимости уже стоят
        install_requirements(force=False)

    if not args.no_start:
        time.sleep(2)
        start_main()
    else:
        log_message("[L] Запуск main.py пропущен.")

    log_message("[L] Работа лаунчера завершена. Окно можно закрыть.")

if __name__ == "__main__":
    main()