#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launcher для автообновления приложения из GitHub репозитория.
Для работы этого скрипта требуется установить библиотеку requests:
    pip3 install requests

Скрипт скачивает последнюю версию репозитория https://github.com/Blue-Kod/R2,
обновляет файлы в своей папке (включая самого себя), устанавливает/обновляет зависимости из requirements.txt.
При отсутствии интернета или ошибке обновления просто завершается (без запуска main.py).
Работает без использования виртуального окружения (venv).

Поддерживает установку/удаление автозапуска в Linux через .desktop файл:
    python3 launcher.py                      # обычный запуск + автоустановка автозагрузки (если ещё не установлена)
    python3 launcher.py --dont-install-autostart   # запуск без автоматической установки автозагрузки
    python3 launcher.py --install-autostart        # принудительно установить в автозагрузку
    python3 launcher.py --remove-autostart         # удалить из автозагрузки

Обновление самого себя:
    Скрипт умеет обновлять собственный код. При обнаружении новой версии launcher.py
    в репозитории он создаёт временную копию, сравнивает содержимое, и если оно отличается,
    заменяет текущий файл новой версией и перезапускается с теми же аргументами.
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
import shlex
import datetime
from pathlib import Path

# Константы
REPO_URL = "https://github.com/Blue-Kod/R2"
ARCHIVE_URL = "https://github.com/Blue-Kod/R2/archive/refs/heads/main.zip"  # предполагаем ветку main
REQUIREMENTS_FILE = "requirements.txt"
MAIN_SCRIPT = "main.py"
LAUNCHER_LOG = "logs.txt"

# Для автозапуска в Linux (только .desktop, без systemd)
AUTOSTART_DESKTOP_FILE = "r2-monitor.desktop"

def log_message(*args):
    """Выводит сообщение в консоль и дописывает в лог-файл с временной меткой."""
    msg = " ".join(str(arg) for arg in args)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    try:
        with open(LAUNCHER_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[!] Не удалось записать в лог-файл {LAUNCHER_LOG}: {e}")

def check_python_version():
    if sys.version_info.major < 3:
        log_message("[!] Ошибка: требуется Python 3")
        sys.exit(1)

def is_internet_available(timeout=3):
    try:
        requests.get("https://github.com", timeout=timeout)
        return True
    except requests.RequestException:
        return False

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

def download_and_extract_repo(target_dir, script_name):
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
        log_message("[L] Установка зависимостей...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
        log_message("[L] Зависимости успешно установлены/обновлены.")
        return True
    except subprocess.CalledProcessError as e:
        log_message(f"[!] Ошибка при установке зависимостей: {e}")
        return False

def setup_autostart_linux():
    """Устанавливает .desktop файл для автозапуска main.py после старта графической сессии."""
    script_path = os.path.abspath(__file__)
    # Команда для запуска: сначала ждём 10 секунд (для полной инициализации), затем запускаем main.py
    cmd = f"/bin/bash -c 'sleep 10 && {sys.executable} {os.path.dirname(script_path)}/{MAIN_SCRIPT}'"
    user_home = os.path.expanduser("~")
    autostart_dir = os.path.join(user_home, ".config", "autostart")
    os.makedirs(autostart_dir, exist_ok=True)
    desktop_file_path = os.path.join(autostart_dir, AUTOSTART_DESKTOP_FILE)
    desktop_content = f"""[Desktop Entry]
Type=Application
Name=Orange Pi Monitor
Exec={cmd}
Path={os.path.dirname(script_path)}
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Phase=Applications
"""
    try:
        with open(desktop_file_path, 'w') as f:
            f.write(desktop_content)
        log_message(f"[L] Автозапуск через .desktop файл установлен: {desktop_file_path}")
        log_message(f"[L] Команда: {cmd}")
        return True
    except Exception as e:
        log_message(f"[!] Не удалось создать .desktop файл: {e}")
        return False

def remove_autostart_linux():
    """Удаляет .desktop файл автозапуска."""
    user_home = os.path.expanduser("~")
    desktop_file_path = os.path.join(user_home, ".config", "autostart", AUTOSTART_DESKTOP_FILE)
    try:
        if os.path.exists(desktop_file_path):
            os.remove(desktop_file_path)
            log_message(f"[L] .desktop файл {desktop_file_path} удалён.")
    except Exception as e:
        log_message(f"[!] Ошибка при удалении .desktop файла: {e}")

def is_autostart_installed():
    """Проверяет, установлен ли .desktop файл."""
    if platform.system() != "Linux":
        return False
    user_home = os.path.expanduser("~")
    desktop_file = os.path.join(user_home, ".config", "autostart", AUTOSTART_DESKTOP_FILE)
    return os.path.exists(desktop_file)

def main():
    check_python_version()

    parser = argparse.ArgumentParser(description="Launcher for R2 project", add_help=False)
    parser.add_argument("--install-autostart", action="store_true", help="Установить автозапуск (.desktop)")
    parser.add_argument("--remove-autostart", action="store_true", help="Удалить автозапуск")
    parser.add_argument("--dont-install-autostart", action="store_true", help="Не устанавливать автозапуск автоматически")
    args, unknown = parser.parse_known_args()

    # Обработка специальных команд автозапуска
    if args.install_autostart or args.remove_autostart:
        if platform.system() != "Linux":
            log_message("[!] Автозапуск поддерживается только в Linux.")
            sys.exit(1)
        if args.install_autostart:
            setup_autostart_linux()
        elif args.remove_autostart:
            remove_autostart_linux()
        sys.exit(0)

    # Основной запуск (только обновление кода и зависимостей)
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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.basename(__file__)
    os.chdir(script_dir)
    log_message(f"[L] Рабочая директория: {script_dir}")
    log_message(f"[L] Используемый интерпретатор: {sys.executable}")

    # Автоматическая установка автозапуска (только Linux и если не запрещено)
    if platform.system() == "Linux" and not args.dont_install_autostart:
        if not is_autostart_installed():
            log_message("[L] Автозапуск не обнаружен. Устанавливаем...")
            setup_autostart_linux()
        else:
            log_message("[L] Автозапуск уже установлен.")

    # Проверка интернета и обновление
    internet_ok = is_internet_available()
    if internet_ok:
        log_message("[L] Интернет доступен, пробуем обновить репозиторий...")
        success = download_and_extract_repo(script_dir, script_name)
        if success:
            install_requirements()
        else:
            log_message("[*] Обновление не удалось.")
    else:
        log_message("[*] Нет интернета, пропускаем обновление.")

    # Лаунчер завершает работу, не запуская main.py (он будет запущен через автозапуск или вручную)
    log_message("[L] Работа лаунчера завершена. main.py будет запущен автоматически при следующем входе в графическую сессию (или вручную).")

if __name__ == "__main__":
    main()
