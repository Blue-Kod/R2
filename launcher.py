#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launcher для автообновления приложения из GitHub репозитория.
Для работы этого скрипта требуется установить библиотеку requests:
    pip3 install requests

Скрипт использует python3 и pip3 (через sys.executable, что гарантирует работу с тем же интерпретатором,
которым запущен скрипт). Он скачивает последнюю версию репозитория https://github.com/Blue-Kod/R2,
обновляет файлы в своей папке (включая самого себя), устанавливает/обновляет зависимости из requirements.txt,
затем запускает main.py в отдельном окне терминала (чтобы визуально видеть его работу).
При отсутствии интернета или ошибке обновления запускает текущую версию.
Работает без использования виртуального окружения (venv).

Поддерживает установку/удаление автозапуска в Linux:
    python3 launcher.py                      # обычный запуск + автоустановка автозагрузки (если ещё не установлена)
    python3 launcher.py --dont-install-autostart   # запуск без автоматической установки автозагрузки
    python3 launcher.py --no-terminal              # запустить main.py в текущем терминале (без нового окна)
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
from pathlib import Path

# Константы
REPO_URL = "https://github.com/Blue-Kod/R2"
ARCHIVE_URL = "https://github.com/Blue-Kod/R2/archive/refs/heads/main.zip"  # предполагаем ветку main
REQUIREMENTS_FILE = "requirements.txt"
MAIN_SCRIPT = "main.py"

# Для автозапуска в Linux
SYSTEMD_SERVICE_NAME = "r2-launcher.service"
AUTOSTART_DESKTOP_FILE = "r2-launcher.desktop"

def check_python_version():
    """Проверяет, что используется Python 3."""
    if sys.version_info.major < 3:
        print("[!] Ошибка: требуется Python 3")
        sys.exit(1)

def is_internet_available(timeout=3):
    """Проверяет доступность GitHub для определения наличия интернета."""
    try:
        requests.get("https://github.com", timeout=timeout)
        return True
    except requests.RequestException:
        return False

def apply_self_update(new_launcher_path):
    """
    Сравнивает текущий скрипт с новой версией из new_launcher_path.
    Если они различаются, заменяет текущий файл новой версией и
    перезапускает себя с теми же аргументами командной строки.
    """
    current_script = os.path.abspath(__file__)
    # Сравниваем содержимое (побайтово)
    if filecmp.cmp(current_script, new_launcher_path, shallow=False):
        print("[L] Текущая версия лаунчера актуальна.")
        os.unlink(new_launcher_path)
        return False  # обновление не требуется

    print("[L] Обнаружена новая версия лаунчера. Выполняю замену и перезапуск...")
    try:
        # Заменяем текущий файл новым (атомарно для POSIX)
        os.replace(new_launcher_path, current_script)
        # Восстанавливаем права на исполнение (если были)
        st = os.stat(current_script)
        os.chmod(current_script, st.st_mode)
        print("[L] Лаунчер успешно обновлён. Перезапускаю...")
        # Перезапускаемся с теми же аргументами (кроме возможных флагов автозапуска, но их оставляем)
        os.execv(sys.executable, [sys.executable, current_script] + sys.argv[1:])
    except Exception as e:
        print(f"[!] Ошибка при самообновлении: {e}")
        # Если не удалось заменить, пробуем удалить временный файл
        try:
            os.unlink(new_launcher_path)
        except:
            pass
        return False

def download_and_extract_repo(target_dir, script_name):
    """
    Скачивает репозиторий как ZIP, распаковывает во временную папку,
    затем копирует все файлы в target_dir.
    Возвращает True при успехе, False при ошибке.
    """
    try:
        print("[L] Скачивание репозитория...")
        response = requests.get(ARCHIVE_URL, stream=True)
        response.raise_for_status()

        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_zip = tmp_file.name

        # Распаковываем во временную папку
        with tempfile.TemporaryDirectory() as tmp_extract_dir:
            with zipfile.ZipFile(tmp_zip, 'r') as zip_ref:
                zip_ref.extractall(tmp_extract_dir)

            # Находим корневую папку репозитория
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

            # Копируем все файлы из репозитория, кроме .git
            new_launcher_tmp = None
            for root, dirs, files in os.walk(repo_root):
                rel_path = os.path.relpath(root, repo_root)
                if rel_path == ".":
                    dest_dir = target_dir
                else:
                    dest_dir = os.path.join(target_dir, rel_path)
                    os.makedirs(dest_dir, exist_ok=True)

                for file in files:
                    src_file = os.path.join(root, file)
                    # Пропускаем .git
                    if '.git' in rel_path.split(os.sep):
                        continue

                    # Обработка самого лаунчера: не копируем сразу, сохраняем отдельно
                    if file == script_name:
                        print(f"[L] Найдена новая версия {script_name}, проверяем необходимость обновления...")
                        # Сохраняем во временный файл
                        fd, new_launcher_tmp = tempfile.mkstemp(prefix="launcher_new_", suffix=".py")
                        os.close(fd)
                        shutil.copy2(src_file, new_launcher_tmp)
                        continue

                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy2(src_file, dest_file)
                    print(f"[L] Скопирован: {os.path.join(rel_path, file) if rel_path != '.' else file}")

        # Удаляем временный zip
        os.unlink(tmp_zip)

        # Если есть обновление для лаунчера, запускаем самообновление
        if new_launcher_tmp:
            # apply_self_update либо заменяет файл и перезапускает процесс (не возвращает управление),
            # либо возвращает False, если обновление не требуется.
            if apply_self_update(new_launcher_tmp):
                # Если обновление произошло, процесс будет перезапущен и сюда управление не дойдёт.
                # Но apply_self_update возвращает False в случае ошибки или если файлы идентичны.
                pass

        return True

    except Exception as e:
        print(f"[!] Ошибка при загрузке/распаковке репозитория: {e}")
        return False

def install_requirements():
    """
    Устанавливает/обновляет зависимости из requirements.txt с помощью pip.
    Использует тот же интерпретатор Python (sys.executable) для вызова pip,
    что гарантирует работу с правильной версией pip (pip3 для Python 3).
    """
    req_path = Path(REQUIREMENTS_FILE)
    if not req_path.exists():
        print("[*] Файл requirements.txt не найден, пропускаем установку зависимостей.")
        return True

    try:
        print("[L] Установка зависимостей...")
        # Используем sys.executable -m pip для вызова pip, соответствующего текущему python3
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE]
        )
        print("[L] Зависимости успешно установлены/обновлены.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] Ошибка при установке зависимостей: {e}")
        return False

def run_main_in_terminal():
    """Запускает main.py в новом окне терминала (если возможно), иначе в текущем."""
    main_path = Path(MAIN_SCRIPT)
    if not main_path.exists():
        print(f"[!] Ошибка: файл {MAIN_SCRIPT} не найден в текущей директории.")
        return False

    # Команда для запуска main.py (используем sys.executable для гарантии python3)
    cmd = [sys.executable, MAIN_SCRIPT]
    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)

    # Проверяем, запущены ли мы в графической среде (имеется DISPLAY или WAYLAND_DISPLAY)
    has_gui = os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')

    # Если есть GUI, пробуем найти доступный терминал
    terminal_found = False
    if has_gui:
        # Список возможных эмуляторов терминала в порядке предпочтения
        terminals = [
            # name, command pattern ({} будет заменено на cmd_str)
            ('gnome-terminal', 'gnome-terminal -- {}'),  # gnome-terminal требует опцию --, иногда --wait
            ('xterm', 'xterm -hold -e {}'),              # -hold оставляет окно открытым после завершения
            ('konsole', 'konsole -e {}'),
            ('xfce4-terminal', 'xfce4-terminal -e {}'),
            ('lxterminal', 'lxterminal -e {}'),
            ('terminator', 'terminator -e {}'),
            ('urxvt', 'urxvt -e {}'),
            ('rxvt', 'rxvt -e {}'),
        ]

        for term_name, term_cmd_template in terminals:
            if shutil.which(term_name):
                # Формируем полную команду
                full_cmd = term_cmd_template.format(cmd_str)
                print(f"[L] Запускаю {MAIN_SCRIPT} в терминале {term_name}...")
                try:
                    # Используем shell=True, так как шаблон может содержать пробелы и опции
                    subprocess.Popen(full_cmd, shell=True)
                    terminal_found = True
                    break
                except Exception as e:
                    print(f"[!] Не удалось запустить {term_name}: {e}")
                    continue

    if not terminal_found:
        # Если нет GUI или не найден ни один терминал, запускаем в текущем
        if not has_gui:
            print("[L] Графическая среда не обнаружена. Запускаю в текущем терминале...")
        else:
            print("[L] Не найден подходящий эмулятор терминала. Запускаю в текущем терминале...")
        try:
            subprocess.run(cmd)
            return True
        except Exception as e:
            print(f"[!] Ошибка при запуске {MAIN_SCRIPT}: {e}")
            return False

    return True

# --- Функции для автозапуска в Linux ---
def is_autostart_installed():
    """Проверяет, установлен ли уже автозапуск (systemd или .desktop)."""
    if platform.system() != "Linux":
        return False
    # Проверка systemd сервиса
    service_path = f"/etc/systemd/system/{SYSTEMD_SERVICE_NAME}"
    if os.path.exists(service_path):
        return True
    # Проверка .desktop файла
    user_home = os.path.expanduser("~")
    desktop_file = os.path.join(user_home, ".config", "autostart", AUTOSTART_DESKTOP_FILE)
    if os.path.exists(desktop_file):
        return True
    return False

def setup_autostart_linux():
    """Устанавливает текущий скрипт в автозагрузку Linux."""
    script_path = os.path.abspath(__file__)
    user_home = os.path.expanduser("~")
    
    if os.geteuid() == 0:
        # Системный systemd юнит
        service_content = f"""[Unit]
Description=R2 Launcher Auto-Update
After=network.target

[Service]
Type=simple
User={os.getenv('SUDO_USER', 'root')}
ExecStart={sys.executable} {script_path}
WorkingDirectory={os.path.dirname(script_path)}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        service_path = f"/etc/systemd/system/{SYSTEMD_SERVICE_NAME}"
        try:
            with open(service_path, 'w') as f:
                f.write(service_content)
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "enable", SYSTEMD_SERVICE_NAME], check=True)
            subprocess.run(["systemctl", "start", SYSTEMD_SERVICE_NAME], check=True)
            print(f"[L] Автозапуск через systemd установлен (сервис {SYSTEMD_SERVICE_NAME})")
            return True
        except Exception as e:
            print(f"[!] Не удалось установить systemd сервис: {e}")
            return False
    else:
        # Пользовательский автозапуск через .desktop файл
        autostart_dir = os.path.join(user_home, ".config", "autostart")
        os.makedirs(autostart_dir, exist_ok=True)
        desktop_file_path = os.path.join(autostart_dir, AUTOSTART_DESKTOP_FILE)
        desktop_content = f"""[Desktop Entry]
Type=Application
Name=R2 Launcher
Exec={sys.executable} {script_path}
Path={os.path.dirname(script_path)}
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
"""
        try:
            with open(desktop_file_path, 'w') as f:
                f.write(desktop_content)
            print(f"[L] Автозапуск через .desktop файл установлен: {desktop_file_path}")
            return True
        except Exception as e:
            print(f"[!] Не удалось создать .desktop файл: {e}")
            return False

def remove_autostart_linux():
    """Удаляет текущий скрипт из автозагрузки Linux."""
    if os.geteuid() == 0:
        service_path = f"/etc/systemd/system/{SYSTEMD_SERVICE_NAME}"
        try:
            if os.path.exists(service_path):
                subprocess.run(["systemctl", "stop", SYSTEMD_SERVICE_NAME], check=False)
                subprocess.run(["systemctl", "disable", SYSTEMD_SERVICE_NAME], check=False)
                os.remove(service_path)
                subprocess.run(["systemctl", "daemon-reload"], check=True)
                print(f"[L] Systemd сервис {SYSTEMD_SERVICE_NAME} удалён.")
        except Exception as e:
            print(f"[!] Ошибка при удалении systemd сервиса: {e}")
    
    user_home = os.path.expanduser("~")
    desktop_file_path = os.path.join(user_home, ".config", "autostart", AUTOSTART_DESKTOP_FILE)
    try:
        if os.path.exists(desktop_file_path):
            os.remove(desktop_file_path)
            print(f"[L] .desktop файл {desktop_file_path} удалён.")
    except Exception as e:
        print(f"[!] Ошибка при удалении .desktop файла: {e}")

def main():
    check_python_version()

    # Разбор аргументов командной строки
    parser = argparse.ArgumentParser(description="Launcher for R2 project", add_help=False)
    parser.add_argument("--install-autostart", action="store_true", help="Установить скрипт в автозагрузку (только Linux)")
    parser.add_argument("--remove-autostart", action="store_true", help="Удалить скрипт из автозагрузки (только Linux)")
    parser.add_argument("--dont-install-autostart", action="store_true", help="Не устанавливать автозагрузку автоматически")
    parser.add_argument("--no-terminal", action="store_true", help="Запустить main.py в текущем терминале (без нового окна)")
    args, unknown = parser.parse_known_args()

    # Обработка специальных команд автозапуска
    if args.install_autostart or args.remove_autostart:
        if platform.system() != "Linux":
            print("[!] Автозапуск поддерживается только в Linux.")
            sys.exit(1)
        if args.install_autostart:
            setup_autostart_linux()
        elif args.remove_autostart:
            remove_autostart_linux()
        sys.exit(0)

    # Основной запуск
    print("""
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

    # Определяем директорию, в которой находится сам скрипт
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.basename(__file__)

    # Переходим в эту директорию (чтобы все пути были относительно неё)
    os.chdir(script_dir)
    print(f"[L] Рабочая директория: {script_dir}")
    print(f"[L] Используемый интерпретатор: {sys.executable}")

    # Автоматическая установка автозапуска (только Linux и если не запрещено и ещё не установлено)
    if platform.system() == "Linux" and not args.dont_install_autostart:
        if not is_autostart_installed():
            print("[L] Автозапуск не обнаружен. Устанавливаем...")
            setup_autostart_linux()
        else:
            print("[L] Автозапуск уже установлен.")

    # Проверяем наличие интернета
    internet_ok = is_internet_available()

    if internet_ok:
        print("[L] Интернет доступен, пробуем обновить репозиторий...")
        success = download_and_extract_repo(script_dir, script_name)
        if success:
            # После обновления устанавливаем зависимости
            install_requirements()
        else:
            print("[*] Обновление не удалось, продолжим с существующими файлами.")
    else:
        print("[*] Нет интернета, пропускаем обновление.")

    # Запускаем main.py в отдельном терминале (если не указано --no-terminal)
    if args.no_terminal:
        print("[L] Запуск main.py в текущем терминале (--no-terminal)...")
        from run_main import run_main
        run_main()  # используем старую функцию run_main (которая была ранее в коде)
    else:
        run_main_in_terminal()

if __name__ == "__main__":
    main()
