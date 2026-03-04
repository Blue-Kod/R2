#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launcher для автообновления приложения из GitHub репозитория.
Для работы этого скрипта требуется установить библиотеку requests:
    pip3 install requests

Скрипт использует python3 и pip3 (через sys.executable, что гарантирует работу с тем же интерпретатором,
которым запущен скрипт). Он скачивает последнюю версию репозитория https://github.com/Blue-Kod/R2,
обновляет файлы в своей папке (включая самого себя), устанавливает/обновляет зависимости из requirements.txt,
затем запускает main.py. При запуске с графическим интерфейсом открывает новое окно терминала,
в противном случае (если графика не готова) запускает main.py в фоновом режиме с сохранением вывода в лог-файл.
Все сообщения лаунчера дублируются в консоль и в файл logs.txt.
При отсутствии интернета или ошибке обновления запускает текущую версию.
Работает без использования виртуального окружения (venv).

Поддерживает установку/удаление автозапуска в Linux:
    python3 launcher.py                      # обычный запуск + автоустановка автозагрузки (если ещё не установлена)
    python3 launcher.py --dont-install-autostart   # запуск без автоматической установки автозагрузки
    python3 launcher.py --no-terminal              # принудительно запустить main.py в текущем терминале (без нового окна)
    python3 launcher.py --background               # запустить main.py в фоне без терминала (лог в файл)
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
import pwd
import time
import datetime
from pathlib import Path

# Константы
REPO_URL = "https://github.com/Blue-Kod/R2"
ARCHIVE_URL = "https://github.com/Blue-Kod/R2/archive/refs/heads/main.zip"  # предполагаем ветку main
REQUIREMENTS_FILE = "requirements.txt"
MAIN_SCRIPT = "main.py"
MAIN_LOG = f"{MAIN_SCRIPT}.log"
LAUNCHER_LOG = "logs.txt"  # файл для логов лаунчера

# Для автозапуска в Linux
SYSTEMD_SERVICE_NAME = "r2-launcher.service"
AUTOSTART_DESKTOP_FILE = "r2-launcher.desktop"

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
    """Проверяет, что используется Python 3."""
    if sys.version_info.major < 3:
        log_message("[!] Ошибка: требуется Python 3")
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
        log_message("[L] Текущая версия лаунчера актуальна.")
        os.unlink(new_launcher_path)
        return False  # обновление не требуется

    log_message("[L] Обнаружена новая версия лаунчера. Выполняю замену и перезапуск...")
    try:
        # Используем shutil.move для копирования между разными устройствами (tmp -> home)
        shutil.move(new_launcher_path, current_script)
        # Восстанавливаем права на исполнение (если были)
        st = os.stat(current_script)
        os.chmod(current_script, st.st_mode)
        log_message("[L] Лаунчер успешно обновлён. Перезапускаю...")
        # Перезапускаемся с теми же аргументами
        os.execv(sys.executable, [sys.executable, current_script] + sys.argv[1:])
    except Exception as e:
        log_message(f"[!] Ошибка при самообновлении: {e}")
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
        log_message("[L] Скачивание репозитория...")
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
                        log_message(f"[L] Найдена новая версия {script_name}, проверяем необходимость обновления...")
                        # Сохраняем во временный файл
                        fd, new_launcher_tmp = tempfile.mkstemp(prefix="launcher_new_", suffix=".py")
                        os.close(fd)
                        shutil.copy2(src_file, new_launcher_tmp)
                        continue

                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy2(src_file, dest_file)
                    log_message(f"[L] Скопирован: {os.path.join(rel_path, file) if rel_path != '.' else file}")

        # Удаляем временный zip
        os.unlink(tmp_zip)

        # Если есть обновление для лаунчера, запускаем самообновление
        if new_launcher_tmp:
            apply_self_update(new_launcher_tmp)

        return True

    except Exception as e:
        log_message(f"[!] Ошибка при загрузке/распаковке репозитория: {e}")
        return False

def install_requirements():
    """
    Устанавливает/обновляет зависимости из requirements.txt с помощью pip.
    Использует тот же интерпретатор Python (sys.executable) для вызова pip,
    что гарантирует работу с правильной версией pip (pip3 для Python 3).
    """
    req_path = Path(REQUIREMENTS_FILE)
    if not req_path.exists():
        log_message("[*] Файл requirements.txt не найден, пропускаем установку зависимостей.")
        return True

    try:
        log_message("[L] Установка зависимостей...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE]
        )
        log_message("[L] Зависимости успешно установлены/обновлены.")
        return True
    except subprocess.CalledProcessError as e:
        log_message(f"[!] Ошибка при установке зависимостей: {e}")
        return False

def get_display_user():
    """Возвращает имя обычного пользователя для запуска графических приложений (если запущено от root)."""
    if os.geteuid() != 0:
        return None
    user = os.environ.get('SUDO_USER')
    if user and user != 'root':
        return user
    for u in pwd.getpwall():
        if 1000 <= u.pw_uid < 65534:
            return u.pw_name
    return None

def run_terminal_as_user(terminal_cmd):
    """Запускает терминал от имени обычного пользователя, если мы root."""
    user = get_display_user()
    if not user:
        subprocess.Popen(terminal_cmd)
        return

    try:
        pw = pwd.getpwnam(user)
        uid = pw.pw_uid
        gid = pw.pw_gid

        pid = os.fork()
        if pid == 0:
            os.setgid(gid)
            os.setuid(uid)
            os.environ['HOME'] = pw.pw_dir
            os.environ['USER'] = user
            os.environ['LOGNAME'] = user
            os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':0')
            xauth = os.path.join(pw.pw_dir, '.Xauthority')
            if os.path.exists(xauth):
                os.environ['XAUTHORITY'] = xauth

            try:
                subprocess.Popen(terminal_cmd)
            except Exception as e:
                log_message(f"Ошибка запуска терминала: {e}")
            finally:
                os._exit(0)
        else:
            pass
    except Exception as e:
        log_message(f"Не удалось переключиться на пользователя {user}: {e}")
        subprocess.Popen(terminal_cmd)

def is_display_ready(timeout=30):
    """
    Проверяет, готова ли графическая среда.
    Возвращает True, если переменная DISPLAY установлена (признак графической сессии).
    Если DISPLAY не установлена, ждёт до timeout секунд её появления.
    """
    start = time.time()
    while time.time() - start < timeout:
        if os.environ.get('DISPLAY'):
            log_message("[L] Графическая среда готова (DISPLAY установлен).")
            return True
        time.sleep(1)
    log_message("[L] Таймаут ожидания графической среды (DISPLAY не появился).")
    return False

def run_main_background():
    """Запускает main.py в фоне, перенаправляя stdout/stderr в лог-файл."""
    main_path = Path(MAIN_SCRIPT)
    if not main_path.exists():
        log_message(f"[!] Ошибка: файл {MAIN_SCRIPT} не найден.")
        return False

    with open(MAIN_LOG, 'a') as log:
        log.write(f"\n--- Запуск main.py в фоне ({datetime.datetime.now()}) ---\n")
    try:
        cmd = [sys.executable, MAIN_SCRIPT]
        with open(MAIN_LOG, 'a') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        log_message(f"[L] main.py запущен в фоне (PID {process.pid}). Лог: {MAIN_LOG}")
        return True
    except Exception as e:
        log_message(f"[!] Ошибка фонового запуска: {e}")
        return False

def run_main_in_terminal():
    """Запускает main.py в новом окне терминала, если графика готова, иначе в фоне."""
    main_path = Path(MAIN_SCRIPT)
    if not main_path.exists():
        log_message(f"[!] Ошибка: файл {MAIN_SCRIPT} не найден в текущей директории.")
        return False

    cmd_str = f"{sys.executable} {MAIN_SCRIPT}"

    # Проверяем готовность графической среды (ждём до 30 секунд появления DISPLAY)
    if not is_display_ready(timeout=30):
        log_message("[L] Графическая среда не готова. Запускаю main.py в фоновом режиме.")
        return run_main_background()

    terminals = [
        ('xterm', ['xterm', '-hold', '-e'], True),
        ('xfce4-terminal', ['xfce4-terminal', '--hold', '-e'], True),
        ('gnome-terminal', ['gnome-terminal', '--'], True),
        ('konsole', ['konsole', '-e'], True),
        ('lxterminal', ['lxterminal', '-e'], True),
        ('terminator', ['terminator', '-e'], True),
        ('urxvt', ['urxvt', '-e'], True),
        ('rxvt', ['rxvt', '-e'], True),
    ]

    for term_name, base_args, use_string in terminals:
        if shutil.which(term_name):
            if use_string:
                full_args = base_args + [cmd_str]
            else:
                full_args = base_args + [sys.executable, MAIN_SCRIPT]
            log_message(f"[L] Запускаю {MAIN_SCRIPT} в терминале {term_name}...")
            try:
                if os.geteuid() == 0:
                    run_terminal_as_user(full_args)
                else:
                    subprocess.Popen(full_args)
                return True
            except Exception as e:
                log_message(f"[!] Не удалось запустить {term_name}: {e}")
                continue

    log_message("[L] Не найден подходящий эмулятор терминала. Запускаю в фоновом режиме.")
    return run_main_background()

def run_main_current_terminal(cmd_str):
    """Запускает main.py в текущем терминале (блокирует лаунчер до завершения)."""
    try:
        cmd_list = shlex.split(cmd_str)
        subprocess.run(cmd_list)
        return True
    except Exception as e:
        log_message(f"[!] Ошибка при запуске {MAIN_SCRIPT}: {e}")
        return False

# --- Функции для автозапуска в Linux ---
def is_autostart_installed():
    """Проверяет, установлен ли уже автозапуск (systemd или .desktop)."""
    if platform.system() != "Linux":
        return False
    service_path = f"/etc/systemd/system/{SYSTEMD_SERVICE_NAME}"
    if os.path.exists(service_path):
        return True
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
            log_message(f"[L] Автозапуск через systemd установлен (сервис {SYSTEMD_SERVICE_NAME})")
            return True
        except Exception as e:
            log_message(f"[!] Не удалось установить systemd сервис: {e}")
            return False
    else:
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
            log_message(f"[L] Автозапуск через .desktop файл установлен: {desktop_file_path}")
            return True
        except Exception as e:
            log_message(f"[!] Не удалось создать .desktop файл: {e}")
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
                log_message(f"[L] Systemd сервис {SYSTEMD_SERVICE_NAME} удалён.")
        except Exception as e:
            log_message(f"[!] Ошибка при удалении systemd сервиса: {e}")
    
    user_home = os.path.expanduser("~")
    desktop_file_path = os.path.join(user_home, ".config", "autostart", AUTOSTART_DESKTOP_FILE)
    try:
        if os.path.exists(desktop_file_path):
            os.remove(desktop_file_path)
            log_message(f"[L] .desktop файл {desktop_file_path} удалён.")
    except Exception as e:
        log_message(f"[!] Ошибка при удалении .desktop файла: {e}")

def main():
    check_python_version()

    parser = argparse.ArgumentParser(description="Launcher for R2 project", add_help=False)
    parser.add_argument("--install-autostart", action="store_true", help="Установить скрипт в автозагрузку (только Linux)")
    parser.add_argument("--remove-autostart", action="store_true", help="Удалить скрипт из автозагрузки (только Linux)")
    parser.add_argument("--dont-install-autostart", action="store_true", help="Не устанавливать автозагрузку автоматически")
    parser.add_argument("--no-terminal", action="store_true", help="Запустить main.py в текущем терминале (без нового окна)")
    parser.add_argument("--background", action="store_true", help="Запустить main.py в фоне без терминала (лог в файл)")
    args, unknown = parser.parse_known_args()

    if args.install_autostart or args.remove_autostart:
        if platform.system() != "Linux":
            log_message("[!] Автозапуск поддерживается только в Linux.")
            sys.exit(1)
        if args.install_autostart:
            setup_autostart_linux()
        elif args.remove_autostart:
            remove_autostart_linux()
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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.basename(__file__)
    os.chdir(script_dir)
    log_message(f"[L] Рабочая директория: {script_dir}")
    log_message(f"[L] Используемый интерпретатор: {sys.executable}")

    if platform.system() == "Linux" and not args.dont_install_autostart:
        if not is_autostart_installed():
            log_message("[L] Автозапуск не обнаружен. Устанавливаем...")
            setup_autostart_linux()
        else:
            log_message("[L] Автозапуск уже установлен.")

    internet_ok = is_internet_available()
    if internet_ok:
        log_message("[L] Интернет доступен, пробуем обновить репозиторий...")
        success = download_and_extract_repo(script_dir, script_name)
        if success:
            install_requirements()
        else:
            log_message("[*] Обновление не удалось, продолжим с существующими файлами.")
    else:
        log_message("[*] Нет интернета, пропускаем обновление.")

    # Запуск main.py
    if args.background:
        log_message("[L] Принудительный фоновый запуск main.py...")
        run_main_background()
    elif args.no_terminal:
        log_message("[L] Запуск main.py в текущем терминале (--no-terminal)...")
        run_main_current_terminal(f"{sys.executable} {MAIN_SCRIPT}")
    else:
        run_main_in_terminal()

if __name__ == "__main__":
    main()
