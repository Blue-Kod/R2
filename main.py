import os
import psutil
import datetime
from flask import Flask, render_template, jsonify

app = Flask(__name__)


def get_cpu_temp():
    """Чтение температуры процессора Orange Pi (RK3399)."""
    try:
        # Стандартный путь для большинства Rockchip плат
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read()) / 1000
            return f"{temp:.1f}°C"
    except:
        return "N/A"


@app.route('/')
def index():
    return render_template('screen.html')


@app.route('/get_system_data')
def get_system_data():
    """Формирование списка логов для фона."""
    now = datetime.datetime.now().strftime("%H:%M:%S")
    cpu_load = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    temp = get_cpu_temp()

    # Генерируем несколько строк логов
    logs = [
        f"[{now}] SYS_MONITOR: CPU_TEMP={temp} | CPU_LOAD={cpu_load}%",
        f"[{now}] MEM_STATS: USED_RAM={ram}% | FREE_VIRTUAL={psutil.virtual_memory().available // 1024 ** 2}MB",
        f"[{now}] KERNEL: RK3399_CORE_STABLE",
        f"[{now}] NETWORK: ETH0_UP | ADDR=192.168.1.x"
    ]
    return jsonify(logs)


if __name__ == '__main__':
    # Запуск на всех интерфейсах, порт 5000
    app.run(host='0.0.0.0', port=5000, debug=False)