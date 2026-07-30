# Руководство по установке

Это руководство предоставляет подробные инструкции по настройке системы.
## Предварительные требования

### Аппаратные требования
- **Orange Pi 4 Pro**
- **Стереокамера** GXIVISION LSM22100 3D стерео камера 720P 120 градусов
- **Драйвер сервоприводов PCA9685** подключенный через I2C
- **Сервоприводы** 6 на 270, 40+ кг | 4 на 180, 10+ кг
- **SD-Карта** 16+ GB, класс 10+

### Программные требования
- Доступ в глобальную сеть (для первоначальной установки)

---

## Подготовка SD-карты

Установите образ **Debian 1.0.6 Bullseye server** для Orange Pi 4 Pro на SD-карту (образ доступен на [Google Drive](https://drive.google.com/drive/folders/1AzF-uTwA328qDFPaVBaKpiP4VjZjkmbS) или официальном сайте Orange Pi).

---

## Подключение к Wi-Fi

Если у вас есть QR-код с настройками Wi-Fi (стандартный формат `WIFI:T:WPA;S:SSID;P:Пароль;;`), робот сможет настроить подключение автоматически при первом запуске.

### Ручная настройка Wi-Fi

#### Обычная (открытая) сеть
```bash
# Через nmcli (если установлен NetworkManager)
nmcli dev wifi connect "Имя_сети" password "Пароль"

# Через wpa_supplicant
wpa_passphrase "Имя_сети" "Пароль" | sudo tee -a /etc/wpa_supplicant/wpa_supplicant.conf
sudo wpa_cli -i wlan0 reconfigure
sudo dhclient wlan0
```

#### Скрытая сеть
```bash
# Через nmcli
nmcli dev wifi connect "Имя_сети" password "Пароль" hidden yes

# Через wpa_supplicant — после добавления в конфиг добавьте scan_ssid=1:
# network={
#     ssid="Имя_сети"
#     psk="Пароль"
#     scan_ssid=1
# }
# Затем:
sudo wpa_cli -i wlan0 reconfigure
sudo dhclient wlan0
```

#### Проверка подключения
```bash
ping -c 2 8.8.8.8
```

---

## Установка

**НЕ ОБНОВЛЯЙТЕ СИСТЕМУ (`apt upgrade`) НИ ПРИ КАКИХ ОБСТОЯТЕЛЬСТВАХ — ОНА ПЕРЕСТАНЕТ ЗАПУСКАТЬСЯ**

1. Подключитесь к Orange Pi через SSH или последовательный порт.

2. Настройте систему:
    ```bash
    sudo orangepi-config
    ```
    Включите все интерфейсы I2C и настройте часовой пояс.

3. Установите базовые пакеты:
    ```bash
    sudo apt update
    sudo apt install -y \
        git \
        python3-pip \
        python3-dev \
        libportaudio2 \
        libportaudiocpp0 \
        portaudio19-dev \
        libopencv-dev \
        python3-opencv \
        i2c-tools \
        build-essential \
        cmake \
        pkg-config
    ```

4. Перезагрузите робота:
    ```bash
    sudo reboot
    ```

5. Скачайте репозиторий в домашнюю папку пользователя (или используйте свой `$HOME`):
    ```bash
    cd ~
    git clone https://github.com/Blue-Kod/R2.git
    cd ~/R2
    ```

6. Установите Python-зависимости:
    ```bash
    sudo pip3 install -r requirements.txt
    ```

7. Запустите лаунчер:
    ```bash
    sudo python3 launcher.py
    ```
    Лаунчер автоматически выполнит:
    - Установку системных зависимостей через apt
    - Установку Python пакетов из requirements.txt
    - Настройку sudoers (для `shutdown`, `amixer`, `aplay`)
    - Запуск main.py

8. Робот готов к работе. Откройте в браузере веб-интерфейс:
    ```
    http://<ip-адрес-робота>/
    ```
    IP-адрес можно узнать через `ip addr` или `hostname -I`.

### Автоматический Wi-Fi через QR-код

Если при запуске интернет недоступен, робот сам скажет:
> *«Я не подключён к интернету. Пожалуйста, покажите QR-код с настройками Wi-Fi перед камерой.»*

Покажите камере QR-код в формате `WIFI:T:WPA;S:SSID;P:Пароль;;`. Робот распознает его, подключится к сети и продолжит загрузку.

---

## Устранение неполадок

#### Камера не обнаруживается
```bash
# Проверка разрешений камеры
ls -la /dev/video*
sudo usermod -a -G video $USER

# Проверка USB устройств
lsusb
dmesg | grep -i usb
```

#### Проблемы с I2C
```bash
# Проверка I2C устройств
sudo i2cdetect -y 1

# Проверка прав доступа
sudo chmod 666 /dev/i2c-1
```

#### Ошибки зависимостей Python
```bash
# Переустановка проблемных пакетов
pip uninstall opencv-python
pip install opencv-python-headless
```
