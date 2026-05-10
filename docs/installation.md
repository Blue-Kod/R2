# Руководство по установке

Это руководство предоставляет подробные инструкции по настройке системы.
## 🎯 Предварительные требования

### Аппаратные требования
- **Orange Pi 4 Pro**
- **Стереокамера** GXIVISION LSM22100 3D стерео камера 720P 120 градусов
- **Драйвер сервоприводов PCA9685** подключенный через I2C
- **Сервоприводы** 6 на 270, 40+ кг | 4 на 180, 10+ кг
- **Дисплей** 7 дюймов, HDMI + USB, сенсорный 1024x600
- **SD-Карта** 16+ GB, класс 10+

### Программные требования
- Доступ в глобальную сеть

---

## Установка
**❗НЕ ОБНОВЛЯЙТЕ СИСТЕМУ (apt upgrade) НИ ПРИ КАКИХ ОБСТОЯТЕЛЬСТВАХ - ОНА ПЕРЕСТАНЕТ ЗАПУСКАТЬСЯ❗**

1. Установите образ [Debian 1.0.6 Bullseye Xfce для Orange Pi 4 Pro](https://drive.google.com/drive/folders/1AzF-uTwA328qDFPaVBaKpiP4VjZjkmbS) на SD-Карту.
2. После установки, подключите экран и измените его ориентацию на портретную (Right).
3. Подключитесь к интернету.
4. Используя ```sudo orangepi-config``` включите все I2C и настройте часовой пояс.
5. Установите все пакеты:
    ```bash
    sudo apt update
    sudo apt install -y \
        git \
        python3-pip \
        python3-pygame \
        python3-dev \
        libsdl2-2.0-0 \
        libsdl2-image-2.0-0 \
        libsdl2-ttf-2.0-0 \
        libportaudio2 \
        libportaudiocpp0 \
        portaudio19-dev \
        unclutter-xfixes \
        libopencv-dev \
        python3-opencv \
        i2c-tools \
        build-essential \
        cmake \
        pkg-config \
        onboard
    ```
6. Перезапустите робота: ```sudo reboot```
7. Скачайте репозиторий: ```git clone https://github.com/Blue-Kod/R2.git```
8. Установите зависимости: ```sudo pip3 install -r R2/requirements.txt```
9. Запустите launcher.py ```sudo python3 R2/launcher.py```. Лаунчер автоматически выполнит:
    - Установку системных зависимостей через apt
    - Установку Python пакетов из requirements.txt
    - Настройку системных сервисов
    - Запуск
10. Робот готов к работе. Если всё прошло успешно, вы увидете новое окно терминала, а затем глаза робота. На экране глаз робота нажмите в нижней части экрана, и откроется меню с IP и кнопкой Выход, которая завершит процесс R2.

### Настройка сенсорного ввода
1. Выполните:
    ```bash
    sudo apt install xinput
    sudo mkdir -p /etc/X11/xorg.conf.d
    sudo cp /usr/share/X11/xorg.conf.d/40-libinput.conf /etc/X11/xorg.conf.d/
    sudo nano /etc/X11/xorg.conf.d/40-libinput.conf
    ```
2. Найдите в файле секцию Identifier "libinput touchscreen catchall" или аналогичную. Внутри этой секции добавьте строку с матрицей трансформации для поворота на 90°:
    ```text
    Option "CalibrationMatrix" "0 -1 1 1 0 0 0 0 1"
    ```
3. Перезапустите робота: ```sudo reboot```

---

## 🐛 Устранение неполадок


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