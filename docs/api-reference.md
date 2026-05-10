# Справочник API

Этот документ предоставляет полную документацию по всем эндпоинтам и Python интерфейсам API робота R2.

## 🌐 REST API Эндпоинты

Робот R2 предоставляет RESTful API через Flask веб-сервер, обычно работающий на порту 80.

### Базовый URL
```
http://<robot-ip>/
```
Вы можете узнать IP робота, нажав внизу его экрана с глазами.

### Аутентификация
В настоящее время аутентификация не требуется. Все эндпоинты доступны без аутентификации.

---

## 📊 Системная информация

### GET `/api/data`
Возвращает снимок состояния системы, включая CPU, память, температуру и последние логи.

**Ответ:**
```json
{
  "cpu_usage": 25.5,
  "memory_usage": 45.2,
  "temperature": "45.0°C",
  "fps": 15.0,
  "logs": ["Запись лога 1", "Запись лога 2", ...]
}
```

### GET `/api/ip`
Возвращает IP-адрес робота.

**Ответ:**
```json
{
  "ip": "192.168.1.100"
}
```

---

## 🎥 Управление камерой

### GET `/video_feed`
Предоставляет MJPEG видеопоток со стереокамеры.

**Ответ:** Multipart поток с JPEG кадрами

### GET `/api/camera/params`
Получить текущие параметры камеры.

**Ответ:**
```json
{
  "depth_enabled": false,
  "alpha_depth": 0.3,
  "show_left": true,
  "num_disp": 128
}
```

### POST `/api/camera/params`
Обновить параметры камеры.

**Тело запроса:**
```json
{
  "depth_enabled": true,
  "alpha_depth": 0.5,
  "show_left": false,
  "num_disp": 64
}
```

### POST `/api/depth`
Получить измерение глубины в указанных координатах.

**Тело запроса:**
```json
{
  "x": 320,
  "y": 240
}
```

**Ответ:**
```json
{
  "depth": 1250.5
}
```

---

## 🦾 Управление сервоприводами

### POST `/api/servo/<channel>/<angle>`
Установить угол сервопривода для указанного канала.

**Параметры:**
- `channel` (int): Канал сервопривода (0-7)
- `angle` (int): Целевой угол в градусах

**Ответ:**
```json
{
  "status": "ok",
  "channel": 0,
  "angle": 90
}
```

**Ответ с ошибкой:**
```json
{
  "error": "Угол должен быть 0-180"
}
```

---

## 😊 Эмоции и отображение

### GET `/api/emote`
Получить текущую эмоцию и список поддерживаемых эмоций.

**Ответ:**
```json
{
  "status": "ok",
  "emote": "happy",
  "supported": ["happy", "neutral", "scared"]
}
```

### POST `/api/emote`
Установить отображение эмоции робота.

**Тело запроса:**
```json
{
  "emotion_name": "happy"
}
```

**Ответ:**
```json
{
  "status": "ok",
  "emote": "happy"
}
```

### GET `/api/eyes`
Получить текущую позицию глаз.

**Ответ:**
```json
{
  "status": "ok",
  "x": 0.0,
  "y": 0.0
}
```

### POST `/api/eyes`
Установить позицию глаз.

**Тело запроса:**
```json
{
  "x": 0.5,
  "y": -0.3
}
```

**Ответ:**
```json
{
  "status": "ok",
  "x": 0.5,
  "y": -0.3
}
```

---

## 🖥️ Терминал и оболочка

### POST `/api/cmd/send`
Отправить команду в оболочку робота.

**Тело запроса:**
```json
{
  "command": "ls -la"
}
```

**Ответ:**
```json
{
  "status": "ok"
}
```

### GET `/api/cmd/output`
Получить вывод команды оболочки.

**Ответ:**
```json
{
  "output": "Вывод команды здесь..."
}
```

### POST `/api/python/exec`
Выполнить Python код на роботе.

**Тело запроса:**
```json
{
  "code": "print('Привет от робота!')"
}
```

**Ответ:**
```json
{
  "stdout": "Привет от робота!\n",
  "stderr": ""
}
```

---

## 📁 Управление файлами

### GET `/api/files`
Получить список файлов и директорий.

**Параметры запроса:**
- `path` (string): Путь к директории (по умолчанию: "/")

**Ответ:**
```json
{
  "path": "/home/robot",
  "items": [
    {
      "name": "documents",
      "path": "/home/robot/documents",
      "type": "directory",
      "size": 0,
      "modified": 1640995200,
      "permissions": "755"
    },
    {
      "name": "config.txt",
      "path": "/home/robot/config.txt",
      "type": "file",
      "size": 1024,
      "modified": 1640995200,
      "permissions": "644"
    }
  ]
}
```

### GET `/api/files/read`
Прочитать содержимое файла.

**Параметры запроса:**
- `path` (string): Путь к файлу

**Ответ:**
```json
{
  "content": "Содержимое файла здесь...",
  "encoding": "utf-8",
  "size": 1024
}
```

### POST `/api/files/write`
Записать содержимое файла.

**Тело запроса:**
```json
{
  "path": "/home/robot/test.txt",
  "content": "Привет, мир!"
}
```

**Ответ:**
```json
{
  "success": true,
  "message": "Файл успешно сохранен",
  "size": 13
}
```

### POST `/api/files/create`
Создать новый файл или директорию.

**Тело запроса:**
```json
{
  "path": "/home/robot",
  "name": "new_directory",
  "type": "directory"
}
```

**Ответ:**
```json
{
  "success": true,
  "message": "Директория успешно создана",
  "path": "/home/robot/new_directory"
}
```

### POST `/api/files/delete`
Удалить файл или директорию.

**Тело запроса:**
```json
{
  "path": "/home/robot/old_file.txt"
}
```

**Ответ:**
```json
{
  "success": true,
  "message": "Успешно удалено"
}
```

### POST `/api/files/rename`
Переименовать файл или директорию.

**Тело запроса:**
```json
{
  "old_path": "/home/robot/old_name.txt",
  "new_name": "new_name.txt"
}
```

**Ответ:**
```json
{
  "success": true,
  "message": "Успешно переименовано",
  "new_path": "/home/robot/new_name.txt"
}
```

### POST `/api/files/upload`
Загрузить файлы.

**Данные формы:**
- `path` (string): Целевая директория
- `files` (file[]): Файлы для загрузки

**Ответ:**
```json
{
  "success": true,
  "uploaded_count": 2,
  "total_files": 2
}
```

### GET `/api/files/download`
Скачать файл.

**Параметры запроса:**
- `path` (string): Путь к файлу

**Ответ:** Скачивание файла с соответствующим MIME типом

---

## 🔧 Управление системой

### POST `/api/update`
Обновить систему робота.

**Ответ:**
```json
{
  "status": "ok",
  "message": "Обновление начато"
}
```

### POST `/api/shutdown`
Выключить систему робота.

**Ответ:**
```json
{
  "status": "ok",
  "message": "Завершение работы"
}
```

---

## 🐍 Python API

Робот также предоставляет Python API для прямой интеграции:

### Основные функции

```python
from robov_core.high_level import *  # В веб управлении не требуется

# Запуск фоновых сервисов
start_background()

# Управление сервоприводами
angle(channel, angle)        # Установить угол сервопривода
get_servo_angles()           # Получить все необработанные углы сервоприводов
get_servo_angles_physical()  # Получить физические углы с корректной инверсией, рекомендуется

# Камера
get_stereo_camera()       # Получить экземпляр камеры
get_raw_frame(left=True)  # Получить сырой кадр камеры

# Эмоции
emote(emotion_name)      # Установить эмоцию
get_emote()              # Получить текущую эмоцию
set_eyes_position(x, y)  # Установить позицию глаз
get_eyes_position()      # Получить позицию глаз

# Система
health_snapshot()      # Получить снимок состояния системы
ip_address()           # Получить IP-адрес
get_logs(count)        # Получить системные логи
```

### API камеры

```python
from robov_core.camera import StereoCamera

camera = StereoCamera("calib_params.json", source=0)

# Получить кадры
left_frame, right_frame = camera.get_rectified_frames()

# Расчет глубины
disparity_map = camera.compute_disparity(left_frame, right_frame)
depth_mm = camera.get_depth_at_point(disparity_map, x, y)

# Координаты реального мира
coords = camera.get_real_coords(x_px, y_px)
# Возвращает: {'x': 0.5, 'y': 0.3, 'z': 1.2} в метрах
```

### API AI

```python
from robov_core.ai import command, get_current_response, enable_ai_audio

# Отправить команду в AI. Он сам способен исполнять код, управлять роботом и выполнять команды. Имеет полный доступ.
command("Скажи привет")

# Получить накопленный ответ
response = get_current_response()

# Включить/выключить аудио
enable_ai_audio(True)
```

---

## 📝 Коды ответов

- **200 OK**: Запрос успешен
- **400 Bad Request**: Недействительные параметры
- **403 Forbidden**: Доступ запрещен
- **404 Not Found**: Ресурс не найден
- **409 Conflict**: Ресурс уже существует
- **500 Internal Server Error**: Ошибка сервера
