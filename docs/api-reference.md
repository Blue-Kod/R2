# Справочник API

Этот документ предоставляет полную документацию по всем эндпоинтам и Python интерфейсам API робота R2.

## REST API Эндпоинты

Робот R2 предоставляет RESTful API через Flask веб-сервер, обычно работающий на порту 80.

### Базовый URL
```
http://<robot-ip>/
```

---

## Системная информация

### GET `/api/data`
Возвращает снимок состояния системы.

**Ответ:**
```json
{
  "cpu_usage": 25.5,
  "memory_usage": 45.2,
  "temperature": "45.0°C",
  "fps": 15.0,
  "logs": ["..."]
}
```

### GET `/api/ip`
Возвращает IP-адрес робота.

---

## Камера и стереозрение

### GET `/video_feed`
MJPEG видеопоток со стереокамеры (с CLAHE выравниванием яркости).

### GET `/api/camera/params`
Получить текущие параметры камеры.

**Ответ:**
```json
{
  "depth_enabled": false,
  "detection_enabled": false,
  "detection_prompts": "",
  "alpha_depth": 0.3,
  "show_left": true,
  "num_disp": 128,
  "window_size": 11
}
```

### POST `/api/camera/params`
Обновить параметры камеры. Доступные ключи:
- `depth_enabled` (bool) — включить наложение карты глубины
- `detection_enabled` (bool) — включить распознавание объектов
- `detection_prompts` (str) — фильтр классов через запятую (например `"person, dog, cup"`)
- `alpha_depth` (float) — прозрачность наложения глубины (0.0–1.0)
- `show_left` (bool) — показать левую камеру
- `num_disp`, `window_size`, `min_disp` — параметры SGBM-матчера

### POST `/api/depth`
Измерение глубины в пикселях.

**Тело:** `{"x": 320, "y": 240}`  
**Ответ:** `{"depth": 1.25}` (метры)

### GET `/api/cursor_xyz`
3D-координаты под курсором мыши. Обновляется при движении мыши в браузере.

---

## Компьютерное зрение

### ObjectDetector (`robov_core/detector.py`)

ONNX-детектор на базе YOLOE-11s-seg с встроенными масками.

**Модели:**
| Модель | Размер | Вход | Скорость (x86) | Описание |
|--------|--------|------|----------------|----------|
| `yoloe-11s-seg-640.onnx` | 39 MB | 640x640 | ~300ms | Точная, по умолчанию |
| `yoloe-11s-seg-320.onnx` | 39 MB | 320x320 | ~80ms | Быстрая |
| `yolov8s.onnx` | 43 MB | 640x640 | ~150ms | COCO 80 классов (fallback) |

**79 распознаваемых классов:**
person, face, hand, dog, cat, rodent, laptop, monitor, keyboard, mouse, phone, remote control, headphones, speaker, charger, pen, pencil, scissors, notebook, book, calculator, cup, bottle, glass, plate, bowl, fork, spoon, knife, pan, kettle, microwave, refrigerator, sink, banana, apple, orange, egg, bread, can, chair, table, desk, sofa, bed, shelf, cabinet, stool, lamp, door, window, trash can, bucket, towel, pillow, blanket, clock, vase, plant pot, hat, glasses, bag, shoe, screwdriver, hammer, wrench, measuring tape, box, key, flashlight, ball, umbrella, toilet, bathtub, soap, toothbrush, comb, pill, hair dryer

```python
from robov_core.detector import ObjectDetector, Detection

det = ObjectDetector()  # Автоматически выбирает лучшую модель
det = ObjectDetector("models/yoloe-11s-seg-320.onnx")  # Указать модель вручную

# Детекция
detections: List[Detection] = det.detect(frame_bgr)
# Detection: name, class_id, confidence, x1,y1,x2,y2, center_x, center_y, mask(np.ndarray|None)

# Поиск по имени
matched = det.find("cup", frame_bgr)  # Возвращает List[Detection], отсортированные по conf

# Смена модели на лету
det.reinit_object_detection("yoloe-11s-seg-320.onnx")  # → True/False
det.model_name  # Текущее имя файла модели

# Визуализация
vis = det.annotate(frame, detections, labels=None)
# Маски отображаются полупрозрачным оверлеем с контурами
# Лейблы: автоматические или кастомные
```

**Декодирование масок:**
Маски декодируются из prototype-мапы (32 канала) через матричное умножение coefficients @ proto → sigmoid → resize → clip to bbox → morphological open → connected components (фильтрация кластеров < 5% bbox-площади).

### StereoCamera (`robov_core/camera.py`)

Стереокамера с клавибровкой, ректификацией, глубиной и детекцией.

```python
camera = StereoCamera("cam_params.json", source=0)

# Калибровка: Kl, Kr, Dl, Dr, R, T, Q — из cam_params.json
# Ректификация: cv2.fisheye.stereoRectify, balance=0.8
# img_size = (640, 360) — размер обработки
# imSize = (1280, 720) — реальное разрешение камеры
```

**Ключевые методы:**

```python
# Получить ректифицированные кадры
left, right = camera.get_rectified_frames()

# Расчёт диспаритета
disp = camera.compute_disparity(left, right)

# Глубина в точке (мм → метры)
depth_m = camera.get_depth_at(disp, x, y) / 1000

# 3D-координаты в метрах
coords = camera.get_real_coords(x_px, y_px)
# → {'x': 0.5, 'y': -0.2, 'z': 1.2, 'depth': 1.2}

# 3D по маске (с эрозией, фильтрацией фона)
m3d = camera._get_mask_3d(mask, erode_px=3)
# → {'x': ..., 'y': ..., 'z': ..., 'vx': ..., 'vy': ..., 'vz': ..., 'n_points': ...}
```

**Детекция + глубина (Python API):**

```python
# find() — найти один объект по имени
result = camera.find("dog")
# → {'name': 'dog', 'confidence': 0.92, 'bbox': {...}, 'x': 0.3, 'y': -0.1, 'z': 1.5,
#    'vx': 0.2, 'vy': 0.4, 'vz': 0.3, 'depth': 1.5}

# scan() — найти все объекты по фильтру
results = camera.scan(prompts="person, dog, cup")
# → [{'name': 'person', 'confidence': 0.95, 'bbox': {...}, 'x': ..., ...}, ...]
```

**Overlay-система:**
Результаты `find()` и `scan()` автоматически отображаются на видеопотоке в браузере как полупрозрачный оверлей на 3 секунды с плавным затуханием.

```python
camera._push_overlay([{"name": "dog", "confidence": 0.9, "bbox": {"x1": 100, ...}}])
camera._overlay_duration = 3.0  # Секунды
```

**CLAHE (компенсация backlit):**
На каждом кадре применяется CLAHE (clipLimit=2.0, tileGridSize=8x8) через L-канал LAB. Выравнивает яркость при_backlit-сценах.

### Глубина (`robov_core/depth_providers.py`)

```python
from robov_core.depth_providers import StereoSGBMDepthProvider

provider = StereoSGBMDepthProvider()
provider.setup(num_disp=160, window_size=11, min_disp=0, wls_enabled=True)
result = provider.compute(gray_left, gray_right)
disp = result.disparity  # int16, /16.0 → пиксели диспаритета
```

---

## Управление сервоприводами

### POST `/api/servo/<channel>/<angle>`
Установить угол сервопривода (0–180).

---

## Эмоции

### GET/POST `/api/emote`
Получить/установить эмоцию. Список: happy, neutral, scared, spooked, sleep.

### GET/POST `/api/eyes`
Получить/установить позицию глаз (x, y от -1.0 до 1.0).

---

## Терминал

### POST `/api/cmd/send`
Отправить команду в shell.

### GET `/api/cmd/output`
Получить вывод shell.

### POST `/api/python/exec`
Выполнить Python-код. Агент имеет доступ ко всем функциям робота.

---

## AI-агент

AI-агент ( EveryLLM ) имеет доступ к следующим функциям через Python-окружение:

```python
# Детекция и глубина
find("person")           # → dict с name, bbox, x,y,z, vx,vy,vz
precise_find("cup, dog") # → List[RealObject]
scan("person, dog")      # → List[RealObject]

# Навигация и манипуляции (stubs)
goto(target)             # → bool (заглушка)
grab(target)             # → bool (заглушка)
move_arm_to(target)      # → bool (заглушка)

# Сервоприводы
set_servo_physical(channel, angle)  # Физический угол с инверсией
get_servo_angles_physical()

# Речь
speak("Привет!")         # TTS через Piper (русский)

# Эмоции и глаза
set_emote("happy")
set_eyes_position(0.5, -0.3)

# Терминал
shell_start(), shell_write(cmd), shell_output()
```

### Dataclass RealObject

```python
@dataclass
class RealObject:
    name: str          # "dog"
    confidence: float  # 0.92
    position: Position # Position(x=0.3, y=-0.1, z=1.5) — в метрах
    bbox: dict         # {"x1": 100, "y1": 50, "x2": 300, "y2": 400}
    depth: float       # 1.5 (метры)
    vx: float          # 0.2 — ширина bounding volume (метры)
    vy: float          # 0.4 — высота
    vz: float          # 0.3 — глубина
```

### Переключение модели детекции

```python
reinit_object_detection("yoloe-11s-seg-320.onnx")  # Быстрая
reinit_object_detection("yoloe-11s-seg-640.onnx")  # Точная (по умолчанию)
reinit_object_detection("yolov8s.onnx")             # COCO fallback
```

### REST API для детекции

**GET `/api/scan?prompts=person,dog`** — сканирование сцену.

**Ответ:**
```json
{
  "objects": [
    {
      "name": "person",
      "confidence": 0.95,
      "bbox": {"x1": 100, "y1": 50, "x2": 300, "y2": 400},
      "x": 0.3, "y": -0.1, "z": 1.5, "depth": 1.5,
      "vx": 0.2, "vy": 0.4, "vz": 0.3
    }
  ],
  "count": 1
}
```

**POST `/api/detection/model`** — смена модели.

**Тело:** `{"model": "yoloe-11s-seg-320.onnx"}`

---

## Коды ответов

- **200 OK**: Запрос успешен
- **400 Bad Request**: Недействительные параметры
- **500 Internal Server Error**: Ошибка сервера
