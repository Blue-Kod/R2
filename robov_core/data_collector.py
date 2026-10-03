"""Сбор данных для VLA-обучения в формат LeRobot (parquet).

Данные пишутся в ``ROOT_DIR/collected_data/<session>/episode_<N>.parquet``
(один файл на эпизод; эпизод пишется row-group'ами по мере сбора, поэтому
память не растёт с длиной эпизода). Сквозные счётчики ``episode_index`` /
``index`` — в ``collected_data/meta.json`` (переживают перезапуск процесса).

Схема колонок (по договорённости для VLA):
- observation.images.left/right — struct {bytes: binary, path: string}
  (объект {'bytes': ..., 'path': ''}, как в LeRobot).
- observation.state  — float32[10] на кадр: [neck, tilt, R_sz, R_sx, R_eb,
  L_sz, L_sx, L_eb, grip_R, grip_L] в градусах (логические команды, те же,
  что в servo.current_angles); грипперы — нормализованные 0..1.
- action            — float32[10]: по контракту action[row t] == state[row t+1];
  последний кадр эпизода — его собственный state (dummy, теряется при маске).
- task              — str, описание задачи (постоянно для эпизода).
- timestamp         — float32, монотонные секунды от начала эпизода.
- frame_index       — int64, 0,1,2,... внутри эпизода.
- episode_index     — int64, глобальный номер эпизода (счётчик из meta.json).
- index             — int64, глобальный сквозной id строки (порядок записи).
"""

import json
import threading
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robov_core.servo import REST_POSE as DEFAULT_ANGLES

DATA_DIR = Path(__file__).resolve().parent.parent / "collected_data"
META_FILE = DATA_DIR / "meta.json"

SAMPLE_HZ = 20.0          # желаемая частота сэмплирования, Гц
BATCH_SIZE = 100          # строк в одном row-group (граница памяти эпизода)
IMG_WH = (1280, 720)      # разрешение ректифицированных JPEG
JPEG_QUALITY = 85         # качество JPEG с камер
COMPRESSION = "zstd"      # сжатие parquet
STATE_DIM = 10

# Порядок каналов в state/action: [neck, tilt, правая рука (sz,sx,eb),
# левая рука (sz,sx,eb), гриппер правый, гриппер левый].
STATE_CHANNELS: List[Tuple[int, str]] = [
    (0,  "neck"),                 # шея
    (3,  "tilt"),                 # наклон
    (4,  "right_shoulder_z"),     # правое плечо (поворот)
    (1,  "right_shoulder_x"),     # правое плечо (вверх)
    (6,  "right_elbow"),          # правый локоть
    (5,  "left_shoulder_z"),      # левое плечо (поворот)
    (2,  "left_shoulder_x"),      # левое плечо (вверх)
    (7,  "left_elbow"),           # левый локоть
    (8,  "grip_right"),           # правый схват (0..1)
    (9,  "grip_left"),            # левый схват (0..1)
]
# Каналы грипперов нормализуются 0..1; остальные хранятся в градусах.
GRIPPER_CHANNELS = {8, 9}


def _schema() -> pa.Schema:
    """Схема parquet-таблицы (см. docstring модуля)."""
    image = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    vector = pa.list_(pa.float32(), STATE_DIM)
    return pa.schema([
        ("observation.images.left", image),
        ("observation.images.right", image),
        ("observation.state", vector),
        ("action", vector),
        ("task", pa.string()),
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
    ])


def _default_meta() -> Dict:
    return {"next_episode": 0, "next_index": 0, "episodes": []}


class DataCollector:
    """Фоновый сборщик эпизодов: стейт из серво + ректифицированные JPEG.

    Самосэмплирует в собственном потоке с частотой SAMPLE_HZ, независимо
    от управления: на каждом кадре фиксируется фактические углы (state),
    а абсолютным таргетом (action) становится следующий сэмпл.
    """

    def __init__(self,
                 state_getter: Callable[[], Dict[int, float]],
                 camera_getter: Callable[[], Optional[object]],
                 limits_getter: Callable[[], Dict[int, Tuple[float, float]]]):
        self._lock = threading.RLock()
        self._state_getter = state_getter
        self._camera_getter = camera_getter
        self._limits_getter = limits_getter

        self._collecting = False
        self._thread: Optional[threading.Thread] = None

        self._episode_index: Optional[int] = None
        self._frame = 0
        self._start_mono: Optional[float] = None
        self._start_wall: Optional[datetime] = None
        self._task = ""
        # Потоковая запись: в памяти только маленький батч + один предыдущий
        # сэмпл (для action[t]=state[t+1]); всё остальное уходит на диск.
        self._writer: Optional[pq.ParquetWriter] = None
        self._pending: List[Dict] = []
        self._prev_sample: Optional[Dict] = None
        self._row_count = 0
        self._first_index = 0
        self._last_ts = 0.0

        self._session_name: Optional[str] = None

        self._meta = _default_meta()
        self._load_meta()

    # --- meta.json (счётчики + индекс эпизодов для просмотра) ---

    def _load_meta(self) -> None:
        try:
            data = json.loads(META_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._meta = data
        except (OSError, ValueError):
            self._meta = _default_meta()

    def _save_meta(self) -> None:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            META_FILE.write_text(
                json.dumps(self._meta, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except OSError:
            pass

    # --- публичный API ---

    def status(self) -> Dict:
        with self._lock:
            return {
                "collecting": self._collecting,
                "episode_index": self._episode_index,
                "frames": self._frame,
                "episode_start": self._start_wall.isoformat(timespec="seconds")
                if self._start_wall else None,
                "since": round(_time.monotonic() - self._start_mono, 1)
                if self._start_mono is not None else 0.0,
                "task": self._task,
                "session_file": self._session_name,
                "next_episode": int(self._meta.get("next_episode", 0)),
                "next_index": int(self._meta.get("next_index", 0)),
                "episodes_total": len(self._meta.get("episodes", [])),
            }

    def start(self, task: str = "") -> Dict:
        """Начать новый эпизод сбора (сбрасывает буферы, берёт номер эпизода)."""
        with self._lock:
            if self._collecting:
                raise RuntimeError("Сбор данных уже идёт")
            self._episode_index = int(self._meta.get("next_episode", 0))
            self._meta["next_episode"] = self._episode_index + 1
            self._task = task or ""
            self._frame = 0
            self._pending = []
            self._prev_sample = None
            self._row_count = 0
            self._last_ts = 0.0
            self._first_index = int(self._meta.get("next_index", 0))
            self._start_mono = _time.monotonic()
            self._start_wall = datetime.now()
            if self._session_name is None:
                self._session_name = f"collect-{self._start_wall:%Y%m%d-%H%M%S}"
            self._open_writer()
            self._save_meta()
            self._collecting = True
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="r2-data-collector")
            self._thread.start()
            return self.status()

    def stop(self) -> Dict:
        """Остановить эпизод, записать action-сдвиг и сбросить row-group на диск."""
        with self._lock:
            if not self._collecting:
                raise RuntimeError("Сбор данных не идёт")
            self._collecting = False
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)
        with self._lock:
            result = self._finalize_episode()
            return {**self.status(), "episode": result}

    def close(self) -> None:
        """Остановить сбор (если шёл) и отбросить незавершённый эпизод."""
        with self._lock:
            if self._collecting:
                self._collecting = False
        if self._thread is not None:
            try:
                self._thread.join(timeout=2.0)
            except Exception:
                pass
        with self._lock:
            self._discard_episode()
        self._collecting = False
        self._thread = None

    # --- внутренняя механика ---

    def _state_vector(self) -> List[float]:
        angles = {}
        if self._state_getter is not None:
            try:
                angles = dict(self._state_getter() or {})
            except Exception:
                angles = {}
        limits = {}
        if self._limits_getter is not None:
            try:
                limits = dict(self._limits_getter() or {})
            except Exception:
                limits = {}
        vec: List[float] = []
        for channel, _ in STATE_CHANNELS:
            value = float(angles.get(channel, DEFAULT_ANGLES.get(channel, 90.0)))
            low, high = limits.get(channel, (0.0, 180.0))
            if channel in GRIPPER_CHANNELS:
                span = max(0.01, float(high) - float(low))
                vec.append(min(1.0, max(0.0, (value - float(low)) / span)))
            else:
                vec.append(value)
        return vec

    def _rectified_jpeg(self, left: bool) -> Dict[str, object]:
        camera = self._camera_getter()
        frame = None
        if camera is not None:
            try:
                frame = camera.get_rectified_frame(left=left, size=IMG_WH)
            except Exception:
                frame = None
        if frame is None:
            return {"bytes": b"", "path": ""}
        ok, jpeg = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return {"bytes": b"", "path": ""}
        return {"bytes": jpeg.tobytes(), "path": ""}

    def _episode_path(self) -> Path:
        fname = f"{self._session_name}/episode_{int(self._episode_index):06d}.parquet"
        return DATA_DIR / fname

    def _open_writer(self) -> None:
        path = self._episode_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(str(path), _schema(),
                                        compression=COMPRESSION)

    def _close_writer(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

    def _table_from(self, rows: List[Dict]) -> pa.Table:
        image = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
        vector = pa.list_(pa.float32(), STATE_DIM)
        episode = int(self._episode_index)
        return pa.Table.from_arrays([
            pa.array([r["img_left"] for r in rows], type=image),
            pa.array([r["img_right"] for r in rows], type=image),
            pa.array([r["state"] for r in rows], type=vector),
            pa.array([r["action"] for r in rows], type=vector),
            pa.array([self._task] * len(rows), type=pa.string()),
            pa.array([r["timestamp"] for r in rows], type=pa.float32()),
            pa.array([r["frame_index"] for r in rows], type=pa.int64()),
            pa.array([episode] * len(rows), type=pa.int64()),
            pa.array([self._first_index + r["frame_index"] for r in rows],
                     type=pa.int64()),
        ], schema=_schema())

    def _flush(self) -> None:
        if not self._pending or self._writer is None:
            return
        rows = self._pending
        try:
            self._writer.write_table(self._table_from(rows))
        except Exception as exc:
            print(f"[DataCollector] write failed: {exc}", flush=True)
        finally:
            # Всегда чистим батч: иначе при ошибке записи он растёт без границ.
            self._pending = []

    def _emit(self, sample: Dict, action: List[float]) -> None:
        self._pending.append({
            "img_left": sample["img_left"],
            "img_right": sample["img_right"],
            "state": sample["state"],
            "action": action,
            "timestamp": sample["timestamp"],
            "frame_index": sample["frame_index"],
        })
        self._row_count += 1

    def _sample_once(self) -> None:
        """Сэмплировать кадр. action[t]=state[t+1], поэтому строку пишем с
        задержкой в один сэмпл (последнюю закрывает _finalize_episode)."""
        left = self._rectified_jpeg(True)
        right = self._rectified_jpeg(False)
        state = self._state_vector()
        sample = {
            "img_left": left,
            "img_right": right,
            "state": state,
            "timestamp": float(_time.monotonic() - self._start_mono),
            "frame_index": self._frame,
        }
        self._frame += 1
        self._last_ts = sample["timestamp"]
        if self._prev_sample is not None:
            self._emit(self._prev_sample, sample["state"])
        self._prev_sample = sample
        if len(self._pending) >= BATCH_SIZE:
            self._flush()

    def _loop(self) -> None:
        interval = 1.0 / SAMPLE_HZ
        while self._collecting:
            try:
                # Кодирование JPEG — вне self._lock: иначе status()/stop()
                # ждут его десятки миллисекунд.
                self._sample_once()
            except Exception:
                pass
            _time.sleep(interval)

    def _finalize_episode(self) -> Dict:
        """Дописать хвост (dummy action), закрыть parquet и записать meta."""
        if self._prev_sample is not None:
            self._emit(self._prev_sample, self._prev_sample["state"])
            self._prev_sample = None
        self._flush()
        self._close_writer()

        n = self._row_count
        episode = int(self._episode_index)
        first_index = self._first_index
        if n == 0:
            self._discard_episode()
            raise RuntimeError("Эпизод пуст — нечего сохранять")

        fname = f"{self._session_name}/episode_{episode:06d}.parquet"
        record = {
            "episode_index": episode,
            "file": fname,
            "session": self._session_name,
            "rows": n,
            "task": self._task,
            "started": self._start_wall.isoformat(timespec="seconds"),
            "duration_s": round(self._last_ts, 3),
            "index_first": first_index,
            "index_last": first_index + n - 1,
            "timestamp": _time.time(),
        }
        episodes = self._meta.setdefault("episodes", [])
        episodes.append(record)
        self._meta["next_index"] = first_index + n
        self._save_meta()

        self._reset_episode_state()
        return record

    def _reset_episode_state(self) -> None:
        self._pending = []
        self._prev_sample = None
        self._row_count = 0
        self._frame = 0
        self._episode_index = None
        self._start_mono = None
        self._start_wall = None

    def _discard_episode(self) -> None:
        """Закрыть и удалить незавершённый эпизод (shutdown/пустой)."""
        self._close_writer()
        if self._episode_index is not None and self._session_name:
            try:
                self._episode_path().unlink()
            except OSError:
                pass
        self._reset_episode_state()

    # --- просмотр / скачивание данных ---

    def list_datasets(self) -> List[Dict]:
        """Индекс собранных эпизодов (из meta.json, отсортировано по времени)."""
        episodes = list(self._meta.get("episodes", []))
        episodes.sort(key=lambda r: r.get("timestamp", 0.0))
        return episodes

    def dataset_file(self, episode_index: int) -> Optional[Path]:
        for record in self._meta.get("episodes", []):
            if record.get("episode_index") == episode_index:
                path = DATA_DIR / record["file"]
                if path.exists():
                    return path
        return None

    def dataset_parquet(self, episode_index: int) -> Optional[pa.Table]:
        """Прочитать эпизод из его parquet-файла."""
        path = self.dataset_file(episode_index)
        if path is None:
            return None
        try:
            return pq.read_table(str(path))
        except Exception:
            return None