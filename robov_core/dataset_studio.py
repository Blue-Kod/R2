"""Офлайн-монтаж собранных эпизодов: вырезание диапазонов кадров в новые
эпизоды (неразрушающе — исходник не трогается).

Два хранилища (каждое — папка со своим ``meta.json``):
- ``library`` — все отснятые эпизоды (пишет ``DataCollector``; он источник
  правды по индексам и файлам);
- ``dataset`` — обучающий набор, пополняется только из студии
  (export → in dataset).

Формат тот же, что пишет ``DataCollector``: один parquet на эпизод, JPEG-байты
в struct-колонках, сквозные ``episode_index``/``index``. Здесь только чтение
исходников и запись новых файлов; живой сбор не затрагивается — для ``library``
аллокация индексов и запись meta идут через lock/состояние ``DataCollector``.
"""

import functools
import json
import shutil
import threading
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robov_core.data_collector import (
    BATCH_SIZE, COMPRESSION, DATA_DIR, NAN4, SAMPLE_HZ, STATE_CHANNELS,
    SYMBOL_COLORS, _schema,
)

LIBRARY = "library"
DATASET = "dataset"
STORES = (LIBRARY, DATASET)

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
DATASET_META = DATASET_DIR / "meta.json"

_dataset_lock = threading.Lock()
_dataset_cache: Dict = {"meta": None, "mtime": -1.0}


# --- dataset store (meta.json в dataset/) ---


def _dataset_meta() -> Dict:
    try:
        mtime = DATASET_META.stat().st_mtime
    except OSError:
        mtime = 0.0
    if _dataset_cache["meta"] is None or _dataset_cache["mtime"] != mtime:
        try:
            data = json.loads(DATASET_META.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError
        except (OSError, ValueError):
            data = {"next_episode": 0, "next_index": 0, "episodes": []}
        _dataset_cache.update(meta=data, mtime=mtime)
    return _dataset_cache["meta"]


def _dataset_save(meta: Dict) -> None:
    try:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        DATASET_META.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    try:
        mtime = DATASET_META.stat().st_mtime
    except OSError:
        mtime = 0.0
    _dataset_cache.update(meta=meta, mtime=mtime)


# --- общий доступ к эпизодам обоих сторов ---


def list_episodes(collector, store: str = LIBRARY) -> List[Dict]:
    if store == DATASET:
        return list(_dataset_meta().get("episodes", []))
    return collector.list_datasets()


def _record(collector, store: str, episode: int) -> Optional[Dict]:
    for rec in list_episodes(collector, store):
        if rec.get("episode_index") == episode:
            return rec
    return None


def episode_path(collector, store: str, episode: int) -> Optional[Path]:
    if store == DATASET:
        rec = _record(collector, store, episode)
        if rec is None:
            return None
        path = DATASET_DIR / rec["file"]
        return path if path.exists() else None
    return collector.dataset_file(episode)


def episode_meta(collector, store: str, episode: int) -> Optional[Dict]:
    """Мета эпизода для таймлайна (rows, fps, task, ...)."""
    rec = _record(collector, store, episode)
    if rec is None:
        return None
    rows = int(rec.get("rows", 0))
    duration = float(rec.get("duration_s", 0.0)) or 0.0
    fps = (rows / duration) if duration > 0 else SAMPLE_HZ
    return {
        "store": store,
        "episode_index": episode,
        "rows": rows,
        "task": rec.get("task", ""),
        "duration_s": duration,
        "fps": round(fps, 3),
        "session": rec.get("session"),
        "file": rec.get("file"),
        "bbox": frame_bbox(collector, store, episode, 0),
    }


def frame_bbox(collector, store: str, episode: int, frame: int = 0
               ) -> Optional[List[float]]:
    """Нормированный bbox цели на кадре (или None)."""
    path = episode_path(collector, store, episode)
    if path is None:
        return None
    try:
        table = pq.read_table(str(path), columns=["observation.target_bbox"],
                              filters=[("frame_index", "=", int(frame))])
    except Exception:
        return None
    if table.num_rows == 0:
        return None
    bb = table.column("observation.target_bbox")[0].as_py()
    if not bb or any(x != x for x in bb):          # пусто или NaN
        return None
    return [float(x) for x in bb]


def set_bbox(collector, store: str, episode: int,
             bbox: Optional[List[float]]) -> List[float]:
    """Записать bbox цели (на кадре 0); перезаписывает parquet эпизода.

    bbox — нормированный [x1,y1,x2,y2] или None (стереть). Возвращает записанное.
    """
    path = episode_path(collector, store, episode)
    if path is None:
        raise ValueError(f"Эпизод {episode} не найден в {store}")
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("bbox: нужно 4 числа")
        x1, y1, x2, y2 = (float(v) for v in bbox)
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        if x2 - x1 < 0.005 or y2 - y1 < 0.005:
            raise ValueError("bbox слишком маленький")
        val = [x1, y1, x2, y2]
    else:
        val = None
    table = pq.read_table(str(path))
    n = table.num_rows
    vals = [list(NAN4) for _ in range(n)]
    if val is not None and n:
        vals[0] = val
    col = pa.array(vals, type=pa.list_(pa.float32(), 4))
    idx = table.schema.get_field_index("observation.target_bbox")
    table = (table.append_column("observation.target_bbox", col) if idx < 0
             else table.set_column(idx, "observation.target_bbox", col))
    # Пишем во временный файл и атомарно переименовываем: обрыв питания
    # посреди записи не должен оставлять битый единственный экземпляр эпизода.
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp), compression=COMPRESSION)
    tmp.replace(path)
    _frame_cached.cache_clear()
    # в meta — для списка/студии
    rec = _record(collector, store, episode)
    if rec is not None:
        rec["bbox0"] = val
        if store == DATASET:
            with _dataset_lock:
                _dataset_save(_dataset_meta())
        else:
            with collector._lock:
                collector._save_meta()
    # None при стирании: [NaN]*4 — не валидный JSON, браузер не парсит NaN.
    return val


# --- чтение кадров/ряда state ---


@functools.lru_cache(maxsize=512)
def _frame_cached(path: str, frame: int, eye: str, width: int) -> bytes:
    col = f"observation.images.{eye}"
    try:
        table = pq.read_table(str(path), columns=[col],
                              filters=[("frame_index", "=", int(frame))])
    except Exception:
        return b""
    if table.num_rows == 0:
        return b""
    value = table.column(col)[0].as_py() or {}
    raw = value.get("bytes", b"") or b""
    if not raw or not width:
        return raw
    try:
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return raw
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (int(width), max(1, round(h * width / w))))
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    except Exception:
        return raw


def frame_jpeg(collector, store: str, episode: int, frame: int,
               eye: str = "left", width: Optional[int] = None,
               bbox: bool = False, symbol: str = "A") -> Optional[bytes]:
    """JPEG кадра. ``width`` — даунскейл, ``bbox`` — нарисовать метку цели."""
    path = episode_path(collector, store, episode)
    if path is None:
        return None
    jpeg = _frame_cached(str(path), int(frame),
                         "right" if eye == "right" else "left",
                         int(width) if width else 0)
    if not jpeg or not bbox:
        return jpeg or None
    bb = frame_bbox(collector, store, episode, frame)
    if bb is None:
        return jpeg
    try:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jpeg
        h, w = img.shape[:2]
        x1, y1 = int(round(bb[0] * w)), int(round(bb[1] * h))
        x2, y2 = int(round(bb[2] * w)), int(round(bb[3] * h))
        cv2.rectangle(img, (x1, y1), (x2, y2),
                      SYMBOL_COLORS.get(symbol, (0, 0, 255)), 2)
        return cv2.imencode(".jpg", img,
                            [cv2.IMWRITE_JPEG_QUALITY, 78])[1].tobytes()
    except Exception:
        return jpeg


def state_series(collector, store: str, episode: int,
                 max_points: int = 1200) -> Optional[Dict]:
    """Ряд state по кадрам (downsample) для дорожки движений серв."""
    path = episode_path(collector, store, episode)
    if path is None:
        return None
    try:
        table = pq.read_table(str(path), columns=["observation.state"])
    except Exception:
        return None
    values = table.column("observation.state").to_pylist()
    rows = len(values)
    step = max(1, (rows + max_points - 1) // max_points) if rows else 1
    return {
        "rows": rows,
        "step": step,
        "channels": [{"ch": ch, "name": name} for ch, name in STATE_CHANNELS],
        "values": values[::step],
    }


# --- запись нового эпизода ---


def _write_segment(pf: pq.ParquetFile, names, out_path: Path, seg: Dict,
                   new_ep: int, base_idx: int) -> Tuple[int, float]:
    """Записать [start, end) как новый эпизод. Возвращает (rows, length_s)."""
    start, end = int(seg["start"]), int(seg["end"])
    task = str(seg.get("task", ""))
    writer = pq.ParquetWriter(str(out_path), _schema(), compression=COMPRESSION)
    written = 0
    ts_offset = None
    last_ts = 0.0
    try:
        for batch in pf.iter_batches(batch_size=max(BATCH_SIZE, 1),
                                     columns=names):
            fi = batch.column("frame_index").to_pylist()
            if not fi or max(fi) < start or min(fi) >= end:
                continue
            keep = [j for j, f in enumerate(fi) if start <= f < end]
            if not keep:
                continue
            tb = batch.take(pa.array(keep, type=pa.int64()))
            n = tb.num_rows
            ts = tb.column("timestamp").to_pylist()
            if ts_offset is None:
                ts_offset = ts[0]

            # action[t]=state[t+1]: у последнего кадра сегмента action должен
            # указывать на себя, иначе он ведёт на вырезанный кадр.
            action = tb.column("action")
            if fi[keep[-1]] == end - 1:
                acts = action.to_pylist()
                acts[-1] = tb.column("observation.state").to_pylist()[-1]
                action = pa.array(acts, type=action.type)

            new_ts = [float(t) - float(ts_offset) for t in ts]
            last_ts = new_ts[-1]
            bb = (tb.column("observation.target_bbox")
                  if "observation.target_bbox" in tb.schema.names
                  else pa.array([list(NAN4)] * n, type=pa.list_(pa.float32(), 4)))
            table = pa.Table.from_arrays([
                tb.column("observation.images.left"),
                tb.column("observation.images.right"),
                tb.column("observation.state"),
                bb,
                action,
                pa.array([task] * n, type=pa.string()),
                pa.array(new_ts, type=pa.float32()),
                pa.array(range(written, written + n), type=pa.int64()),
                pa.array([new_ep] * n, type=pa.int64()),
                pa.array(range(base_idx + written, base_idx + written + n),
                         type=pa.int64()),
            ], schema=_schema())
            writer.write_table(table)
            written += n
    finally:
        writer.close()
    if written == 0:
        try:
            out_path.unlink()
        except OSError:
            pass
    return written, last_ts


def _reserve(collector, target: str, count: int) -> Tuple[int, int]:
    if target == DATASET:
        with _dataset_lock:
            meta = _dataset_meta()
            base_ep = int(meta.get("next_episode", 0))
            base_idx = int(meta.get("next_index", 0))
            meta["next_episode"] = base_ep + count
            _dataset_save(meta)
            return base_ep, base_idx
    with collector._lock:
        meta = collector._meta
        base_ep = int(meta.get("next_episode", 0))
        base_idx = int(meta.get("next_index", 0))
        meta["next_episode"] = base_ep + count
        return base_ep, base_idx


def _commit(collector, target: str, records: List[Dict], idx_cursor: int) -> None:
    if target == DATASET:
        with _dataset_lock:
            meta = _dataset_meta()
            meta.setdefault("episodes", []).extend(records)
            meta["next_index"] = idx_cursor
            _dataset_save(meta)
        return
    with collector._lock:
        meta = collector._meta
        meta.setdefault("episodes", []).extend(records)
        meta["next_index"] = idx_cursor
        collector._save_meta()


def _rollback(collector, target: str, base_ep: int) -> None:
    if target == DATASET:
        with _dataset_lock:
            meta = _dataset_meta()
            meta["next_episode"] = base_ep
            _dataset_save(meta)
        return
    with collector._lock:
        collector._meta["next_episode"] = base_ep
        collector._save_meta()


def apply_segments(collector, source_store: str, episode: int,
                   segments: List[Dict], target_store: str = LIBRARY,
                   out_dir: Optional[Path] = None) -> List[Dict]:
    """Каждый сегмент {start,end,task} → отдельный новый эпизод в target_store.

    Неразрушающе: исходный эпизод остаётся. Возвращает записи новых эпизодов.
    """
    if target_store not in STORES:
        raise ValueError(f"Неизвестный target: {target_store}")
    rec = _record(collector, source_store, episode)
    if rec is None:
        raise ValueError(f"Эпизод {episode} не найден в {source_store}")
    src = episode_path(collector, source_store, episode)
    if src is None:
        raise ValueError(f"Файл эпизода {episode} не найден")
    rows = int(rec.get("rows", 0))

    clean: List[Dict] = []
    for seg in segments or []:
        try:
            start, end = int(seg["start"]), int(seg["end"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Каждый сегмент: start и end — целые числа")
        if not (0 <= start < end <= rows):
            raise ValueError(f"Сегмент [{start}, {end}) вне 0..{rows}")
        clean.append({"start": start, "end": end, "task": str(seg.get("task", ""))})
    if not clean:
        raise ValueError("Нет сегментов для экспорта")

    root = Path(out_dir) if out_dir is not None else (
        DATASET_DIR if target_store == DATASET else DATA_DIR)
    prefix = "ds" if target_store == DATASET else "edit"
    session = f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}"
    session_dir = root / session

    base_ep, base_idx = _reserve(collector, target_store, len(clean))
    names = _schema().names
    pf = pq.ParquetFile(str(src))
    session_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []
    idx_cursor = base_idx
    try:
        for i, seg in enumerate(clean):
            new_ep = base_ep + i
            out_path = session_dir / f"episode_{new_ep:06d}.parquet"
            n, length_s = _write_segment(pf, names, out_path, seg, new_ep,
                                         idx_cursor)
            if n == 0:
                raise ValueError(f"Пустой сегмент [{seg['start']}, {seg['end']})")
            records.append({
                "episode_index": new_ep,
                "file": f"{session}/{out_path.name}",
                "session": session,
                "rows": n,
                "task": seg["task"],
                "started": datetime.now().isoformat(timespec="seconds"),
                "duration_s": round(length_s, 3),
                "index_first": idx_cursor,
                "index_last": idx_cursor + n - 1,
                "timestamp": _time.time(),
                "source_store": source_store,
                "source_episode": episode,
                "source_range": [seg["start"], seg["end"]],
            })
            idx_cursor += n
    except Exception:
        _rollback(collector, target_store, base_ep)
        shutil.rmtree(session_dir, ignore_errors=True)
        raise

    _commit(collector, target_store, records, idx_cursor)
    return records
