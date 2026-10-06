"""Self-check for dataset_studio: split a synthetic episode and verify that
frame/episode/index remap, action boundary fix, timestamp reset, and export to
both stores (library / dataset) are correct.

Run:  .venv/bin/python test_dataset_studio.py
"""

import tempfile
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from robov_core.data_collector import _schema
from robov_core import dataset_studio as ds

SCHEMA = _schema()
STATE_DIM = SCHEMA.field("observation.state").type.list_size
N = 10


def _build_source(path: Path) -> None:
    state = [[float(i)] * STATE_DIM for i in range(N)]
    action = [state[i + 1] if i + 1 < N else state[i] for i in range(N)]
    imgs = [{"bytes": f"L{i}".encode(), "path": ""} for i in range(N)]
    table = pa.Table.from_arrays([
        pa.array(imgs, type=SCHEMA.field("observation.images.left").type),
        pa.array(imgs, type=SCHEMA.field("observation.images.right").type),
        pa.array(state, type=SCHEMA.field("observation.state").type),
        pa.array(action, type=SCHEMA.field("action").type),
        pa.array(["orig"] * N, type=pa.string()),
        pa.array([i * 0.1 for i in range(N)], type=pa.float32()),
        pa.array(range(N), type=pa.int64()),
        pa.array([0] * N, type=pa.int64()),
        pa.array(range(N), type=pa.int64()),
    ], schema=SCHEMA)
    pq.write_table(table, str(path), row_group_size=4)


class FakeCollector:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.RLock()
        self._meta = {
            "next_episode": 1,
            "next_index": 0,
            "episodes": [{
                "episode_index": 0,
                "file": "src/episode_000000.parquet",
                "session": "src",
                "rows": N,
                "task": "orig",
                "duration_s": N * 0.1,
                "timestamp": 0.0,
            }],
        }

    def list_datasets(self):
        return list(self._meta["episodes"])

    def dataset_file(self, episode):
        return self._path if episode == 0 else None

    def _save_meta(self):
        pass


def _test_library(fc, tmp: Path) -> None:
    records = ds.apply_segments(fc, ds.LIBRARY, 0, [
        {"start": 1, "end": 4, "task": "a"},
        {"start": 6, "end": 10, "task": "b"},
    ], target_store=ds.LIBRARY, out_dir=tmp / "out")

    assert len(records) == 2, records
    a = pq.read_table(str(tmp / "out" / records[0]["file"]))
    b = pq.read_table(str(tmp / "out" / records[1]["file"]))
    assert (a.num_rows, b.num_rows) == (3, 4), (a.num_rows, b.num_rows)

    assert a.column("frame_index").to_pylist() == [0, 1, 2]
    assert a.column("episode_index").to_pylist() == [1, 1, 1]
    assert a.column("index").to_pylist() == [0, 1, 2]
    assert a.column("task").to_pylist() == ["a", "a", "a"]
    assert a.column("timestamp").to_pylist()[0] == 0.0
    assert a.column("observation.state").to_pylist()[0] == [1.0] * STATE_DIM
    # action последнего кадра сегмента указывает на себя (не на вырезанный)
    assert a.column("action").to_pylist()[-1] == a.column("observation.state").to_pylist()[-1]

    assert b.column("frame_index").to_pylist() == [0, 1, 2, 3]
    assert b.column("task").to_pylist() == ["b"] * 4
    assert b.column("timestamp").to_pylist()[0] == 0.0
    assert b.column("action").to_pylist()[-1] == b.column("observation.state").to_pylist()[-1]
    assert records[1]["index_first"] == 3
    assert records[1]["source_range"] == [6, 10]

    assert pq.read_table(str(fc._path)).num_rows == N          # исходник цел
    assert fc._meta["next_episode"] == 3                       # индексы сдвинуты
    assert fc._meta["next_index"] == 7
    assert len(fc._meta["episodes"]) == 3
    assert ds.list_episodes(fc, ds.LIBRARY)[-1]["task"] == "b"


def _test_dataset(fc, tmp: Path) -> None:
    # dataset — отдельный стор; не трогаем реальную папку
    ds.DATASET_DIR = tmp / "dataset"
    ds.DATASET_META = ds.DATASET_DIR / "meta.json"
    ds._dataset_cache = {"meta": None, "mtime": -1.0}

    records = ds.apply_segments(fc, ds.LIBRARY, 0, [
        {"start": 0, "end": 2, "task": "curated"},
    ], target_store=ds.DATASET)

    assert len(records) == 1
    assert records[0]["episode_index"] == 0                    # dataset-счётчик свой
    listed = ds.list_episodes(fc, ds.DATASET)
    assert len(listed) == 1
    assert listed[0]["task"] == "curated"
    out = pq.read_table(str(ds.DATASET_DIR / records[0]["file"]))
    assert out.num_rows == 2
    # library не изменился от экспорта в dataset
    assert fc._meta["next_episode"] == 3


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "episode_000000.parquet"
        _build_source(src)
        fc = FakeCollector(src)

        _test_library(fc, tmp)

        # точечное чтение кадра / ряд state (путь UI)
        assert ds.frame_jpeg(fc, ds.LIBRARY, 0, 3, eye="left") == b"L3"
        assert ds.frame_jpeg(fc, ds.LIBRARY, 0, 9, eye="right") == b"L9"
        series = ds.state_series(fc, ds.LIBRARY, 0)
        assert series["rows"] == N
        assert len(series["channels"]) == STATE_DIM
        assert len(series["values"]) == N
        assert series["values"][2][0] == 2.0

        _test_dataset(fc, tmp)

    print("dataset_studio self-check OK")


if __name__ == "__main__":
    main()
