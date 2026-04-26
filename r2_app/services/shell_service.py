import platform
import subprocess
import threading
from collections import deque
from typing import Optional


class ShellService:
    def __init__(self, logger) -> None:
        self._logger = logger
        self._proc = None
        self._buffer = deque(maxlen=2000)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if platform.system() == "Windows":
            try:
                self._proc = subprocess.Popen(
                    ["powershell", "-NoLogo"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            except Exception as exc:
                self._logger.log(f"Не удалось запустить PowerShell: {exc}")
                self._running = False
                return
        else:
            try:
                import ptyprocess

                self._proc = ptyprocess.PtyProcess.spawn(["/bin/bash", "-i"])
                self._proc.setwinsize(24, 80)
            except Exception as exc:
                self._logger.log(f"Не удалось запустить shell: {exc}")
                self._running = False
                return

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        try:
            while self._running:
                if self._proc is None:
                    data = b""
                elif platform.system() == "Windows":
                    data = self._proc.stdout.read1(1024) if self._proc.stdout else b""
                else:
                    data = self._proc.read(1024)
                if not data:
                    break
                text = self._decode_output(data)
                with self._lock:
                    self._buffer.append(text)
        except Exception:
            pass
        finally:
            self._running = False

    @staticmethod
    def _decode_output(data: bytes) -> str:
        for enc in ("utf-8", "cp866", "cp1251"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    def write(self, command: str) -> bool:
        if not self._running or self._proc is None:
            return False
        try:
            if not command.endswith("\n"):
                command += "\n"
            if platform.system() == "Windows":
                if not self._proc.stdin:
                    return False
                self._proc.stdin.write(command.encode("utf-8"))
                self._proc.stdin.flush()
            else:
                self._proc.write(command.encode("utf-8"))
            return True
        except Exception:
            return False

    def output(self) -> str:
        with self._lock:
            return "".join(self._buffer)

    def stop(self) -> None:
        self._running = False
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass

