import os
import socket
import subprocess
from collections import deque
from typing import List

import psutil


class SystemService:
    def cpu_temp(self) -> str:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r", encoding="utf-8") as file:
                temp = int(file.read()) / 1000
                return f"{temp:.1f}°C"
        except Exception:
            return "N/A"

    def ip_address(self) -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except Exception:
            return "127.0.0.1"
        finally:
            sock.close()

    def recent_logs(self, n: int = 500) -> List[str]:
        for candidate in ("/var/log/syslog", "/var/log/messages"):
            if os.path.exists(candidate) and os.access(candidate, os.R_OK):
                try:
                    with open(candidate, "rb") as file:
                        file.seek(0, os.SEEK_END)
                        pos = file.tell()
                        block_size = 4096
                        lines: deque[str] = deque()
                        while len(lines) < n and pos > 0:
                            read_size = min(block_size, pos)
                            pos -= read_size
                            file.seek(pos, os.SEEK_SET)
                            chunk = file.read(read_size).decode("utf-8", errors="ignore")
                            lines.extendleft(reversed(chunk.splitlines()))
                        return list(lines)[-n:]
                except Exception:
                    pass

        try:
            output = subprocess.check_output(
                ["journalctl", "-n", str(n), "--no-pager"],
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
            )
            return output.splitlines()
        except Exception:
            return ["Нет доступа к системным логам"]

    def health_snapshot(self) -> dict:
        return {
            "cpu": psutil.cpu_percent(),
            "ram": psutil.virtual_memory().percent,
            "temp": self.cpu_temp(),
            "logs": self.recent_logs(500),
        }

