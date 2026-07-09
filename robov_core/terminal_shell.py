import threading
import queue
import platform
import subprocess
import time as _time


class TerminalShell:
    def __init__(self):
        self._proc = None
        self._out_buf = queue.Queue()
        self._running = False
        self._mode = None
        self._reader_thread = None
        self._lock = threading.Lock()
        self._master_fd = None
        self._slave_fd = None
        self._ssh_client = None
        self._ssh_channel = None

    def start_local(self):
        self.close()
        self._mode = "local"
        if platform.system() == "Windows":
            self._proc = subprocess.Popen(
                ["powershell", "-NoLogo", "-NoProfile"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
        else:
            import pty
            import os
            self._master_fd, self._slave_fd = pty.openpty()
            self._proc = subprocess.Popen(
                ["/bin/bash", "-i"],
                stdin=self._slave_fd,
                stdout=self._slave_fd,
                stderr=self._slave_fd,
                close_fds=True,
            )
            os.close(self._slave_fd)
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def start_ssh(self, host: str, port: int, user: str, password: str):
        self.close()
        import paramiko
        self._mode = "ssh"
        self._ssh_client = paramiko.SSHClient()
        self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh_client.connect(
            host, port=port, username=user, password=password,
            look_for_keys=False, allow_agent=False, timeout=10,
        )
        self._ssh_channel = self._ssh_client.invoke_shell(
            term="xterm-256color", width=80, height=24,
        )
        self._ssh_channel.settimeout(0.05)
        self._running = True
        self._reader_thread = threading.Thread(target=self._ssh_reader_loop, daemon=True)
        self._reader_thread.start()

    def write(self, data: bytes | str):
        if isinstance(data, str):
            data = data.encode("utf-8")
        with self._lock:
            if not self._running:
                return
            try:
                if self._mode == "local":
                    if platform.system() == "Windows":
                        if self._proc and self._proc.stdin:
                            self._proc.stdin.write(data)
                            self._proc.stdin.flush()
                    else:
                        if self._master_fd is not None:
                            import os
                            os.write(self._master_fd, data)
                elif self._mode == "ssh":
                    if self._ssh_channel and self._ssh_channel.active:
                        self._ssh_channel.send(data)
            except Exception:
                pass

    def read(self) -> str:
        out = []
        while True:
            try:
                out.append(self._out_buf.get_nowait())
            except queue.Empty:
                break
        return "".join(out)

    def resize(self, cols: int, rows: int):
        with self._lock:
            if not self._running:
                return
            if self._mode == "local" and platform.system() != "Windows":
                import fcntl
                import struct
                import termios
                import os
                try:
                    buf = struct.pack("HHHH", rows, cols, 0, 0)
                    fcntl.ioctl(self._master_fd, termios.TIOCSWINSZ, buf)
                except Exception:
                    pass
            elif self._mode == "ssh":
                if self._ssh_channel and self._ssh_channel.active:
                    try:
                        self._ssh_channel.resize_pty(width=cols, height=rows)
                    except Exception:
                        pass

    def close(self):
        self._running = False
        with self._lock:
            if self._mode == "local":
                if platform.system() == "Windows":
                    if self._proc:
                        try:
                            self._proc.terminate()
                            self._proc.wait(timeout=3)
                        except Exception:
                            try:
                                self._proc.kill()
                            except Exception:
                                pass
                else:
                    import os
                    if self._master_fd is not None:
                        try:
                            os.close(self._master_fd)
                        except Exception:
                            pass
                    if self._proc:
                        try:
                            self._proc.terminate()
                            self._proc.wait(timeout=3)
                        except Exception:
                            try:
                                self._proc.kill()
                            except Exception:
                                pass
            elif self._mode == "ssh":
                if self._ssh_channel:
                    try:
                        self._ssh_channel.close()
                    except Exception:
                        pass
                if self._ssh_client:
                    try:
                        self._ssh_client.close()
                    except Exception:
                        pass
        self._mode = None
        self._proc = None
        self._master_fd = None
        self._slave_fd = None
        self._ssh_client = None
        self._ssh_channel = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def mode(self) -> str | None:
        return self._mode

    def _reader_loop(self):
        try:
            while self._running:
                try:
                    if platform.system() == "Windows":
                        if self._proc and self._proc.stdout:
                            chunk = self._proc.stdout.read(4096)
                            if not chunk:
                                break
                            self._out_buf.put(chunk.decode("utf-8", errors="replace"))
                    else:
                        import select
                        import os
                        r, _, _ = select.select([self._master_fd], [], [], 0.1)
                        if r:
                            chunk = os.read(self._master_fd, 4096)
                            if not chunk:
                                break
                            self._out_buf.put(chunk.decode("utf-8", errors="replace"))
                except (OSError, ValueError):
                    break
        finally:
            self._running = False
            self._out_buf.put("\r\n\x1b[31m[Shell closed]\x1b[0m\r\n")

    def _ssh_reader_loop(self):
        try:
            while self._running and self._ssh_channel and self._ssh_channel.active:
                try:
                    if self._ssh_channel.recv_ready():
                        chunk = self._ssh_channel.recv(4096)
                        if not chunk:
                            break
                        self._out_buf.put(chunk.decode("utf-8", errors="replace"))
                    else:
                        _time.sleep(0.02)
                except Exception:
                    break
        finally:
            self._running = False
            self._out_buf.put("\r\n\x1b[31m[SSH disconnected]\x1b[0m\r\n")
