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
        self._pid = None
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
            # pty.fork() gives the child its own session + controlling terminal,
            # so bash enables job control and full-screen apps (top, vim) work.
            import fcntl
            import os
            import pty
            import struct
            import termios

            env = os.environ.copy()
            env["TERM"] = "xterm-256color"
            env["COLORTERM"] = "truecolor"

            pid, master_fd = pty.fork()
            if pid == 0:
                os.environ.update(env)
                try:
                    os.execvpe("/bin/bash", ["/bin/bash", "-i"], env)
                except Exception:
                    os._exit(127)
            self._pid = pid
            self._master_fd = master_fd
            try:
                fcntl.ioctl(master_fd, termios.TIOCSWINSZ,
                            struct.pack("HHHH", 24, 80, 0, 0))
            except Exception:
                pass
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def start_ssh(self, host: str, port: int, user: str, password: str,
                  key_filename: str | None = None):
        self.close()
        import os
        import paramiko
        self._mode = "ssh"
        self._ssh_client = paramiko.SSHClient()
        self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # Prefer local SSH keys / agent (passwordless), fall back to the password.
        kwargs = dict(
            port=port, username=user, timeout=10,
            banner_timeout=10, auth_timeout=10,
            look_for_keys=True, allow_agent=True,
        )
        if password:
            kwargs["password"] = password
        keyfile = key_filename or os.environ.get("R2_SSH_KEY")
        if keyfile:
            kwargs["key_filename"] = keyfile
        self._ssh_client.connect(host, **kwargs)
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
                    import signal
                    if self._pid:
                        try:
                            os.kill(self._pid, signal.SIGHUP)
                        except Exception:
                            pass
                        try:
                            os.kill(self._pid, signal.SIGTERM)
                        except Exception:
                            pass
                        try:
                            os.waitpid(self._pid, 0)
                        except Exception:
                            pass
                    if self._master_fd is not None:
                        try:
                            os.close(self._master_fd)
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
        self._pid = None
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
