from datetime import datetime


class AppLogger:
    """Simple logger that outputs to stdout only."""

    def log(self, *parts: object) -> str:
        message = " ".join(str(part) for part in parts)
        full_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(full_message)
        return full_message

