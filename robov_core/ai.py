"""
AI Library for R2 Robot – Filler version with dummy implementations.
All functions retain original signatures but contain no real logic, returning filler data as needed.
"""

# Global state variables (kept for compatibility, no real logic)
_audio_enabled = True
_current_response = ""
_ws = None
_loop = None
_running = False
_executing = False

# -----------------------------------------------------------------------------
# Filler functions available for #EXECUTE (dummy implementations)
# -----------------------------------------------------------------------------
def log(message: str) -> None:
    """Filler log function, performs no real logging."""
    pass


def get_system_stats() -> dict:
    """Filler get_system_stats, returns dummy system stats dict."""
    return {
        "cpu_usage": 20,
        "memory_usage": 30,
        "temperature": "45.0°C"
    }

_AI_EXEC_GLOBALS = {
    "log": log,
    "get_system_stats": get_system_stats,
}

# -----------------------------------------------------------------------------
# Public interface functions (dummy implementations)
# -----------------------------------------------------------------------------
def command(text: str):
    """Filler command, sends no real data."""
    pass

def send_frame():
    """Filler send_frame, captures no real camera data."""
    pass

def get_current_response():
    """Filler get_current_response, returns empty string."""
    return ""

def enable_ai_audio(enabled: bool):
    """Filler enable_ai_audio, modifies no real state."""
    pass

def cleanup():
    """Filler cleanup, performs no real cleanup."""
    pass