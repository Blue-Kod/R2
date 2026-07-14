#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Face Display using a single face texture (PNG).
Texture touches left/right screen edges.
Blinking = vertical squash of the whole face.
Smooth emote change: close -> swap texture -> open.
AI overlay: semi-transparent reasoning text behind the face.
Ticker: scrolling answer text at bottom synced to audio duration.
"""

import math
import os
import sys
import subprocess
import threading
import logging
import socket
import time
import random

try:
    import pygame
except ImportError:
    print("Pygame not installed. Install with: pip install pygame")
    sys.exit(1)

try:
    from robov_core.keyboard import PygameKeyboard
except ImportError:
    PygameKeyboard = None

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)


class FaceState:
    """
    Manages the face emote and blink animation.
    Animation phases: idle, closing, opening.
    During closing the current face shrinks vertically,
    at closed we swap to a new emote (if requested),
    then opening grows the new face back.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self._emote = "neutral"
        self._anim_phase = "idle"       # 'idle', 'closing', 'opening'
        self._anim_t = 0.0              # 0..1 progress of current phase
        self._half_duration = 0.05      # seconds per half-blink
        self._target_emote = None       # emote to swap to after closing
        self._pending_smooth_emote = None   # queued smooth change if busy
        self._last_idle_blink_time = 0.0
        self._min_blink_interval = 1.2  # seconds between idle blinks
        self._running = True

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------
    def get_emote(self):
        with self.lock:
            return self._emote

    def get_blink_scale(self):
        """Return vertical scale factor for rendering (1 = fully open)."""
        with self.lock:
            if self._anim_phase == "closing":
                return 1.0 - self._anim_t
            elif self._anim_phase == "opening":
                return self._anim_t
            else:
                return 1.0

    def request_emote_change(self, new_emote, smooth=True):
        """
        Ask to change the displayed emote.
        smooth=True:  close -> change -> open (natural blink)
        smooth=False: instantly swap (stops any running animation)
        """
        with self.lock:
            if new_emote == self._emote and not self._target_emote and not self._pending_smooth_emote:
                return  # nothing to do

            if not smooth:
                # Instant change: abort any animation, set immediately
                self._anim_phase = "idle"
                self._anim_t = 0.0
                self._target_emote = None
                self._pending_smooth_emote = None
                self._emote = new_emote
                return

            # Smooth change
            if self._anim_phase == "idle":
                # Start a new blink that will swap after closing
                self._anim_phase = "closing"
                self._anim_t = 0.0
                self._target_emote = new_emote
                self._pending_smooth_emote = None
            else:
                # Already blinking (closing/opening) – queue for later
                self._pending_smooth_emote = new_emote

    def trigger_idle_blink(self):
        """Start a simple blink (no emote change) if idle and interval ok."""
        with self.lock:
            now = time.time()
            if (self._anim_phase == "idle" and
                now - self._last_idle_blink_time >= self._min_blink_interval):
                self._anim_phase = "closing"
                self._anim_t = 0.0
                self._target_emote = None
                self._last_idle_blink_time = now
                return True
        return False

    def is_running(self):
        with self.lock:
            return self._running

    def stop(self):
        with self.lock:
            self._running = False

    # ------------------------------------------------------------------
    # Animation update – called every frame with delta time (seconds)
    # ------------------------------------------------------------------
    def update_logic(self, dt):
        with self.lock:
            was_idle = (self._anim_phase == "idle")

            if self._anim_phase == "closing":
                self._anim_t += dt / self._half_duration
                if self._anim_t >= 1.0:
                    self._anim_t = 1.0
                    # Closing finished – swap emote if requested
                    if self._target_emote is not None:
                        self._emote = self._target_emote
                        self._target_emote = None
                    # Start opening
                    self._anim_phase = "opening"
                    self._anim_t = 0.0

            elif self._anim_phase == "opening":
                self._anim_t += dt / self._half_duration
                if self._anim_t >= 1.0:
                    self._anim_t = 1.0
                    # Opening finished – back to idle
                    self._anim_phase = "idle"

                    # Check for a queued smooth change
                    if self._pending_smooth_emote is not None:
                        pending = self._pending_smooth_emote
                        self._pending_smooth_emote = None
                        # Start a new blink for it right away
                        self._anim_phase = "closing"
                        self._anim_t = 0.0
                        self._target_emote = pending

            # If idle, maybe trigger a random idle blink
            if self._anim_phase == "idle":
                now = time.time()
                if (now - self._last_idle_blink_time >= self._min_blink_interval
                        and random.random() < dt * 0.3):   # ~0.3 blinks per second
                    self._anim_phase = "closing"
                    self._anim_t = 0.0
                    self._target_emote = None
                    self._last_idle_blink_time = now


class RobotFace:
    def __init__(self, scale_factor=1.5):
        self.state = FaceState()
        self._exit_btn_rect = pygame.Rect(0, 0, 0, 0)
        self._textures = {}   # cache: emote_name -> pygame.Surface
        self.scale_factor = scale_factor  # Масштаб спрайта лица
        
        # Jiggle effect parameters
        self._jiggle_x = 0.0
        self._jiggle_y = 0.0
        self._jiggle_target_x = 0.0
        self._jiggle_target_y = 0.0
        self._jiggle_intensity = 13.0  # max pixels to jiggle
        self._jiggle_smooth = 2.0  # how fast to move towards target (higher = faster)
        self._jiggle_change_interval = 1  # seconds between target changes
        self._jiggle_timer = 0.0

        self._menu_visible = False
        self._menu_rect = pygame.Rect(0, 0, 0, 0)

        # Command input state
        self._cmd_buf = ""
        self._cmd_active = False
        self._cmd_input_rect = pygame.Rect(0, 0, 0, 0)
        self._cmd_send_rect = pygame.Rect(0, 0, 0, 0)
        self._close_btn_rect = pygame.Rect(0, 0, 0, 0)
        self._shutdown_btn_rect = pygame.Rect(0, 0, 0, 0)
        self._pygame_keyboard = None

    def _preload_all_emotions(self):
        """Загружает все PNG из папки emotions/ при старте."""
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotions")
        if not os.path.isdir(base_path):
            return
        for filename in os.listdir(base_path):
            if filename.lower().endswith(".png"):
                emote_name = os.path.splitext(filename)[0]
                log.info(f"Preloading: {emote_name}")
                self._get_texture_for_emote(emote_name)

    def _load_texture(self, emote_name):
        """Load PNG from emotions/ folder."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(script_dir, "emotions")
        os.makedirs(base_path, exist_ok=True)

        filename = os.path.join(base_path, f"{emote_name}.png")
        if not os.path.isfile(filename):
            filename = os.path.join(base_path, "neutral.png")
            if not os.path.isfile(filename):
                log.error(f"No texture for '{emote_name}' and no neutral.png in {base_path}")
                # fallback: grey rectangle with text
                surf = pygame.Surface((400, 400), pygame.SRCALPHA)
                surf.fill((100, 100, 100, 255))
                try:
                    font = pygame.font.SysFont("Arial", 24)
                    txt = font.render(emote_name, True, (255, 255, 255))
                    surf.blit(txt, (200 - txt.get_width() // 2, 200 - txt.get_height() // 2))
                except Exception:
                    pass
                return surf

        try:
            img = pygame.image.load(filename).convert_alpha()
            log.info(f"Loaded face texture: {emote_name} ({img.get_width()}x{img.get_height()})")
            return img
        except Exception as e:
            log.error(f"Failed to load {filename}: {e}")
            surf = pygame.Surface((400, 400), pygame.SRCALPHA)
            surf.fill((255, 0, 255, 200))
            return surf

    def _get_texture_for_emote(self, emote_name):
        if emote_name in self._textures:
            return self._textures[emote_name]
        tex = self._load_texture(emote_name)
        self._textures[emote_name] = tex
        return tex

    def _get_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 1))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _run(self):
        pygame.init()
        pygame.mixer.quit()
        pygame.display.init()

        info = pygame.display.Info()
        sw, sh = info.current_w, info.current_h

        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        sw, sh = screen.get_size()
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(True)
        clock = pygame.time.Clock()

        # Preload default and normal textures (display is ready)
        self._preload_all_emotions()
        
        # Initialize pygame keyboard
        if PygameKeyboard is not None:
            self._pygame_keyboard = PygameKeyboard(sw, sh)
            self._pygame_keyboard.set_callback(self._on_keyboard_input)
        
        font = pygame.font.SysFont("Arial", 28, bold=True)
        overlay_font = pygame.font.SysFont("Consolas", 28)
        subtitle_font = pygame.font.SysFont("Arial", 32, bold=True)

        while self.state.is_running():
            # delta time in seconds
            dt = clock.tick(60) / 1000.0

            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state.stop()
                elif event.type == pygame.KEYDOWN:
                    if self._cmd_active:
                        if event.key == pygame.K_ESCAPE:
                            self._cmd_active = False
                            if self._pygame_keyboard:
                                self._pygame_keyboard.hide()
                        elif event.key == pygame.K_BACKSPACE:
                            self._cmd_buf = self._cmd_buf[:-1]
                        elif event.key == pygame.K_RETURN:
                            if self._cmd_buf.strip():
                                self._send_command(self._cmd_buf.strip())
                                self._cmd_buf = ""
                        elif event.unicode and event.unicode.isprintable():
                            self._cmd_buf += event.unicode
                    else:
                        if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                            self.state.stop()
                elif event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.FINGERDOWN:
                    mx, my = event.pos
                    if event.type == pygame.FINGERDOWN:
                        mx = int(mx * sw)
                        my = int(my * sh)

                    # Handle pygame keyboard first if visible
                    if self._pygame_keyboard and self._pygame_keyboard.visible:
                        key_result = self._pygame_keyboard.handle_event(event)
                        if key_result:
                            action, value = key_result
                            if action == "char":
                                self._cmd_buf += value
                            elif action == "backspace":
                                self._cmd_buf = self._cmd_buf[:-1]
                            elif action == "enter":
                                if self._cmd_buf.strip():
                                    self._send_command(self._cmd_buf.strip())
                                    self._cmd_buf = ""
                            elif action == "close":
                                self._cmd_active = False
                                self._pygame_keyboard.hide()
                        continue

                    if my > sh * 0.8 and not self._menu_rect.collidepoint(mx, my):
                        self._menu_visible = not self._menu_visible
                        self.state._menu_visible = self._menu_visible
                    elif self._menu_visible:
                        if self._exit_btn_rect.collidepoint(mx, my):
                            log.info("Exit via menu")
                            self.state.stop()
                        elif self._close_btn_rect.collidepoint(mx, my):
                            self._menu_visible = False
                            self.state._menu_visible = False
                            self._cmd_active = False
                            if self._pygame_keyboard:
                                self._pygame_keyboard.hide()
                        elif self._shutdown_btn_rect.collidepoint(mx, my):
                            self._shutdown()
                            self.state.stop()
                        elif self._cmd_send_rect.collidepoint(mx, my):
                            if self._cmd_buf.strip():
                                self._send_command(self._cmd_buf.strip())
                                self._cmd_buf = ""
                        elif self._cmd_input_rect.collidepoint(mx, my):
                            if not self._cmd_active:
                                self._cmd_active = True
                                if self._pygame_keyboard:
                                    self._pygame_keyboard.show()
                        else:
                            if self._cmd_active:
                                self._cmd_active = False
                                if self._pygame_keyboard:
                                    self._pygame_keyboard.hide()

            # Update animation (including random idle blinks)
            self.state.update_logic(dt)

            # Update jiggle effect
            self._jiggle_timer += dt
            if self._jiggle_timer >= self._jiggle_change_interval:
                self._jiggle_timer = 0.0
                self._jiggle_target_x = random.uniform(-self._jiggle_intensity, self._jiggle_intensity)
                self._jiggle_target_y = random.uniform(-self._jiggle_intensity, self._jiggle_intensity)
            
            # Smoothly interpolate towards target position
            self._jiggle_x += (self._jiggle_target_x - self._jiggle_x) * min(1.0, self._jiggle_smooth * dt)
            self._jiggle_y += (self._jiggle_target_y - self._jiggle_y) * min(1.0, self._jiggle_smooth * dt)

            # Clear screen
            screen.fill((0, 0, 0))

            # --- AI reasoning overlay (behind the face) ---
            ai_state = self._get_ai_state()
            if ai_state:
                reasoning = ai_state.get("reasoning_text", "")
                tools = ai_state.get("tools_text", "")
                history = ai_state.get("reasoning_history", "")

                overlay_text = ""
                if history:
                    overlay_text = history
                if reasoning:
                    overlay_text += ("\n\n" if overlay_text else "") + reasoning
                if tools:
                    overlay_text += ("\n\n" if overlay_text else "") + tools

                if overlay_text:
                    overlay_surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
                    lines = self._wrap_text(overlay_font, overlay_text, sw - 80)
                    max_lines = (sh - 120) // 32
                    if len(lines) > max_lines:
                        paragraphs = overlay_text.split("\n\n")
                        keep = max(1, len(paragraphs) // 4)
                        trimmed = "\n\n".join(paragraphs[-keep:])
                        lines = self._wrap_text(overlay_font, trimmed, sw - 80)
                    y = 20
                    for line in lines:
                        if y > sh - 80:
                            break
                        surf = overlay_font.render(line, True, (50, 50, 50, 120))
                        overlay_surf.blit(surf, (40, y))
                        y += 32
                    screen.blit(overlay_surf, (0, 0))

            # Get current emote texture and blink scale
            emote = self.state.get_emote()
            tex = self._get_texture_for_emote(emote)
            blink_scale = self.state.get_blink_scale()

            # Scale texture to fill screen width with scale factor
            target_width = int(sw * self.scale_factor)
            # original aspect-ratio height
            orig_height = int(tex.get_height() * (target_width / tex.get_width()))
            # apply blink squish
            target_height = int(orig_height * blink_scale)

            # Prevent zero/negative height (pygame would crash)
            if target_height < 1:
                target_height = 1

            scaled_tex = pygame.transform.scale(tex, (target_width, target_height))

            # Draw centered both horizontally and vertically with jiggle effect
            x_offset = (sw - target_width) // 2 + int(self._jiggle_x)
            y_offset = (sh - target_height) // 2 + int(self._jiggle_y)
            screen.blit(scaled_tex, (x_offset, y_offset))

            # --- Subtitle overlay (centered, 10% above bottom) ---
            if ai_state:
                ticker_text = ai_state.get("ticker_text", "")
                if ticker_text:
                    max_w = int(sw * 0.85)
                    lines = self._wrap_text(subtitle_font, ticker_text, max_w)[:5]
                    line_h = subtitle_font.get_height() + 4
                    total_h = len(lines) * line_h
                    base_y = sh - int(sh * 0.10) - total_h
                    for i, line in enumerate(lines):
                        surf = subtitle_font.render(line, True, (255, 255, 255))
                        shadow = subtitle_font.render(line, True, (0, 0, 0))
                        lx = (sw - surf.get_width()) // 2
                        ly = base_y + i * line_h
                        screen.blit(shadow, (lx + 2, ly + 2))
                        screen.blit(surf, (lx, ly))

            # Menu overlay (simplified – touch bottom area toggles)
            if getattr(self, '_menu_visible', False):
                overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 220))
                screen.blit(overlay, (0, 0))

                m_w, m_h = 400, 460
                self._menu_rect = pygame.Rect((sw - m_w) // 2, 10, m_w, m_h)
                m_rect = self._menu_rect
                pygame.draw.rect(screen, (25, 25, 25), m_rect)
                pygame.draw.rect(screen, (200, 200, 200), m_rect, 2)

                y_cur = m_rect.y + 20

                ip_label = font.render(f"IP: {self._get_ip()}", True, (255, 255, 255))
                screen.blit(ip_label, (m_rect.x + 40, y_cur))
                y_cur += 50

                # Command input field
                input_color = (50, 50, 80) if self._cmd_active else (40, 40, 40)
                self._cmd_input_rect = pygame.Rect(m_rect.x + 20, y_cur, 280, 40)
                pygame.draw.rect(screen, input_color, self._cmd_input_rect)
                pygame.draw.rect(screen, (100, 140, 200) if self._cmd_active else (120, 120, 120), self._cmd_input_rect, 2)
                placeholder = self._cmd_buf if self._cmd_buf else "Команда..."
                cmd_color = (255, 255, 255) if self._cmd_buf else (120, 120, 120)
                cmd_surf = overlay_font.render(placeholder[-22:], True, cmd_color)
                screen.blit(cmd_surf, (self._cmd_input_rect.x + 8, self._cmd_input_rect.y + 6))
                # Blinking cursor when active
                if self._cmd_active and int(time.time() * 2) % 2 == 0:
                    cx = self._cmd_input_rect.x + 8 + cmd_surf.get_width()
                    pygame.draw.line(screen, (255, 255, 255), (cx, y_cur + 8), (cx, y_cur + 32), 2)

                # Send button
                self._cmd_send_rect = pygame.Rect(m_rect.x + 310, y_cur, 70, 40)
                pygame.draw.rect(screen, (40, 100, 40), self._cmd_send_rect)
                pygame.draw.rect(screen, (80, 200, 80), self._cmd_send_rect, 2)
                send_txt = overlay_font.render(">>>", True, (255, 255, 255))
                screen.blit(send_txt, send_txt.get_rect(center=self._cmd_send_rect.center))
                y_cur += 60

                # Exit button
                self._exit_btn_rect = pygame.Rect(m_rect.x + 50, y_cur, 300, 50)
                pygame.draw.rect(screen, (180, 40, 40), self._exit_btn_rect)
                pygame.draw.rect(screen, (255, 100, 100), self._exit_btn_rect, 2)
                txt = font.render("Exit", True, (255, 255, 255))
                screen.blit(txt, txt.get_rect(center=self._exit_btn_rect.center))
                y_cur += 65

                # Close menu button
                self._close_btn_rect = pygame.Rect(m_rect.x + 50, y_cur, 300, 45)
                pygame.draw.rect(screen, (50, 50, 80), self._close_btn_rect)
                pygame.draw.rect(screen, (100, 120, 200), self._close_btn_rect, 2)
                close_txt = font.render("Close", True, (255, 255, 255))
                screen.blit(close_txt, close_txt.get_rect(center=self._close_btn_rect.center))
                y_cur += 55

                # Shutdown button
                self._shutdown_btn_rect = pygame.Rect(m_rect.x + 50, y_cur, 300, 45)
                pygame.draw.rect(screen, (100, 30, 30), self._shutdown_btn_rect)
                pygame.draw.rect(screen, (200, 60, 60), self._shutdown_btn_rect, 2)
                shd_txt = font.render("Shutdown", True, (255, 255, 255))
                screen.blit(shd_txt, shd_txt.get_rect(center=self._shutdown_btn_rect.center))

            # --- Current LLM model (top-right corner) ---
            if ai_state:
                model_name = ai_state.get("current_model", "")
                if model_name:
                    model_surf = overlay_font.render(model_name, True, (120, 120, 120))
                    screen.blit(model_surf, (sw - model_surf.get_width() - 12, 10))

            # Draw pygame keyboard if visible
            if self._pygame_keyboard and self._pygame_keyboard.visible:
                self._pygame_keyboard.draw(screen)

            pygame.display.flip()

        log.info("Shutting down...")
        pygame.quit()

    def start(self):
        self._run()

    def stop(self):
        self.state.stop()

    def update_emote(self, name, smooth=True):
        self.state.request_emote_change(name, smooth)

    # kept for backwards compatibility (does nothing)
    def update_eyes_position(self, x, y):
        pass

    def _get_ai_state(self):
        """Get AI display state from the shared agent (thread-safe)."""
        try:
            from robov_core.ai import agent
            if agent and agent.display:
                return agent.display.get_all()
        except Exception:
            pass
        return None

    @staticmethod
    def _send_command(text: str):
        try:
            from robov_core.high_level import command
            command(text)
        except Exception as e:
            log.error(f"Command failed: {e}")

    @staticmethod
    def _shutdown():
        log.info("Shutdown requested — cleaning up...")
        try:
            from robov_core.high_level import cleanup
            cleanup()
        except Exception as e:
            log.error(f"Cleanup before shutdown failed: {e}")
        log.info("Launching system shutdown")
        try:
            subprocess.Popen(
                ["sudo", "shutdown", "-P", "now"],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            log.error(f"Shutdown failed: {e}")

    def _on_keyboard_input(self, text):
        """Callback for pygame keyboard input."""
        if text == "backspace":
            self._cmd_buf = self._cmd_buf[:-1]
        elif text == "enter":
            if self._cmd_buf.strip():
                self._send_command(self._cmd_buf.strip())
                self._cmd_buf = ""
        else:
            self._cmd_buf += text

    @staticmethod
    def _wrap_text(font, text, max_width):
        """Word-wrap text to fit within max_width pixels."""
        lines = []
        for paragraph in text.split("\n"):
            if not paragraph.strip():
                lines.append("")
                continue
            words = paragraph.split(" ")
            current = ""
            for word in words:
                test = (current + " " + word).strip()
                if font.size(test)[0] <= max_width:
                    current = test
                else:
                    if current:
                        lines.append(current)
                    current = word
            if current:
                lines.append(current)
        return lines


# ----------------------------------------------------------------------
# Backward compatibility wrappers (same as original)
# ----------------------------------------------------------------------
class EyeAPI:
    def __init__(self, robot_face):
        self._face = robot_face

    def update_emote(self, name):
        self._face.update_emote(name)

    def update_eyes_position(self, x, y):
        self._face.update_eyes_position(x, y)


class EyeDisplay:
    def __init__(self, scale_factor=1):
        self._face = RobotFace(scale_factor)
        self._api = EyeAPI(self._face)

    @property
    def api(self):
        return self._api

    def start(self):
        self._face.start()

    def stop(self):
        self._face.stop()


def optimize_for_arm():
    pass


def main():
    log.info("Robot Face Display (single texture, full width)")
    log.info("Place your face PNGs in 'emotions/' folder (neutral.png, ...)")
    face = RobotFace()
    face.start()
    try:
        while face.state.is_running():
            time.sleep(0.5)
    except KeyboardInterrupt:
        face.stop()


if __name__ == "__main__":
    main()