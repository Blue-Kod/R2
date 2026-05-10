#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Face Display using a single face texture (PNG).
Texture touches left/right screen edges.
Blinking = vertical squash of the whole face.
Smooth emote change: close -> swap texture -> open.
"""

import math
import os
import sys
import threading
import logging
import socket
import time
import random

os.environ['SDL_VIDEO_CENTERED'] = '1'

try:
    import pygame
except ImportError:
    print("Pygame not installed. Install with: pip install pygame")
    sys.exit(1)

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
        pygame.display.init()

        info = pygame.display.Info()
        sw, sh = info.current_w, info.current_h

        screen = pygame.display.set_mode((sw, sh), pygame.FULLSCREEN)
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(True)
        clock = pygame.time.Clock()

        # Preload default and normal textures (display is ready)
        self._preload_all_emotions()
        
        font = pygame.font.SysFont("Arial", 28, bold=True)

        while self.state.is_running():
            # delta time in seconds
            dt = clock.tick(60) / 1000.0

            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state.stop()
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                        self.state.stop()
                elif event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.FINGERDOWN:
                    mx, my = event.pos
                    if self._exit_btn_rect.collidepoint(mx, my):
                        log.info("Exit via menu")
                        self.state.stop()
                    else:
                        # toggle menu on bottom 20% of screen
                        if my > sh * 0.8:
                            self._menu_visible = not getattr(self, '_menu_visible', False)
                            self.state._menu_visible = self._menu_visible  # keep in sync

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

            # Menu overlay (simplified – touch bottom area toggles)
            if getattr(self, '_menu_visible', False):
                overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 220))
                screen.blit(overlay, (0, 0))

                m_w, m_h = 400, 250
                m_rect = pygame.Rect((sw - m_w) // 2, (sh - m_h) // 2, m_w, m_h)
                pygame.draw.rect(screen, (25, 25, 25), m_rect)
                pygame.draw.rect(screen, (200, 200, 200), m_rect, 2)

                ip_label = font.render(f"IP: {self._get_ip()}", True, (255, 255, 255))
                screen.blit(ip_label, (m_rect.x + 40, m_rect.y + 50))

                self._exit_btn_rect = pygame.Rect(m_rect.x + 50, m_rect.y + 130, 300, 70)
                pygame.draw.rect(screen, (180, 40, 40), self._exit_btn_rect)
                pygame.draw.rect(screen, (255, 100, 100), self._exit_btn_rect, 2)

                txt = font.render("Exit", True, (255, 255, 255))
                screen.blit(txt, txt.get_rect(center=self._exit_btn_rect.center))

            pygame.display.flip()

        log.info("Shutting down...")
        pygame.quit()
        os._exit(0)

    def start(self):
        self._run()

    def stop(self):
        self.state.stop()

    def update_emote(self, name, smooth=True):
        self.state.request_emote_change(name, smooth)

    # kept for backwards compatibility (does nothing)
    def update_eyes_position(self, x, y):
        pass


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