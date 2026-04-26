#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Face Display using a single face texture (PNG).
Texture touches left/right screen edges.
Blinking = vertical squash of the whole face.
"""
import math
import os
import sys
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

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)


class FaceState:
    def __init__(self):
        self.lock = threading.Lock()
        self._emote = "normal"
        self._blink_state = 0.0
        self._menu_visible = False
        self._running = True
        self._last_blink_time = 0.0
        self._min_blink_interval = 1.5
        self._pending_emote = None
        self._need_emote_change = False

    def toggle_menu(self):
        with self.lock:
            self._menu_visible = not self._menu_visible
            return self._menu_visible

    def get_emote(self):
        with self.lock:
            return self._emote

    def update_logic(self):
        with self.lock:
            was_blinking = self._blink_state > 0.0
            self._blink_state *= 0.75
            if self._blink_state < 0.01:
                self._blink_state = 0.0
            if was_blinking and self._blink_state == 0.0:
                self._apply_pending_emote()

    def trigger_blink(self):
        with self.lock:
            now = time.time()
            if now - self._last_blink_time >= self._min_blink_interval:
                self._blink_state = 1.0
                self._last_blink_time = now
                return True
        return False

    def request_emote_change(self, new_emote):
        with self.lock:
            if new_emote == self._emote:
                return
            self._pending_emote = new_emote
            self._need_emote_change = True
            if self._blink_state == 0.0:
                self.trigger_blink()

    def _apply_pending_emote(self):
        if self._need_emote_change and self._pending_emote is not None:
            self._emote = self._pending_emote
            self._need_emote_change = False
            self._pending_emote = None
            return True
        return False

    def is_running(self):
        with self.lock:
            return self._running

    def stop(self):
        with self.lock:
            self._running = False


class RobotFace:
    def __init__(self):
        self.state = FaceState()
        self._exit_btn_rect = pygame.Rect(0, 0, 0, 0)
        self._textures = {}          # cache: emote_name -> pygame.Surface

    def _load_texture(self, emote_name):
        """Load PNG from emotions/ (relative to script)"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(script_dir, "emotions")
        os.makedirs(base_path, exist_ok=True)

        filename = os.path.join(base_path, f"{emote_name}.png")
        if not os.path.isfile(filename):
            filename = os.path.join(base_path, "default.png")
            if not os.path.isfile(filename):
                log.error(f"No texture for '{emote_name}' and no default.png in {base_path}")
                # fallback: grey rectangle with text
                surf = pygame.Surface((400, 400), pygame.SRCALPHA)
                surf.fill((100, 100, 100, 255))
                try:
                    font = pygame.font.SysFont("Arial", 24)
                    txt = font.render(emote_name, True, (255,255,255))
                    surf.blit(txt, (200 - txt.get_width()//2, 200 - txt.get_height()//2))
                except:
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
        except:
            return "127.0.0.1"

    def _run(self):
        pygame.init()
        pygame.display.init()

        info = pygame.display.Info()
        sw, sh = info.current_w, info.current_h

        screen = pygame.display.set_mode((sw, sh), pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE)
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(True)
        clock = pygame.time.Clock()

        # Preload normal and default textures (display is ready)
        self._get_texture_for_emote("normal")
        self._get_texture_for_emote("default")

        font = pygame.font.SysFont("Arial", 28, bold=True)

        while self.state.is_running():
            pygame.event.pump()

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state.stop()
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                        self.state.stop()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if self.state._menu_visible:
                        if self._exit_btn_rect.collidepoint(mx, my):
                            log.info("Exit via menu")
                            self.state.stop()
                        else:
                            self.state.toggle_menu()
                    else:
                        if my > sh * 0.80:
                            self.state.toggle_menu()

            # Animation update
            self.state.update_logic()

            # Random blink
            if random.random() < 0.005:
                self.state.trigger_blink()

            # Clear screen
            screen.fill((0, 0, 0))

            # Get current face texture
            emote = self.state.get_emote()
            tex = self._get_texture_for_emote(emote)

            # Blink factor: vertical scale (1 = fully open, 0 = fully closed)
            blink_scale = 1.0 - self.state._blink_state
            if blink_scale < 0.05:
                blink_scale = 0.05

            # Scale texture to fill screen width (touch left/right edges)
            target_width = sw
            target_height = int(tex.get_height() * (target_width / tex.get_width()))
            # Apply blink: reduce height
            target_height = int(target_height * blink_scale)

            # If height exceeds screen height, limit (but keep aspect)
            if target_height > sh:
                target_height = sh
                # Optionally recalc width to keep aspect, but user wants full width touch - ignore

            scaled_tex = pygame.transform.scale(tex, (target_width, target_height))

            # Center vertically
            y_offset = (sh - target_height) // 2
            screen.blit(scaled_tex, (0, y_offset))

            # Draw menu if visible
            if self.state._menu_visible:
                overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 220))
                screen.blit(overlay, (0, 0))

                m_w, m_h = 400, 250
                m_rect = pygame.Rect((sw - m_w)//2, (sh - m_h)//2, m_w, m_h)
                pygame.draw.rect(screen, (25,25,25), m_rect, border_radius=20)
                pygame.draw.rect(screen, (200,200,200), m_rect, 2, border_radius=20)

                ip_label = font.render(f"IP: {self._get_ip()}", True, (255,255,255))
                screen.blit(ip_label, (m_rect.x+40, m_rect.y+50))

                self._exit_btn_rect = pygame.Rect(m_rect.x+50, m_rect.y+130, 300, 70)
                pygame.draw.rect(screen, (180,40,40), self._exit_btn_rect, border_radius=15)
                pygame.draw.rect(screen, (255,100,100), self._exit_btn_rect, 2, border_radius=15)

                txt = font.render("Exit", True, (255,255,255))
                screen.blit(txt, txt.get_rect(center=self._exit_btn_rect.center))

            pygame.display.flip()
            clock.tick(60)

        log.info("Shutting down...")
        pygame.quit()
        os._exit(0)

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.state.stop()

    def update_emote(self, name):
        self.state.request_emote_change(name)

    # gaze position not used with single face, but keep for compatibility
    def update_eyes_position(self, x, y):
        pass


# Backward compatibility wrappers
class EyeAPI:
    def __init__(self, robot_face):
        self._face = robot_face
    def update_emote(self, name):
        self._face.update_emote(name)
    def update_eyes_position(self, x, y):
        self._face.update_eyes_position(x, y)


class EyeDisplay:
    def __init__(self):
        self._face = RobotFace()
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
    log.info("Place your face PNGs in 'emotions/' folder (normal.png, default.png, ...)")
    face = RobotFace()
    face.start()
    try:
        while face.state.is_running():
            time.sleep(0.5)
    except KeyboardInterrupt:
        face.stop()


if __name__ == "__main__":
    main()