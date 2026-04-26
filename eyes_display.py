#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Eye Display using Pygame.
Native rendering without HTML/CSS - all drawn with Pygame primitives.
"""

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# Emotion configurations
EMOTION_CONFIG = {
    "normal": {
        "eye_scale_y": 1.5,
        "eye_color": (255, 255, 255),
        "eye_border": None,
        "brow_opacity": 0.0,
        "brow_rotation": 0,
        "brow_y_offset": 0,
        "eye_rotation": 0,
    },
    "sad": {
        "eye_scale_y": 1.5,
        "eye_color": (255, 255, 255),
        "eye_border": None,
        "brow_opacity": 1.0,
        "brow_rotation": -8,
        "brow_y_offset": -8,
        "eye_rotation": -6,
        "eye_right_rotation": 6,
    },
    "excited": {
        "eye_scale_y": 1.5,
        "eye_color": (255, 255, 255),
        "eye_border": None,
        "brow_opacity": 0.0,
        "brow_rotation": 0,
        "brow_y_offset": 0,
        "eye_rotation": 0,
        "eye_scale": 1.03,
        "glow_intensity": 0.32,
    },
    "spooked": {
        "eye_scale_y": 1.2,
        "eye_color": (0, 0, 0, 0),
        "eye_border": (255, 255, 255),
        "eye_border_width": 4,
        "brow_opacity": 0.0,
        "brow_rotation": 0,
        "brow_y_offset": 0,
        "eye_rotation": 0,
    },
    "unamused": {
        "eye_scale_y": 0.62,
        "eye_color": (255, 255, 255),
        "eye_border": None,
        "brow_opacity": 1.0,
        "brow_rotation": 0,
        "brow_y_offset": 0,
        "eye_rotation": 0,
    },
    "worried": {
        "eye_scale_y": 0.75,
        "eye_color": (255, 255, 255),
        "eye_border": None,
        "brow_opacity": 1.0,
        "brow_rotation": -14,
        "brow_y_offset": 0,
        "eye_rotation": -7,
        "eye_right_rotation": 7,
    },
    "woozy": {
        "eye_scale_y": 0.8,
        "eye_color": (255, 255, 255),
        "eye_border": None,
        "brow_opacity": 0.0,
        "brow_rotation": 0,
        "brow_y_offset": 0,
        "eye_rotation": -10,
        "eye_right_rotation": 10,
    },
    "angry": {
        "eye_scale_y": 0.72,
        "eye_color": (255, 255, 255),
        "eye_border": None,
        "brow_opacity": 1.0,
        "brow_rotation": 18,
        "brow_y_offset": -10,
        "eye_right_rotation": -18,
        "eye_rotation": 0,
    },
    "wince": {
        "eye_scale_y": 0.4,
        "eye_color": (255, 255, 255),
        "eye_border": None,
        "brow_opacity": 1.0,
        "brow_rotation": 10,
        "brow_y_offset": 0,
        "eye_right_rotation": -10,
        "eye_rotation": 0,
    },
}

SUPPORTED_EMOTES = set(EMOTION_CONFIG.keys())


class EyeState:
    """Thread-safe container for eye state."""
    def __init__(self):
        self.lock = threading.Lock()
        self._emote = "normal"
        self._target_x = 0.0
        self._target_y = 0.0
        self._current_x = 0.0
        self._current_y = 0.0
        self._blink_state = 0.0
        self._blink_target = 0.0
        self._menu_visible = False
        self._running = True

    def set_emote(self, name):
        with self.lock:
            if name in SUPPORTED_EMOTES:
                self._emote = name

    def get_emote(self):
        with self.lock:
            return self._emote

    def set_position(self, x, y):
        with self.lock:
            self._target_x = max(-1.0, min(1.0, float(x)))
            self._target_y = max(-1.0, min(1.0, float(y)))

    def get_position(self):
        with self.lock:
            return self._current_x, self._current_y

    def update_interpolation(self, smooth_factor=0.08):
        with self.lock:
            self._current_x += (self._target_x - self._current_x) * smooth_factor
            self._current_y += (self._target_y - self._current_y) * smooth_factor

    def trigger_blink(self):
        with self.lock:
            self._blink_target = 1.0

    def update_blink(self):
        with self.lock:
            if self._blink_target > 0:
                self._blink_state = self._blink_target
                self._blink_target = 0.0
            else:
                self._blink_state *= 0.85
                if self._blink_state < 0.01:
                    self._blink_state = 0.0

    def get_blink_scale(self):
        with self.lock:
            return max(0.08, 1.0 - self._blink_state * 0.92)

    def toggle_menu(self):
        with self.lock:
            self._menu_visible = not self._menu_visible
            return self._menu_visible

    def is_menu_visible(self):
        with self.lock:
            return self._menu_visible

    def set_running(self, value):
        with self.lock:
            self._running = value

    def is_running(self):
        with self.lock:
            return self._running


class RobotEyes:
    """Main robot eyes display using Pygame."""
    
    def __init__(self):
        self.state = EyeState()
        self.thread = None
        self._next_blink = time.time() + random.uniform(2.5, 4.5)
        self._menu_card_rect = None
        self._exit_btn_rect = None

    def start(self):
        """Start the display in a separate thread."""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        log.info("[RobotEyes] Display thread started")

    def _get_ip(self):
        """Get local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            ip = "127.0.0.1"
        return ip

    def _run(self):
        """Main Pygame loop - runs in separate thread."""
        pygame.init()
        pygame.mouse.set_visible(False)

        info = pygame.display.Info()
        screen_width, screen_height = info.current_w, info.current_h

        screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
        pygame.display.set_caption("R2 Eyes")
        clock = pygame.time.Clock()

        log.info("[RobotEyes] Pygame initialized: %dx%d", screen_width, screen_height)

        # Eye dimensions
        eye_size = int(min(screen_width, screen_height) * 0.15)
        eye_spacing = eye_size * 2.5

        # Fonts
        font = pygame.font.SysFont("Arial", 18)
        menu_font = pygame.font.SysFont("Arial", 24)

        running = True
        while running and self.state.is_running():
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    if self.state.is_menu_visible():
                        # Check if exit button clicked
                        if self._exit_btn_rect and self._exit_btn_rect.collidepoint(mouse_x, mouse_y):
                            log.info("[RobotEyes] Exit button clicked")
                            running = False
                            self.state.set_running(False)
                        # Check if clicked outside menu card to close
                        elif self._menu_card_rect and not self._menu_card_rect.collidepoint(mouse_x, mouse_y):
                            self.state.toggle_menu()
                    else:
                        # Bottom zone click (15% from bottom)
                        if mouse_y > screen_height * 0.85:
                            self.state.toggle_menu()

            # Update state
            self.state.update_interpolation()
            self.state.update_blink()

            # Procedural blink
            now = time.time()
            if now >= self._next_blink:
                self.state.trigger_blink()
                self._next_blink = now + random.uniform(2.5, 4.5)
                if random.random() > 0.82:
                    self._next_blink += 0.14  # Double blink

            # Clear screen
            screen.fill((0, 0, 0))

            # Get current state
            emote = self.state.get_emote()
            current_x, current_y = self.state.get_position()
            blink_scale = self.state.get_blink_scale()
            config = EMOTION_CONFIG.get(emote, EMOTION_CONFIG["normal"])

            # Calculate eye positions with movement
            offset_x = current_x * (screen_width * 0.10)
            offset_y = current_y * (screen_height * 0.07)

            face_center_x = screen_width // 2
            face_center_y = screen_height // 2

            # Draw eyes
            self._draw_eyes(screen, face_center_x, face_center_y, offset_x, offset_y,
                          eye_size, eye_spacing, blink_scale, config)

            # Draw brows if visible
            if config.get("brow_opacity", 0) > 0:
                self._draw_brows(screen, face_center_x, face_center_y, offset_x, offset_y,
                               eye_size, eye_spacing, config)

            # Draw menu if visible
            if self.state.is_menu_visible():
                self._draw_menu(screen, screen_width, screen_height, menu_font)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        log.info("[RobotEyes] Pygame loop ended")

    def _draw_eyes(self, screen, center_x, center_y, offset_x, offset_y,
                   eye_size, spacing, blink_scale, config):
        """Draw the robot eyes."""
        eye_scale_y = config.get("eye_scale_y", 1.5) * blink_scale
        eye_scale = config.get("eye_scale", 1.0)
        eye_color = config.get("eye_color", (255, 255, 255))
        border_color = config.get("eye_border")
        border_width = int(config.get("eye_border_width", 0))
        glow = config.get("glow_intensity", 0)

        # Left eye
        left_x = center_x - spacing - (eye_size * eye_scale) / 2 + offset_x
        left_y = center_y - (eye_size * eye_scale_y) / 2 + offset_y
        left_rect = pygame.Rect(left_x, left_y, eye_size * eye_scale, eye_size * eye_scale_y)

        # Right eye
        right_x = center_x + spacing - (eye_size * eye_scale) / 2 + offset_x
        right_y = center_y - (eye_size * eye_scale_y) / 2 + offset_y
        right_rect = pygame.Rect(right_x, right_y, eye_size * eye_scale, eye_size * eye_scale_y)

        # Draw glow effect for excited emotion
        if glow > 0:
            glow_surf = pygame.Surface((eye_size * 3, eye_size * 3), pygame.SRCALPHA)
            glow_alpha = int(255 * glow)
            glow_color = (eye_color[0], eye_color[1], eye_color[2], glow_alpha)
            pygame.draw.ellipse(glow_surf, glow_color, glow_surf.get_rect())
            screen.blit(glow_surf, (int(left_rect.centerx - eye_size * 1.5),
                                   int(left_rect.centery - eye_size * 1.5)))

        # Draw eyes (left and right)
        for rect, rotation in [(left_rect, config.get("eye_rotation", 0)),
                               (right_rect, config.get("eye_right_rotation",
                                                       config.get("eye_rotation", 0)))]:
            if rotation != 0:
                # Create surface for rotated eye
                surf = pygame.Surface((int(rect.width), int(rect.height)), pygame.SRCALPHA)
                pygame.draw.ellipse(surf, eye_color, (0, 0, int(rect.width), int(rect.height)))
                if border_color:
                    pygame.draw.ellipse(surf, border_color, (0, 0, int(rect.width), int(rect.height)), border_width)
                rotated = pygame.transform.rotate(surf, rotation)
                screen.blit(rotated, rotated.get_rect(center=rect.center))
            else:
                pygame.draw.ellipse(screen, eye_color, rect)
                if border_color:
                    pygame.draw.ellipse(screen, border_color, rect, border_width)

    def _draw_brows(self, screen, center_x, center_y, offset_x, offset_y,
                    eye_size, spacing, config):
        """Draw eyebrows."""
        brow_width = int(eye_size * 0.97)
        brow_height = 12
        brow_opacity = int(255 * config.get("brow_opacity", 0))
        brow_color = (255, 255, 255, brow_opacity)

        for side, rotation in [("left", config.get("brow_rotation", 0)),
                               ("right", config.get("brow_right_rotation",
                                                    config.get("brow_rotation", 0)))]:
            x_offset = spacing + eye_size / 2
            if side == "left":
                brow_x = center_x - x_offset - brow_width / 2 + offset_x
            else:
                brow_x = center_x + x_offset - brow_width / 2 + offset_x

            brow_y = center_y - eye_size * 0.75 + config.get("brow_y_offset", 0) + offset_y

            surf = pygame.Surface((brow_width, brow_height), pygame.SRCALPHA)
            pygame.draw.rect(surf, brow_color, (0, 0, brow_width, brow_height),
                           border_radius=999)
            if rotation != 0:
                rotated = pygame.transform.rotate(surf, rotation)
                screen.blit(rotated, rotated.get_rect(center=(brow_x + brow_width/2,
                                                               brow_y + brow_height/2)))
            else:
                screen.blit(surf, (brow_x, brow_y))

    def _draw_menu(self, screen, width, height, font):
        """Draw system menu overlay."""
        # Overlay background
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 183))  # 72% opacity
        screen.blit(overlay, (0, 0))

        # Menu card
        card_width = min(int(width * 0.88), 380)
        card_height = 140
        card_x = (width - card_width) // 2
        card_y = (height - card_height) // 2

        # Store card rect for click detection
        self._menu_card_rect = pygame.Rect(card_x, card_y, card_width, card_height)

        pygame.draw.rect(screen, (16, 16, 16), self._menu_card_rect, border_radius=14)
        pygame.draw.rect(screen, (59, 59, 59), self._menu_card_rect, width=1, border_radius=14)

        # Title
        title = font.render("System", True, (189, 189, 189))
        screen.blit(title, (card_x + 16, card_y + 16))

        # IP address
        ip_text = self._get_ip()
        ip_surface = font.render("IP: " + ip_text, True, (255, 255, 255))
        screen.blit(ip_surface, (card_x + 16, card_y + 42))

        # Exit button
        btn_width = (card_width - 40) // 2
        btn_height = 40
        btn_x = card_x + 16
        btn_y = card_y + card_height - btn_height - 16

        self._exit_btn_rect = pygame.Rect(btn_x + btn_width + 8, btn_y, btn_width, btn_height)
        pygame.draw.rect(screen, (40, 18, 18), self._exit_btn_rect, border_radius=10)
        pygame.draw.rect(screen, (122, 46, 46), self._exit_btn_rect, width=1, border_radius=10)
        exit_text = font.render("Exit", True, (255, 255, 255))
        screen.blit(exit_text, exit_text.get_rect(center=self._exit_btn_rect.center))

    def update_emote(self, name):
        """Called from any thread to update emote."""
        self.state.set_emote(name)
        log.info("[RobotEyes] Emote updated: %s", name)

    def update_eyes_position(self, x, y):
        """Called from any thread to update eye position."""
        self.state.set_position(x, y)
        log.info("[RobotEyes] Eyes position updated: x=%s, y=%s", x, y)

    def stop(self):
        """Stop the display."""
        self.state.set_running(False)
        if self.thread:
            self.thread.join(timeout=2.0)
        log.info("[RobotEyes] Display stopped")


class EyeAPI:
    """API bridge for backward compatibility with EmoteService."""
    
    def __init__(self, robot_eyes):
        self._robot_eyes = robot_eyes
    
    def update_emote(self, name):
        """Update emote - called by EmoteService."""
        self._robot_eyes.update_emote(name)
    
    def update_eyes_position(self, x, y):
        """Update eyes position - called by EmoteService."""
        self._robot_eyes.update_eyes_position(x, y)


class EyeDisplay:
    """Backward-compatible wrapper for RobotEyes.
    Used by r2_app/high_level.py for integration.
    """
    
    def __init__(self):
        self._robot_eyes = RobotEyes()
        self._api = EyeAPI(self._robot_eyes)
    
    @property
    def api(self):
        """Return API bridge for EmoteService."""
        return self._api
    
    def start(self):
        """Start the display."""
        self._robot_eyes.start()
    
    def stop(self):
        """Stop the display."""
        self._robot_eyes.stop()


def optimize_for_arm():
    """Dummy function for backward compatibility with r2_app/high_level.py."""
    pass


def main():
    """Main entry point for standalone testing."""
    log.info("=" * 40)
    log.info("  R2 Eye Display (Pygame)")
    log.info("=" * 40)

    eyes = RobotEyes()
    eyes.start()

    # Keep main thread alive for testing
    try:
        while eyes.state.is_running():
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("[Main] Keyboard interrupt received")
        eyes.stop()


if __name__ == "__main__":
    main()