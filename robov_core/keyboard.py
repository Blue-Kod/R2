#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pygame on-screen keyboard with Russian/English support.
"""

import pygame

# Keyboard layouts
LAYOUTS = {
    "en": {
        "name": "EN",
        "rows": [
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "="],
            ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "[", "]"],
            ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";", "'", "\\"],
            ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/"],
        ],
        "shift_map": {
            "1": "!", "2": "@", "3": "#", "4": "$", "5": "%",
            "6": "^", "7": "&", "8": "*", "9": "(", "0": ")",
            "-": "_", "=": "+", "[": "{", "]": "}", ";": ":",
            "'": "\"", "\\": "|", ",": "<", ".": ">", "/": "?",
        },
    },
    "ru": {
        "name": "RU",
        "rows": [
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "="],
            ["й", "ц", "у", "к", "е", "н", "г", "ш", "щ", "з", "х", "ъ"],
            ["ф", "ы", "в", "а", "п", "р", "о", "л", "д", "ж", "э"],
            ["я", "ч", "с", "м", "и", "т", "ь", "б", "ю", "."],
        ],
        "shift_map": {
            "1": "!", "2": "\"", "3": "№", "4": ";", "5": "%",
            "6": ":", "7": "?", "8": "*", "9": "(", "0": ")",
            "-": "_", "=": "+",
        },
    },
}


class PygameKeyboard:
    def __init__(self, screen_width, screen_height):
        self.screen_w = screen_width
        self.screen_h = screen_height
        self.visible = False
        self.layout = "en"
        self.shift = False
        self.input_callback = None  # Called with entered character

        self._btn_color = (60, 60, 80)
        self._btn_active_color = (80, 100, 140)
        self._btn_special_color = (40, 60, 100)
        self._btn_border_color = (100, 120, 160)
        self._text_color = (255, 255, 255)

        self._keys = []  # List of (Rect, char_or_action)
        self._build_layout()

    def set_callback(self, callback):
        self.input_callback = callback

    def _build_layout(self):
        self._keys = []
        font = pygame.font.SysFont("Arial", 20, bold=True)
        layout_data = LAYOUTS[self.layout]

        kb_x = 10
        kb_y = self.screen_h - 260
        key_w = (self.screen_w - 20) // 12
        key_h = 38
        gap = 3

        # Row 0: numbers
        y = kb_y
        row = layout_data["rows"][0]
        for i, ch in enumerate(row):
            x = kb_x + i * (key_w + gap)
            rect = pygame.Rect(x, y, key_w, key_h)
            display = ch
            if self.shift and ch in layout_data.get("shift_map", {}):
                display = layout_data["shift_map"][ch]
            self._keys.append((rect, ch, display))

        # Row 1: top letter row
        y += key_h + gap
        row = layout_data["rows"][1]
        x_offset = key_w // 2
        for i, ch in enumerate(row):
            x = kb_x + x_offset + i * (key_w + gap)
            rect = pygame.Rect(x, y, key_w, key_h)
            display = ch.upper() if self.shift else ch
            self._keys.append((rect, ch, display))

        # Row 2: home row
        y += key_h + gap
        row = layout_data["rows"][2]
        x_offset = key_w
        for i, ch in enumerate(row):
            x = kb_x + x_offset + i * (key_w + gap)
            rect = pygame.Rect(x, y, key_w, key_h)
            display = ch.upper() if self.shift else ch
            self._keys.append((rect, ch, display))

        # Row 3: bottom row
        y += key_h + gap
        row = layout_data["rows"][3]
        x_offset = key_w * 2
        for i, ch in enumerate(row):
            x = kb_x + x_offset + i * (key_w + gap)
            rect = pygame.Rect(x, y, key_w, key_h)
            display = ch.upper() if self.shift else ch
            self._keys.append((rect, ch, display))

        # Special keys
        y = kb_y
        special_h = key_h * 3 + gap * 2

        # Shift key (left side)
        shift_w = key_w * 1
        shift_x = kb_x
        shift_y = kb_y + (key_h + gap) * 3
        self._keys.append((
            pygame.Rect(shift_x, shift_y, key_w, key_h),
            "shift", "⇧" if not self.shift else "⇧↑"
        ))

        # Space bar
        space_w = key_w * 5
        space_x = kb_x + key_w * 2 + gap * 2
        space_y = kb_y + (key_h + gap) * 3
        self._keys.append((
            pygame.Rect(space_x, space_y, space_w, key_h),
            "space", " "
        ))

        # Backspace
        bs_w = key_w * 2
        bs_x = self.screen_w - kb_x - bs_w
        bs_y = kb_y
        self._keys.append((
            pygame.Rect(bs_x, bs_y, bs_w, key_h),
            "backspace", "⌫"
        ))

        # Enter
        enter_w = key_w * 2
        enter_x = self.screen_w - kb_x - enter_w
        enter_y = kb_y + key_h + gap
        self._keys.append((
            pygame.Rect(enter_x, enter_y, enter_w, key_h),
            "enter", "↵"
        ))

        # Layout toggle
        layout_w = key_w * 1
        layout_x = kb_x
        layout_y = kb_y
        layout_label = LAYOUTS[self.layout]["name"]
        self._keys.append((
            pygame.Rect(layout_x, layout_y, key_w, key_h),
            "layout", layout_label
        ))

    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self._build_layout()

    def show(self):
        self.visible = True
        self._build_layout()

    def hide(self):
        self.visible = False

    def handle_event(self, event):
        if not self.visible:
            return None

        if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.FINGERDOWN:
            mx, my = event.pos
            sw, sh = self.screen_w, self.screen_h
            if event.type == pygame.FINGERDOWN:
                mx = int(mx * sw)
                my = int(my * sh)

            for rect, char, display in self._keys:
                if rect.collidepoint(mx, my):
                    return self._on_key_press(char)
        return None

    def _on_key_press(self, char):
        if char == "shift":
            self.shift = not self.shift
            self._build_layout()
            return None
        elif char == "layout":
            self.layout = "ru" if self.layout == "en" else "en"
            self.shift = False
            self._build_layout()
            return None
        elif char == "backspace":
            return ("backspace", "")
        elif char == "enter":
            return ("enter", "")
        elif char == "space":
            return ("char", " ")
        else:
            layout_data = LAYOUTS[self.layout]
            if self.shift and char in layout_data.get("shift_map", {}):
                ch = layout_data["shift_map"][char]
            elif self.shift:
                ch = char.upper()
            else:
                ch = char
            self.shift = False
            self._build_layout()
            return ("char", ch)

    def draw(self, screen):
        if not self.visible:
            return

        # Semi-transparent background
        bg = pygame.Surface((self.screen_w, 260), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 200))
        screen.blit(bg, (0, self.screen_h - 260))

        font = pygame.font.SysFont("Arial", 20, bold=True)

        for rect, char, display in self._keys:
            if char == "space":
                color = self._btn_color
            elif char in ("shift", "layout", "backspace", "enter"):
                color = self._btn_special_color
            else:
                color = self._btn_color

            if char == "shift" and self.shift:
                color = self._btn_active_color

            pygame.draw.rect(screen, color, rect, border_radius=4)
            pygame.draw.rect(screen, self._btn_border_color, rect, 1, border_radius=4)

            txt = font.render(display, True, self._text_color)
            txt_rect = txt.get_rect(center=rect.center)
            screen.blit(txt, txt_rect)
