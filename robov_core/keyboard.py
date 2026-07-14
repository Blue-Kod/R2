#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pygame on-screen keyboard with Russian/English support.
Designed for 600-1024px wide touchscreen displays.
"""

import pygame

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
            "1": "!", "2": "\"", "3": "N", "4": ";", "5": "%",
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
        self.input_callback = None

        self._btn_color = (60, 60, 80)
        self._btn_active_color = (80, 100, 140)
        self._btn_special_color = (40, 60, 100)
        self._btn_close_color = (140, 40, 40)
        self._btn_border_color = (100, 120, 160)
        self._text_color = (255, 255, 255)

        self._keys = []
        self._build_layout()

    def set_callback(self, callback):
        self.input_callback = callback

    def _get_font(self):
        for name in ("DejaVuSans", "DejaVu Sans", "Liberation Sans", "Arial", "FreeSans"):
            try:
                f = pygame.font.SysFont(name, 20, bold=True)
                if f:
                    return f
            except Exception:
                continue
        return pygame.font.Font(None, 24)

    def _build_layout(self):
        self._keys = []
        font = self._get_font()
        layout_data = LAYOUTS[self.layout]

        kb_margin = 6
        gap = 3
        key_h = 38

        max_keys = max(len(row) for row in layout_data["rows"])
        key_w = (self.screen_w - kb_margin * 2 - (max_keys - 1) * gap) // max_keys

        kb_y = self.screen_h - 260

        def make_row(y, row_keys, x_offset_cols=0):
            x_start = kb_margin + x_offset_cols * (key_w + gap)
            for i, ch in enumerate(row_keys):
                x = x_start + i * (key_w + gap)
                if x + key_w > self.screen_w - kb_margin:
                    break
                rect = pygame.Rect(x, y, key_w, key_h)
                display = ch
                if self.shift and ch in layout_data.get("shift_map", {}):
                    display = layout_data["shift_map"][ch]
                self._keys.append((rect, ch, display))

        # Row 0: numbers (left-aligned)
        y = kb_y
        make_row(y, layout_data["rows"][0])

        # Backspace: rightmost 2 columns of row 0
        bs_w = key_w * 2 + gap
        bs_x = self.screen_w - kb_margin - bs_w
        self._keys.append((
            pygame.Rect(bs_x, y, bs_w, key_h),
            "backspace", "BS"
        ))

        # Row 1: top letters (offset by 0.5 col)
        y += key_h + gap
        x_offset_1 = 0.5
        make_row(y, layout_data["rows"][1], x_offset_1)

        # Enter: rightmost 2 columns of row 1
        ent_w = key_w * 2 + gap
        ent_x = self.screen_w - kb_margin - ent_w
        self._keys.append((
            pygame.Rect(ent_x, y, ent_w, key_h),
            "enter", "Ent"
        ))

        # Row 2: home row (offset by 1 col)
        y += key_h + gap
        make_row(y, layout_data["rows"][2], 1)

        # Row 3: bottom row
        y += key_h + gap
        row3 = layout_data["rows"][3]

        # Shift at leftmost position
        shift_x = kb_margin
        self._keys.append((
            pygame.Rect(shift_x, y, key_w, key_h),
            "shift", "Sft" if not self.shift else "SFT"
        ))

        # Letter keys in the middle
        x_start_3 = kb_margin + (key_w + gap)
        for i, ch in enumerate(row3):
            x = x_start_3 + i * (key_w + gap)
            if x + key_w > self.screen_w - kb_margin - (key_w + gap) * 5:
                break
            rect = pygame.Rect(x, y, key_w, key_h)
            display = ch.upper() if self.shift else ch
            self._keys.append((rect, ch, display))

        # Space bar (5 cols wide)
        space_w = key_w * 5 + gap * 4
        space_x = self.screen_w - kb_margin - space_w - (key_w + gap) * 2
        space_x = max(space_x, kb_margin + (key_w + gap))
        self._keys.append((
            pygame.Rect(space_x, y, space_w, key_h),
            "space", " "
        ))

        # Close button: bottom-right corner
        close_w = key_w * 2 + gap
        close_x = self.screen_w - kb_margin - close_w
        self._keys.append((
            pygame.Rect(close_x, y, close_w, key_h),
            "close", "X"
        ))

        # Layout toggle: top-left corner (small)
        self._keys.append((
            pygame.Rect(kb_margin, kb_y - key_h - gap, key_w * 2 + gap, key_h),
            "layout", LAYOUTS[self.layout]["name"]
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
        elif char == "close":
            return ("close", "")
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

        bg_h = 260
        bg = pygame.Surface((self.screen_w, bg_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 200))
        screen.blit(bg, (0, self.screen_h - bg_h))

        font = self._get_font()

        for rect, char, display in self._keys:
            if char == "space":
                color = self._btn_color
            elif char == "close":
                color = self._btn_close_color
            elif char in ("shift", "layout", "backspace", "enter"):
                color = self._btn_special_color
            else:
                color = self._btn_color

            if char == "shift" and self.shift:
                color = self._btn_active_color

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, self._btn_border_color, rect, 1)

            txt = font.render(display, True, self._text_color)
            txt_rect = txt.get_rect(center=rect.center)
            screen.blit(txt, txt_rect)
