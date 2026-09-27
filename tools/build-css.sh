#!/usr/bin/env bash
# Build static/css/r2.css with the Tailwind v4 standalone CLI.
#
# Dev machine only: the robot has no Node/Tailwind, so the compiled r2.css is
# committed and shipped. Run this after changing input.css or any class names
# used in templates/ and static/js/.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=tools/tailwindcss
if [ ! -x "$BIN" ]; then
    echo "Downloading Tailwind standalone CLI (linux-x64)..."
    curl -fsSL -o "$BIN" \
        https://github.com/tailwindlabs/tailwindcss/releases/latest/download/tailwindcss-linux-x64
    chmod +x "$BIN"
fi

"$BIN" -i static/css/input.css -o static/css/r2.css --minify
echo "Built static/css/r2.css"
