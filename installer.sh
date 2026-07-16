#!/usr/bin/env bash
# R2 Robot - Clean System Installer
# Based on: https://github.com/Blue-Kod/R2/blob/main/docs/installation.md
#
# Usage:
#   wget -qO- https://raw.githubusercontent.com/Blue-Kod/R2/main/installer.sh | sudo bash
#
# Or download and run:
#   wget https://raw.githubusercontent.com/Blue-Kod/R2/main/installer.sh
#   sudo bash installer.sh

set -euo pipefail

REPO_URL="https://github.com/Blue-Kod/R2.git"
REPO_DIR="/opt/R2"
APT_PACKAGES=(
    git
    python3-pip
    python3-pygame
    python3-dev
    libsdl2-2.0-0
    libsdl2-image-2.0-0
    libsdl2-ttf-2.0-0
    libportaudio2
    libportaudiocpp0
    portaudio19-dev
    unclutter-xfixes
    libopencv-dev
    python3-opencv
    i2c-tools
    build-essential
    cmake
    pkg-config
    onboard
)

DISABLE_SERVICES=(
    bluetooth.service
    containerd.service
    lircd.service
    cups.service
    cups-browsed.service
    ModemManager.service
    avahi-daemon.service
    triggerhappy.service
    avahi-daemon.socket
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

check_root() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "[!] This script must be run as root. Use: sudo bash $0"
        exit 1
    fi
}

check_internet() {
    log "Checking internet connection..."
    if ! ping -c 1 -W 3 8.8.8.8 >/dev/null 2>&1; then
        log "[!] No internet connection. Please connect to the internet and try again."
        exit 1
    fi
    log "Internet OK."
}

disable_unnecessary_services() {
    log "Disabling unnecessary services..."
    for svc in "${DISABLE_SERVICES[@]}"; do
        if systemctl list-unit-files "$svc" >/dev/null 2>&1; then
            systemctl disable "$svc" 2>/dev/null || true
            systemctl stop "$svc" 2>/dev/null || true
            log "[+] Disabled: $svc"
        else
            log "[-] $svc not found, skipping."
        fi
    done
}

setup_onboard_autostart() {
    log "Setting up Onboard on-screen keyboard autostart..."
    local autostart_dir="/etc/xdg/autostart"
    mkdir -p "$autostart_dir"

    cat > "$autostart_dir/onboard-autostart.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Onboard
Exec=onboard --no-delay
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Phase=Applications
EOF

    log "[+] Onboard autostart installed: $autostart_dir/onboard-autostart.desktop"
}

install_apt_packages() {
    log "Updating package lists..."
    apt-get update -y

    log "Installing system packages..."
    for pkg in "${APT_PACKAGES[@]}"; do
        if dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
            log "[+] $pkg already installed."
        else
            log "[+] Installing $pkg..."
            DEBIAN_FRONTEND=noninteractive apt-get install -y -f "$pkg"
        fi
    done
    log "System packages installed."
}

clone_repo() {
    if [ -d "$REPO_DIR" ]; then
        log "Repository already exists at $REPO_DIR, updating..."
        cd "$REPO_DIR"
        git pull --ff-only || true
    else
        log "Cloning R2 repository..."
        git clone "$REPO_URL" "$REPO_DIR"
    fi
}

install_pip_packages() {
    log "Installing Python dependencies..."
    pip3 install --break-system-packages -r "$REPO_DIR/requirements.txt"
    log "Python packages installed."
}

main() {
    check_root
    check_internet

    log "=== R2 Robot Installer ==="
    install_apt_packages
    disable_unnecessary_services
    setup_onboard_autostart
    clone_repo
    install_pip_packages

    log ""
    log "============================================="
    log "  Installation complete!"
    log "============================================="
    log ""
    log "The following steps must be done manually:"
    log ""
    log "1. SCREEN ORIENTATION"
    log "   After first boot, set display to portrait (Right)"
    log "   using your desktop display settings."
    log ""
    log "2. I2C & TIMEZONE"
    log "   Run: sudo orangepi-config"
    log "   Enable all I2C interfaces and set your timezone."
    log ""
    log "3. REBOOT"
    log "   Run: sudo reboot"
    log ""
    log "4. TOUCHSCREEN ROTATION (after reboot)"
    log "   sudo apt install xinput"
    log "   sudo mkdir -p /etc/X11/xorg.conf.d"
    log "   sudo cp /usr/share/X11/xorg.conf.d/40-libinput.conf \\"
    log "           /etc/X11/xorg.conf.d/"
    log "   Edit /etc/X11/xorg.conf.d/40-libinput.conf:"
    log "   Find section with Identifier \"libinput touchscreen catchall\""
    log "   Add inside it:"
    log "     Option \"CalibrationMatrix\" \"0 -1 1 1 0 0 0 0 1\""
    log "   Then reboot again: sudo reboot"
    log ""
    log "5. START R2"
    log "   sudo python3 $REPO_DIR/launcher.py"
    log ""
    log "============================================="
}

main "$@"
