#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# MNQ Futures Trader — VPS Initial Setup
# ═══════════════════════════════════════════════════════════════════════
# Run this ONCE on a fresh Ubuntu 22.04/24.04 VPS:
#   ssh root@your-vps-ip 'bash -s' < scripts/setup-vps.sh
#
# What it does:
#   1. Creates a 'trader' system user (no sudo, locked down)
#   2. Installs Python 3.11, git, tmux
#   3. Clones the repo to /opt/futures-trader
#   4. Creates virtualenv and installs dependencies
#   5. Installs systemd service for auto-start/restart
#   6. Sets up log rotation
#   7. Opens only necessary outbound ports
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

DEPLOY_PATH="/opt/futures-trader"
SERVICE_USER="trader"
REPO_URL="${REPO_URL:-}"  # Pass as env var or set after clone

echo "══════════════════════════════════════════"
echo "  MNQ Futures Trader — VPS Setup"
echo "══════════════════════════════════════════"

# ── 1. System packages ───────────────────────────────────────────────
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3.11 python3.11-venv python3.11-dev \
    git tmux curl jq \
    > /dev/null 2>&1

# ── 2. Create service user ──────────────────────────────────────────
echo "[2/7] Creating service user '${SERVICE_USER}'..."
if ! id -u "${SERVICE_USER}" &>/dev/null; then
    useradd --system --shell /bin/bash --home-dir "/home/${SERVICE_USER}" --create-home "${SERVICE_USER}"
fi

# ── 3. Set up deploy directory ──────────────────────────────────────
echo "[3/7] Setting up ${DEPLOY_PATH}..."
mkdir -p "${DEPLOY_PATH}"
mkdir -p "${DEPLOY_PATH}/data"
mkdir -p "${DEPLOY_PATH}/logs"

# If repo URL provided, clone. Otherwise assume deploy.sh will handle it.
if [[ -n "${REPO_URL}" ]]; then
    if [[ -d "${DEPLOY_PATH}/.git" ]]; then
        echo "  Repo already exists, pulling latest..."
        cd "${DEPLOY_PATH}" && git pull --ff-only
    else
        echo "  Cloning repo..."
        git clone "${REPO_URL}" "${DEPLOY_PATH}"
    fi
fi

chown -R "${SERVICE_USER}:${SERVICE_USER}" "${DEPLOY_PATH}"

# ── 4. Python virtual environment ───────────────────────────────────
echo "[4/7] Setting up Python virtualenv..."
sudo -u "${SERVICE_USER}" python3.11 -m venv "${DEPLOY_PATH}/.venv"
sudo -u "${SERVICE_USER}" "${DEPLOY_PATH}/.venv/bin/pip" install --upgrade pip -q
if [[ -f "${DEPLOY_PATH}/pyproject.toml" ]]; then
    sudo -u "${SERVICE_USER}" "${DEPLOY_PATH}/.venv/bin/pip" install -e "${DEPLOY_PATH}" -q
fi

# ── 5. Install systemd service ──────────────────────────────────────
echo "[5/7] Installing systemd service..."
cp "${DEPLOY_PATH}/scripts/futures-trader.service" /etc/systemd/system/futures-trader.service 2>/dev/null || \
cat > /etc/systemd/system/futures-trader.service << 'UNIT'
[Unit]
Description=MNQ Futures Trading System
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=trader
Group=trader
WorkingDirectory=/opt/futures-trader
Environment=PATH=/opt/futures-trader/.venv/bin:/usr/bin:/bin
EnvironmentFile=/opt/futures-trader/.env
ExecStart=/opt/futures-trader/.venv/bin/python -m src.main --paper
Restart=on-failure
RestartSec=30
StartLimitBurst=5
StartLimitIntervalSec=300

# Graceful shutdown — send SIGTERM, wait 60s for flatten, then SIGKILL
KillSignal=SIGTERM
TimeoutStopSec=60

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/futures-trader/data /opt/futures-trader/logs /tmp
PrivateTmp=yes

# Logging
StandardOutput=append:/opt/futures-trader/logs/trader.log
StandardError=append:/opt/futures-trader/logs/trader.log

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable futures-trader

# ── 6. Log rotation ─────────────────────────────────────────────────
echo "[6/7] Setting up log rotation..."
cat > /etc/logrotate.d/futures-trader << 'LOGROTATE'
/opt/futures-trader/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    su trader trader
}
LOGROTATE

# ── 7. Firewall (allow outbound only) ───────────────────────────────
echo "[7/7] Configuring firewall..."
if command -v ufw &>/dev/null; then
    ufw --force enable
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow ssh
    echo "  UFW configured: SSH in, all outbound allowed"
fi

echo ""
echo "══════════════════════════════════════════"
echo "  Setup complete!"
echo "══════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Copy your .env to ${DEPLOY_PATH}/.env"
echo "     scp .env trader@your-vps:${DEPLOY_PATH}/.env"
echo ""
echo "  2. Start the service:"
echo "     systemctl start futures-trader"
echo ""
echo "  3. Check status:"
echo "     systemctl status futures-trader"
echo "     tail -f ${DEPLOY_PATH}/logs/trader.log"
echo ""
echo "  4. Set up GitHub Actions secrets:"
echo "     VPS_HOST     = your VPS IP"
echo "     VPS_USER     = root (or user with sudo)"
echo "     VPS_SSH_KEY  = your SSH private key"
echo "     DEPLOY_PATH  = ${DEPLOY_PATH}"
echo ""
