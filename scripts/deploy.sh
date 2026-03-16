#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# MNQ Futures Trader — Deploy Script
# ═══════════════════════════════════════════════════════════════════════
# Called by GitHub Actions on push to main, or manually:
#   SSH_HOST=x SSH_USER=root DEPLOY_PATH=/opt/futures-trader ./scripts/deploy.sh
#
# What it does:
#   1. SSH into VPS
#   2. Pull latest code
#   3. Install any new dependencies
#   4. Gracefully restart the trading service
#      (systemd sends SIGTERM → system flattens positions → clean exit → restart)
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

SSH_HOST="${SSH_HOST:?SSH_HOST is required}"
SSH_USER="${SSH_USER:-root}"
SSH_KEY_FILE="${SSH_KEY_FILE:-}"
DEPLOY_PATH="${DEPLOY_PATH:-/opt/futures-trader}"
BRANCH="${BRANCH:-main}"

# Build SSH command
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
if [[ -n "${SSH_KEY_FILE}" ]]; then
    SSH_OPTS="${SSH_OPTS} -i ${SSH_KEY_FILE}"
fi
SSH_CMD="ssh ${SSH_OPTS} ${SSH_USER}@${SSH_HOST}"

echo "══════════════════════════════════════════"
echo "  Deploying to ${SSH_HOST}"
echo "  Path: ${DEPLOY_PATH}"
echo "  Branch: ${BRANCH}"
echo "══════════════════════════════════════════"

# ── 1. Check if service is mid-trade ────────────────────────────────
echo "[1/5] Checking trading state..."
POSITION_CHECK=$(${SSH_CMD} "cat ${DEPLOY_PATH}/data/position_state.json 2>/dev/null || echo '{}'")
if echo "${POSITION_CHECK}" | grep -q '"has_position": true'; then
    echo "  ⚠️  System has an OPEN POSITION. Deploy will trigger graceful flatten."
    echo "  Proceeding in 5 seconds... (Ctrl+C to abort)"
    sleep 5
else
    echo "  No open position. Safe to deploy."
fi

# ── 2. Pull latest code ─────────────────────────────────────────────
echo "[2/5] Pulling latest code..."
${SSH_CMD} << REMOTE_PULL
    cd ${DEPLOY_PATH}
    git fetch origin ${BRANCH}
    git reset --hard origin/${BRANCH}
REMOTE_PULL

# ── 3. Install dependencies ─────────────────────────────────────────
echo "[3/5] Installing dependencies..."
${SSH_CMD} << REMOTE_INSTALL
    cd ${DEPLOY_PATH}
    .venv/bin/pip install -e . -q 2>&1 | tail -3
REMOTE_INSTALL

# ── 4. Restart service ──────────────────────────────────────────────
echo "[4/5] Restarting service..."
${SSH_CMD} << REMOTE_RESTART
    # Graceful restart — systemd sends SIGTERM, waits for flatten, then starts new instance
    sudo systemctl restart futures-trader

    # Wait for it to come up
    sleep 3
    if systemctl is-active --quiet futures-trader; then
        echo "  ✓ Service is running"
    else
        echo "  ✗ Service failed to start!"
        sudo journalctl -u futures-trader --no-pager -n 20
        exit 1
    fi
REMOTE_RESTART

# ── 5. Verify ───────────────────────────────────────────────────────
echo "[5/5] Verifying deployment..."
${SSH_CMD} << REMOTE_VERIFY
    echo "Service status:"
    systemctl status futures-trader --no-pager | head -10
    echo ""
    echo "Last 5 log lines:"
    tail -5 ${DEPLOY_PATH}/logs/trader.log 2>/dev/null || echo "(no logs yet)"
REMOTE_VERIFY

echo ""
echo "══════════════════════════════════════════"
echo "  Deploy complete ✓"
echo "══════════════════════════════════════════"
