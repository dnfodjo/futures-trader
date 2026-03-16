#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# MNQ Futures Trader — Health Check
# ═══════════════════════════════════════════════════════════════════════
# Run on VPS to verify the system is ready to trade.
# Can be run remotely:
#   SSH_HOST=x SSH_USER=root SSH_PASSWORD=x ./scripts/health_check.sh --remote
#
# Or locally on VPS:
#   /opt/futures-trader/scripts/health_check.sh
# ═══════════════════════════════════════════════════════════════════════
set -uo pipefail

DEPLOY_PATH="${DEPLOY_PATH:-/opt/futures-trader}"
REMOTE_MODE=false
PASSED=0
FAILED=0
WARNED=0

# ── Parse args ─────────────────────────────────────────────────────
if [[ "${1:-}" == "--remote" ]]; then
    REMOTE_MODE=true
    SSH_HOST="${SSH_HOST:?SSH_HOST is required for remote mode}"
    SSH_USER="${SSH_USER:-root}"
    SSH_KEY_FILE="${SSH_KEY_FILE:-}"
    SSH_PASSWORD="${SSH_PASSWORD:-}"

    SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
    if [[ -n "${SSH_KEY_FILE}" ]]; then
        SSH_CMD="ssh ${SSH_OPTS} -i ${SSH_KEY_FILE} ${SSH_USER}@${SSH_HOST}"
    elif [[ -n "${SSH_PASSWORD}" ]]; then
        SSH_CMD="sshpass -p '${SSH_PASSWORD}' ssh ${SSH_OPTS} ${SSH_USER}@${SSH_HOST}"
    else
        SSH_CMD="ssh ${SSH_OPTS} ${SSH_USER}@${SSH_HOST}"
    fi
fi

# ── Helper functions ───────────────────────────────────────────────
run_cmd() {
    if [[ "${REMOTE_MODE}" == true ]]; then
        eval ${SSH_CMD} "$1"
    else
        eval "$1"
    fi
}

pass() { echo "  ✅ $1"; ((PASSED++)); }
fail() { echo "  ❌ $1"; ((FAILED++)); }
warn() { echo "  ⚠️  $1"; ((WARNED++)); }

echo "══════════════════════════════════════════════════════"
echo "  MNQ Futures Trader — Health Check"
echo "══════════════════════════════════════════════════════"
echo ""

# ── 1. System basics ──────────────────────────────────────────────
echo "📦 System"
if run_cmd "python3 --version" 2>/dev/null | grep -q "3.1[1-9]"; then
    pass "Python 3.11+ installed"
else
    fail "Python 3.11+ not found"
fi

if run_cmd "test -d ${DEPLOY_PATH}" 2>/dev/null; then
    pass "Deploy directory exists: ${DEPLOY_PATH}"
else
    fail "Deploy directory missing: ${DEPLOY_PATH}"
fi

if run_cmd "test -d ${DEPLOY_PATH}/.venv" 2>/dev/null; then
    pass "Virtual environment exists"
else
    fail "Virtual environment missing"
fi

if run_cmd "test -d ${DEPLOY_PATH}/data" 2>/dev/null; then
    pass "Data directory exists"
else
    fail "Data directory missing"
fi

if run_cmd "test -d ${DEPLOY_PATH}/logs" 2>/dev/null; then
    pass "Logs directory exists"
else
    fail "Logs directory missing"
fi

echo ""

# ── 2. Environment variables ──────────────────────────────────────
echo "🔑 Environment (.env)"
if run_cmd "test -f ${DEPLOY_PATH}/.env" 2>/dev/null; then
    pass ".env file exists"
else
    fail ".env file missing — copy from local!"
fi

# Check critical env vars are set (non-empty)
for var in ANTHROPIC_API_KEY DB_API_KEY QL_WEBHOOK_URL QL_USER_ID TRADE_SYMBOL; do
    val=$(run_cmd "grep '^${var}=' ${DEPLOY_PATH}/.env 2>/dev/null | head -1 | cut -d= -f2-" 2>/dev/null || echo "")
    if [[ -n "${val}" && "${val}" != "your_"* ]]; then
        pass "${var} is set"
    else
        fail "${var} is missing or placeholder"
    fi
done

echo ""

# ── 3. Python dependencies ───────────────────────────────────────
echo "🐍 Dependencies"
if run_cmd "${DEPLOY_PATH}/.venv/bin/python -c 'import anthropic; print(\"ok\")'" 2>/dev/null | grep -q ok; then
    pass "anthropic SDK installed"
else
    fail "anthropic SDK missing"
fi

if run_cmd "${DEPLOY_PATH}/.venv/bin/python -c 'import databento; print(\"ok\")'" 2>/dev/null | grep -q ok; then
    pass "databento SDK installed"
else
    fail "databento SDK missing"
fi

if run_cmd "${DEPLOY_PATH}/.venv/bin/python -c 'import pydantic; print(\"ok\")'" 2>/dev/null | grep -q ok; then
    pass "pydantic installed"
else
    fail "pydantic missing"
fi

if run_cmd "${DEPLOY_PATH}/.venv/bin/python -c 'import structlog; print(\"ok\")'" 2>/dev/null | grep -q ok; then
    pass "structlog installed"
else
    fail "structlog missing"
fi

echo ""

# ── 4. API connectivity ──────────────────────────────────────────
echo "🌐 API Connectivity"

# Test Anthropic API
ANTHROPIC_CHECK=$(run_cmd "${DEPLOY_PATH}/.venv/bin/python -c \"
import os
from dotenv import load_dotenv
load_dotenv('${DEPLOY_PATH}/.env')
import anthropic
c = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
r = c.messages.create(model='claude-haiku-4-5-20251001', max_tokens=5, messages=[{'role':'user','content':'hi'}])
print('ok')
\"" 2>&1 || echo "error")

if echo "${ANTHROPIC_CHECK}" | grep -q "ok"; then
    pass "Anthropic API — connected"
else
    fail "Anthropic API — connection failed"
fi

# Test Databento API (just import + key validation — no subscription cost)
DB_CHECK=$(run_cmd "${DEPLOY_PATH}/.venv/bin/python -c \"
import os
from dotenv import load_dotenv
load_dotenv('${DEPLOY_PATH}/.env')
import databento as db
key = os.getenv('DB_API_KEY', '')
if key and not key.startswith('your_'):
    print('ok')
else:
    print('no_key')
\"" 2>&1 || echo "error")

if echo "${DB_CHECK}" | grep -q "ok"; then
    pass "Databento API key — present"
else
    fail "Databento API key — missing or placeholder"
fi

# Test QuantLynk webhook endpoint is reachable (HEAD request only)
QL_CHECK=$(run_cmd "curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 https://lynk.quantvue.io" 2>/dev/null || echo "000")
if [[ "${QL_CHECK}" != "000" ]]; then
    pass "QuantLynk endpoint — reachable"
else
    fail "QuantLynk endpoint — unreachable"
fi

echo ""

# ── 5. Systemd service ───────────────────────────────────────────
echo "⚙️  Service"
if run_cmd "systemctl is-enabled futures-trader" 2>/dev/null | grep -q enabled; then
    pass "futures-trader service enabled"
else
    warn "futures-trader service not enabled"
fi

SERVICE_STATUS=$(run_cmd "systemctl is-active futures-trader" 2>/dev/null || echo "unknown")
if [[ "${SERVICE_STATUS}" == "active" ]]; then
    pass "futures-trader service running"
elif [[ "${SERVICE_STATUS}" == "inactive" ]]; then
    warn "futures-trader service inactive (not started)"
else
    warn "futures-trader service status: ${SERVICE_STATUS}"
fi

echo ""

# ── 6. Disk & permissions ────────────────────────────────────────
echo "💾 Storage"
DISK_AVAIL=$(run_cmd "df -h ${DEPLOY_PATH} | tail -1 | awk '{print \$4}'" 2>/dev/null || echo "unknown")
echo "  ℹ️  Available disk: ${DISK_AVAIL}"

OWNER=$(run_cmd "stat -c '%U' ${DEPLOY_PATH}/data 2>/dev/null" 2>/dev/null || echo "unknown")
if [[ "${OWNER}" == "trader" ]]; then
    pass "data/ owned by trader user"
elif [[ "${OWNER}" == "root" ]]; then
    warn "data/ owned by root (should be 'trader')"
else
    warn "data/ owner: ${OWNER}"
fi

echo ""

# ── 7. Time & timezone ───────────────────────────────────────────
echo "🕐 Time"
SYSTEM_TIME=$(run_cmd "date '+%Y-%m-%d %H:%M:%S %Z'" 2>/dev/null || echo "unknown")
echo "  ℹ️  System time: ${SYSTEM_TIME}"

ET_TIME=$(run_cmd "TZ='US/Eastern' date '+%Y-%m-%d %H:%M:%S %Z'" 2>/dev/null || echo "unknown")
echo "  ℹ️  Eastern time: ${ET_TIME}"

# Verify NTP is running
if run_cmd "timedatectl show --property=NTPSynchronized --value" 2>/dev/null | grep -q yes; then
    pass "NTP synchronized"
else
    warn "NTP may not be synchronized"
fi

echo ""

# ── 8. Recent logs (if service has run) ───────────────────────────
echo "📋 Recent Activity"
if run_cmd "test -f ${DEPLOY_PATH}/logs/trader.log" 2>/dev/null; then
    echo "  Last 5 log lines:"
    run_cmd "tail -5 ${DEPLOY_PATH}/logs/trader.log" 2>/dev/null | while read -r line; do
        echo "    ${line}"
    done
else
    echo "  ℹ️  No logs yet (service hasn't run)"
fi

# Check journal.db exists
if run_cmd "test -f ${DEPLOY_PATH}/data/journal.db" 2>/dev/null; then
    pass "Trade journal exists"
else
    echo "  ℹ️  No trade journal yet (will be created on first run)"
fi

echo ""

# ── Summary ───────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════"
echo "  Results: ✅ ${PASSED} passed | ❌ ${FAILED} failed | ⚠️  ${WARNED} warnings"
echo "══════════════════════════════════════════════════════"

if [[ ${FAILED} -gt 0 ]]; then
    echo ""
    echo "  ❌ System NOT ready — fix failures above"
    exit 1
else
    echo ""
    echo "  ✅ System ready for trading!"
    exit 0
fi
