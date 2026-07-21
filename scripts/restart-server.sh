#!/usr/bin/env bash
# Restart the voice_tutor server: kill any running instance, start fresh.
#
# Usage:
#   scripts/restart-server.sh           # restart, tail log on Ctrl+C
#   scripts/restart-server.sh --no-log  # restart, return immediately
#
# The server writes to logs/server.log. Health is checked after startup.

set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="logs"
LOG_FILE="$LOG_DIR/server.log"
HEALTH_URL="http://127.0.0.1:8888/health"
HEALTH_TIMEOUT=20   # seconds to wait for healthy startup
TAIL=${1:-"--log"}

mkdir -p "$LOG_DIR"

# ─── 1. Kill old server ─────────────────────────────────────────────
# Match the uvicorn worker process (child of `uv run`), not the parent.
OLD_PIDS=$(pgrep -f "python3? -m voice_tutor.server" || true)
if [[ -n "$OLD_PIDS" ]]; then
    echo "Killing old server: $(echo "$OLD_PIDS" | tr '\n' ' ')"
    kill $OLD_PIDS 2>/dev/null || true
    # Wait for them to actually exit (up to 5s)
    for _ in 1 2 3 4 5; do
        sleep 1
        REMAINING=$(pgrep -f "python3? -m voice_tutor.server" || true)
        [[ -z "$REMAINING" ]] && break
        kill -9 $REMAINING 2>/dev/null || true
    done
else
    echo "No old server found."
fi

# ─── 2. Start new server ────────────────────────────────────────────
echo "Starting new server..."
: > "$LOG_FILE"   # truncate
nohup uv run python -m voice_tutor.server > "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "Started PID $NEW_PID (log: $LOG_FILE)"

# ─── 3. Wait for health ─────────────────────────────────────────────
echo "Waiting for health..."
for i in $(seq 1 $HEALTH_TIMEOUT); do
    if BODY=$(curl -sf "$HEALTH_URL" 2>/dev/null); then
        echo "✅ Healthy after ${i}s: $BODY"
        break
    fi
    # Check if process died
    if ! kill -0 $NEW_PID 2>/dev/null; then
        echo "❌ Server process died. Last log lines:"
        tail -20 "$LOG_FILE"
        exit 1
    fi
    sleep 1
    [[ $i -eq $HEALTH_TIMEOUT ]] && {
        echo "❌ Health check timed out after ${HEALTH_TIMEOUT}s. Last log lines:"
        tail -20 "$LOG_FILE"
        exit 1
    }
done

# ─── 4. Optionally tail the log ─────────────────────────────────────
if [[ "$TAIL" == "--log" ]]; then
    echo
    echo "Tailing log (Ctrl+C to stop tailing — server keeps running)..."
    tail -f "$LOG_FILE"
fi
