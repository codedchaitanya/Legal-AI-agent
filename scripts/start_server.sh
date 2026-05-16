#!/usr/bin/env bash
# ILLA full-server start script (no-Docker fallback)
#
# Installs and launches MongoDB, Redis, PostgreSQL and MinIO as background
# daemons inside the current container, then starts the FastAPI + Gradio UI.
#
# Use this only when Docker is not available (e.g. RunPod, SageMaker).
# Otherwise prefer `docker compose -f docker/docker-compose.yml up --build`.
#
# Run:
#   chmod +x scripts/start_server.sh
#   ./scripts/start_server.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/.server-logs"
mkdir -p "$LOG_DIR"

echo "════════════════════════════════════════════════════════════════════"
echo "  ILLA — Full Server Start (no-Docker)"
echo "════════════════════════════════════════════════════════════════════"

# ── 1. Sanity checks ────────────────────────────────────────────────────
if [[ ! -f "$REPO_ROOT/.env" ]]; then
  echo "✗ .env not found. Run: cp .env.example .env, then fill in keys."
  exit 1
fi

if ! command -v python &>/dev/null; then
  echo "✗ python not found in PATH. Activate your venv first."
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "⚠ No active venv detected. Activating .venv/bin/activate…"
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

# Load .env into shell
set -a; source "$REPO_ROOT/.env"; set +a

# ── 2. Install system dependencies (idempotent) ─────────────────────────
echo ""
echo "[1/5] Ensuring system services are installed…"

NEED_APT_UPDATE=0
for pkg in redis-server postgresql postgresql-contrib mongodb; do
  if ! dpkg -s "$pkg" &>/dev/null; then
    NEED_APT_UPDATE=1
  fi
done

if [[ $NEED_APT_UPDATE -eq 1 ]]; then
  apt-get update -qq
  # mongodb on newer Debian/Ubuntu may be named differently — try both
  apt-get install -y -qq redis-server postgresql postgresql-contrib \
    || true
  apt-get install -y -qq mongodb || apt-get install -y -qq mongodb-org || \
    echo "⚠ MongoDB package unavailable in repo. Run separately if needed."
fi

# ── 3. Start MongoDB ────────────────────────────────────────────────────
echo ""
echo "[2/5] Starting MongoDB on :27017…"
if pgrep -x mongod >/dev/null; then
  echo "    already running"
else
  mkdir -p /var/lib/mongodb /var/log/mongodb
  mongod --fork --bind_ip 127.0.0.1 --port 27017 \
    --dbpath /var/lib/mongodb \
    --logpath "$LOG_DIR/mongod.log" \
    || echo "    ⚠ mongod failed to start — check $LOG_DIR/mongod.log"
fi

# ── 4. Start Redis ──────────────────────────────────────────────────────
echo ""
echo "[3/5] Starting Redis on :6379…"
if pgrep -x redis-server >/dev/null; then
  echo "    already running"
else
  redis-server --daemonize yes --port 6379 \
    --logfile "$LOG_DIR/redis.log" \
    --maxmemory 2gb --maxmemory-policy allkeys-lru
fi

# ── 5. Start PostgreSQL ─────────────────────────────────────────────────
echo ""
echo "[4/5] Starting PostgreSQL on :5432…"
PG_DATA=/var/lib/postgresql/data
if [[ ! -d "$PG_DATA/base" ]]; then
  mkdir -p "$PG_DATA"
  chown -R postgres:postgres "$PG_DATA" 2>/dev/null || true
  su - postgres -c "/usr/lib/postgresql/*/bin/initdb -D $PG_DATA" 2>&1 \
    | tail -3 || true
fi

if ! pgrep -x postgres >/dev/null; then
  su - postgres -c "/usr/lib/postgresql/*/bin/pg_ctl -D $PG_DATA -l $LOG_DIR/postgres.log start" \
    || echo "    ⚠ postgres failed to start — check $LOG_DIR/postgres.log"
  sleep 2
fi

# Create role + DB if missing
su - postgres -c "psql -tc \"SELECT 1 FROM pg_roles WHERE rolname='legalai'\" | grep -q 1 \
  || psql -c \"CREATE USER legalai WITH PASSWORD '${POSTGRES_PASSWORD:-changeme}' CREATEDB;\"" \
  2>/dev/null || true
su - postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='legalai'\" | grep -q 1 \
  || psql -c \"CREATE DATABASE legalai OWNER legalai;\"" \
  2>/dev/null || true

# ── 6. MinIO (optional — only if binary present) ────────────────────────
if command -v minio &>/dev/null; then
  echo ""
  echo "  ↳ Starting MinIO on :9000 / console :9001…"
  if ! pgrep -x minio >/dev/null; then
    MINIO_ROOT_USER="${MINIO_ACCESS_KEY:-minioadmin}" \
    MINIO_ROOT_PASSWORD="${MINIO_SECRET_KEY:-changeme}" \
    nohup minio server /tmp/minio-data --console-address ":9001" \
      >"$LOG_DIR/minio.log" 2>&1 &
  fi
else
  echo ""
  echo "  ↳ MinIO not installed. Document storage will be disabled."
  echo "     (Install: wget https://dl.min.io/server/minio/release/linux-amd64/minio && chmod +x minio && mv minio /usr/local/bin/)"
fi

# ── 7. Start the FastAPI app ────────────────────────────────────────────
echo ""
echo "[5/5] Starting FastAPI on :8000…"
echo "     Logs:   $LOG_DIR/api.log"
echo "     Health: http://localhost:8000/health"
echo "     Docs:   http://localhost:8000/docs"
echo ""
echo "     (Press Ctrl+C to stop the API; backing services keep running.)"
echo "════════════════════════════════════════════════════════════════════"

cd "$REPO_ROOT"
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info
