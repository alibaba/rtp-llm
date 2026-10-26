#!/usr/bin/env bash
set -uo pipefail

# Run the external sCR controller flow for a GPU-only checkpoint.  This script
# intentionally never invokes CRIU or a CPU dump.  It also does not stop model
# processes or delete a checkpoint directory.
#
# Usage:
#   dump_restore_gpu_only.sh /path/to/checkpoint [wait_timeout_seconds]

DUMP_DIR="${1:?usage: $0 CHECKPOINT_DIR [WAIT_TIMEOUT_SECONDS]}"
WAIT_TIMEOUT="${2:-1800}"
CONTROLLER="${SCR_CONTROLLER:-/home/yuziqu.yzq/scr_controller}"
LOG_DIR="${SCR_CONTROLLER_LOG_DIR:-${DUMP_DIR%/*}/controller_logs}"
mkdir -p "$DUMP_DIR" "$LOG_DIR"

if [[ "${SCR_AS_ROOT:-1}" == "1" ]]; then
  SUDO=(/usr/bin/sudo -n)
else
  SUDO=()
fi

run_controller() {
  "${SUDO[@]}" "$CONTROLLER" "$@"
}

run_step() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "[$(date -Is)] controller $*" | tee "$log"
  run_controller "$@" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  echo "[$(date -Is)] ${name}_rc=${rc}" | tee -a "$log"
  return "$rc"
}

echo "checkpoint_dir=$DUMP_DIR"
echo "controller=$CONTROLLER"
echo "wait_timeout=$WAIT_TIMEOUT"
df -h "$DUMP_DIR"

# The check is informational.  In this sCR build it can report false after an
# earlier unblock even when all Epsilon participants have reached Running.
run_step check_before check || true

# Both --path and --bypass-cr-path point to the same directory.  With the
# machine scheduler configured with bypass_dump_restore=true this is the
# GPU-memory bypass path; no CPU/CRIU dump is requested here.
if ! run_step dump dump --path "$DUMP_DIR" --bypass-cr-path "$DUMP_DIR" \
    --block-timeout-ms 120000; then
  echo "dump command failed; not attempting restore" >&2
  exit 1
fi

# wait-cr-done in controller 1.6.0 accepts only --timeout (not path flags).
# Keep going after a persist/CR wrapper error so the log records the exact
# boundary between GPU completion and process-level persistence.
run_step wait_dump wait-cr-done --timeout "$WAIT_TIMEOUT" || true

if ! run_step restore restore --path "$DUMP_DIR" --bypass-cr-path "$DUMP_DIR"; then
  echo "restore command failed" >&2
  exit 1
fi
run_step wait_restore wait-cr-done --timeout "$WAIT_TIMEOUT" || true
run_step check_after check || true

echo "[$(date -Is)] checkpoint files:"
find "$DUMP_DIR" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
du -sh "$DUMP_DIR"
