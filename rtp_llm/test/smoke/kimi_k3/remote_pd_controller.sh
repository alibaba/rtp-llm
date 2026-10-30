#!/usr/bin/env bash
# Run two K3 remote shards with an invocation-scoped rendezvous controller.
set -euo pipefail
workspace=${1:?Bazel workspace is required}
target=${2:?native smoke target is required}
shift 2
[[ $# -gt 0 ]] || { echo "test command is required" >&2; exit 2; }
workspace=$(cd "$workspace" && pwd -P)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
[[ "$target" == //*:* ]] || { echo "expected //package:target" >&2; exit 2; }
artifact_path=${target#//}
artifact_path=${artifact_path/:/\/}
umask 077
outputs=$(mktemp -d "$workspace/kimi-k3-ci.XXXXXX")
broker_pid=""
archive_controller() {
  local destination="$workspace/bazel-testlogs/$artifact_path/controller"
  local name
  mkdir -p "$(dirname "$destination")" || return 1
  # Bazel may expose cached artifacts only after creating its testlogs symlink.
  # Keep the old directory separate so rendezvous files cannot mix across runs.
  if [[ -d "$destination" && ! -L "$destination" ]]; then
    mv -- "$destination" "$outputs/previous-controller" || return 1
  fi
  mkdir "$destination" || return 1
  for name in broker.log bazel-exit-code.txt invocation.json; do
    if [[ -f "$outputs/$name" ]]; then
      cp -- "$outputs/$name" "$destination/" || return 1
    fi
  done
  if [[ -d "$outputs/rendezvous" ]]; then
    cp -R -- "$outputs/rendezvous" "$destination/" || return 1
  fi
}
print_verdict() {
  python3 - "$outputs/rendezvous/state.json" "$1" "${K3_PD_RUN_ID:-}" <<'PY'
import json
import sys
from pathlib import Path

try:
    roles = json.loads(Path(sys.argv[1]).read_text())["roles"]
except (OSError, ValueError, KeyError):
    roles = {}

def verdict(outcome):
    if outcome is None:
        return "unknown"
    return "passed" if outcome["passed"] else "failed"

summary = {"run_id": sys.argv[3] or None, "bazel_exit_code": int(sys.argv[2]), "roles": {}}
for name in ("prefill", "decode"):
    role = roles.get(name, {})
    summary["roles"][name] = {
        "workload": verdict(role.get("result")),
        "cleanup": verdict(role.get("cleanup")),
    }
print("K3 PD verdict: " + json.dumps(summary))
PY
}
cleanup() {
  local result=$?
  local cleanup_failed=0 name
  trap - EXIT
  trap '' INT TERM
  # Try every cleanup step even when stopping or archiving fails.
  set +e
  if [[ -n "$broker_pid" ]]; then
    kill -TERM "$broker_pid" 2>/dev/null || true
    wait "$broker_pid" 2>/dev/null || true
  fi
  rm -f -- "$outputs/task-token" || cleanup_failed=1
  printf '%s\n' "$result" > "$outputs/bazel-exit-code.txt" || cleanup_failed=1
  print_verdict "$result" || true
  if [[ -d "$workspace/bazel-testlogs" ]]; then
    archive_controller || cleanup_failed=1
  else
    # Startup may fail before Bazel creates its testlogs symlink. Preserve
    # diagnostics in the CI console without creating a directory in its place.
    for name in broker.log bazel-exit-code.txt; do
      if [[ -f "$outputs/$name" ]]; then
        printf 'K3 controller diagnostic: %s\n' "$name" >&2
        cat -- "$outputs/$name" >&2 || cleanup_failed=1
      fi
    done
  fi
  if [[ "$cleanup_failed" == 0 ]]; then
    # Only this invocation's mktemp tree, including superseded K3 testlogs.
    rm -rf -- "$outputs" || cleanup_failed=1
  fi
  if [[ "$cleanup_failed" != 0 ]]; then
    printf 'K3 controller cleanup failed; diagnostics retained at %s\n' "$outputs" >&2
    if [[ "$result" == 0 ]]; then result=1; fi
  fi
  exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Do not upload a previous invocation's native smoke if this controller fails early.
previous="$workspace/bazel-testlogs/$artifact_path"
if [[ -e "$previous" ]]; then mv "$previous" "$outputs/previous-testlogs"; fi
python3 -c 'import secrets,sys; open(sys.argv[1],"x").write(secrets.token_hex(32))' "$outputs/task-token"
export K3_PD_BROKER_TOKEN K3_PD_RUN_ID K3_PD_BROKER_URL
K3_PD_BROKER_TOKEN=$(cat "$outputs/task-token")
K3_PD_RUN_ID=$(python3 -c 'import secrets; print("k3-pd-" + secrets.token_hex(12))')
python3 - "$outputs/invocation.json" "$K3_PD_RUN_ID" "$target" <<'PY'
import json
import sys
from pathlib import Path

record = json.dumps({"run_id": sys.argv[2], "target": sys.argv[3]})
Path(sys.argv[1]).write_text(record + "\n")
print("K3 PD invocation: " + record)
PY
controller_ip=${K3_PD_CONTROLLER_IP:-$(hostname -I | awk '{print $1}')}
# Duplicate the live CI output only for structured progress. Broker stderr and
# all other broker output keep going to the original artifact file.
python3 "$script_dir/remote_pd_broker.py" --run-id "$K3_PD_RUN_ID" \
  --token-file "$outputs/task-token" --advertise-ip "$controller_ip" \
  --output "$outputs/rendezvous" --timeout 43200 --progress-fd 3 \
  3>&1 > "$outputs/broker.log" 2>&1 &
broker_pid=$!
for attempt in {1..100}; do
  if [[ -s "$outputs/rendezvous/ready.json" ]]; then break; fi
  kill -0 "$broker_pid"
  sleep 0.1
done
test -s "$outputs/rendezvous/ready.json"
K3_PD_BROKER_URL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["url"])' "$outputs/rendezvous/ready.json")
# The caller may supply additional targets alongside the K3 shards.
# Only the K3 target inherits the broker environment; other smoke cache keys
# must not depend on this invocation's random token or run ID.
export BAZEL_TEST_TARGET=${BAZEL_TEST_TARGET:-$target}
"$@" --strategy=TestRunner=remote
