#!/usr/bin/env bash
set -euo pipefail
umask 077

host_id=${1:?usage: run_feat_build_detached_114115.sh 114\|115}
case "$host_id" in
  114|115) ;;
  *) echo 'only 114 and 115 are configured' >&2; exit 2 ;;
esac

base=/data0/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927
build="$base/build_feat_a9bf_114115_20260929.sh"
log="$base/build-feat-a9bf-${host_id}-20260929.log"
exit_file="$base/build-feat-a9bf-${host_id}-20260929.exit"

test "$(id -un)" = luohaocheng.lhc
test -f "$build"
test ! -e "$log"
test ! -e "$exit_file"
docker inspect lhc_GPU --format '{{.State.Running}}' | grep -qx true

set +e
docker exec -u luohaocheng.lhc lhc_GPU bash "$build" "$host_id" >"$log" 2>&1
status=$?
printf '%s\n' "$status" >"$exit_file"
exit "$status"
