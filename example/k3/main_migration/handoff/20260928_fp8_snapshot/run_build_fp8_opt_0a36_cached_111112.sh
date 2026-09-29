#!/usr/bin/env bash
set -uo pipefail

case "${1:-}" in
  111) base=/data6/luohaocheng.lhc ;;
  112) base=/data1/luohaocheng.lhc ;;
  *) echo 'usage: run_build_fp8_opt_0a36_cached_111112.sh 111|112' >&2; exit 2 ;;
esac

task="$base/artifacts/k3-fp8-opt-20260927"
log="$task/build-fp8-opt-0a36-cached-$1.log"
exit_file="$task/build-fp8-opt-0a36-cached-$1.exit"
test "$(id -un)" = luohaocheng.lhc || exit 2
test ! -e "$log" && test ! -e "$exit_file" || exit 2

docker exec -u luohaocheng.lhc lhc_GPU bash "$task/build_fp8_opt_0a36_cached_111112.sh" "$1" > "$log" 2>&1
status=$?
printf '%s\n' "$status" > "$exit_file"
exit "$status"
