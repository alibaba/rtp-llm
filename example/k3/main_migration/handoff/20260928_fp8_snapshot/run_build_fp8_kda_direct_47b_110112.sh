#!/usr/bin/env bash
set -uo pipefail

case "${1:-}" in
  110) base=/data7/luohaocheng.lhc ;;
  112) base=/data1/luohaocheng.lhc ;;
  *) echo 'usage: run_build_fp8_kda_direct_47b_110112.sh 110|112' >&2; exit 2 ;;
esac

task="$base/artifacts/k3-fp8-opt-20260927"
log="$task/build-fp8-kda-direct-47b-$1.log"
exit_file="$task/build-fp8-kda-direct-47b-$1.exit"
test "$(id -un)" = luohaocheng.lhc || exit 2
test ! -e "$log" && test ! -e "$exit_file" || exit 2

docker exec -u luohaocheng.lhc lhc_GPU_k3_3fs_20260927 \
  bash "$task/build_fp8_kda_direct_47b_110112.sh" "$1" > "$log" 2>&1
status=$?
printf '%s\n' "$status" > "$exit_file"
exit "$status"
