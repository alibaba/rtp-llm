#!/usr/bin/env bash
set -euo pipefail
umask 077

role=${1:-}
case "$role" in
  prefill) base=/data6/luohaocheng.lhc ;;
  decode) base=/data1/luohaocheng.lhc ;;
  *) echo 'usage: feat_role_3fs_parallel_111112.sh prefill|decode' >&2; exit 2 ;;
esac
task="$base/artifacts/k3-fp8-opt-20260927"
test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test -f "$task/libparallel_3fs_pread.so"
test -x "$task/feat_smoke_role_3fs.sh"

export K3_3FS_PREAD_THREADS=64
export FASTSAFETENSORS_DEBUG=true
export LD_PRELOAD="$task/libparallel_3fs_pread.so${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$task/feat_smoke_role_3fs.sh" "$role"
