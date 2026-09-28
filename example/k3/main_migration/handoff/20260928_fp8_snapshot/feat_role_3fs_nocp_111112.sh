#!/usr/bin/env bash
# Task-local no-CP control for the fixed feat build. The original role script
# and its source checkout stay untouched.
set -euo pipefail
umask 077

role=${1:-}
case "$role" in
  prefill) base=/data6/luohaocheng.lhc ;;
  decode) base=/data1/luohaocheng.lhc ;;
  *) echo 'usage: feat_role_3fs_nocp_111112.sh prefill|decode' >&2; exit 2 ;;
esac
task="$base/artifacts/k3-fp8-opt-20260927"
source_role="$task/feat_smoke_role_3fs.sh"
patched_role="$task/feat_smoke_role_3fs_nocp_r9.sh"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test -f "$task/libparallel_3fs_pread.so"
test -f "$source_role"
test -w "$task"

python3 - "$source_role" "$patched_role" <<'PY'
import hashlib
import os
from pathlib import Path
import sys

source, destination = map(Path, sys.argv[1:])
original = source.read_bytes()
expected = "f19e7abbeb875eacea1f27008c93fb1745759b7d1fca86e44c6605eeadc952fe"
actual = hashlib.sha256(original).hexdigest()
if actual != expected:
    raise SystemExit(f"fixed feat role script changed: {actual}")

replacements = {
    'expected["PREFILL_CP_KV_CACHE_SHARDED"] = "1"':
        'expected["PREFILL_CP_KV_CACHE_SHARDED"] = "0"',
    '"DECODE_CP_KV_CACHE_SHARDED": "1",':
        '"DECODE_CP_KV_CACHE_SHARDED": "0",',
    '"DECODE_CP_Q_REPLICATED": decode_q_replicated,':
        '"DECODE_CP_Q_REPLICATED": "0",',
    'export PREFILL_CP_KV_CACHE_SHARDED=1':
        'export PREFILL_CP_KV_CACHE_SHARDED=0',
    'export DECODE_CP_KV_CACHE_SHARDED=1':
        'export DECODE_CP_KV_CACHE_SHARDED=0',
    'export DECODE_CP_Q_REPLICATED="${smoke_decode_q_replicated}"':
        'export DECODE_CP_Q_REPLICATED=0',
    'smoke_reuse_unit_tokens=$((smoke_block_size * smoke_prefill_tp_size))':
        'smoke_reuse_unit_tokens=$((smoke_block_size))',
    '--decode-page-rr 1 --proposal-tokens':
        '--decode-page-rr 0 --proposal-tokens',
}
text = original.decode()
for old, new in replacements.items():
    if text.count(old) != 1:
        raise SystemExit(f"expected exactly one source occurrence: {old}")
    text = text.replace(old, new)
content = text.encode()
if destination.exists() and destination.read_bytes() == content:
    pass
else:
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(content)
    os.chmod(temporary, 0o700)
    temporary.replace(destination)
print(f"fixed feat no-CP role SHA256={hashlib.sha256(content).hexdigest()}")
PY
bash -n "$patched_role"

export HOME="$base"
export K3_3FS_PREAD_THREADS=64
export FASTSAFETENSORS_DEBUG=true
export LD_PRELOAD="$task/libparallel_3fs_pread.so${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$patched_role" "$role"
