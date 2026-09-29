#!/usr/bin/env bash
# Task-local no-CP control for the fixed feat build. The original role script
# and its source checkout stay untouched. The Decode post-request check uses
# the no-CP evidence contract instead of the original DCP/Page-RR contract.
set -euo pipefail
umask 077

role=${1:-}
case "$role" in
  prefill|decode) ;;
  *) echo 'usage: feat_role_3fs_nocp_r10.sh prefill|decode' >&2; exit 2 ;;
esac
task=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
base=${task%/artifacts/k3-fp8-opt-20260927}
test "$base" != "$task"
source_role="$task/feat_smoke_role_3fs.sh"
patched_role="$task/feat_smoke_role_3fs_nocp_r12.sh"
auditor="$task/audit_feat_nocp_decode_runtime.py"

test "$(id -un)" = luohaocheng.lhc
test -f /.dockerenv
test -f "$task/libparallel_3fs_pread_20260929.so"
test -f "$source_role"
test -f "$auditor"
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
    'expected["PREFILL_CP_SIZE"] = prefill_tp_size':
        'expected["PREFILL_CP_SIZE"] = "1"',
    'export PREFILL_CP_KV_CACHE_SHARDED=1':
        'export PREFILL_CP_KV_CACHE_SHARDED=0',
    'export DECODE_CP_KV_CACHE_SHARDED=1':
        'export DECODE_CP_KV_CACHE_SHARDED=0',
    'export DECODE_CP_Q_REPLICATED="${smoke_decode_q_replicated}"':
        'export DECODE_CP_Q_REPLICATED=0',
    'export PREFILL_CP_SIZE="${smoke_prefill_tp_size}"':
        'export PREFILL_CP_SIZE=1',
    'source_cache=page-rr/${smoke_prefill_tp_size} destination_cache=page-rr/${smoke_decode_tp_size}':
        'source_cache=rank-local/1 destination_cache=rank-local/1',
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
start = "verify_decode_graph_log() {\n"
end = "\nverify_smoke_runtime_coverage() {"
if text.count(start) != 1 or text.count(end) != 1:
    raise SystemExit("fixed feat Decode verifier boundary changed")
before, rest = text.split(start, 1)
discarded, after = rest.split(end, 1)
if "MLA_DCP" not in discarded or "K3_PAGE_RR_TARGET" not in discarded:
    raise SystemExit("expected original DCP/Page-RR verifier is absent")
text = before + '''verify_decode_graph_log() {
    [[ "${role}" == "decode" ]] || return 0
    python3 "${K3_NOCP_DECODE_AUDITOR:?}" \\
        --service-env "${role_dir}/service.env" \\
        --engine-log "${role_dir}/runtime/work/decode/logs/engine.log" \\
        --tp-size "${smoke_tp_size}" \\
        --proposal-tokens "${smoke_proposal_tokens}" \\
        --output "${role_dir}/decode-nocp-runtime-audit.json" \\
        || die "Decode no-CP runtime audit failed"
}
''' + end + after
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
if [[ "${K3_NOCP_PATCH_ONLY:-0}" == 1 ]]; then
    exit 0
fi

export HOME="$base"
export K3_3FS_PREAD_THREADS=64
export FASTSAFETENSORS_DEBUG=true
export K3_NOCP_DECODE_AUDITOR="$auditor"
export LD_PRELOAD="$task/libparallel_3fs_pread_20260929.so${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$patched_role" "$role"
