#!/usr/bin/env bash
# SM120 lock-lane coverage, sibling of update_cuda13_locks_test.sh (which
# is left unchanged).
#
# The SM120 requirements_lock_torch_gpu_cuda13_sm120_dsv4.txt pin file is a
# hand-assembled artifact with accepted manual/OSS sha provenance
# (refresh steps are in that lock's header). There is deliberately NO
# sm120 regeneration lane: the .update target is not supported, so the
# helper accepts x86|arm only.
set -euo pipefail

here=${TEST_SRCDIR:-}/rtp_deps
if [[ ! -x "$here/update_cuda13_locks.sh" ]]; then
    # For an external-repository test, its runfiles are under rtp_deps,
    # not under the main workspace's TEST_WORKSPACE label.
    here=${TEST_SRCDIR:-}/${TEST_WORKSPACE:-}/deps
fi
if [[ ! -x "$here/update_cuda13_locks.sh" ]]; then
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi
script="$here/update_cuda13_locks.sh"
[[ -x "$script" ]]

# The update helper exposes only the real regeneration lanes.
usage_out=$($script 2>&1 || true)
[[ "$usage_out" == *"x86|arm"* ]]
[[ "$usage_out" != *"sm120 --print-command"* ]]
[[ "$usage_out" != *"sm120-internal --print-command"* ]]

# The dead sm120 lanes are rejected, not silently aliased to x86.
if "$script" sm120 --print-command >/dev/null 2>&1; then
    echo "dead sm120 regeneration lane is still accepted" >&2
    exit 1
fi
if "$script" sm120-internal --print-command >/dev/null 2>&1; then
    echo "removed sm120-internal lane is still accepted" >&2
    exit 1
fi

# The real lanes still resolve to their expected compile commands.
x86=$($script x86 --print-command)
[[ "$x86" == *"//deps:requirements_torch_gpu_cuda13.update"* ]]

arm=$($script arm --print-command)
[[ "$arm" == *"//deps:requirements_cuda13_arm.update"* ]]

# The SM120 pin artifacts are present as checked-in files; nothing here
# regenerates them.
[[ -s "$here/requirements_torch_gpu_cuda13_sm120_dsv4.txt" ]]
[[ -s "$here/requirements_lock_torch_gpu_cuda13_sm120_dsv4.txt" ]]
