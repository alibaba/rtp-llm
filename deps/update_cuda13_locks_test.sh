#!/usr/bin/env bash
# Guards the command construction of update_cuda13_locks.sh (repo paths,
# platform, overrides) against rename drift.  Lock-file drift itself is
# checked by `update_cuda13_locks.sh <platform> --check`, which needs uv
# 0.11.1 plus package-index access and is deliberately not wired into this
# sandboxed unit test.
set -euo pipefail

script=${TEST_SRCDIR:-}/${TEST_WORKSPACE:-}/deps/update_cuda13_locks.sh
if [[ ! -x "$script" ]]; then
    script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/update_cuda13_locks.sh
fi

x86=$($script x86 --print-command)
arm=$($script arm --print-command)
internal=$($script internal-x86 --print-command)

[[ "$x86" == *"requirements_torch_gpu_cuda13.txt"* ]]
[[ "$x86" == *"requirements_cuda13_overrides.txt"* ]]
[[ "$x86" == *"x86_64-manylinux_2_28"* ]]
[[ "$x86" == *"requirements_torch_gpu_cuda13.update"* ]]
[[ "$arm" == *"requirements_cuda13_arm.txt"* ]]
[[ "$arm" == *"requirements_cuda13_overrides.txt"* ]]
[[ "$arm" == *"aarch64-manylinux_2_28"* ]]
[[ "$arm" == *"requirements_cuda13_arm.update"* ]]
[[ "$internal" == *"internal_source/deps/requirements_torch_gpu_cuda13.txt"* ]]
[[ "$internal" == *"internal_source/deps/requirements_lock_torch_gpu_cuda13.txt"* ]]
[[ "$internal" == *"requirements_cuda13_overrides.txt"* ]]
[[ "$internal" == *"@rtp_deps//:requirements_torch_gpu_cuda13.update"* ]]

# Invoking from another directory must not embed machine paths into the lock.
from_elsewhere=$(cd /tmp && "$script" x86 --print-command)
[[ "$from_elsewhere" == "$x86" ]]

sm90=$($script sm90-x86 --print-command)
internal_sm90=$($script internal-sm90-x86 --print-command)
[[ "$sm90" == *"deps/requirements_torch_gpu_cuda13_sm90.txt"* ]]
[[ "$sm90" == *"deps/requirements_lock_torch_gpu_cuda13_sm90.txt"* ]]
[[ "$sm90" == *"deps/requirements_cuda13_sm90_overrides.txt"* ]]
[[ "$sm90" == *"@rtp_public_deps//:requirements_torch_gpu_cuda13_sm90.update"* ]]
[[ "$internal_sm90" == *"internal_source/deps/requirements_torch_gpu_cuda13_sm90.txt"* ]]
[[ "$internal_sm90" == *"internal_source/deps/requirements_lock_torch_gpu_cuda13_sm90.txt"* ]]
[[ "$internal_sm90" == *"@rtp_deps//:requirements_torch_gpu_cuda13_sm90.update"* ]]
# The Hopper closure is x86 only; ARM continues its existing upstream lineage.
if "$script" sm90-arm --print-command >/dev/null 2>&1; then
    echo "Unexpected SM90 ARM lock target" >&2
    exit 1
fi
