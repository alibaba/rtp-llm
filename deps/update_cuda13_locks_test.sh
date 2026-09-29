#!/usr/bin/env bash
set -euo pipefail

script=${TEST_SRCDIR:-}/${TEST_WORKSPACE:-}/deps/update_cuda13_locks.sh
if [[ ! -x "$script" ]]; then
    script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/update_cuda13_locks.sh
fi

x86=$($script x86 --print-command)
arm=$($script arm --print-command)

[[ "$x86" == *"requirements_torch_gpu_cuda13.txt"* ]]
[[ "$x86" == *"requirements_cuda13_overrides.txt"* ]]
[[ "$x86" == *"x86_64-manylinux_2_28"* ]]
[[ "$x86" == *"requirements_torch_gpu_cuda13.update"* ]]
[[ "$arm" == *"requirements_cuda13_arm.txt"* ]]
[[ "$arm" == *"requirements_cuda13_overrides.txt"* ]]
[[ "$arm" == *"aarch64-manylinux_2_28"* ]]
[[ "$arm" == *"requirements_cuda13_arm.update"* ]]
