#!/usr/bin/env bash
# Guards the command construction of update_cuda13_locks.sh (repo paths,
# platform, overrides) against rename drift.  Lock-file drift itself is
# checked by `update_cuda13_locks.sh <platform> --check`, which needs uv
# 0.11.1 plus package-index access and is deliberately not wired into this
# sandboxed unit test. Wheel metadata must also match the native/Python
# XGrammar release used by these locks, or tokenizer deserialization fails.
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

workspace=$(dirname "$(dirname "$script")")
xgrammar=$(sed -n 's/^\(xgrammar==[^[:space:]]*\)$/\1/p' "$workspace/deps/requirements_base.txt")
[[ -n "$xgrammar" ]]
for source in "$workspace/arch_config/arch_select.bzl" "$workspace/rtp_llm/BUILD"; do
    wheel_pin=$(grep -oE 'xgrammar==[0-9.]+' "$source" | sort -u)
    if [[ "$wheel_pin" != "$xgrammar" ]]; then
        echo "$source: wheel pin $wheel_pin differs from CUDA13 runtime $xgrammar" >&2
        exit 1
    fi
done
