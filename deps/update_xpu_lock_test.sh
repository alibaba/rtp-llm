#!/usr/bin/env bash
# Guards the command construction of update_xpu_lock.sh (repo paths, platform,
# overrides, XPU index) against rename drift. Lock drift itself is checked by
# `update_xpu_lock.sh --check`, which needs uv 0.11.1 and package-index access.
set -euo pipefail

script=${TEST_SRCDIR:-}/${TEST_WORKSPACE:-}/deps/update_xpu_lock.sh
if [[ ! -x "$script" ]]; then
    script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/update_xpu_lock.sh
fi

cmd=$($script --print-command)

[[ "$cmd" == *"requirements_xpu.txt"* ]]
[[ "$cmd" == *"requirements_xpu_overrides.txt"* ]]
[[ "$cmd" == *"requirements_lock_xpu.txt"* ]]
[[ "$cmd" == *"x86_64-manylinux_2_28"* ]]
[[ "$cmd" == *"download.pytorch.org/whl/xpu"* ]]
[[ "$cmd" == *"requirements_xpu.update"* ]]
