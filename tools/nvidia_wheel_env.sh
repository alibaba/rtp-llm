#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wheel_lib_dirs=()
torch_lib_dir=""
for _ in {1..14}; do
    for base in "$root" "$root/external"; do
        found=("$base"/pip_*_torch_nvidia_*/site-packages/nvidia/*/lib)
        if [[ -e "${found[0]}" ]]; then
            wheel_lib_dirs=("${found[@]}")
        fi
        if [[ -z "$torch_lib_dir" ]]; then
            for candidate in "$base"/pip_*_torch_torch/site-packages/torch/lib; do
                [[ ! -d "$candidate" ]] || torch_lib_dir="$candidate"
            done
        fi
        if [[ ${#wheel_lib_dirs[@]} -gt 0 && -n "$torch_lib_dir" ]]; then
            break 2
        fi
    done
    root="$(dirname "$root")"
    [[ "$root" != / ]] || break
done
base_paths=(/opt/rh/gcc-toolset-13/root/usr/lib64 /opt/rh/gcc-toolset-12/root/usr/lib64 /usr/local/cuda/lib64 /opt/conda310/lib /usr/local/nvidia/lib64 /usr/lib64)
prefix="$(IFS=:; printf '%s' "${base_paths[*]}")"
if [[ ${#wheel_lib_dirs[@]} -gt 0 ]]; then
    prefix="$(IFS=:; printf '%s' "${wheel_lib_dirs[*]}"):$prefix"
fi
[[ -z "$torch_lib_dir" ]] || prefix="$torch_lib_dir:$prefix"
export LD_LIBRARY_PATH="$prefix${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

roots=("${TEST_SRCDIR:-}" "${RUNFILES_DIR:-}")
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _ in {1..14}; do
    roots+=("$root")
    root="$(dirname "$root")"
    [[ "$root" != / ]] || break
done
for base in "${roots[@]}"; do
    [[ -n "$base" ]] || continue
    for candidate in "$base"/rtp_llm/test/utils/gpu_lock "$base"/external/rtp_llm/test/utils/gpu_lock; do
        if [[ -x "$candidate" ]]; then
            exec "$candidate" "$@"
        fi
    done
done
printf '%s\n' 'nvidia_wheel_env: gpu_lock helper missing; refusing unlocked execution.' >&2
exit 1
