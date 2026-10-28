#!/usr/bin/env bash
# --run_under wrapper for tests: make the bazel-fetched NVIDIA wheel
# libraries resolvable by the dynamic loader.
#
# The dev images ship only the CUDA toolkit; nccl/cudnn/cublas/... arrive as
# pip_parse'd wheel repos.  torch's .so RPATHs ($ORIGIN/../../nvidia/*/lib)
# expect pip's co-located site-packages layout, which the per-repo runfiles
# layout does not provide, so locate the wheel lib dirs in the runfiles tree
# and prepend them to LD_LIBRARY_PATH.  The gcc-toolset-12 lib64 goes ahead
# of the system dirs as well: the JIT link lines use its ld, which needs its
# own libctf (LIBCTF_1.1).
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$here"
wheel_lib_dirs=()
torch_lib_dir=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    for base in "$root" "$root/external"; do
        found=("$base"/pip_*_torch_nvidia_*/site-packages/nvidia/*/lib)
        if [ -e "${found[0]}" ]; then
            wheel_lib_dirs=("${found[@]}")
        fi
        if [ -z "$torch_lib_dir" ]; then
            torch_lib_dir=$(ls -d "$base"/pip_*_torch_torch/site-packages/torch/lib 2>/dev/null | head -1)
        fi
        if [ "${#wheel_lib_dirs[@]}" -gt 0 ] && [ -n "$torch_lib_dir" ]; then
            break 2
        fi
    done
    root="$(dirname "$root")"
    [ "$root" = "/" ] && break
done

base_paths=(/opt/rh/gcc-toolset-12/root/usr/lib64 /usr/local/cuda/lib64 /opt/conda310/lib /usr/local/nvidia/lib64 /usr/lib64)
prefix="$(IFS=:; echo "${base_paths[*]}")"
if [ "${#wheel_lib_dirs[@]}" -gt 0 ]; then
    prefix="$(IFS=:; echo "${wheel_lib_dirs[*]}"):$prefix"
fi
# torch's own lib dir must be searchable too: engine tests LD_PRELOAD
# libtorch_cpu.so, and the preload is silently ignored (crashes follow in the
# embedded-interpreter suites) when the loader cannot resolve it.
if [ -n "$torch_lib_dir" ]; then
    prefix="$torch_lib_dir:$prefix"
fi
export LD_LIBRARY_PATH="$prefix${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# CI lanes previously wrapped every test with --run_under=//rtp_llm/test/utils:gpu_lock;
# this wrapper replaces that flag, so delegate to the gpu_lock device wrapper
# when it is present in the runfiles (it is a data dep of this target).
# The runfiles root is usually exported: prefer it, then fall back to walking
# up from this script.  Looking in the wrong place used to exec the test
# directly, which silently dropped both the GPU lock and the hang watchdog.
gpu_lock=""
roots=()
[ -n "${TEST_SRCDIR:-}" ] && roots+=("$TEST_SRCDIR")
[ -n "${RUNFILES_DIR:-}" ] && roots+=("$RUNFILES_DIR")
root="$here"
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    roots+=("$root")
    root="$(dirname "$root")"
    [ "$root" = "/" ] && break
done
for base in "${roots[@]}"; do
    for candidate in "$base"/rtp_llm/test/utils/gpu_lock "$base"/external/rtp_llm/test/utils/gpu_lock; do
        if [ -x "$candidate" ] || [ -f "$candidate" ]; then
            gpu_lock="$candidate"
            break 2
        fi
    done
done
if [ -n "$gpu_lock" ]; then
    exec "$gpu_lock" "$@"
fi
echo "nvidia_wheel_env: WARNING gpu_lock helper NOT found in runfiles (searched: ${roots[*]}); running the test WITHOUT the GPU lock and WITHOUT the hang watchdog. Set RTP_TEST_WATCHDOG_SECONDS to bound the test, and check that //rtp_llm/test/utils:gpu_lock is a data dep of this run_under target." >&2
exec "$@"
