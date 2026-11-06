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
# Toolkit stub last: on driver-less workers the stub libcuda.so satisfies the
# DT_NEEDED of device-linked test binaries so CPU-only tests (GPU_COUNT=0)
# can load and run; real driver paths above win wherever a driver exists.
for _stub in /usr/local/cuda/lib64/stubs /usr/local/cuda-13.2/lib64/stubs; do
    [ -d "$_stub" ] && base_paths+=("$_stub")
done
# Final fallback for driver-less workers (CPU-only test execution with
# GPU_COUNT=0): device-linked test binaries DT_NEED driver SONAMEs such as
# libcuda.so.1 and libnvidia-ml.so.1. The toolkit stubs dir ships only the
# dev names (libcuda.so, libnvidia-ml.so, ...), which satisfy neither the
# loader nor a SONAME existence check: link EVERY toolkit stub to its
# .so.1 SONAME in a temp dir appended to LD_LIBRARY_PATH; if a stub file is
# missing entirely, synthesize an empty one - lazy binding resolves symbols
# only at first call, and CPU-only tests never call driver APIs, so the
# load succeeds and the CPU assertions run.
_have_driver=no
for _d in "${base_paths[@]}"; do
    [ -e "$_d/libcuda.so.1" ] && _have_driver=yes
done
ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1' && _have_driver=yes
if [ "$_have_driver" = no ]; then
    _stub=""
    for _cand in /usr/local/cuda/compat /usr/local/cuda*/lib64/stubs; do
        if [ -e "$_cand/libcuda.so.1" ]; then
            _stub="$_cand"
            break
        fi
    done
    if [ -z "$_stub" ]; then
        _gen=/tmp/rtp-libcuda-stub.$$
        mkdir -p "$_gen" 2>/dev/null || true
        _linked=0
        for _pair in \
            'libcuda.so.1 libcuda.so' \
            'libnvidia-ml.so.1 libnvidia-ml.so'
        do
            set -- $_pair
            _soname=$1
            _dev=$2
            _src=""
            for _cand in /usr/local/cuda*/lib64/stubs /usr/local/cuda/lib64/stubs; do
                [ -e "$_cand/$_dev" ] && _src="$_cand/$_dev" && break
            done
            if [ -z "$_src" ] && command -v gcc >/dev/null 2>&1; then
                printf '' | gcc -shared -x c -o "$_gen/$_dev" - 2>/dev/null || true
                _src="$_gen/$_dev"
            fi
            if [ -n "$_src" ] && [ ! -e "$_gen/$_soname" ]; then
                ln -sn "$_src" "$_gen/$_soname" 2>/dev/null || true
                [ -e "$_gen/$_soname" ] && _linked=$((_linked+1))
            fi
        done
        if [ "$_linked" -gt 0 ]; then
            _stub="$_gen"
            echo "nvidia_wheel_env: no driver; linked $_linked stub SONAME(s) in $_stub" >&2
        else
            rm -rf "$_gen" 2>/dev/null || true
        fi
    fi
    if [ -n "$_stub" ]; then
        base_paths+=("$_stub")
    fi
fi
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
root="$here"
gpu_lock=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    for base in "$root" "$root/external"; do
        gpu_lock=$(ls "$base"/rtp_llm/test/utils/gpu_lock 2>/dev/null | head -1)
        [ -n "$gpu_lock" ] && break 2
    done
    root="$(dirname "$root")"
    [ "$root" = "/" ] && break
done
if [ -n "$gpu_lock" ]; then
    exec "$gpu_lock" "$@"
fi
exec "$@"
