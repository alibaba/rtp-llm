#!/usr/bin/env bash
set -euo pipefail

wrapper_source="${TEST_SRCDIR}/${TEST_WORKSPACE}/tools/nvidia_wheel_env.sh"
wrapper="${TEST_TMPDIR}/nvidia_wheel_env.sh"
target_runfiles="${TEST_TMPDIR}/wrapped_test.runfiles"
cudnn_lib_dir="${target_runfiles}/pip_gpu_cuda13_torch_nvidia_cudnn_cu13/site-packages/nvidia/cudnn/lib"
torch_lib_dir="${target_runfiles}/pip_gpu_cuda13_torch_torch/site-packages/torch/lib"

cp "$wrapper_source" "$wrapper"
mkdir -p "$cudnn_lib_dir" "$torch_lib_dir"
touch "$cudnn_lib_dir/libcudnn_adv.so.9" "$torch_lib_dir/libtorch.so"

actual=$(
    RUNFILES_DIR="$target_runfiles" \
    TEST_SRCDIR="" \
    LD_LIBRARY_PATH="/sentinel" \
        bash "$wrapper" bash -c 'printf "%s|%s" "$LD_LIBRARY_PATH" "$RTP_LLM_RUN_UNDER_LIBRARY_PATH"'
)
actual_library_path="${actual%%|*}"
actual_run_under_path="${actual#*|}"

expected_prefix="${torch_lib_dir}:${cudnn_lib_dir}:"
if [[ "$actual_library_path" != "$expected_prefix"* ]]; then
    echo "wheel paths from RUNFILES_DIR were not prepended to LD_LIBRARY_PATH" >&2
    echo "expected prefix: $expected_prefix" >&2
    echo "actual: $actual_library_path" >&2
    exit 1
fi

if [[ ":$actual_library_path:" != *":/sentinel:"* ]]; then
    echo "existing LD_LIBRARY_PATH was not preserved: $actual_library_path" >&2
    exit 1
fi

if [[ "$actual_run_under_path" != "$expected_prefix"* ]]; then
    echo "wrapper-injected paths were not recorded separately" >&2
    echo "expected prefix: $expected_prefix" >&2
    echo "actual: $actual_run_under_path" >&2
    exit 1
fi

wrapper_launcher="${TEST_TMPDIR}/nvidia_wheel_env"
wrapper_runfiles="${wrapper_launcher}.runfiles"
wrapper_cudnn_lib_dir="${wrapper_runfiles}/pip_gpu_cuda13_torch_nvidia_cudnn_cu13/site-packages/nvidia/cudnn/lib"
wrapper_torch_lib_dir="${wrapper_runfiles}/pip_gpu_cuda13_torch_torch/site-packages/torch/lib"

cp "$wrapper_source" "$wrapper_launcher"
mkdir -p "$wrapper_cudnn_lib_dir" "$wrapper_torch_lib_dir"
touch "$wrapper_cudnn_lib_dir/libcudnn_adv.so.9" "$wrapper_torch_lib_dir/libtorch.so"

actual=$(
    RUNFILES_DIR="" \
    TEST_SRCDIR="" \
    LD_LIBRARY_PATH="/sentinel" \
        bash "$wrapper_launcher" bash -c 'printf "%s|%s" "$LD_LIBRARY_PATH" "$RTP_LLM_RUN_UNDER_LIBRARY_PATH"'
)
actual_library_path="${actual%%|*}"
actual_run_under_path="${actual#*|}"

expected_prefix="${wrapper_torch_lib_dir}:${wrapper_cudnn_lib_dir}:"
if [[ "$actual_library_path" != "$expected_prefix"* ]]; then
    echo "wheel paths from the wrapper runfiles were not prepended to LD_LIBRARY_PATH" >&2
    echo "expected prefix: $expected_prefix" >&2
    echo "actual: $actual_library_path" >&2
    exit 1
fi

if [[ "$actual_run_under_path" != "$expected_prefix"* ]]; then
    echo "wrapper runfiles paths were not recorded separately" >&2
    echo "expected prefix: $expected_prefix" >&2
    echo "actual: $actual_run_under_path" >&2
    exit 1
fi
