#!/usr/bin/env bash
set -euo pipefail

wrapper="${TEST_SRCDIR}/${TEST_WORKSPACE}/tools/cache_perf_collect_env.sh"
work_dir="${TEST_TMPDIR}/cache_perf_collect_env_test"
deps_dir="${work_dir}/deps"
fake_python="${work_dir}/python3"
probe_state="${work_dir}/probe-ready"
python_log="${work_dir}/python.log"
command_output="${work_dir}/command.out"
command="${work_dir}/command.sh"

mkdir -p "$work_dir"

cat >"$fake_python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$FAKE_PYTHON_LOG"
if [[ "${1-}" == "-c" ]]; then
    [[ -f "$FAKE_PYTHON_PROBE_STATE" ]]
    exit
fi
if [[ "${1-}" == "-m" && "${2-}" == "pip" ]]; then
    touch "$FAKE_PYTHON_PROBE_STATE"
    exit 0
fi
echo "unexpected fake Python invocation: $*" >&2
exit 99
EOF
chmod +x "$fake_python"

cat >"$command" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$PYTHONPATH" >"$COMMAND_OUTPUT"
printf '%s\n' "$*" >>"$COMMAND_OUTPUT"
EOF
chmod +x "$command"

export CACHE_PERF_COLLECT_PYTHON="$fake_python"
export CACHE_PERF_COLLECT_DEPS_DIR="$deps_dir"
export CACHE_PERF_COLLECT_ARCH=x86_64
export FAKE_PYTHON_LOG="$python_log"
export FAKE_PYTHON_PROBE_STATE="$probe_state"
export COMMAND_OUTPUT="$command_output"

# A healthy environment must execute the command without invoking pip.
touch "$probe_state"
: >"$python_log"
PYTHONPATH=/sentinel "$wrapper" "$command" alpha beta
if grep -q -- '-m pip' "$python_log"; then
    echo "pip was invoked for an already healthy collect environment" >&2
    exit 1
fi
if [[ "$(sed -n '1p' "$command_output")" != "$deps_dir:/sentinel" ]]; then
    echo "collect dependency directory was not prepended to PYTHONPATH" >&2
    exit 1
fi
if [[ "$(sed -n '2p' "$command_output")" != "alpha beta" ]]; then
    echo "wrapped command arguments were not preserved" >&2
    exit 1
fi

# A missing environment must install the pinned CPU wheel and Plotly lock.
rm "$probe_state"
: >"$python_log"
PYTHONPATH=/sentinel "$wrapper" "$command" gamma
if [[ "$(grep -c -- '-m pip install' "$python_log")" -ne 2 ]]; then
    echo "expected separate Torch and Plotly pip installs" >&2
    cat "$python_log" >&2
    exit 1
fi
if ! grep -q 'torch-2.1.2%2Bcpu-cp310-cp310-linux_x86_64.whl' "$python_log"; then
    echo "x86_64 CPU Torch wheel was not pinned" >&2
    exit 1
fi
if ! grep -q -- '--require-hashes.*requirements_plotly_lock.txt' "$python_log"; then
    echo "Plotly lock was not installed with hash verification" >&2
    exit 1
fi

# An unsupported controller architecture must fail before running pip or command.
rm "$probe_state"
: >"$python_log"
rm "$command_output"
if CACHE_PERF_COLLECT_ARCH=ppc64le "$wrapper" "$command" 2>"${work_dir}/error.log"; then
    echo "unsupported architecture unexpectedly succeeded" >&2
    exit 1
fi
if ! grep -q 'unsupported cache-perf collect architecture: ppc64le' "${work_dir}/error.log"; then
    echo "unsupported architecture error was not actionable" >&2
    cat "${work_dir}/error.log" >&2
    exit 1
fi
if [[ -e "$command_output" ]]; then
    echo "wrapped command ran after architecture validation failed" >&2
    exit 1
fi
