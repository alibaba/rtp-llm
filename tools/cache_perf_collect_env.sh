#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
python=${CACHE_PERF_COLLECT_PYTHON:-python3}
deps_dir=${CACHE_PERF_COLLECT_DEPS_DIR:-${TMPDIR:-/tmp}/rtp_llm_cache_perf_collect_deps}
plotly_lock="${repo_root}/rtp_llm/test/perf_test/requirements_plotly_lock.txt"
original_pythonpath=${PYTHONPATH-}
collect_pythonpath="$deps_dir"
if [[ -n "$original_pythonpath" ]]; then
    collect_pythonpath="${collect_pythonpath}:${original_pythonpath}"
fi

probe='import plotly, torch; assert plotly.__version__ == "7.1.0"; assert torch.__version__.split("+")[0] == "2.1.2"; assert torch.version.cuda is None; torch.empty(1, device="cpu")'

if ! PYTHONPATH="$collect_pythonpath" "$python" -c "$probe"; then
    arch=${CACHE_PERF_COLLECT_ARCH:-$(uname -m)}
    case "$arch" in
        x86_64)
            torch_wheel='https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.1.2%2Bcpu-cp310-cp310-linux_x86_64.whl#sha256=bf3ca897f8c7c218dd6c4b1cc5eec57b4f4e71106b0b8120e92f5fdaf4acf6cd'
            ;;
        aarch64)
            torch_wheel='https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.1.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl#sha256=bef6996c27d8f6e92ea4e13a772d89611da0e103b48790de78131e308cf73076'
            ;;
        *)
            echo "unsupported cache-perf collect architecture: $arch" >&2
            exit 2
            ;;
    esac

    echo "Preparing isolated cache-perf collect dependencies in $deps_dir"
    mkdir -p "$deps_dir"
    "$python" -m pip install \
        --target "$deps_dir" \
        --upgrade \
        --disable-pip-version-check \
        --no-input \
        --index-url https://artlab.alibaba-inc.com/1/PYPI/simple/ \
        --trusted-host artlab.alibaba-inc.com \
        "$torch_wheel"
    "$python" -m pip install \
        --target "$deps_dir" \
        --upgrade \
        --disable-pip-version-check \
        --no-input \
        --require-hashes \
        -r "$plotly_lock"

    PYTHONPATH="$collect_pythonpath" "$python" -c "$probe"
fi

export PYTHONPATH="$collect_pythonpath"
exec "$@"
