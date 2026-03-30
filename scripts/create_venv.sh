#!/bin/bash
# Create a venv and install RTP-LLM deps. Platform wheels are resolved by setup.py
# (detect_build_config); this script only sets RTP_BAZEL_CONFIG when the venv name
# implies a target stack (optional hint on mixed hosts).
#
# Why does github-opensource/.venv sometimes appear?  `uv run` / `uv sync` default to a project-local
# .venv. This script installs into $HOME/venvs/<platform> instead. To align ad-hoc uv commands:
#   export UV_PROJECT_ENVIRONMENT=$HOME/venvs/rtp   # or your venv path
# Or remove stray envs: rm -rf .venv
#
# Usage (run INSIDE the target container):
#   bash scripts/create_venv.sh <cuda129|rocm|ppu>
#
set -euo pipefail

PLATFORM="${1:?Usage: $0 <cuda129|rocm|ppu>}"
UV=/home/liukan.lk/bin/uv
VENV="$HOME/venvs/${PLATFORM}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

case "$PLATFORM" in
    cuda129) export RTP_BAZEL_CONFIG='--config=cuda12_9' ;;
    rocm)    export RTP_BAZEL_CONFIG='--config=rocm' ;;
    ppu)     export RTP_BAZEL_CONFIG='--config=ppu' ;;
    *)       echo "Unknown platform: $PLATFORM"; exit 1 ;;
esac

INDEX_URL="https://artlab.alibaba-inc.com/1/PYPI/simple/"
EXTRA_INDEXES=(
    "http://artlab.alibaba-inc.com/1/pypi/py-central"
    "https://artlab.alibaba-inc.com/1/pypi/huiwa_rtp_internal"
    "http://artlab.alibaba-inc.com/1/pypi/rtp_diffusion"
    "https://artlab.alibaba-inc.com/1/PYPI/pytorch/whl"
)

UV_INDEX_ARGS=(
    --index-url "$INDEX_URL"
    --allow-insecure-host "artlab.alibaba-inc.com"
    --index-strategy unsafe-best-match
)
for idx in "${EXTRA_INDEXES[@]}"; do
    UV_INDEX_ARGS+=(--extra-index-url "$idx")
done

echo "=== Creating venv: $VENV (platform=$PLATFORM, RTP_BAZEL_CONFIG=$RTP_BAZEL_CONFIG) ==="

PYTHON3=$(command -v python3 || echo /opt/conda310/bin/python3)
echo "Using Python: $PYTHON3 ($($PYTHON3 --version 2>&1))"

$UV venv --python "$PYTHON3" --clear "$VENV"
echo "Venv created at $VENV"

$UV pip install --python "$VENV/bin/python3" "${UV_INDEX_ARGS[@]}" \
    'setuptools>=64.0,<82' wheel "tomli"
echo "Build deps installed"

cd "$REPO_DIR"
export RTP_SKIP_BAZEL_BUILD=1
export UV_SKIP_WHEEL_FILENAME_CHECK=1
$UV pip install --python "$VENV/bin/python3" --no-build-isolation \
    "${UV_INDEX_ARGS[@]}" \
    -e '.[dev]'
RTP_SKIP_BAZEL_BUILD=1 $UV pip install --python "$VENV/bin/python3" --no-build-isolation \
    "${UV_INDEX_ARGS[@]}" \
    -e . --no-deps
unset RTP_SKIP_BAZEL_BUILD

echo ""
echo "=== Done: $PLATFORM ==="
echo "Activate with: source $VENV/bin/activate"
echo "Packages installed: $($UV pip list --python "$VENV/bin/python3" 2>/dev/null | wc -l)"
