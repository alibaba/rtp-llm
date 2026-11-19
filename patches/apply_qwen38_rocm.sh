#!/usr/bin/env bash
# Install the production ATOM 0.1.1.dev1+g0189b3506 overlays in dependency order.
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
python=${1:?usage: apply_qwen38_rocm.sh PYTHON FLYDSL_WHEEL}
wheel=${2:?provide the pinned FlyDSL wheel}
base=("$python" "$root/apply_qwen38_dependency_patches.py" --atom-main-dir "$root/atom/main_0189b3506" --flydsl-wheel "$wheel")
"${base[@]}"
"${base[@]}" --verify-only
site=$("$python" -c 'import importlib.metadata as m; print(m.distribution("atom").locate_file(""))')
for patch in topk512/topk512 draft_loop/draft_loop atom_rtp_model_api; do
    git -C "$site" --work-tree="$site" apply --check "$root/atom/$patch.patch"
    git -C "$site" --work-tree="$site" apply "$root/atom/$patch.patch"
    git -C "$site" --work-tree="$site" apply --reverse --check "$root/atom/$patch.patch"
done
