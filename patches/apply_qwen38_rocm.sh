#!/usr/bin/env bash
# Install the production ATOM 0.1.8 overlays in dependency order.
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
python=${1:?usage: apply_qwen38_rocm.sh PYTHON FLYDSL_WHEEL}
wheel=${2:?provide the pinned FlyDSL wheel}
base=("$python" "$root/apply_qwen38_dependency_patches.py" --atom-patch "$root/atom/qwen38_atom.patch" --flydsl-wheel "$wheel")
"${base[@]}"
"${base[@]}" --verify-only
for overlay in mtp_hc topk512 draft_loop ssm_state_compaction request_state_pool; do
    "$python" "$root/atom/$overlay/apply.py"
    "$python" "$root/atom/$overlay/apply.py" --verify-only
done
site=$("$python" -c 'import importlib.metadata as m; print(m.distribution("atom").locate_file(""))')
for patch in atom_rtp_model_api nonpd_prefill_sequence_parallel; do
    git -C "$site" --work-tree="$site" apply --check "$root/atom/$patch.patch"
    git -C "$site" --work-tree="$site" apply "$root/atom/$patch.patch"
    git -C "$site" --work-tree="$site" apply --reverse --check "$root/atom/$patch.patch"
done
