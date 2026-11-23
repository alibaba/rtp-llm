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
apply_atom_overlay() {
    local patch="$root/atom/$1.patch"
    git -C "$site" --work-tree="$site" apply --check "$patch"
    git -C "$site" --work-tree="$site" apply "$patch"
    git -C "$site" --work-tree="$site" apply --reverse --check "$patch"
}
for patch in topk512/topk512 draft_loop/draft_loop atom_rtp_model_api main_0189b3506/mtp_hc_sp; do
    apply_atom_overlay "$patch"
done
for feature in ssm_state_compaction request_state_pool; do
    "$python" "$root/atom/$feature/apply.py"
    "$python" "$root/atom/$feature/apply.py" --verify-only
done
apply_atom_overlay main_0189b3506/native_pd_cache
apply_atom_overlay main_0189b3506/pd_single_verify_moe
apply_atom_overlay topk512/verify_route
# Build the MI308X verifier kernel before runtime graph capture.
route_source="$site/atom/plugin/rtpllm/utils/qwen4_verify_route"
"${ROCM_PATH:-/opt/rocm}/bin/hipcc" --genco -O3 --offload-arch=gfx942 \
    "$route_source.hip" -o "$route_source.hsaco"
