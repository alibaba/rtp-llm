#!/usr/bin/env bash
# Package the patched rtp-llm-rdma wheel into an immutable runtime for Scheme B.
# Run on the build machine after: bazelisk build //rtp_llm:rtp_llm --config=cuda12_9
set -Eeuo pipefail
SRC=/home/admin/project/rtp-llm-rdma
RUNTIME_PATH=${1:-/home/admin/rtp-hol/runtime/rtp-b-offload-$(date +%Y%m%d)}
BASE_RUNTIME=/home/admin/rtp-hol/runtime/rtp-rdma-stepmetrics-ae442576d9f6
stage="${RUNTIME_PATH}.staging.$$"
trap 'rm -rf "$stage"' EXIT

[[ -d "$RUNTIME_PATH" ]] && { echo "runtime exists: $RUNTIME_PATH"; exit 0; }
wheel="$(find -L "$SRC/bazel-bin/rtp_llm" -maxdepth 1 -type f -name 'rtp_llm-*.whl' -print -quit)"
test -s "$wheel"
mkdir -p "$stage/site-packages"
/opt/conda310/bin/python -m pip install --no-deps -q --target "$stage/site-packages" "$wheel"
/opt/conda310/bin/python -m pip install -q --target "$stage/site-packages" "pydantic==2.13.4"
so="$stage/site-packages/rtp_llm/libs/libth_transformer.so"
test -s "$so"
# carry over the offload python modules + hook import
D=/home/admin/workspace/aop_lab/app_source/latency_analysis/2026-08-12_dsv32-longctx-p1
cp "$D/v32_offload.py" "$D/v32_offload_hook.py" "$D/v32_capacity.py" "$stage/site-packages/rtp_llm/"
MLA="$stage/site-packages/rtp_llm/models_py/modules/hybrid/mla_attention.py"
grep -q v32_offload_hook "$MLA" || printf '\nimport rtp_llm.v32_offload_hook  # noqa: E402,F401\n' >> "$MLA"
sha256sum "$wheel" | awk '{print $1}' > "$stage/wheel.sha256"
git -C "$SRC" rev-parse HEAD > "$stage/source_commit.txt"
git -C "$SRC" diff > "$stage/source_worktree.diff"
mv "$stage" "$RUNTIME_PATH"
trap - EXIT
echo "RUNTIME_OK $RUNTIME_PATH (base A runtime untouched: $BASE_RUNTIME)"
