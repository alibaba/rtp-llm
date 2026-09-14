#!/usr/bin/env bash
set -euo pipefail

# Run the MiMo V2.5 E2E test with a valid local ACCL-EP repository override.
# The public tree declares @accl_ep_rpm, while the actual DeepEP RPM is supplied
# by the internal build environment and is not available from the GitHub repo.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

is_rpm() {
    local candidate="$1"
    [[ -f "$candidate" ]] || return 1
    command -v rpm2cpio >/dev/null 2>&1 || {
        echo "rpm2cpio is required to validate ACCL-EP RPMs" >&2
        return 1
    }
    rpm2cpio "$candidate" >/dev/null 2>&1
}

find_accl_ep_rpm() {
    local candidate

    # Explicit paths take priority when the package is outside the Bazel cache.
    for candidate in \
        "${ACCL_EP_RPM_PATH:-}" \
        "${ACCL_EP_REPO:-}/file/file" \
        "${ACCL_EP_REPO:-}/file/downloaded"; do
        [[ -n "$candidate" ]] || continue
        if is_rpm "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    # A previous successful build normally leaves the downloaded DeepEP RPM in
    # one of these Bazel external-repository caches.
    local cache_root
    for cache_root in \
        /root/.cache/bazel \
        /data1/renkun.ren/.cache/bazel \
        /home/*/.cache/bazel; do
        [[ -d "$cache_root" ]] || continue
        while IFS= read -r -d '' candidate; do
            if is_rpm "$candidate"; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done < <(
            find "$cache_root" -type f \
                -path '*/external/accl_ep_rpm/file/*' -print0 2>/dev/null
        )
    done

    return 1
}

ACCL_EP_RPM_PATH="$(find_accl_ep_rpm || true)"
if [[ -z "$ACCL_EP_RPM_PATH" ]]; then
    cat >&2 <<'EOF'
Unable to find a valid ACCL-EP/DeepEP RPM.
The old placeholder file is not usable: 3rdparty/accl_ep extracts it with rpm2cpio.
Provide the real package with:
  ACCL_EP_RPM_PATH=/path/to/DeepEP-*.rpm ./run_mimo_v25_test.sh
EOF
    exit 1
fi

# Keep a stable repository directory so the --override_repository argument does
# not need to be assembled manually. Do not overwrite an unrelated directory.
ACCL_EP_REPO_DIR="${ACCL_EP_REPO_DIR:-/tmp/mimo_v25_accl_ep_rpm}"
if [[ -e "$ACCL_EP_REPO_DIR" && ! -f "$ACCL_EP_REPO_DIR/.managed_by_mimo_v25" ]]; then
    ACCL_EP_REPO_DIR="$(mktemp -d /tmp/mimo_v25_accl_ep.XXXXXX)"
fi
mkdir -p "$ACCL_EP_REPO_DIR/file"
printf 'managed by run_mimo_v25_test.sh\n' > "$ACCL_EP_REPO_DIR/.managed_by_mimo_v25"
printf 'workspace(name = "accl_ep_rpm")\n' > "$ACCL_EP_REPO_DIR/WORKSPACE"
cat > "$ACCL_EP_REPO_DIR/file/BUILD" <<'EOF'
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "file",
    srcs = ["downloaded"],
)
EOF
ln -sfn "$ACCL_EP_RPM_PATH" "$ACCL_EP_REPO_DIR/file/downloaded"

echo "Using ACCL-EP RPM: $ACCL_EP_RPM_PATH"
echo "Using Bazel repository override: $ACCL_EP_REPO_DIR"

# Bazel's @cutlass_h_moe fetch can stall while GitHub prepares/transfers the
# pack for this exact historical commit. Reuse a complete local CUTLASS clone
# and create the repository shape that new_git_repository normally produces.
CUTLASS_H_MOE_COMMIT="19b4c5e065e7e5bbc8082dfc7dbd792bdac850fc"
CUTLASS_SOURCE_REPO="${CUTLASS_SOURCE_REPO:-$SCRIPT_DIR/cutlass}"
CUTLASS_H_MOE_REPO_DIR="${CUTLASS_H_MOE_REPO_DIR:-/tmp/mimo_v25_cutlass_h_moe}"

if ! git -C "$CUTLASS_SOURCE_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    cat >&2 <<EOF
Unable to find the local CUTLASS clone at:
  $CUTLASS_SOURCE_REPO
Clone https://github.com/NVIDIA/cutlass.git there, or set:
  CUTLASS_SOURCE_REPO=/path/to/cutlass ./run_mimo_v25_test.sh
EOF
    exit 1
fi

if ! git -C "$CUTLASS_SOURCE_REPO" cat-file -e "${CUTLASS_H_MOE_COMMIT}^{commit}" 2>/dev/null; then
    cat >&2 <<EOF
The local CUTLASS clone does not contain the required commit:
  $CUTLASS_H_MOE_COMMIT
Fetch that commit into $CUTLASS_SOURCE_REPO before running this script.
EOF
    exit 1
fi

prepare_cutlass_h_moe_repo() {
    local repo_dir="$CUTLASS_H_MOE_REPO_DIR"

    # Never overwrite a directory that was not created by this script.
    if [[ -e "$repo_dir" && ! -f "$repo_dir/.managed_by_mimo_v25" ]]; then
        repo_dir="$(mktemp -d /tmp/mimo_v25_cutlass_h_moe.XXXXXX)"
    fi

    # A stale managed checkout is left intact; create a fresh one beside it.
    if [[ -f "$repo_dir/.managed_by_mimo_v25" ]] &&
       ! git -C "$repo_dir" rev-parse --verify "$CUTLASS_H_MOE_COMMIT^{commit}" >/dev/null 2>&1; then
        repo_dir="$(mktemp -d /tmp/mimo_v25_cutlass_h_moe.XXXXXX)"
    fi

    if [[ ! -d "$repo_dir/.git" ]]; then
        git clone --shared --no-checkout "$CUTLASS_SOURCE_REPO" "$repo_dir"
        git -C "$repo_dir" checkout --detach "$CUTLASS_H_MOE_COMMIT"
    elif [[ "$(git -C "$repo_dir" rev-parse HEAD)" != "$CUTLASS_H_MOE_COMMIT" ]]; then
        repo_dir="$(mktemp -d /tmp/mimo_v25_cutlass_h_moe.XXXXXX)"
        git clone --shared --no-checkout "$CUTLASS_SOURCE_REPO" "$repo_dir"
        git -C "$repo_dir" checkout --detach "$CUTLASS_H_MOE_COMMIT"
    fi

    # new_git_repository injects this BUILD file after cloning. A local
    # repository override must provide the equivalent file itself.
    cp "$SCRIPT_DIR/3rdparty/cutlass/cutlass.BUILD" "$repo_dir/BUILD.bazel"
    printf 'workspace(name = "cutlass_h_moe")\n' > "$repo_dir/WORKSPACE"
    printf 'managed by run_mimo_v25_test.sh\n' > "$repo_dir/.managed_by_mimo_v25"
    CUTLASS_H_MOE_REPO_DIR="$repo_dir"
}

prepare_cutlass_h_moe_repo
echo "Using local CUTLASS H-MoE override: $CUTLASS_H_MOE_REPO_DIR"

MIMO_TEST_TARGET="${MIMO_TEST_TARGET:-//rtp_llm/test/model_test:test_mimo_v25}"

exec bazelisk test "$MIMO_TEST_TARGET" \
    --override_repository=accl_ep_rpm="$ACCL_EP_REPO_DIR" \
    --override_repository=cutlass_h_moe="$CUTLASS_H_MOE_REPO_DIR" \
    --verbose_failures \
    --config=cuda12_9 \
    --define=use_remote_kv_cache=false \
    --test_output=all \
    --cache_test_results=no \
    --test_env="TP_SIZE=4" \
    --test_env="LOG_LEVEL=INFO" \
    --test_env="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    --test_env="CHECKPOINT_PATH=${CHECKPOINT_PATH:-/home/renkun.ren/models/MiMo-V2.5}" \
    --test_env="REUSE_CACHE=1" \
    --test_env="GSM8K_NUM_EXAMPLES=${GSM8K_NUM_EXAMPLES:-200}" \
    --test_env="GSM8K_NUM_THREADS=${GSM8K_NUM_THREADS:-8}" \
    --test_env="GSM8K_MAX_TOKENS=${GSM8K_MAX_TOKENS:-4096}" \
    --test_env="GSM8K_MAX_SEQ_LEN=${GSM8K_MAX_SEQ_LEN:-8192}" \
    --test_env="GSM8K_DATA_PATH=${GSM8K_DATA_PATH:-/home/renkun.ren/dataset/gsm8k_test.jsonl}" \
    --test_env="MIMO_BENCHMARK_LOG_DIR=${MIMO_BENCHMARK_LOG_DIR:-/home/renkun.ren/log/mimov25}" \
    --jobs="${BAZEL_JOBS:-64}"
