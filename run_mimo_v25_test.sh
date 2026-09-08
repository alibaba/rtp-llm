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

exec bazelisk test //rtp_llm/test/model_test:test_mimo_v25 \
    --override_repository=accl_ep_rpm="$ACCL_EP_REPO_DIR" \
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
    --jobs="${BAZEL_JOBS:-64}"
