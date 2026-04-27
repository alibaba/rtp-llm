#!/bin/bash
# Upload large files to OSS and replace with pointer files.
# Usage: oss-lfs-push.sh <file1> [file2] ...
#   or:  oss-lfs-push.sh --all   (scan for matching patterns)
#
# =============================================================================
# OFFLINE-ONLY MAINTENANCE SCRIPT — DO NOT WIRE INTO CI OR PRODUCTION.
# =============================================================================
#
# Who runs this:
#   Only RTP-LLM team members, by hand on a personal developer machine, when
#   seeding/refreshing OSS objects (e.g. adding a new test fixture, rotating
#   a stale checkpoint). Never invoked by automation. The corresponding
#   read path (oss-lfs-pull.sh) is the credentialless one used by CI and
#   open-source contributors.
#
# Credentials on the command line — INTENTIONAL DESIGN, NOT A BUG:
#   $OSS_LFS_ACCESS_KEY_ID / $OSS_LFS_ACCESS_KEY_SECRET are passed via
#   `--access-key-id` / `--access-key-secret` to `ossutil`, so they appear
#   in /proc/<pid>/cmdline and `ps -ef` for the lifetime of each upload.
#   This is the simplest path that works with stock ossutil and is
#   acceptable here precisely because:
#     1. The script never runs in CI or on a shared box.
#     2. The keys belong to the operator personally, not a service.
#     3. Process listings on a personal dev box are not a meaningful threat.
#   If this script is ever adapted to run unattended (CI, cron, shared host),
#   STOP and switch to `ossutil --config-file <tempfile>` or an env-var-only
#   credential profile before doing so. Do not try to "fix" the cmdline
#   exposure piecemeal — the threat model changes the moment the operator
#   stops being the only viewer.
#
# Quoting on $OSS_COMMON_ARGS:
#   The variable is intentionally word-split (no quotes around expansion at
#   the call site). Access keys, endpoint, and region are alphanumeric/dot
#   strings by OSS spec — they do not contain whitespace or shell-special
#   characters, so word-splitting is safe and avoids the awkwardness of
#   building an array just for this one-line script.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/oss-lfs-config.sh"

OSS_PREFIX="oss://${OSS_LFS_BUCKET}/lfs"
OSS_COMMON_ARGS="--access-key-id $OSS_LFS_ACCESS_KEY_ID --access-key-secret $OSS_LFS_ACCESS_KEY_SECRET --endpoint $OSS_LFS_ENDPOINT --region $OSS_LFS_REGION"

is_pointer() {
    local file="$1"
    [ -f "$file" ] && head -1 "$file" 2>/dev/null | grep -q "^oss-lfs v1$"
}

push_file() {
    local file="$1"
    if is_pointer "$file"; then
        echo "  skip (already a pointer): $file"
        return 0
    fi

    local sha256
    sha256=$(sha256sum "$file" | awk '{print $1}')
    local size
    size=$(stat -c%s "$file")
    local oss_path="${OSS_PREFIX}/${sha256}"

    if ! ossutil stat "$oss_path" $OSS_COMMON_ARGS >/dev/null 2>&1; then
        echo "  upload: $file ($size bytes) -> $oss_path"
        ossutil cp "$file" "$oss_path" $OSS_COMMON_ARGS >/dev/null 2>&1
    else
        echo "  exists: $oss_path"
    fi

    cat > "$file" << EOF
oss-lfs v1
oss://${OSS_LFS_BUCKET}/lfs/${sha256}
sha256:${sha256}
size:${size}
EOF
    echo "  pointer: $file"
}

files=()
if [ "${1:-}" = "--all" ]; then
    while IFS= read -r -d '' f; do
        files+=("$f")
    done < <(find "$REPO_ROOT/rtp_llm/test/smoke/data" \( -name "*.pt" -o -name "*.bin" -o -name "*.model" -o -name "*.safetensors" \) -print0 2>/dev/null)
else
    files=("$@")
fi

if [ ${#files[@]} -eq 0 ]; then
    echo "oss-lfs-push: no files specified" >&2
    exit 1
fi

echo "oss-lfs: pushing ${#files[@]} file(s)..."
for f in "${files[@]}"; do
    push_file "$f"
done
echo "oss-lfs: done."
