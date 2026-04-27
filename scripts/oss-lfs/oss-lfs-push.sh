#!/bin/bash
# Upload large files to OSS and replace with pointer files.
# Usage: oss-lfs-push.sh <file1> [file2] ...
#   or:  oss-lfs-push.sh --all   (scan for matching patterns)
#
# OFFLINE-ONLY MAINTENANCE SCRIPT — not invoked by CI or production.
# Credentials are passed on the ossutil command line for simplicity, which
# means they are visible in /proc/<pid>/cmdline and `ps` while the upload
# runs. This is acceptable here because the script is only ever run by hand
# on a developer machine to seed/refresh OSS objects; it must not be wired
# into any automated pipeline. If that ever changes, switch to ossutil
# config file (--config-file) or environment variables.
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
