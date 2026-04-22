#!/bin/bash
# Download large files from OSS by reading pointer files.
# Uses public HTTP — no credentials needed.
# Usage: oss-lfs-pull.sh [file1] [file2] ...
#   or:  oss-lfs-pull.sh          (scan for all pointer files)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OSS_HTTP_BASE="https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/lfs"
MAX_PARALLEL="${OSS_LFS_PARALLEL:-8}"

is_pointer() {
    local file="$1"
    [ -f "$file" ] && [ "$(wc -l < "$file")" -le 5 ] && head -1 "$file" 2>/dev/null | grep -q "^oss-lfs v1$"
}

pull_file() {
    local file="$1"
    local sha256 size
    sha256=$(sed -n '3p' "$file" | sed 's/^sha256://')
    size=$(sed -n '4p' "$file" | sed 's/^size://')
    local url="${OSS_HTTP_BASE}/${sha256}"
    local tmp="${file}.oss-lfs-tmp"

    if ! curl -sS -f -o "$tmp" "$url"; then
        echo "  FAIL: $file (download failed)" >&2
        rm -f "$tmp"
        return 1
    fi

    local actual_sha256
    actual_sha256=$(sha256sum "$tmp" | awk '{print $1}')
    if [ "$actual_sha256" != "$sha256" ]; then
        echo "  FAIL: $file (sha256 mismatch)" >&2
        rm -f "$tmp"
        return 1
    fi

    mv "$tmp" "$file"
    echo "  ok: $file ($size B)"
}

files=()
if [ $# -gt 0 ]; then
    files=("$@")
else
    while IFS= read -r -d '' f; do
        files+=("$f")
    done < <(find "$REPO_ROOT/rtp_llm/test/smoke/data" \( -name "*.pt" -o -name "*.bin" -o -name "*.model" -o -name "*.safetensors" \) -print0 2>/dev/null)
fi

pointers=()
for f in "${files[@]}"; do
    if is_pointer "$f"; then
        pointers+=("$f")
    fi
done

if [ ${#pointers[@]} -eq 0 ]; then
    exit 0
fi

echo "oss-lfs: pulling ${#pointers[@]} file(s) (${MAX_PARALLEL} parallel)..."

failed=0
if [ ${#pointers[@]} -le 1 ] || ! command -v xargs >/dev/null 2>&1; then
    for f in "${pointers[@]}"; do
        pull_file "$f" || failed=$((failed + 1))
    done
else
    export -f pull_file is_pointer
    export OSS_HTTP_BASE
    printf '%s\0' "${pointers[@]}" | xargs -0 -P "$MAX_PARALLEL" -I{} bash -c 'pull_file "$@"' _ {} || failed=1
fi

if [ "$failed" -gt 0 ]; then
    echo "oss-lfs: some files failed to download." >&2
    exit 1
fi

# Mark downloaded files as assume-unchanged so git status stays clean
for f in "${pointers[@]}"; do
    git -C "$REPO_ROOT" update-index --assume-unchanged "$f" 2>/dev/null || true
done

echo "oss-lfs: done (${#pointers[@]} files)."
