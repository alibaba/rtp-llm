#!/usr/bin/env bash
# Git pre-commit step: formats the changed lines of staged FlexLB Java files and stages the result.
# A staged file that also has unstaged edits is not formatted, because staging the formatted file
# would commit those edits too; the commit stops so the file can be fully staged or stashed first.
set -euo pipefail

tool_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

files=$(git diff --cached --name-only --diff-filter=ACMR -- 'rtp_llm/flexlb/*.java')
[ -n "$files" ] || exit 0

partially_staged=$(git diff --name-only -- $files)
if [ -n "$partially_staged" ]; then
    echo "format-changed-lines: these staged files also have unstaged edits:" >&2
    echo "$partially_staged" | sed 's/^/  /' >&2
    echo "Stage or stash those edits, or commit with --no-verify to skip formatting." >&2
    exit 1
fi

"$tool_dir/format-changed-lines.sh" $files
git add -- $files
