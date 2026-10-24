#!/usr/bin/env bash
# Formats the lines of FlexLB Java files that differ from HEAD, using eclipse-java-formatter.xml.
# Files added after HEAD are formatted completely; lines unchanged since HEAD are left as written.
#
# Usage: format-changed-lines.sh [file ...]
# Without arguments it formats every changed or untracked Java file under rtp_llm/flexlb.
set -euo pipefail

tool_dir=$(cd "$(dirname "$0")" && pwd)
flexlb_dir=$(cd "$tool_dir/../.." && pwd)
repo_root=$(git -C "$flexlb_dir" rev-parse --show-toplevel)
classpath_file="$tool_dir/target/classpath.txt"

if [ ! -s "$classpath_file" ]; then
    (cd "$flexlb_dir" && ./mvnw -q -f "$tool_dir/pom.xml" dependency:build-classpath \
        -Dmdep.outputFile="$classpath_file")
fi

cd "$repo_root"
if [ $# -eq 0 ]; then
    files=$({
        git diff --name-only HEAD -- 'rtp_llm/flexlb/*.java'
        git ls-files --others --exclude-standard -- 'rtp_llm/flexlb/*.java'
    } | sort -u)
else
    files=$(for f in "$@"; do git ls-files --full-name --others --cached -- "$f"; done | sort -u)
fi

targets=()
for file in $files; do
    [ -f "$file" ] || continue
    if git cat-file -e "HEAD:$file" 2>/dev/null; then
        ranges=$(git diff -U0 HEAD -- "$file" \
            | sed -nE 's/^@@ -[0-9,]+ \+([0-9]+)(,([0-9]+))? @@.*/\1 \3/p' \
            | awk '{ n = ($2 == "" ? 1 : $2); if (n > 0) printf "%s%d-%d", (c++ ? "," : ""), $1, $1 + n - 1 }')
        [ -n "$ranges" ] || continue
        targets+=("$repo_root/$file:$ranges")
    else
        targets+=("$repo_root/$file:all")
    fi
done

[ ${#targets[@]} -gt 0 ] || exit 0
java -cp "$(cat "$classpath_file")" "$tool_dir/FormatChangedLines.java" \
    "$flexlb_dir/eclipse-java-formatter.xml" "${targets[@]}"
