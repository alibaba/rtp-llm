#!/usr/bin/env bash
set -euo pipefail

blocked=$(git diff --cached --name-only --no-renames --diff-filter=AMT -- \
    ':(top,glob)**/*.mckidx' \
    ':(top,exclude,literal)rtp_llm/flexlb/mock-framework/data/traffic/basic.mckidx')

if [[ -n "$blocked" ]]; then
    printf 'Staged replay datasets are blocked:\n%s\n' "$blocked" >&2
    printf '%s\n' 'Store local datasets under <shared-root>/data/traffic and use mock-case --shared-root.' >&2
    exit 1
fi
