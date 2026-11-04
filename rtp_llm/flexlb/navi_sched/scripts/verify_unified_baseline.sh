#!/usr/bin/env bash

set -euo pipefail

readonly REQUIRED_BRANCH=navi_sched
readonly MODEL_HEADER=aios/navi_sched/resource/NonLinearPrefillModelR.h
readonly VLLM_ADAPTER=aios/navi_sched/navi_sched/adapters/vllm/VLLMEngineObserver.py

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "$REPO_ROOT"

git rev-parse --verify HEAD >/dev/null
source_branch=${NAVI_SCHED_SOURCE_BRANCH:-$(git branch --show-current)}
if [[ "$source_branch" != "$REQUIRED_BRANCH" ]]; then
    printf 'NaviSched builds require branch %s, got %s\n' \
        "$REQUIRED_BRANCH" "${source_branch:-detached HEAD}" >&2
    exit 1
fi
test -f "$VLLM_ADAPTER"
grep -Eq 'PARAMETER_COUNT[[:space:]]*=[[:space:]]*10;' "$MODEL_HEADER"
grep -Eq 'CAPACITY_SPLIT_PARAMETER_COUNT[[:space:]]*=[[:space:]]*19;' \
    "$MODEL_HEADER"
grep -Eq 'BINARY_VERSION[[:space:]]*=[[:space:]]*24;' "$MODEL_HEADER"

printf 'Verified NaviSched source branch: %s\n' "$source_branch"
printf 'Verified NaviSched source commit: %s\n' "$(git rev-parse HEAD)"
printf 'Verified latency model: PARAMETER_COUNT=10 CAPACITY_SPLIT_PARAMETER_COUNT=19 BINARY_VERSION=24\n'
