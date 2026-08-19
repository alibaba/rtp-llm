#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo "Usage: $0 <commit-sha> <security> <owner/repository> <branch>" >&2
}

[[ $# -eq 4 ]] || {
    usage
    exit 2
}

commit_sha="$1"
security="$2"
repository="$3"
branch="$4"

[[ "${commit_sha}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "invalid commit SHA" >&2
    exit 2
}
[[ "${repository}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || {
    echo "invalid GitHub repository" >&2
    exit 2
}
[[ "${branch}" == "feat/k3_dev" ]] || {
    echo "refusing unexpected branch: ${branch}" >&2
    exit 2
}

: "${K3_AONE_PIPELINE_ID:?K3_AONE_PIPELINE_ID is required}"
project_id="${K3_AONE_PROJECT_ID:-2654816}"
status_url="${K3_CI_STATUS_URL:-https://get-tasend-back-twkvcdsbpj.cn-hangzhou-vpc.fcapp.run}"
repository_url="https://github.com/${repository}.git"

payload="$({
    jq -n \
        --arg commit_id "${commit_sha}" \
        --arg repository_url "${repository_url}" \
        --arg project_id "${project_id}" \
        --arg pipeline_id "${K3_AONE_PIPELINE_ID}" \
        '{
            type: "RETRIEVE-TASK-STATUS",
            commitId: $commit_id,
            repositoryUrl: $repository_url,
            aone: {
                projectId: $project_id,
                pipelineId: $pipeline_id
            }
        }'
})"

response="$(curl --silent --show-error --fail-with-body \
    --header 'Content-Type: application/json' \
    --header "Authorization: Basic ${security}" \
    --data-binary "${payload}" \
    "${status_url}")"

task_id="$(jq -r '.taskId // .data.taskId // empty' <<<"${response}" 2>/dev/null || true)"
status="$(jq -c '.status // "UNKNOWN"' <<<"${response}" 2>/dev/null || printf '%s' '"UNKNOWN"')"
task_url="https://code.alibaba-inc.com/foundation_models/RTP-LLM/ci/jobs?pipelineId=${K3_AONE_PIPELINE_ID}${task_id:+&pipelineRunId=${task_id}}&createType=yaml"

echo "K3 CI status: commit=${commit_sha} status=${status}${task_id:+ task_id=${task_id}}"
echo "K3 CI task: ${task_url}"

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo "- Aone status: \`${status}\`"
        echo "- [Aone task](${task_url})"
    } >>"${GITHUB_STEP_SUMMARY}"
fi

if [[ "${K3_STATUS_ONCE:-0}" == "1" ]]; then
    exit 0
fi

case "${status}" in
    *SUCCESS*|*DONE*|*PASS*) exit 0 ;;
    *FAILED*|*ERROR*|*TIMEOUT*|*CANCELLED*|*CANCELED*) exit 1 ;;
    *) exit 3 ;;
esac
