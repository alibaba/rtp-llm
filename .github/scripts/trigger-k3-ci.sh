#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo "Usage: $0 <commit-sha> <security> <owner/repository> <branch> <github-run-id> <github-run-attempt>" >&2
}

[[ $# -eq 6 ]] || {
    usage
    exit 2
}

commit_sha="$1"
security="$2"
repository="$3"
branch="$4"
github_run_id="$5"
github_run_attempt="$6"

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
[[ "${github_run_id}" =~ ^[0-9]+$ && "${github_run_attempt}" =~ ^[0-9]+$ ]] || {
    echo "invalid GitHub run identity" >&2
    exit 2
}

: "${K3_AONE_PIPELINE_ID:?K3_AONE_PIPELINE_ID is required}"
[[ "${K3_AONE_PIPELINE_ID}" =~ ^[0-9]+$ ]] || {
    echo "K3_AONE_PIPELINE_ID must be numeric" >&2
    exit 2
}

project_id="${K3_AONE_PROJECT_ID:-2654816}"
trigger_url="${K3_CI_TRIGGER_URL:-https://triggerid-to-mq-wjrdhcgbie.cn-hangzhou-vpc.fcapp.run}"
repository_url="https://github.com/${repository}.git"
request_id="k3-ci:${repository}:${branch}:${commit_sha}"
aone_task_url="https://code.alibaba-inc.com/foundation_models/RTP-LLM/ci/jobs?pipelineId=${K3_AONE_PIPELINE_ID}&createType=yaml"

payload="$({
    jq -n \
        --arg commit_id "${commit_sha}" \
        --arg repository_url "${repository_url}" \
        --arg repository "${repository}" \
        --arg branch "${branch}" \
        --arg github_run_id "${github_run_id}" \
        --arg github_run_attempt "${github_run_attempt}" \
        --arg request_id "${request_id}" \
        --arg aone_task_url "${aone_task_url}" \
        --arg project_id "${project_id}" \
        --arg pipeline_id "${K3_AONE_PIPELINE_ID}" \
        '{
            type: "CREATE-TASK",
            commitId: $commit_id,
            repositoryUrl: $repository_url,
            prId: "0",
            aone: {
                projectId: $project_id,
                pipelineId: $pipeline_id
            },
            params: {
                "cancel-in-progress": "false",
                github_commit: $commit_id,
                github_branch: $branch,
                github_repository: $repository,
                github_run_id: $github_run_id,
                github_run_attempt: $github_run_attempt,
                k3_request_id: $request_id,
                commit_sha: $commit_id,
                branch: $branch,
                repository: $repository,
                request_id: $request_id,
                aone_task_url: $aone_task_url,
                k3_prefill_host: "106",
                k3_decode_host: "142"
            }
        }'
})"

response_file="$(mktemp)"
trap 'rm -f "${response_file}"' EXIT

http_code="$(curl --silent --show-error \
    --output "${response_file}" \
    --write-out '%{http_code}' \
    --header 'Content-Type: application/json' \
    --header "Authorization: Basic ${security}" \
    --data-binary "${payload}" \
    "${trigger_url}")"

case "${http_code}" in
    2??) ;;
    *)
        echo "K3 CI enqueue failed with HTTP ${http_code}" >&2
        sed -n '1,40p' "${response_file}" >&2
        exit 1
        ;;
esac

task_id="$(jq -r '.taskId // .data.taskId // .id // empty' "${response_file}" 2>/dev/null || true)"
echo "K3 CI accepted: request_id=${request_id} commit=${commit_sha} pipeline=${K3_AONE_PIPELINE_ID}${task_id:+ task_id=${task_id}}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    printf 'request_id=%s\n' "${request_id}" >>"${GITHUB_OUTPUT}"
    printf 'task_id=%s\n' "${task_id}" >>"${GITHUB_OUTPUT}"
fi
