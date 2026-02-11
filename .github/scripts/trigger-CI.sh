#!/bin/bash

set -x
# Check if two arguments are provided
if [ $# -ne 5 ]; then
    echo "Usage: $0 <COMMIT_ID> <SECURITY> <GITHUB_SOURCE_REPO> <GITHUB_PR_ID> <GITHUB_RUN_ID>"
    exit 1
fi

# Read commitId, repositoryUrl, and security from parameters or environment variables
COMMIT_ID=$1
SECURITY=$2
REPO_URL="https://github.com/${GITHUB_REPOSITORY}.git"
PROJECT_ID="2654816"
BRANCH_REF="main-internal"
CANCEL_IN_PROGRESS="true"
PIPELINE_ID="1346"
GITHUB_COMMIT_ID="${COMMIT_ID}"
GITHUB_SOURCE_REPO=$3
GITHUB_PR_ID=$4
BRANCH_NAME="open_merge/${GITHUB_PR_ID}"
CURRENT_INTERNAL_COMMITID="UNKNOWN"
CURRENT_GITHUB_RUN_ID=$5

# Call get-branch-info.sh to get CURRENT_INTERNAL_COMMITID
BRANCH_INFO=$(sh ./get-branch-info.sh "${BRANCH_NAME}" "${GITHUB_REPOSITORY}")
if [ $? -eq 0 ]; then
    CURRENT_INTERNAL_COMMITID=$(echo "$BRANCH_INFO" | jq -r '.commit.id')
fi


# Get current timestamp
timestamp=$(date +%s)

# Concatenate the parameters with timestamp
combined="${COMMIT_ID}${SECURITY}${timestamp}"

# Calculate the MD5 hash
base64_hash=$(echo -n "${combined}" | base64)

# Return the MD5 hash as the script's exit code
echo "${SECURITY}"

# 发送 CREATE-TASK 请求
curl -v -H "Content-Type: application/json" \
     -H "Authorization: Basic ${SECURITY}" \
     -d "{
            \"type\": \"CREATE-TASK\",
            \"commitId\": \"${COMMIT_ID}\",
            \"currentInternalCommitId\": \"${CURRENT_INTERNAL_COMMITID}\",
            \"repositoryUrl\": \"${REPO_URL}\",
            \"prId\": \"${GITHUB_PR_ID}\",
            \"aone\": { \"projectId\": \"${PROJECT_ID}\", \"pipelineId\": \"${PIPELINE_ID}\"},
            \"newBranch\": { \"name\": \"${BRANCH_NAME}\", \"ref\": \"${BRANCH_REF}\", \"head\": \"UNKNOWN\" },
            \"params\": {\"cancel-in-progress\": \"${CANCEL_IN_PROGRESS}\", \"github_commit\":\"${GITHUB_COMMIT_ID}\", \"github_source_repo\": \"${GITHUB_SOURCE_REPO}\",\"github_run_id\": \"${CURRENT_GITHUB_RUN_ID}\",\"aone_branch_name\": \"${BRANCH_NAME}\",\"aone_branch_ref\": \"${BRANCH_REF}\"}
         }" \
     "https://triggero-mq-pre-rbmuaqmqmz.cn-hangzhou.fcapp.run"