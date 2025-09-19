#!/bin/bash

# Check if two arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <COMMIT_ID> <SECURITY>"
    exit 1
fi

# Read commitId, repositoryUrl, and security from parameters or environment variables
COMMIT_ID=$1
SECURITY=$2
REPO_URL="https://github.com/${GITHUB_REPOSITORY}.git"
PROJECT_ID="2654816"
BRANCH_REF="main-internal"
BRANCH_NAME="open_merge_${COMMIT_ID}"
CANCEL_IN_PROGRESS="true"
PIPELINE_ID="1346"

# Get current timestamp
timestamp=$(date +%s)

# Concatenate the parameters with timestamp
combined="${COMMIT_ID}${SECURITY}${timestamp}"

# Calculate the MD5 hash
base64_hash=$(echo -n "${combined}" | base64)

# Output the MD5 hash
echo "MD5 hash of '${COMMIT_ID}' and '${SECURITY}' combined with timestamp is: ${md5_hash}"

# Return the MD5 hash as the script's exit code
echo "${SECURITY}"

# 发送 CREATE-TASK 请求
curl -v -H "Content-Type: application/json" \
     -H "Authorization: Basic ${SECURITY}" \
     -d "{
            \"type\": \"CREATE-TASK\",
            \"commitId\": \"${COMMIT_ID}\",
            \"repositoryUrl\": \"${REPO_URL}\",
            \"aone\": { \"projectId\": \"${PROJECT_ID}\" },
            \"newBranch\": { \"name\": \"${BRANCH_NAME}\", \"ref\": \"${BRANCH_REF}\" },
            \"pipelineId\": \"${PIPELINE_ID}\",
            \"params\": {\"cancel-in-progress\": \"${CANCEL_IN_PROGRESS}\"}
         }" \
     "https://triggerid-to-mq-wjrdhcgbie.cn-hangzhou.fcapp.run"