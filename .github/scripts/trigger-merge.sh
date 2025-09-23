#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <COMMIT_ID> <SECURITY>"
    exit 1
fi

COMMIT_ID=$1
SECURITY=$2
REPO_URL="https://github.com/${GITHUB_REPOSITORY}.git"
AONE_PROJECT_ID="2654816"
AUTHOR_NAME=$(git log -1 --pretty=format:%an ${COMMIT_ID})
AUTHOR_EMAIL=$(git log -1 --pretty=format:%ae ${COMMIT_ID})
MERGE_MESSAGE="auto-merge: github commit $(git log -1 --pretty=format:%B ${COMMIT_ID})"
MERGE_TYPE="SQUASH"
SOURCE_BRANCH="open_merge/${COMMIT_ID}"
TARGET_BRANCH="main"

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


JSON_BODY=$(cat <<EOF
{
  "type": "MERGE-TASK",
  "repositoryUrl": "${REPO_URL}",
  "commitId": "${COMMIT_ID}",
  "aone": {
    "projectId": "${AONE_PROJECT_ID}"
  },
  "authorEmail": "${AUTHOR_EMAIL}",
  "authorName": "${AUTHOR_NAME}",
  "mergeMessage": "${MERGE_MESSAGE}",
  "mergeType": "${MERGE_TYPE}",
  "sourceBranch": "${SOURCE_BRANCH}",
  "targetBranch": "${TARGET_BRANCH}"
}
EOF
)

echo "Sending MERGE-TASK for commitId: ${COMMIT_ID}"

# 调用 HTTP 函数发送消息
curl -v -H "Content-Type: application/json" \
     -H "Authorization: Basic ${SECURITY}" \
     -d "${JSON_BODY}" \
     "https://triggerid-to-mq-wjrdhcgbie.cn-hangzhou.fcapp.run"