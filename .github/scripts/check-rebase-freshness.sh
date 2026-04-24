#!/bin/bash
# Check if PR branch is up to date with main using GitHub API.
# Usage: check-rebase-freshness.sh <PR_NUMBER> <COMMIT_ID> <REPOSITORY>
# Requires: GITHUB_TOKEN env var
set -euo pipefail

PR_NUMBER=$1
COMMIT_ID=$2
REPOSITORY=$3

COMPARE=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
  "https://api.github.com/repos/${REPOSITORY}/compare/${COMMIT_ID}...main")

if [ $? -ne 0 ] || [ -z "$COMPARE" ]; then
  echo "::warning::Failed to query GitHub compare API, skipping rebase check"
  exit 0
fi

BEHIND_BY=$(echo "$COMPARE" | jq '.ahead_by // 0')

if [ "$BEHIND_BY" -eq 0 ] 2>/dev/null; then
  echo "PR #${PR_NUMBER} is up to date with main"
  exit 0
fi

echo "::error::PR #${PR_NUMBER} is ${BEHIND_BY} commit(s) behind main. Please rebase before CI can run."
exit 1
