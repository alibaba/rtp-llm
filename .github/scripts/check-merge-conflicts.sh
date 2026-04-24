#!/bin/bash
# Check if PR has merge conflicts with main using GitHub API.
# Only blocks CI when there are actual textual conflicts.
# Being "behind main" is a non-blocking warning, not a blocker.
#
# Replaces check-rebase-freshness.sh which blocked CI if the PR was
# even 1 commit behind main -- causing a painful rebase loop.
#
# Usage: check-merge-conflicts.sh <PR_NUMBER> <COMMIT_ID> <REPOSITORY>
# Requires: GITHUB_TOKEN env var
set -euo pipefail

PR_NUMBER=$1
COMMIT_ID=$2
REPOSITORY=$3

MAX_RETRIES=3
RETRY_INTERVAL=5

# --- 1. Check PR mergeable status (conflict detection) ---

MERGEABLE="null"
for attempt in $(seq 1 $MAX_RETRIES); do
  PR_DATA=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
    "https://api.github.com/repos/${REPOSITORY}/pulls/${PR_NUMBER}")

  if [ $? -ne 0 ] || [ -z "$PR_DATA" ]; then
    echo "::warning::Failed to query GitHub PR API (attempt ${attempt}/${MAX_RETRIES})"
    if [ "$attempt" -lt "$MAX_RETRIES" ]; then
      sleep $RETRY_INTERVAL
      continue
    fi
    echo "::warning::All attempts failed, skipping conflict check"
    exit 0
  fi

  MERGEABLE=$(echo "$PR_DATA" | jq -r '.mergeable // "null"')
  MERGEABLE_STATE=$(echo "$PR_DATA" | jq -r '.mergeable_state // "unknown"')

  # GitHub computes mergeable asynchronously; null means "still computing"
  if [ "$MERGEABLE" = "null" ]; then
    echo "Mergeable status not yet computed (attempt ${attempt}/${MAX_RETRIES}), retrying..."
    sleep $RETRY_INTERVAL
    continue
  fi
  break
done

if [ "$MERGEABLE" = "false" ]; then
  echo "::error::PR #${PR_NUMBER} has merge conflicts with main. Please rebase or resolve conflicts."
  echo ""
  echo "The PR branch has textual merge conflicts with the main branch."
  echo "Run: git fetch origin main && git rebase origin/main"
  exit 1
fi

if [ "$MERGEABLE" = "null" ]; then
  echo "::warning::GitHub could not determine mergeable status for PR #${PR_NUMBER} after ${MAX_RETRIES} attempts, proceeding anyway"
fi

# --- 2. Staleness check (non-blocking warning) ---

COMPARE=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
  "https://api.github.com/repos/${REPOSITORY}/compare/${COMMIT_ID}...main")

if [ $? -eq 0 ] && [ -n "$COMPARE" ]; then
  BEHIND_BY=$(echo "$COMPARE" | jq '.ahead_by // 0')

  if [ "$BEHIND_BY" -gt 0 ] 2>/dev/null; then
    echo "::warning::PR #${PR_NUMBER} is ${BEHIND_BY} commit(s) behind main (no conflicts, CI will proceed)"
  else
    echo "PR #${PR_NUMBER} is up to date with main"
  fi
fi

exit 0
