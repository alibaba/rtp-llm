#!/bin/bash
# Single-poll check of internal merge status for a given commit.
# Exit 0 ONLY when the internal merge is confirmed already successful
# (i.e. pre-merge-gate already did the work). Exit non-zero for anything
# else (pending, failed, not-found, network error) — caller should then
# proceed with its own merge trigger as a backstop.
#
# Usage: check-merge-done.sh <COMMIT_ID> <SECURITY> <REPOSITORY>

set -uo pipefail

if [ $# -ne 3 ]; then
    echo "Usage: $0 <COMMIT_ID> <SECURITY> <REPOSITORY>" >&2
    exit 2
fi

COMMIT_ID=$1
SECURITY=$2
REPOSITORY=$3

response=$(curl -s -H "Content-Type: application/json" \
    -H "Authorization: Basic ${SECURITY}" \
    -d "{\"type\": \"RETRIEVE-MERGE-STATUS\", \"repositoryUrl\": \"${REPOSITORY}\", \"commitId\": \"${COMMIT_ID}\"}" \
    "https://get-tasend-back-twkvcdsbpj.cn-hangzhou-vpc.fcapp.run")

if [ -z "$response" ]; then
    echo "Empty response from merge service — treating as not-merged" >&2
    exit 1
fi

echo "Response: $response" >&2

status_raw=$(echo "$response" | jq -r '.status' 2>/dev/null)
if [ -z "$status_raw" ] || [ "$status_raw" = "null" ]; then
    echo "No status in response — treating as not-merged" >&2
    exit 1
fi

if [ "$status_raw" = "PENDING" ]; then
    echo "Internal merge still PENDING — backstop should wait/retrigger" >&2
    exit 1
fi

success=$(echo "$status_raw" | jq -r '.success' 2>/dev/null)
if [ "$success" = "true" ]; then
    echo "Internal merge already completed successfully — skip backstop" >&2
    exit 0
fi

echo "Internal merge not in success state (success=$success) — backstop should run" >&2
exit 1
