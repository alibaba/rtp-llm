#!/bin/bash

# 参数说明
CURRENT_BRANCH=$1
PR_ID=$2
REPOSITORY=$3
COMMENT=$4
SOURCE=$5
MAIN_BRANCH=${6:-"main-internal"}  # 默认值为 main-internal
COMMITID=$7
PROJECT_ID="2654816"
MAX_RETRIES=5
RETRY_INTERVAL=5

# 参数验证
if [ -z "$CURRENT_BRANCH" ] || [ -z "$PR_ID" ] || [ -z "$REPOSITORY" ] || [ -z "$COMMENT" ] || [ -z "$SOURCE" ]; then
    echo "Usage: $0 <current_branch> <pr_id> <repository> <comment> <source> [main_branch]" >&2
    echo "Example: $0 feature-branch 123 owner/repo 'This is a comment' github main-internal" >&2
    echo "" >&2
    echo "Note: Branch info for both current_branch and main_branch must be cached in Redis first." >&2
    echo "      Use RETRIEVE-BRANCH-INFO to cache branch info before calling this script." >&2
    exit 1
fi

# 函数：添加评论（带重试机制）
add_comment() {
    local retry_count=0

    while [ $retry_count -lt $MAX_RETRIES ]; do
        echo "Adding comment to PR #$PR_ID (attempt $((retry_count + 1))/$MAX_RETRIES)" >&2

        local response=$(curl -s -H "Content-Type: application/json" \
                            -H "Authorization: Basic ${SECURITY}" \
                            -d "{
                                \"type\": \"ADD-COMMENT\",
                                \"aone\": {\"projectId\": \"${PROJECT_ID}\"},
                                \"currentBranch\": \"${CURRENT_BRANCH}\",
                                \"prId\": \"${PR_ID}\",
                                \"repositoryUrl\": \"${REPOSITORY}\",
                                \"commitId\": \"${COMMITID}\",
                                \"comment\": \"${COMMENT}\",
                                \"source\": \"${SOURCE}\",
                                \"mainBranch\": \"${MAIN_BRANCH}\"
                            }" "https://get-tasend-back-twkvcdsbpj.cn-hangzhou-vpc.fcapp.run")

        echo "Response: $response" >&2

        # 检查curl是否成功
        if [ $? -ne 0 ]; then
            echo "Error: Failed to add comment (attempt $((retry_count + 1)))" >&2
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $MAX_RETRIES ]; then
                echo "Retrying in ${RETRY_INTERVAL}s..." >&2
                sleep $RETRY_INTERVAL
            fi
            continue
        fi

        # 检查响应是否为空
        if [ -z "$response" ]; then
            echo "Error: Empty response from service (attempt $((retry_count + 1)))" >&2
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $MAX_RETRIES ]; then
                echo "Retrying in ${RETRY_INTERVAL}s..." >&2
                sleep $RETRY_INTERVAL
            fi
            continue
        fi

        # 解析响应
        local success=$(echo "$response" | jq -r '.success')
        if [ "$success" != "true" ]; then
            local error=$(echo "$response" | jq -r '.error // .errorMsg // "Unknown error"')

            # 检查是否是分支信息未找到的错误
            if echo "$error" | grep -q "Branch info not found"; then
                echo "Error: Branch info not cached in Redis" >&2
                echo "Please run RETRIEVE-BRANCH-INFO for branches: $CURRENT_BRANCH and $MAIN_BRANCH first" >&2
                return 1
            fi

            echo "Error: Failed to add comment - $error (attempt $((retry_count + 1)))" >&2
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $MAX_RETRIES ]; then
                echo "Retrying in ${RETRY_INTERVAL}s..." >&2
                sleep $RETRY_INTERVAL
            fi
            continue
        fi

        # 成功，处理响应
        local status=$(echo "$response" | jq -r '.status')
        local message=$(echo "$response" | jq -r '.message // empty')

        case "$status" in
            "SUCCESS")
                echo "✓ Comment added successfully" >&2
                local currentCommitId=$(echo "$response" | jq -r '.currentCommitId')
                local mainCommitId=$(echo "$response" | jq -r '.mainCommitId')
                echo "  Current branch ($CURRENT_BRANCH) commit: $currentCommitId" >&2
                echo "  Main branch ($MAIN_BRANCH) commit: $mainCommitId" >&2
                echo "$response" | jq '.'
                return 0
                ;;
            "SKIPPED")
                echo "○ Comment skipped: $message" >&2
                local commitId=$(echo "$response" | jq -r '.commitId')
                echo "  Both branches have the same commit ID: $commitId" >&2
                echo "$response" | jq '.'
                return 0
                ;;
            "FAILED")
                local error=$(echo "$response" | jq -r '.error')
                echo "Error: $error (attempt $((retry_count + 1)))" >&2
                retry_count=$((retry_count + 1))
                if [ $retry_count -lt $MAX_RETRIES ]; then
                    echo "Retrying in ${RETRY_INTERVAL}s..." >&2
                    sleep $RETRY_INTERVAL
                fi
                continue
                ;;
            *)
                echo "Unknown status: $status (attempt $((retry_count + 1)))" >&2
                retry_count=$((retry_count + 1))
                if [ $retry_count -lt $MAX_RETRIES ]; then
                    echo "Retrying in ${RETRY_INTERVAL}s..." >&2
                    sleep $RETRY_INTERVAL
                fi
                continue
                ;;
        esac
    done

    echo "Error: Failed to add comment after $MAX_RETRIES attempts" >&2
    return 1
}

# 主流程
echo "================================================" >&2
echo "Adding GitHub Comment" >&2
echo "================================================" >&2
echo "Repository: $REPOSITORY" >&2
echo "PR ID: $PR_ID" >&2
echo "Current Branch: $CURRENT_BRANCH" >&2
echo "Main Branch: $MAIN_BRANCH" >&2
echo "Source: $SOURCE" >&2
echo "================================================" >&2

if ! add_comment; then
    exit 1
fi

exit 0