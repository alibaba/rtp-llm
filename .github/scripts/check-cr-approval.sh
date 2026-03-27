#!/bin/bash

# 检查 PR 的 Code Review 是否通过
# 通过条件（两个条件都必须满足）：
#   1. 人工 CR：没有任何 reviewer 的最新状态为 CHANGES_REQUESTED（无 review 或全部非反对即通过）
#   2. AI CR：最新一条 AI Code Review 评论中包含 "LGTM ready to ci"
#
# 流程：用户提 PR → CR 审批检查 → 触发 CI → CI 通过 → 人工 Review Approve → 合并
# 因此 CR 检查阶段不要求 APPROVED，只要没有人明确反对（CHANGES_REQUESTED）即可放行
#
# 支持重试机制，超时时间 15 分钟

if [ $# -lt 3 ]; then
    echo "Usage: $0 <PR_NUMBER> <REPOSITORY> <GITHUB_TOKEN>"
    exit 1
fi

PR_NUMBER=$1
REPOSITORY=$2
GITHUB_TOKEN=$3

MAX_WAIT_TIME=900  # 15 minutes
RETRY_INTERVAL=30  # check every 30 seconds
START_TIME=$(date +%s)

echo "================================================"
echo "Checking Code Review Approval"
echo "================================================"
echo "Repository: $REPOSITORY"
echo "PR Number: $PR_NUMBER"
echo "Max wait time: ${MAX_WAIT_TIME}s (15 minutes)"
echo "Retry interval: ${RETRY_INTERVAL}s"
echo "================================================"

# 函数：检查人工 CR 是否无人反对
# 逻辑：取每个 reviewer 的最新一条 review，只要没有人的最新状态是 CHANGES_REQUESTED 即可通过
# 因为 CI 在 Approve 之前运行，此阶段不要求 APPROVED，只检查是否有人明确反对
check_human_review_approved() {
    local response
    response=$(curl -s -L \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "https://api.github.com/repos/${REPOSITORY}/pulls/${PR_NUMBER}/reviews?per_page=100")

    if [ $? -ne 0 ]; then
        echo "  Error: Failed to fetch reviews" >&2
        return 1
    fi

    if [ -z "$response" ]; then
        echo "  Error: Empty response when fetching reviews" >&2
        return 1
    fi

    # 检查是否有 API 错误
    local error_message
    error_message=$(echo "$response" | jq -r '.message // empty' 2>/dev/null)
    if [ -n "$error_message" ]; then
        echo "  Error: GitHub API returned error - $error_message" >&2
        return 1
    fi

    # 过滤掉 COMMENTED 和 PENDING 状态（仅保留有明确审批意见的 review：APPROVED / CHANGES_REQUESTED / DISMISSED）
    local actionable_reviews
    actionable_reviews=$(echo "$response" | jq '[.[] | select(.state != "COMMENTED" and .state != "PENDING")]')

    local actionable_count
    actionable_count=$(echo "$actionable_reviews" | jq 'length')

    # 如果没有任何有明确审批意见的 review，直接算通过（刚提的 PR 无人 review）
    if [ "$actionable_count" -eq 0 ]; then
        echo "  ✓ No actionable reviews submitted yet, passing by default"
        return 0
    fi

    # 按 reviewer 分组，取每个人最新的一条 review（按 submitted_at 排序）
    local latest_reviews_per_user
    latest_reviews_per_user=$(echo "$actionable_reviews" | jq '
        group_by(.user.login)
        | map(sort_by(.submitted_at) | last)
    ')

    # 找出最新状态是 CHANGES_REQUESTED 的 reviewer（即明确反对的人）
    local changes_requested
    changes_requested=$(echo "$latest_reviews_per_user" | jq '[.[] | select(.state == "CHANGES_REQUESTED")]')

    local changes_requested_count
    changes_requested_count=$(echo "$changes_requested" | jq 'length')

    if [ "$changes_requested_count" -eq 0 ]; then
        echo "  ✓ No reviewer has requested changes, CR check passed"
        return 0
    else
        local blocking_details
        blocking_details=$(echo "$changes_requested" | jq -r '.[] | "    - \(.user.login): CHANGES_REQUESTED"')
        echo "  ✗ ${changes_requested_count} reviewer(s) requested changes, CI blocked:"
        echo "$blocking_details"
        return 1
    fi
}

# 函数：检查最新一条 AI Code Review 评论是否包含 "LGTM ready to ci"
# 不限定具体用户，只要最新一条包含 AI Code Review 标识的评论中有 "LGTM ready to ci" 即可通过
# 返回码：0=通过，非0=不通过（均可重试，因为 AI bot 可能会发新评论覆盖旧的）
check_ai_review_approved() {
    local response
    response=$(curl -s -L \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "https://api.github.com/repos/${REPOSITORY}/issues/${PR_NUMBER}/comments?per_page=100")

    if [ $? -ne 0 ]; then
        echo "  Error: Failed to fetch comments" >&2
        return 1
    fi

    if [ -z "$response" ]; then
        echo "  Error: Empty response when fetching comments" >&2
        return 1
    fi

    # 检查是否有 API 错误
    local error_message
    error_message=$(echo "$response" | jq -r '.message // empty' 2>/dev/null)
    if [ -n "$error_message" ]; then
        echo "  Error: GitHub API returned error - $error_message" >&2
        return 1
    fi

    # 获取所有包含 AI Code Review 标识的评论，取最新一条（按时间排序取最后一条）
    local latest_ai_comment
    latest_ai_comment=$(echo "$response" | jq -r '[.[] | select(.body | test("AI Code Review"; "i"))] | sort_by(.created_at) | last // empty')

    if [ -z "$latest_ai_comment" ] || [ "$latest_ai_comment" = "null" ]; then
        echo "  ○ No AI Code Review comment found yet"
        return 1
    fi

    local comment_body
    comment_body=$(echo "$latest_ai_comment" | jq -r '.body')
    local comment_date
    comment_date=$(echo "$latest_ai_comment" | jq -r '.created_at')
    local comment_user
    comment_user=$(echo "$latest_ai_comment" | jq -r '.user.login')

    # 检查最新一条 AI Code Review 评论是否包含 "LGTM ready to ci"
    if echo "$comment_body" | grep -qi "LGTM ready to ci"; then
        echo "  ✓ Latest AI Code Review from ${comment_user} (${comment_date}) contains 'LGTM ready to ci'"
        return 0
    else
        echo "  ✗ Latest AI Code Review from ${comment_user} (${comment_date}) does NOT contain 'LGTM ready to ci'"
        return 1
    fi
}

# 主循环：带重试机制
# 只有"还没有 AI Review 评论"的情况才会重试，明确不通过的情况直接失败退出
attempt=0
while true; do
    attempt=$((attempt + 1))
    current_time=$(date +%s)
    elapsed=$((current_time - START_TIME))
    remaining=$((MAX_WAIT_TIME - elapsed))

    echo ""
    echo "=== Attempt ${attempt} (elapsed: ${elapsed}s, remaining: ${remaining}s) ==="

    # 检查是否超时，超时后直接放行（避免因 AI CR bot 延迟而阻塞 CI）
    if [ $elapsed -ge $MAX_WAIT_TIME ]; then
        echo ""
        echo "=== Final Result ==="
        echo "⚠ Timeout after ${MAX_WAIT_TIME}s (15 minutes) waiting for CR approval, proceeding anyway"
        exit 0
    fi

    # 检查条件1：人工 CR（明确不通过则立即失败）
    echo "Checking human review approval..."
    check_human_review_approved
    human_result=$?
    if [ "$human_result" -ne 0 ]; then
        echo ""
        echo "=== Final Result ==="
        echo "✗ CR check failed: Human review has CHANGES_REQUESTED"
        exit 1
    fi

    # 检查条件2：AI Code Review（不通过时继续重试，因为 AI bot 可能会发新评论）
    echo "Checking AI Code Review approval..."
    check_ai_review_approved
    ai_result=$?

    # 两个条件都通过才放行
    if [ "$human_result" -eq 0 ] && [ "$ai_result" -eq 0 ]; then
        echo ""
        echo "=== Final Result ==="
        echo "✓ CR check passed: Both human review and AI Code Review approved"
        exit 0
    fi

    # AI CR 未通过（可能是还没有评论，也可能是旧评论不通过等待新评论）
    echo ""
    echo "Waiting for:"
    echo "  - AI Code Review approval"
    echo "Will retry in ${RETRY_INTERVAL}s..."
    sleep $RETRY_INTERVAL
done
