#!/bin/bash

# 检查参数
if [ $# -ne 2 ]; then
    echo "Usage: $0 <COMMIT_ID> <SECURITY>"
    exit 1
fi

COMMIT_ID=$1
SECURITY=$2

# 设置最大等待时间
MAX_WAIT_TIME=7200
START_TIME=$(date +%s)

while true; do
    echo "Querying CI status for commitId: ${COMMIT_ID} ..."

    response=$(curl -s  -H "Content-Type: application/json" \
                        -H "Authorization: Basic ${SECURITY}" \
                        -d "{\"type\": \"RETRIEVE-TASK-STATUS\", \"commitId\": \"${COMMIT_ID}\"}" "https://get-tasend-back-twkvcdsbpj.cn-hangzhou.fcapp.run")
    echo "Response: $response"

    # 检查curl是否成功
    if [ $? -ne 0 ]; then
        echo "Error: Failed to query CI status"
        exit 1
    fi

    # 检查响应是否为空
    if [ -z "$response" ]; then
        echo "Error: Empty response from CI service"
        exit 1
    fi

    # 检查是否超时
    CURRENT_TIME=$(date +%s)
    ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
    if [ $ELAPSED_TIME -gt $MAX_WAIT_TIME ]; then
        echo "Error: Timeout waiting for CI completion (waited ${ELAPSED_TIME} seconds)"
        exit 1
    fi

    status_raw=$(echo "$response" | jq -r '.status')

    # 判断 status_raw 是否为合法 JSON 对象
    if echo "$status_raw" | jq empty 2>/dev/null; then
        # 已经是对象，直接用
        status_json="$status_raw"
    elif [[ "$status_raw" =~ ^\{ ]]; then
        # 是字符串且以 { 开头，尝试 fromjson
        status_json=$(echo "$status_raw" | jq 'fromjson')
    else
        # 普通字符串
        status_json="$status_raw"
    fi

    main_status="UNKNOWN"
    if [[ "$status_json" == "null" ]]; then
        main_status="UNKNOWN"
    elif echo "$status_json" | jq -e '.jobs' >/dev/null 2>&1; then
        # 有 jobs 字段，说明是对象
        if echo "$status_json" | jq -e '.jobs[].status | select(. == "PENDING")' >/dev/null; then
            main_status="PENDING"
        elif echo "$status_json" | jq -e '.jobs[].status | select(. == "CANCELED")' >/dev/null; then
            main_status="CANCELED"
        elif echo "$status_json" | jq -e '.jobs[].status | select(. == "RUNNING")' >/dev/null; then
            main_status="RUNNING"
        elif echo "$status_json" | jq -e '.jobs[].status | select(. == "FAILED")' >/dev/null; then
            main_status="FAILED"
        elif echo "$status_json" | jq -e '.jobs[].status | select(. == "NOT_RUN")' >/dev/null; then
            main_status="NOT_RUN"
        elif echo "$status_json" | jq -e '.jobs[].status | select(. == "UNKNOWN")' >/dev/null; then
            main_status="UNKNOWN"
        elif [[ $(echo "$status_json" | jq '[.jobs[].status] | all(. == "SUCCESS")') == "true" ]]; then
            main_status="DONE"
        else
            main_status="UNKNOWN"
        fi
    else
        main_status="$status_json"
    fi

    echo "Current main status: $main_status"

    if [[ "$main_status" == "DONE"     || 
          "$main_status" == "FAILED"   || 
          "$main_status" == "UNKNOWN"  || 
          "$main_status" == "CANCELED" || 
          "$main_status" == "NOT_RUN"     ]]; then
        status_summary=$(echo "$status_raw" | jq -c '.status')
        echo "Current status: $status_summary"
        if [[ "$main_status" == "DONE" || "$main_status" == "NOT_RUN" ]]; then
            echo "CI completed successfully"
            exit 0
        else
            echo "CI failed with status: $main_status"
            exit 1
        fi
        break
    fi

    sleep 120
done