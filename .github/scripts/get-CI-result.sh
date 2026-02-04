#!/bin/bash

# 检查参数
if [ $# -ne 2 ]; then
    echo "Usage: $0 <COMMIT_ID> <SECURITY>"
    exit 1
fi

COMMIT_ID=$1
SECURITY=$2
PIPELINE_ID="1346"
PROJECT_ID="2654816"

# 设置最大等待时间
MAX_WAIT_TIME=7200
START_TIME=$(date +%s)

while true; do
    echo "Querying CI status for commitId: ${COMMIT_ID} ..."

    response=$(curl -s  -H "Content-Type: application/json" \
                        -H "Authorization: Basic ${SECURITY}" \
                        -d "{\"type\": \"RETRIEVE-TASK-STATUS\", \"aone\": { \"projectId\": \"${PROJECT_ID}\", \"pipelineId\": \"${PIPELINE_ID}\"}, \"commitId\": \"${COMMIT_ID}\"}" "https://get-tasend-back-twkvcdsbpj.cn-hangzhou-vpc.fcapp.run")
    echo "Response: $response"

    # 检查curl是否成功
    if [ $? -ne 0 ]; then
        echo "Error: Failed to query CI status!"
        exit 1
    fi

    # 检查响应是否为空
    if [ -z "$response" ]; then
        echo "Error: Empty response from CI service!"
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
    commitId=$(echo "$response" | jq -r '.commitId')
    taskId=$(echo "$response" | jq -r '.taskId')
    echo "Current commitId: $commitId, taskId: $taskId"

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

    status_summary=""
    if echo "$status_json" | jq empty 2>/dev/null; then
        status_summary=$(echo "$status_raw" | jq -c '.status')
    else
        status_summary="$status_json"
    fi
    main_status="UNKNOWN"
    if [[ "$status_json" == "null" ]]; then
        main_status="UNKNOWN"
    elif [[ "$status_summary" != "" ]]; then
        # 有 jobs 字段，说明是对象
        echo "Current status: $status_summary"
        # 首先检查是否有失败状态（最高优先级）
        if echo "$status_summary" | grep -qE "FAILED|ERROR|TIMEOUT|CANCELLED"; then
            main_status="FAILED"
        elif echo "$status_summary" | grep -q "RUNNING"; then
            main_status="RUNNING"
        elif echo "$status_summary" | grep -q "PENDING"; then
            main_status="PENDING"
        # 兼容旧逻辑：捕获其他未明确列举的失败状态
        elif echo "$status_summary" | sed 's/SUCCESS//g' | sed 's/NOT_RUN//g' | grep -q '[a-zA-Z]'; then
            main_status="FAILED"
        else
            main_status="DONE"
        fi
    else
        main_status="$status_json"
    fi

    echo "Current main status: $main_status"

    if [[ "$main_status" == "DONE" || "$main_status" == "FAILED"  ]]; then
        echo "Current status: $status_summary"
        if [[ "$main_status" == "DONE" ]]; then
            echo "CI completed successfully"
            exit 0
        else
            echo "CI failed with commitId: ${COMMIT_ID}, status: $status_summary, task link: https://code.alibaba-inc.com/foundation_models/RTP-LLM/ci/jobs?pipelineId=${PIPELINE_ID}&pipelineRunId=${taskId}&createType=yaml"
            exit 1
        fi
        break
    fi

    sleep 20
done