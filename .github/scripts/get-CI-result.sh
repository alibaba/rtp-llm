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

    status=$(echo "$response" | jq -r '.status | if test("^\\{") then fromjson.status else . end')

    # 检查是否超时
    CURRENT_TIME=$(date +%s)
    ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
    if [ $ELAPSED_TIME -gt $MAX_WAIT_TIME ]; then
        echo "Error: Timeout waiting for CI completion (waited ${ELAPSED_TIME} seconds)"
        exit 1
    fi

    if [[ "$status" == "DONE" || "$status" == "FAILED" || "$status" == "UNKNOWN" || "$status" == "CANCELED" ]]; then
        echo "Current status: $status"
        if [[ "$status" == "DONE" ]]; then
            echo "CI completed successfully"
            exit 0
        elif [[ "$status" == "FAILED" ]]; then
            echo "CI failed"
            exit 1
        elif [[ "$status" == "UNKNOWN" ]]; then
            echo "CI status is unknown"
            exit 1
        elif [[ "$status" == "CANCELED" ]]; then
            echo "CI was canceled"
            exit 1
        else
            echo "Unexpected status: $status"
            exit 1
        fi
        break
    fi

    sleep 5
done