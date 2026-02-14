#!/bin/bash

# 检查参数
if [ $# -ne 3 ]; then
    echo "Usage: $0 <COMMIT_ID> <SECURITY> <REPOSITORY>"
    exit 1
fi

COMMIT_ID=$1
SECURITY=$2
REPOSITORY=$3

# 设置最大等待时间 2h
MAX_WAIT_TIME=7200
START_TIME=$(date +%s)

while true; do
    echo "Querying merge status for commitId: ${COMMIT_ID} ..."

    response=$(curl -s  -H "Content-Type: application/json" \
                        -H "Authorization: Basic ${SECURITY}" \
                        -d "{\"type\": \"RETRIEVE-MERGE-STATUS\", \"repositoryUrl\": \"${REPOSITORY}\", \"commitId\": \"${COMMIT_ID}\"}" "https://get-tasend-back-twkvcdsbpj.cn-hangzhou-vpc.fcapp.run")

    # 检查curl是否成功
    if [ $? -ne 0 ]; then
        echo "Error: Failed to query merge status"
        exit 1
    fi

    # 检查响应是否为空
    if [ -z "$response" ]; then
        echo "Error: Empty response from merge service"
        exit 1
    fi

    # 检查是否超时
    CURRENT_TIME=$(date +%s)
    ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
    if [ $ELAPSED_TIME -gt $MAX_WAIT_TIME ]; then
        echo "Error: Timeout waiting for merge completion (waited ${ELAPSED_TIME} seconds)"
        exit 1
    fi

    status_json=$(echo "$response" | jq -r '.status')

    if [ "$status_json" == "PENDING" ]; then
        echo "Merge is still pending..."
        sleep 5
        continue
    else
        success=$(echo "$status_json" | jq -r '.success')
        echo "Merge completed with status: $success"
        echo "Response: $response"
        if [ "$success" = "true" ]; then
            exit 0
        else
            exit 1
        fi
        break
    fi
done