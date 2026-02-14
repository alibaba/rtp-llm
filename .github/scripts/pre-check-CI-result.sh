#!/bin/bash

# 检查参数
if [ $# -ne 3 ]; then
    echo "Usage: $0 <COMMIT_ID> <SECURITY> <REPOSITORY>"
    exit 1
fi

COMMIT_ID=$1
SECURITY=$2
REPOSITORY=$3
PIPELINE_ID="1346"
PROJECT_ID="2654816"
MAX_ATTEMPTS=6
SLEEP_INTERVAL=20

echo "Pre-checking CI status for commitId: ${COMMIT_ID} ..."
echo "Will check ${MAX_ATTEMPTS} times with ${SLEEP_INTERVAL}s interval"

# 循环检查 3 次
for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo ""
    echo "=== Attempt ${attempt}/${MAX_ATTEMPTS} ==="

    response=$(curl -s  -H "Content-Type: application/json" \
                        -H "Authorization: Basic ${SECURITY}" \
                        -d "{\"type\": \"RETRIEVE-TASK-STATUS\", \"aone\": { \"projectId\": \"${PROJECT_ID}\", \"pipelineId\": \"${PIPELINE_ID}\"}, \"commitId\": \"${COMMIT_ID}\",\"repositoryUrl\": \"${REPOSITORY}\"}" "https://get-tasback-pre-aiffqmsbgj.cn-hangzhou.fcapp.run")
    echo "Response: $response"

    # 检查curl是否成功
    if [ $? -ne 0 ]; then
        echo "Error: Failed to query CI status!"
        if [ $attempt -lt $MAX_ATTEMPTS ]; then
            echo "Will retry in ${SLEEP_INTERVAL} seconds..."
            sleep $SLEEP_INTERVAL
            continue
        else
            echo "All attempts failed, need to run CI"
            exit 1
        fi
    fi

    # 检查响应是否为空
    if [ -z "$response" ]; then
        echo "Error: Empty response from CI service!"
        if [ $attempt -lt $MAX_ATTEMPTS ]; then
            echo "Will retry in ${SLEEP_INTERVAL} seconds..."
            sleep $SLEEP_INTERVAL
            continue
        else
            echo "All attempts failed, need to run CI"
            exit 1
        fi
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
        if echo "$status_summary" | grep -q "PENDING"; then
            main_status="PENDING"
        elif echo "$status_summary" | grep -q "RUNNING"; then
            main_status="RUNNING"
        elif echo "$status_summary" | sed 's/SUCCESS//g' | sed 's/NOT_RUN//g' | grep -q '[a-zA-Z]'; then
            main_status="FAILED"
        else
            main_status="DONE"
        fi
    else
        main_status="$status_json"
    fi

    echo "Current main status: $main_status"

    # 如果状态为 DONE，立即返回成功
    if [[ "$main_status" == "DONE" ]]; then
        echo "CI already completed successfully for this commit"
        echo "Skipping CI trigger"
        exit 0
    fi

    # 如果不是最后一次尝试，等待后继续
    if [ $attempt -lt $MAX_ATTEMPTS ]; then
        echo "CI status is ${main_status}, will check again in ${SLEEP_INTERVAL} seconds..."
        sleep $SLEEP_INTERVAL
    fi
done

# 所有尝试都完成，但没有发现 DONE 状态
echo ""
echo "=== Final Result ==="
echo "After ${MAX_ATTEMPTS} attempts, CI is not in DONE status"
echo "Need to run CI"
exit 1
