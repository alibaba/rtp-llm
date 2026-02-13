#!/bin/bash

BRANCH_NAME=$1
REPOSITORY=$2
COMMITID=$3
PROJECT_ID="2654816"

# 记录开始时间
START_TIME=$(date +%s)
END_TIME=$((START_TIME + 120))  # 2分钟后的时间戳

while [ $(date +%s) -lt $END_TIME ]; do

    response=$(curl -s  -H "Content-Type: application/json" \
                        -H "Authorization: Basic ${SECURITY}" \
                        -d "{\"type\": \"RETRIEVE-BRANCH-INFO\",\"commitId\": \"${COMMITID}\",\"repositoryUrl\": \"${REPOSITORY}\", \"aone\": { \"projectId\": \"${PROJECT_ID}\"}, \"branchName\": \"${BRANCH_NAME}\", \"clearCache\": \"false\"}" "https://get-tasend-back-twkvcdsbpj.cn-hangzhou-vpc.fcapp.run")
    echo "Response: $response"  >&2
    # 检查curl是否成功
    if [ $? -ne 0 ]; then
        echo "Error: Failed to query CI status" >&2
        exit 1
    fi

    # 检查响应是否为空
    if [ -z "$response" ]; then
        echo "Error: Empty response from CI service" >&2
        exit 1
    fi

    # 使用jq解析响应
    success=$(echo "$response" | jq -r '.success')
    if [ "$success" != "true" ]; then
        errorCode=$(echo "$response" | jq -r '.errorCode')
        if [ "$errorCode" = "SYSTEM_NOT_FOUND_ERROR" ]; then
            echo "Error: Branch not found" >&2
        else
            errorMsg=$(echo "$response" | jq -r '.errorMsg')
            echo "Error: Failed to query CI status - $errorMsg" >&2
        fi
        exit 1
    fi

    # 获取分支信息的命令（这里需要根据实际需求填写）
    info_raw=$(echo "$response" | jq -r '.internal_branch_info')

    # 检查分支信息是否为UNKNOWN
    if [ "$info_raw" != "UNKNOWN" ] && [ -n "$info_raw" ]; then
        echo "$info_raw"
        exit 0
    fi
    echo "Waiting for branch info..." >&2
    # 等待一段时间再重试（避免过于频繁的尝试）
    sleep 5
done

echo "Timeout: Could not retrieve valid branch info within 2 minutes" >&2
exit 1