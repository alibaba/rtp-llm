#!/bin/bash

# 检查参数
if [ $# -ne 2 ]; then
    echo "Usage: $0 <COMMIT_ID> <SECURITY>"
    exit 1
fi

COMMIT_ID=$1
SECURITY=$2


while true; do
    echo "Querying CI status for commitId: ${COMMIT_ID} ..."

    response=$(curl -s  -H "Content-Type: application/json" \
                        -H "Authorization: Basic ${SECURITY}" \
                        -d "{\"type\": \"RETRIEVE-TASK-STATUS\", \"commitId\": \"${COMMIT_ID}\"}" "https://get-tasend-back-twkvcdsbpj.cn-hangzhou.fcapp.run")
    echo "Response: $response"

    status=$(echo "$response" | jq -r '.status | if test("^\\{") then fromjson.status else . end')

    if [[ "$status" == "DONE" || "$status" == "FAILED" || "$status" == "UNKNOWN" || "$status" == "CANCELED" ]]; then
        echo "Current status: $status"
        if [[ "$status" == "DONE" ]]; then
            echo "CI completed successfully"
        elif [[ "$status" == "FAILED" ]]; then
            echo "CI failed"
        elif [[ "$status" == "UNKNOWN" ]]; then
            echo "CI status is unknown"
        elif [[ "$status" == "CANCELED" ]]; then
            echo "CI was canceled"
        else
            echo "Unexpected status: $status"
        fi
        break
    fi

    sleep 5
done