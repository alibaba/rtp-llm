#!/bin/bash
# workspace is $GITHUB_WORKSPACE. /mnt/raid0/rtp-actions-runner-yuzho


compile_rtp(){
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    source ${SCRIPT_DIR}/compile_rtp.sh
    COMPILE_EXIT_CODE=$?

    # 如果编译失败，直接退出
    if [ $COMPILE_EXIT_CODE -ne 0 ]; then
        echo "错误：编译失败，退出代码：$COMPILE_EXIT_CODE"
        exit $COMPILE_EXIT_CODE
    fi
}


set_e2e_para(){
    export PORT=8011
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    #export RTP_PATH=/mnt/raid0/junyyang/rtp-llm
    #export WORKSPACE=/mnt/raid0/junyyang
    # yum --disablerepo="*" --enablerepo=alinux3-os install -y jq
    cp ${RTP_PATH}/bazel-bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2* ${RTP_PATH}/rtp_llm/cpp/model_rpc/proto/
    REGRESSION_CASES_FILE="${WORKSPACE}/rocm_benchmark/rtp_llm_benchmark/ci/regression_cases.json"
}

start_and_wait_server() {
    local case_tag=$1
    local server_params=$2
    local model_timeout=$3
    
    CURRENT_LOG_FILE="${LOG_DIR}/server_${case_tag}.log"
    CURRENT_RESULT_FILE="${LOG_DIR}/client_${case_tag}.log"
    
    echo "=========================================="
    echo "测试用例: ${case_tag}"
    echo "=========================================="
    
    echo "启动服务中..."
    local server_cmd="${server_params} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} START_PORT=${PORT} /opt/conda310/bin/python3.10 -m rtp_llm.start_server"
    echo "running server command: ${server_cmd}" | tee -a ${CURRENT_LOG_FILE}
    eval "${server_cmd} >> ${CURRENT_LOG_FILE} 2>&1 &"
    SERVER_PID=$!
    
    echo "等待服务就绪（健康检查: http://localhost:${PORT}/health）..."

    local waited=0
    while [ $waited -lt $model_timeout ]; do
        # 通过健康检查端点判断服务是否就绪
        local health_response=$(curl -s -w "\n%{http_code}" http://localhost:${PORT}/health 2>/dev/null)
        local health_status=$(echo "${health_response}" | tail -n 1)
        
        if [ "${health_status}" = "200" ]; then
            echo "✓ 服务已就绪（已等待 ${waited} 秒）"
            sleep 20
            break
        fi
        sleep 5
        waited=$((waited + 5))
        echo "已等待 ${waited} 秒... (健康检查状态码: ${health_status:-"连接失败"})"
    done
    
    if [ $waited -ge $model_timeout ]; then
        echo "✗ 错误: 服务启动超时（${model_timeout}秒内健康检查未通过）"
        stop_server
        
        return 1
    fi
    
    sleep 10
}

send_curl_request() {
    local prompt=$1
    local generate_config=$2
    
    # 构建请求payload
    local request_payload="{\"prompt\": \"${prompt}\", \"generate_config\":${generate_config}}"
    echo "request_payload: ${request_payload}"
    
    # 发送请求
    echo "发送测试请求..."
    local response=$(curl -XPOST http://localhost:${PORT} -d "${request_payload}")
    echo "response: ${response}"
    
    # 提取response字段
    local actual_response=$(echo "${response}" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('response', ''))" 2>/dev/null || echo "${response}" | grep -o '"response":"[^"]*"' | cut -d'"' -f4)
    echo "actual_response: ${actual_response}"
    
    # 返回response和actual_response（通过全局变量）
    RESPONSE="${response}"
    ACTUAL_RESPONSE="${actual_response}"
}

validate_curl_response() {
    local response=$1
    local actual_response=$2
    local expected_response=$3
    local result_file=$4
    local case_tag=$5
    local server_params=$6
    local prompt=$7
    local generate_config=$8
    
    # 以便在日志文件中单行显示
    local actual_response_single_line=$(echo "${actual_response}" | tr '\n' ' ')
    
    cat > ${result_file} << EOF
CASE: ${case_tag}
SERVER_PARAMS: ${server_params}
TEST_TIME: $(date -Iseconds)
PROMPT: ${prompt}
GENERATE_CONFIG: ${generate_config}
EXPECTED_RESPONSE: ${expected_response}
ACTUAL_RESPONSE: ${actual_response_single_line}
EOF
    
    echo "完整响应已保存到: ${result_file}"
    
    # 校验结果
    echo "校验结果..."
    if echo "${response}"; then
        echo "✓ 请求成功完成"
        
        # 检查 actual_response 是否为空（包括仅包含空格）
        local actual_response_trimmed=$(echo "${actual_response}" | tr -d '[:space:]')
        local expected_response_trimmed=$(echo "${expected_response}" | tr -d '[:space:]')
        if [ -z "${actual_response_trimmed}" ]; then
            echo "✗ 错误: 响应内容为空"
            echo "CHECK: FAILED" >> ${result_file}
            return 1
        elif [ -n "${expected_response_trimmed}" ]; then
            if echo "${actual_response_trimmed}" | grep -qF "${expected_response_trimmed}"; then
                echo "✓ 响应内容匹配预期"
                echo "  预期: ${expected_response}"
                echo "  实际: ${actual_response:0:100}..."
                echo "CHECK: MATCH" >> ${result_file}
                return 0
            else
                echo "⚠ 响应内容与预期不同"
                echo "  预期: ${expected_response}"
                echo "  实际: ${actual_response:0:100}..."
                echo "  需要人工确认是否合理"
                echo "CHECK: MISMATCH" >> ${result_file}
                return 0
            fi
        fi
    else
        echo "✗ 请求失败或未完成"
        echo "CHECK: FAILED" >> ${result_file}
        return 1
    fi
}

stop_server() {
    if [ -n "${SERVER_PID}" ]; then
        echo "停止服务 (PID: ${SERVER_PID})及其子进程..."
        ps -ef | grep backend | grep ${SERVER_PID} | awk '{print $2}' | xargs -r kill -9
        # for multi process cases:
        pstree -p ${SERVER_PID} | grep -E "rtp_llm_backend_server|rtp_llm_rank" | grep -oP '\(\d+\)' | grep -oP '\d+' | xargs -r kill -9
        sleep 30
    fi
}

install_and_test_e2e() {
    echo "=========================================="
    echo "RTP LLM 回归测试"
    echo "=========================================="
    set_e2e_para
    
    echo "从 ${REGRESSION_CASES_FILE} 读取测试用例..."
    
    local test_count=$(python3 -c "import json; data=json.load(open('${REGRESSION_CASES_FILE}')); print(len(data['test_cases']))")
    
    echo "找到 ${test_count} 个测试用例"
    
    for i in $(seq 0 $((test_count - 1))); do
        SERVER_PID=""
        CURRENT_LOG_FILE=""
        CURRENT_RESULT_FILE=""
        RESPONSE=""
        ACTUAL_RESPONSE=""
        
        local case_tag=$(python3 -c "import json; data=json.load(open('${REGRESSION_CASES_FILE}')); print(data['test_cases'][${i}]['case_tag'])")
        local server_params=$(python3 -c "import json; data=json.load(open('${REGRESSION_CASES_FILE}')); print(data['test_cases'][${i}]['server_params'])")
        local model_timeout=$(python3 -c "import json; data=json.load(open('${REGRESSION_CASES_FILE}')); print(data['test_cases'][${i}]['timeout'])")
        local generate_config=$(python3 -c "import json; data=json.load(open('${REGRESSION_CASES_FILE}')); print(json.dumps(data['test_cases'][${i}]['generate_config']))")
        # 如果是数组则转为JSON字符串，如果是字符串则保持原样
        local prompt=$(python3 -c "import json; data=json.load(open('${REGRESSION_CASES_FILE}')); resp=data['test_cases'][${i}].get('prompt', ''); print(json.dumps(resp) if isinstance(resp, (list, dict)) else resp)")
        local expected_response=$(python3 -c "import json; data=json.load(open('${REGRESSION_CASES_FILE}')); resp=data['test_cases'][${i}].get('expected_response', ''); print(json.dumps(resp) if isinstance(resp, (list, dict)) else resp)")
        
        if ! start_and_wait_server "${case_tag}" "${server_params}" "${model_timeout}"; then
            echo "✗ 服务启动失败，跳过此测试用例"
            continue
        fi
        
        # 发请求
        send_curl_request "${prompt}" "${generate_config}"
        # 验证结果
        validate_curl_response "${RESPONSE}" "${ACTUAL_RESPONSE}" "${expected_response}" "${CURRENT_RESULT_FILE}" "${case_tag}" "${server_params}" "${prompt}" "${generate_config}"
        # 停止服务
        stop_server
        
        echo "✓ ${case_tag} 测试完成"
        echo ""
    done
    
    echo "=========================================="
    echo "所有回归案例测试完成！"
    echo "=========================================="
}

compile_rtp
install_and_test_e2e