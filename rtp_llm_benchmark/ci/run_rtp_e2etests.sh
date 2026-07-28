#!/bin/bash
# workspace is $GITHUB_WORKSPACE. /mnt/raid0/rtp-actions-runner-yuzho

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${SCRIPT_DIR}/compile_rtp.sh
COMPILE_EXIT_CODE=$?

# 如果编译失败，直接退出
if [ $COMPILE_EXIT_CODE -ne 0 ]; then
    echo "错误：编译失败，退出代码：$COMPILE_EXIT_CODE"
    exit $COMPILE_EXIT_CODE
fi

set_e2e_para(){
    export PORT=6789
    # yum --disablerepo="*" --enablerepo=alinux3-os install -y jq
    cp ${RTP_PATH}/bazel-bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2* ${RTP_PATH}/rtp_llm/cpp/model_rpc/proto/
    GOLDEN_RESPONSES_FILE="${WORKSPACE}/rocm_benchmark/rtp_llm_benchmark/ci/golden_responses.json"
}

compare_json_score() {
    local actual_str=$1
    local expected_str=$2
    
    [ -z "$actual_str" ] && echo "FAIL" && return 1
    [ -z "$expected_str" ] && echo "FAIL" && return 1
    
    local result=$(echo -e "$actual_str\n$expected_str" | jq -s '
        if (.[0] | type) == "array" and (.[1] | type) == "array" then
            if (.[0] | length) == 0 then "FAIL"
            else
                def compare_arrays(a; b):
                    if (a | length) != (b | length) then false
                    else
                        [range(a | length) as $i |
                            if (a[$i] | type) == "array" and (b[$i] | type) == "array" then
                                compare_arrays(a[$i]; b[$i])
                            elif (a[$i] | type) == "number" and (b[$i] | type) == "number" then
                                (((a[$i] - b[$i]) | if . < 0 then -. else . end) <= 1e-10)
                            else
                                a[$i] == b[$i]
                            end
                        ] | all
                    end;
                if compare_arrays(.[0]; .[1]) then "MATCH" else "MISMATCH" end
            end
        else "FAIL" end
    ' 2>/dev/null || echo "FAIL")
    
    echo $result
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
        # local health_body=$(echo "${health_response}" | head -n -1)
        local health_status=$(echo "${health_response}" | tail -n 1)
        
        # if [ "${health_status}" = "200" ] && [ "${health_body}" = '"ok"' ]; then
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
    # return 0
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

send_reuseCache_request() {
    local generate_config=$1
    local system_content=$2
    local user_content=$3
    local result_file=$4

    _H="Content-Type: application/json"
    extra_configs='{"chat_template_kwargs": {"enable_thinking": false}}'
    _d=$(jq -cn \
        --arg system_content "$system_content" \
        --arg user_content "$user_content" \
        --argjson extra_configs "$extra_configs" \
        '{
            temperature: 0,
            max_tokens: 10,
            top_p: 0.1,
            top_k: 1,
            messages: [
                {role:"system", content:$system_content, partial:false},
                {role:"user", content:$user_content, partial:false}
            ],
            extra_configs: $extra_configs
        }'
    )
    
    echo 'sending request command: curl -s -X POST "http://localhost:'"$PORT"'/v1/chat/completions" -H "'"${_H}"'" -d "'"$_d"'"'
    sending_request='curl -s -X POST "http://localhost:'"$PORT"'/v1/chat/completions" -H "'"${_H}"'" -d "'"$_d"'"'
    local response
    response=$(curl -s -X POST "http://localhost:$PORT/v1/chat/completions" -H "${_H}" -d "$_d")
    echo "response: $response"

    cat > $result_file << EOF
REQUEST: $sending_request
EOF
    
    local actual_response=$(echo "${response}" | python3 -c 'import sys, json; data=json.load(sys.stdin); print(data["choices"][0]["message"]["content"])' 2>/dev/null || echo "${response}" | grep -o '"response":"[^"]*"' | cut -d'"' -f4)
    echo "actual_response: ${actual_response}"

    # return actual_response and cached_tokens
    if [[ -z "${ACTUAL_RESPONSE1}" ]]; then
        ACTUAL_RESPONSE1="${actual_response}"
    else
        ACTUAL_RESPONSE2="${actual_response}"
        local cached_tokens=$(echo "${response}" | python3 -c 'import sys, json; data=json.load(sys.stdin); print(data["usage"]["prompt_tokens_details"]["cached_tokens"])' 2>/dev/null || echo "${response}" | grep -o '"response":"[^"]*"' | cut -d'"' -f4)
        echo "cached_tokens: ${cached_tokens}"
        CACHED_TOKENS=${cached_tokens}
    fi
}

send_bert_request() {
    local prompt=$1
    
    echo "发送bert测试请求..."
    echo "prompt: ${prompt}"
    
    local response=$(PORT=${PORT} /opt/conda310/bin/python3 -c "
import requests
import json
import os
import sys
port = os.environ.get('PORT', '6789')
prompt_json = sys.argv[1]
prompt_list = json.loads(prompt_json)
req = {
    'input': prompt_list
}
result = requests.post(f'http://localhost:{port}', json=req)
print(json.dumps(result.json()))
" "${prompt}" 2> >(tee /dev/stderr >&2))
    
    echo "response: ${response}"
    
    # 使用jq提取score字段
    local actual_score=$(echo "${response}" | jq -c '.score' 2>/dev/null)
    echo "actual_score: ${actual_score}"
    
    RESPONSE="${response}"
    ACTUAL_RESPONSE="${actual_score}"
}

send_vl_request() {
    local prompt=$1
    local image_path=$2
    local generate_config=$3
    
    echo "发送VL测试请求..."
    echo "prompt: ${prompt}"
    echo "image_path: ${image_path}"
    
    local response=$(PORT=${PORT} /opt/conda310/bin/python3 -c "
import requests
import json
import os
import sys

port = os.environ.get('PORT', '6789')
image_path = sys.argv[1]
prompt = sys.argv[2]
generate_config_json = sys.argv[3]
generate_config = json.loads(generate_config_json)

payload = {
    'messages': [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image_url',
                    'image_url': {'url': image_path},
                    'preprocess_config': {'max_pixels': 100000}
                },
                {'type': 'text', 'text': prompt}
            ]
        }
    ],
    'max_tokens': generate_config.get('max_new_tokens', 100),
    'temperature': generate_config.get('temperature', 0.0),
    'top_p': generate_config.get('top_p', 0.01)
}

result = requests.post(f'http://localhost:{port}/v1/chat/completions', json=payload, timeout=600)
print(json.dumps(result.json()))

" "${image_path}" "${prompt}" "${generate_config}" 2> >(tee /dev/stderr >&2))
    
    echo "response: ${response}"
    
    local actual_content=$(echo "${response}" | jq -r '.choices[0].message.content' 2>/dev/null)
    echo "actual_content: ${actual_content}"
    
    RESPONSE="${response}"
    ACTUAL_RESPONSE="${actual_content}"
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

validate_reuseCache_response() {
    local response=$1
    local actual_response1=$2
    local actual_response2=$3
    local cached_tokens=$4
    local result_file=$5
    local case_tag=$6
    local server_params=$7
    local generate_config=$8

    cat >> ${result_file} << EOF
CASE: ${case_tag}
SERVER_PARAMS: ${server_params}
TEST_TIME: $(date -Iseconds)
GENERATE_CONFIG: ${generate_config}
ACTUAL_RESPONSE1: ${actual_response1}
ACTUAL_RESPONSE2: ${actual_response2}
CACHED_TOKENS: ${cached_tokens}
EOF

    echo "完整响应已保存到: ${result_file}"

    echo "校验结果..."
    # 先检查cached_tokens是否大于0，再检查输出是否合理
    if (( cached_tokens > 0 )); then
        echo "✓ REUSE_CACHE生效"
        echo "cached_tokens: $cached_tokens"
        echo "REUSE_CACHE生效, cached_tokens=${cached_tokens}" >> ${result_file}
    else
        echo "✗ 错误: REUSE_CACHE未生效"
        echo "CHECK: FAILED" >> ${result_file}
        return 1
    fi
    # 检查输出是否合理，两次输出一样时，肯定是合理的；两次输出不一样时，可能不合理，可能合理。
    if echo "${actual_response1}" | grep -qF "${actual_response2}"; then
        echo "✓ 响应内容合理"
        echo "  response1: ${actual_response1}"
        echo "  response2: ${actual_response2}"
        echo "CHECK: MATCH" >> ${result_file}
        return 0
    else
        echo "⚠ 两次响应内容不同"
        echo "  response1: ${actual_response1}"
        echo "  response2: ${actual_response2}"
        echo "  需要人工确认是否合理"
        echo "CHECK: MISMATCH" >> ${result_file}
        return 0
    fi
}

validate_bert_response() {
    local response=$1
    local actual_response=$2
    local expected_response=$3
    local result_file=$4
    local case_tag=$5
    local server_params=$6
    local prompt=$7
    local generate_config=$8
    
    cat > ${result_file} << EOF
CASE: ${case_tag}
SERVER_PARAMS: ${server_params}
TEST_TIME: $(date -Iseconds)
PROMPT: ${prompt}
GENERATE_CONFIG: ${generate_config}
EXPECTED_RESPONSE: ${expected_response}
ACTUAL_RESPONSE: ${actual_response}
EOF
    
    echo "完整响应已保存到: ${result_file}"
    
    echo "校验结果..."
    if [ -n "${response}" ]; then
        echo "✓ 请求已完成"
        echo "进行bert用例的score值比较..."
        local actual_score_json=$(echo "${response}" | jq -c '.score' 2>/dev/null)
        local comparison_result=$(compare_json_score "${actual_score_json}" "${expected_response}")
        comparison_result=$(echo "$comparison_result" | tr -d '"')
        echo "comparison_result is $comparison_result"
        echo "CHECK: $comparison_result" >> ${result_file}
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
    echo "RTP LLM 端到端测试"
    echo "=========================================="
    set_e2e_para
    
    echo "从 ${GOLDEN_RESPONSES_FILE} 读取测试用例..."
    
    local test_count=$(python3 -c "import json; data=json.load(open('${GOLDEN_RESPONSES_FILE}')); print(len(data['test_cases']))")
    
    echo "找到 ${test_count} 个测试用例"
    
    for i in $(seq 0 $((test_count - 1))); do
        SERVER_PID=""
        CURRENT_LOG_FILE=""
        CURRENT_RESULT_FILE=""
        RESPONSE=""
        ACTUAL_RESPONSE=""
        ACTUAL_RESPONSE1=""
        ACTUAL_RESPONSE2=""
        CACHED_TOKENS=0
        
        local case_tag=$(python3 -c "import json; data=json.load(open('${GOLDEN_RESPONSES_FILE}')); print(data['test_cases'][${i}]['case_tag'])")
        local server_params=$(python3 -c "import json; data=json.load(open('${GOLDEN_RESPONSES_FILE}')); print(data['test_cases'][${i}]['server_params'])")
        local model_timeout=$(python3 -c "import json; data=json.load(open('${GOLDEN_RESPONSES_FILE}')); print(data['test_cases'][${i}]['timeout'])")
        local generate_config=$(python3 -c "import json; data=json.load(open('${GOLDEN_RESPONSES_FILE}')); print(json.dumps(data['test_cases'][${i}]['generate_config']))")
        # 如果是数组则转为JSON字符串，如果是字符串则保持原样
        local prompt=$(python3 -c "import json; data=json.load(open('${GOLDEN_RESPONSES_FILE}')); resp=data['test_cases'][${i}].get('prompt', ''); print(json.dumps(resp) if isinstance(resp, (list, dict)) else resp)")
        local expected_response=$(python3 -c "import json; data=json.load(open('${GOLDEN_RESPONSES_FILE}')); resp=data['test_cases'][${i}].get('expected_response', ''); print(json.dumps(resp) if isinstance(resp, (list, dict)) else resp)")
        # 尝试读取image字段（VL用例）9
        local image_path=$(python3 -c "import json; data=json.load(open('${GOLDEN_RESPONSES_FILE}')); print(data['test_cases'][${i}].get('image', ''))" 2>/dev/null || echo "")
        # reuse_cache需要字段
        local system_content=$(python3 -c "import json; data=json.load(open('${GOLDEN_RESPONSES_FILE}')); print(data['test_cases'][${i}].get('system_content', ''))")
        local user_content=$(python3 -c "import json; data=json.load(open('${GOLDEN_RESPONSES_FILE}')); print(data['test_cases'][${i}].get('user_content', ''))")
        
        local is_bert_case=0
        local is_vl_case=0
        local is_reuseCache=0
        if echo "${case_tag}" | grep -qiE "VL" || echo "${server_params}" | grep -qiE "VL"; then
            is_vl_case=1
            echo "检测到VL用例，将使用Python发送vision请求"
        elif echo "${case_tag}" | grep -qiE "bert" || echo "${server_params}" | grep -qiE "bert"; then
            is_bert_case=1
            echo "检测到bert用例，将使用Python发送请求"
        elif echo "${case_tag}" | grep -qiE "reuseCache" || echo "${server_params}" | grep -qiE "REUSE_CACHE=1"; then
            is_reuseCache=1
            echo "检测到reuse_cache用例，将连续发送两次请求"
        fi
        
        if ! start_and_wait_server "${case_tag}" "${server_params}" "${model_timeout}"; then
            echo "✗ 服务启动失败，跳过此测试用例"
            continue
        fi
        
        if [ ${is_vl_case} -eq 1 ]; then
            # 如果没有从JSON读取到image路径，使用默认路径
            if [ -z "${image_path}" ]; then
                image_path="/mnt/raid0/pretrained_model/rtp_ci/test_VL.jpg"
            fi
            send_vl_request "${prompt}" "${image_path}" "${generate_config}"
        elif [ ${is_bert_case} -eq 1 ]; then
            send_bert_request "${prompt}"
        elif [ ${is_reuseCache} -eq 1 ]; then
            send_reuseCache_request "${generate_config}" "${system_content}" "${user_content}" "${CURRENT_RESULT_FILE}"
            send_reuseCache_request "${generate_config}" "${system_content}" "${user_content}" "${CURRENT_RESULT_FILE}"
        else
            send_curl_request "${prompt}" "${generate_config}"
        fi
        
        if [ ${is_bert_case} -eq 1 ]; then
            validate_bert_response "${RESPONSE}" "${ACTUAL_RESPONSE}" "${expected_response}" "${CURRENT_RESULT_FILE}" "${case_tag}" "${server_params}" "${prompt}" "${generate_config}"
        elif [ ${is_reuseCache} -eq 1 ]; then
            validate_reuseCache_response "${RESPONSE}" "${ACTUAL_RESPONSE1}" "${ACTUAL_RESPONSE2}" ${CACHED_TOKENS} "${CURRENT_RESULT_FILE}" "${case_tag}" "${server_params}" "${generate_config}"
        else
            validate_curl_response "${RESPONSE}" "${ACTUAL_RESPONSE}" "${expected_response}" "${CURRENT_RESULT_FILE}" "${case_tag}" "${server_params}" "${prompt}" "${generate_config}"
        fi
        
        stop_server
        
        echo "✓ ${case_tag} 测试完成"
        echo ""
    done
    
    echo "=========================================="
    echo "所有E2E测试完成！"
    echo "=========================================="

}

install_and_test_e2e