#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

LOG_DIR=${1}
if [[ -z "${LOG_DIR}" ]]; then
    echo "ERROR: The input LOG_DIR must be set, but is not set."
    exit 1
fi

summarize_results() {
    echo "| case_name | status | duration |" > "${LOG_DIR}/summary.md"
    echo "|-----------|--------|----------|" >> "${LOG_DIR}/summary.md"

    # 读取注册的test cases并检查状态
    local case_index=1
    if [ -f "${LOG_DIR}/test_cases_registry.txt" ]; then
        while IFS= read -r case_entry; do
            # case_entry 格式为 "case_name in logfile_path"
            case_name=$(echo "$case_entry" | sed 's/ in .*//')
            display_name=$(echo "$case_name" | sed 's/.*://')
            logname=$(echo "$case_entry" | sed 's/.* in //')
            logfile="${LOG_DIR}/${logname}"
            
            # 检查logfile中是否有同时包含case_name和"PASSED in"的行
            if [ -f "$logfile" ]; then
                # 使用grep查找包含case_name的行，然后检查该行是否也包含"PASSED in"
                passed_line=$(grep -F "$case_name" "$logfile" | grep "PASSED in" | head -n1)
                
                if [ -n "$passed_line" ]; then
                    # 提取时间（最后一个字段）
                    duration=$(echo "$passed_line" | awk '{print $NF}')
                    echo "| ${case_index}. ${display_name} (in ${logname}) | PASSED | ${duration} |" >> "${LOG_DIR}/summary.md"
                else
                    echo "| ${case_index}. ${display_name} (in ${logname}) | FAILED | N/A |" >> "${LOG_DIR}/summary.md"
                fi
            else
                echo "| ${case_index}. ${display_name} (in ${logname}) | FAILED | N/A |" >> "${LOG_DIR}/summary.md"
            fi
            
            case_index=$((case_index + 1))
        done < "${LOG_DIR}/test_cases_registry.txt"
    fi

    # summarize E2E cases
    # 先检查有 server 日志但没有 client 日志的情况（失败）
    for server_file in ${LOG_DIR}/server_*.log; do
        if [ -f "${server_file}" ]; then
            model_name=$(basename "${server_file}" .log | sed 's/^server_//')
            client_file="${LOG_DIR}/client_${model_name}.log"
            
            if [ ! -f "${client_file}" ]; then
                echo "| ${model_name} | FAILED | - |" >> "${LOG_DIR}/summary.md"
            fi
        fi
    done
    
    # 再处理有 client 日志的情况
    for client_file in ${LOG_DIR}/client_*.log; do
        if [ -f "${client_file}" ]; then
            model_name=$(basename "${client_file}" .log | sed 's/^client_//')
            
            if grep -q "CHECK: FAILED" "${client_file}"; then
                echo "| ${model_name} | FAILED | - |" >> "${LOG_DIR}/summary.md"
            elif grep -q "CHECK: MATCH" "${client_file}"; then
                echo "| ${model_name} | MATCH | - |" >> "${LOG_DIR}/summary.md"
            elif grep -q "CHECK: MISMATCH" "${client_file}"; then
                echo "| ${model_name} | MISMATCH | - |" >> "${LOG_DIR}/summary.md"
            else
                echo "| ${model_name} | UNKNOWN | - |" >> "${LOG_DIR}/summary.md"
            fi
        fi
    done

    # pretty-print
    awk -F'|' '
    {
        for(i=2;i<=NF;i++){
            gsub(/^ +| +$/,"",$i);
            if(length($i) > max[i]) max[i] = length($i);
            row[NR,i] = $i
        }
        ncols = NF-1
    }
    END{
        for(r=1;r<=NR;r++){
            printf "|"
            for(c=2;c<=ncols+1;c++){
                printf " %-"max[c]"s |", row[r,c]
            }
            print ""
        }
    }' "${LOG_DIR}/summary.md" > "${LOG_DIR}/summary_tmp.md"
    mv "${LOG_DIR}/summary_tmp.md" "${LOG_DIR}/summary.md"

}

turn_results_to_htmls() {
 cd ${SCRIPT_DIR}/
    # /opt/conda310/bin/python3 -m pip install recommonmark sphinx-markdown-tables sphinx_pdj_theme
    cp -r ${LOG_DIR}/* ./sphinx_reg_result/
    sphinx-build -M html ./sphinx_reg_result/ ${LOG_DIR}/_docs_build/ --show-traceback
}

summarize_results
turn_results_to_htmls