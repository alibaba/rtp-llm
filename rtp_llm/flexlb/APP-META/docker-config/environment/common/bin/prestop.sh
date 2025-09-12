#!/bin/bash

# 默认等待时间是60s, 如果用户传递了参数则使用用户传递的参数
wait_seconds=60

removeHealthCheckNode() {
  echo "removeHealthCheckNode" >> "$log_path"
    echo "INFO: ${APP_NAME} `cat "$pid_path"` try to remove the health check node..." >> "$log_path"
    kill -s SIGUSR2 `cat "$pid_path"`
    now=`date "+%Y-%m-%d %H:%M:%S"`
    echo "INFO: ${APP_NAME} start to sleep ${wait_seconds}s for the running tasks to finish... $now" >> "$log_path"
    # 循环等待wait_seconds秒
    i=0
    while [ $i -lt $wait_seconds ]; do
      echo -n "$i " >> "$log_path"  # 使用 -n 参数防止自动换行
      sleep 1
      i=$((i+1))
    done
    echo >> "$log_path"  # 在循环结束后添加一个换行符

    now=`date "+%Y-%m-%d %H:%M:%S"`
    echo "INFO: ${APP_NAME} sleep end. $now" >> "$log_path"
  return
}

log_path="/home/admin/logs/prestop.log"
pid_path="/home/admin/ai-whale/.default/ai-whale.pid"

main() {
  touch "$log_path"
  echo "prestop.sh start" > "$log_path"
  # 如果参数1存在, 则覆盖等待时间, 否则忽略
  if [ -n "$1" ]; then
    wait_seconds=$1
  fi
  echo "wait_seconds:" "$wait_seconds" >> "$log_path"
  now=`date "+%Y-%m-%d %H:%M:%S"`
  echo "time is $now" >> "$log_path"
  echo "pid: " `cat "$pid_path"` >> "$log_path"
  # 此段脚本用于检测并处理服务进程的状态
    if [ -f "$pid_path" ]; then # 检查指定的PID文件是否存在
        if [ -s "$pid_path" ]; then # 检查PID文件是否非空
            kill -0 `cat "$pid_path"` >/dev/null 2>&1; # 尝试发送信号0到PID文件中的进程以验证其存在性
            if [ $? -gt 0 ]; then # 判断上一步操作的返回值, 如果大于0表示进程不存在
                # 输出信息说明PID文件存在但未找到对应进程
                echo "PID file found but no matching process was found. Stop aborted." >> "$log_path";
            else
                removeHealthCheckNode
            fi
        else
            # 输出信息说明PID文件为空将被忽略
            echo "PID file is empty and has been ignored." >> "$log_path";
        fi
    else
        # 输出信息说明设置的\$pid_path变量对应的文件不存在, 可能服务并未运行
        echo "\$pid_path was set but the specified file does not exist. Is Service running? Stop aborted." >> "$log_path";
    fi

  echo "prestop.sh end" >> "$log_path"
  echo "time is $now" >> "$log_path"
}

main "$1"



