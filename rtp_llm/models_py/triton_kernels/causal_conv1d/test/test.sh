#!/bin/bash
cmd="/opt/conda310/bin/python test_casual_conv1d_decode.py"

for i in $(seq 1 10); do
    echo "第 $i 次执行: $cmd"
    $cmd
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "命令返回非零值 $ret，停止执行。"
        exit $ret
    fi
done

echo "命令已成功执行 10 次。"
exit 0
