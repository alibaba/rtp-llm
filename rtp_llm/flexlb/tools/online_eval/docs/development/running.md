# 运行测试

功能、性能和故障场景共用 `scripts/commands/run_cases.py`。先完成[编译与运行](build-and-runtime.md)；workload 需要 Prometheus。命令从 `rtp_llm/flexlb` 执行。

## 选择与预览

```bash
python3 tools/online_eval/scripts/commands/list_cases.py \
  --source tools/online_eval/config/scenarios \
  --profile batch-window --suite all --list-json

python3 tools/online_eval/scripts/commands/run_cases.py \
  --profile batch-window --suite workload \
  --instances '<case>::<variant>::<profile>' --parallel 1 --dry-run
```

`core` 根据 `config/suites.yaml` 选 CI 实例；`functional` 和 `workload` 按 `test.kind` 筛选；`all` 选择全部。默认 suite 也来自该配置，不在文档维护实例数量。`--case-dir` 可指向一份 YAML 或目录，`--instances` 使用清单中的完整 ID。

`--dry-run` 显示执行、端口及报告视图计划，不启动服务。核对 profile、worker 端口容量、lane 和内存预算后再执行；编译与执行使用同一组端口容量参数。列表顺序和 dry-run 成功不代表测试通过。

## 执行与收尾

```bash
OUT=/path/to/new-run
python3 tools/online_eval/scripts/commands/run_cases.py \
  --profile batch-window --suite workload \
  --instances '<case>::<variant>::<profile>' --parallel 1 \
  --out-dir "$OUT" --json "$OUT/aggregate.json" --archive "$OUT.zip"
```

每轮使用独占的新输出目录。性能或故障对照通常保持 `--parallel 1`；功能实例的并行度按资源计划选择。不要在 runner 之前手工启动同一套服务。拓扑、流量、窗口与门槛由 YAML 声明，步骤和分支由 Python program 定义；改变输入或规模后重新校验门槛。

入口退出码：0 表示所选检查和清理通过或命中显式 finding，1 表示运行失败，2 表示配置或选择错误。finding 不能吞掉启动、证据、超时与清理错误；少跑或零检查不能成为 PASS。

完成后按[结果与指标](results.md)核对有效性并收齐报告。常用参数见[参数参考](parameters.md)，对比和离线重判见[命令入口](entrypoints.md)。

## 干预与恢复

保留独立的基线、过渡和恢复窗口。确认成员及路由收敛后取得恢复请求证据；故障前晚完成的请求不能代替恢复流量。停止发新请求后等待在途请求有界排空，核对已发与终态账目，不用杀进程代替排空。取消失败、丢记录、线程未退出和资源未收敛都必须保留为失败。

校准标准来自独立有效的基线，不用干预结果回调门槛。原始请求、监控序列和采集错误保留在运行目录，具体窗口及合同以 YAML 和 program 为准。
