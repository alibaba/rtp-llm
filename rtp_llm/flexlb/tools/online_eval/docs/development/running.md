# 运行测试

功能、性能和故障场景共用 `scripts/commands/run_cases.py`。先完成[编译与运行](build-and-runtime.md)；workload 需要 Prometheus。命令从 `rtp_llm/flexlb` 执行。

## 选择与预览

```bash
python3 tools/online_eval/scripts/commands/list_cases.py \
  --source tools/online_eval/config/scenarios \
  --profile batch-window --all --list-json

python3 tools/online_eval/scripts/commands/run_cases.py \
  --profile batch-window --suite workload \
  --instances '<case>::<variant>::<profile>' --parallel 1 --dry-run
```

`--suite` 只选择 `config/suites.yaml` 中的固定组合：`quick` 用于检查 mock 系统是否具备基本运行能力，`workload` 用于默认 workload 验证。`core` 保留为空，尚未定义核心行为验证用例；显式选择空 suite 会报错，不产生空跑成功。`--kind functional|workload` 选择该性质的全部变体，`--all` 选择全量；这三种选择互斥。列表与执行入口共用配置中的默认 suite 和 profile。列表可用 `--profile all` 展开所有支持的 profile；执行入口一次运行一个 profile。

`--case-dir` 可指向一份 YAML 或目录，`--instances` 使用清单中的完整 ID，并在 suite 或 kind 的选择范围内进一步筛选。固定组合缺少成员时直接失败，不静默少跑。

CI 使用同一份配置的 `ci_jobs`：suite 和 profiles 定义运行集，parallel 定义 lane 数；可选的 workers 只声明资源分配条件，不另建子 suite 或复制成员清单。`cases` 或 `exclude_cases` 按 case 分组，worker 的 profiles 默认为 job 的 profiles。规划器先编译全部选例并检查每个实例恰好分配一次；缺失、重复分配或非法 ID 会在任何 worker 启动前失败。

`plan_ci.py` 和 `run_ci.py` 共用 `CiPlanner`。前者输出完整实例清单、worker 分组和选中 worker 的实例；后者只执行该计划中的完整 ID，逐 profile 串行运行。`--suite`、`--profiles` 可覆盖 job 默认范围；显式 `--instances` 覆盖 suite/profile 选择，但仍须符合 job 支持的 profile。`--worker` 选择一个资源组，不隐式增加测试点。

```bash
python3 tools/online_eval/scripts/commands/plan_ci.py \
  --job workload --json /path/to/ci-plan.json
python3 tools/online_eval/scripts/commands/run_ci.py \
  --job workload --worker <worker> --out-dir /path/to/new-ci-run --dry-run
```

外层 CI 只准备环境、申请资源、传递选例参数并调用这两个入口，不自行枚举 profile、解析实例 ID 或按 case 编写分配条件。每个 worker 的完整运行计划保存在 `ci-plan.json`，执行状态保存在 `ci-results.json`；具体门禁以各 profile 的 `aggregate.json` 为准。失败时停止后续 profile，保留已执行范围。配置声明不等于平台接入成功；接入状态需要核对实际流水线配置及运行结果。

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

## 报告交互测试

Python 回归不执行 Node 测试。修改报告浏览器交互时，需在安装 Node.js 的环境中从 `online_eval` 目录手动执行：

```bash
node tests/js/test_multi_curve.js
node tests/js/test_legend_interaction.js
```

这些测试覆盖选区统计、缺失样本、刷选和图例交互，不需要启动服务或运行 workload。
