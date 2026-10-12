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

CI 执行策略在同一配置的 `ci_jobs` 中声明 suite、profiles 和 parallel，不复制成员清单。使用下面的入口先预览，再执行；profile 按配置顺序串行运行，产物分别保存在 profile 子目录，任一失败停止后续执行。`ci-results.json` 记录实际执行范围与退出码，具体门禁以各 profile 的 `aggregate.json` 为准。

```bash
python3 tools/online_eval/scripts/commands/run_ci.py \
  --job quick --out-dir /path/to/new-ci-run --dry-run
python3 tools/online_eval/scripts/commands/run_ci.py \
  --job workload --out-dir /path/to/new-workload-run
```

手动流水线可先调用 `scripts/commands/plan_ci.py --suite <suite> --profiles <profile列表> --json /path/to/selection.json` 编译完整选例。`--instances <case::variant::profile列表>` 可覆盖 suite/profile 选择；未知或重复 ID 在启动执行组前失败。输出按配置的缩容 case 划分两个 profile 执行组，其余实例归入第三组；额外的缩容对照变体仍进入对应缩容组。空组由流水线跳过，各组使用独立 job，组内串行、profile 输出目录分开。这是固定分组，运行中不迁移任务。

这些策略和入口不会自动创建外部 CI 任务；流水线需要显式调用 `run_ci.py --job <job>`，不能仅凭 `ci_jobs` 配置认为已接入。

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

默认 `--scheduling dynamic`：启动前固定候选实例与预计耗时排序，空闲航道从公共队列领取下一条，每实例使用独立子 runner 和产物目录。实际航道归属以 `aggregate.json` 为准；dry-run 的 `dispatch_queue` 表示领取优先级，不预言实际归属。`--scheduling static` 固定每航道的实例列表，用于需要固定排列的执行。

航道保留独立端口租约；动态调度按全部候选实例的最大 worker 需求预留容量。每条实例完成后核对清理结果与租约端口，清理失败、子 runner 未可靠完成或端口未释放会退役该航道。剩余航道仍可领取任务；无可复用航道时，剩余实例记为 `ERROR`、`not_started: true`，不伪造执行结果。航道退役会使汇总退出码非零，即使最后一条实例已完成；该实例的业务判定保持原值。业务门禁 FAIL 且清理成功不阻止后续实例。端口隔离不提供 CPU、内存或磁盘带宽隔离，性能对照仍优先串行。

默认仅在现场清理通过、实验工件提交成功并校验 TSDB 原文件与压缩附件一致后回收 `telemetry/<epoch>/data`。原始 TSDB 保留在 lossless raw 附件中；`--retain-tsdb` 保留解压目录用于现场查询。归档、校验或清理失败不回收现场文件；回收受归档预算约束，失败保留剩余目录与完整原始附件。

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
