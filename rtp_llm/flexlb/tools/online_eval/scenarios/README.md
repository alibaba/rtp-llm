# Python case 配置目录

每份 YAML 只声明环境、profile、Python 变体选择和参数，格式版本为 `schema_version: 2`。
编排与断言位于 [`case_programs/`](../flexlb_test_framework/case_programs)；
公共 action 位于 `scenario/actions/`。已删除的扩展功能 case 仍可通过 Git 历史追溯。

先读 [框架结构](../docs/reference/architecture.md) 了解执行和资源模型；
新增用例按 [如何添加新 case](../docs/reference/adding-cases.md) 操作。

从仓库根目录列出全部可执行实例：

```sh
python3 rtp_llm/flexlb/tools/online_eval/scenario_runner.py \
  --source rtp_llm/flexlb/tools/online_eval/scenarios \
  --profile batch-window --suite core --list-json
```

日常入口 `test_runner.py` 默认只运行 `suites.yaml` 登记的 5 个核心功能实例。
直接使用底层 `scenario_runner.py` 时应显式指定 suite；`functional` 与 `core`
现在都是 5 个核心合同，持续负载使用 `--suite workload`，两类全集使用 `--suite all`。

| 类别 | YAML 文件 |
|---|---|
| balance | [balance_distribution.yaml](balance/balance_distribution.yaml) |
| core | [request_completion.yaml](core/request_completion.yaml) |
| elastic | [concurrent_mutation.yaml](elastic/concurrent_mutation.yaml), [lifecycle.yaml](elastic/lifecycle.yaml), [pending_drain.yaml](elastic/pending_drain.yaml) |
| engine_fault | [engine_fault_recovery.yaml](engine_fault/engine_fault_recovery.yaml) |
| kv | [cache_affinity.yaml](kv/cache_affinity.yaml), [cache_capacity_recovery.yaml](kv/cache_capacity_recovery.yaml), [cache_churn.yaml](kv/cache_churn.yaml) |
| master | [client_fallback_failback.yaml](master/client_fallback_failback.yaml), [master_ha_failover.yaml](master/master_ha_failover.yaml), [master_lifecycle.yaml](master/master_lifecycle.yaml) |
| workload | [cache_scale_in.yaml](workload/cache_scale_in.yaml), [trace_scale_out.yaml](workload/trace_scale_out.yaml) |

`scale_cases/` 保存未来接入的在线规模实验，其中包括 codex13 的分阶段缓存缩容场景；
它不进入默认功能回归。

更多配置及资源语义见 [执行器说明](../flexlb_test_framework/scenario/README.md)。
