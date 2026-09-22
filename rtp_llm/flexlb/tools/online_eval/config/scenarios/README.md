# Python case 配置目录

每份 YAML 声明环境、profile、Python 变体选择、参数和测试元数据，格式版本为 `schema_version: 2`。
编排与断言位于 [`case_programs/`](../../src/cases/programs)；
公共 action 位于 `src/scenario/actions/`。已删除的扩展功能 case 仍可通过 Git 历史追溯。

先读 [框架结构](../../docs/reference/architecture.md) 了解执行和资源模型；
新增用例按 [如何添加新 case](../../docs/reference/adding-cases.md) 操作。

从仓库根目录列出当前 CI 必跑实例：

```sh
python3 rtp_llm/flexlb/tools/online_eval/scripts/commands/list_cases.py \
  --source rtp_llm/flexlb/tools/online_eval/config/scenarios \
  --profile batch-window --suite core --list-json
```

`run_cases.py` 默认读取 `config/suites.yaml` 的 `default_suite`；当前 `core` 清单包含 5 个实例。
`--suite functional` / `workload` 按实例自己的 `test.kind` 筛选，`--suite all` 选择全部实例。
文件顶层 `test` 提供公共默认值；`variants[].test` 可以覆盖 kind、description、collection 和 monitoring。

所有 YAML 直接放在本目录，文件名采用稳定的 case 名；目录不参与分类或 CI 选例。
- [balance_distribution.yaml](balance_distribution.yaml)
- [cache_affinity.yaml](cache_affinity.yaml)
- [cache_capacity_recovery.yaml](cache_capacity_recovery.yaml)
- [cache_churn.yaml](cache_churn.yaml)
- [cache_scale_in.yaml](cache_scale_in.yaml)
- [client_fallback_failback.yaml](client_fallback_failback.yaml)
- [elastic_concurrent_mutation.yaml](elastic_concurrent_mutation.yaml)
- [elastic_lifecycle.yaml](elastic_lifecycle.yaml)
- [elastic_pending_drain.yaml](elastic_pending_drain.yaml)
- [engine_fault_recovery.yaml](engine_fault_recovery.yaml)
- [master_ha_failover.yaml](master_ha_failover.yaml)
- [master_lifecycle.yaml](master_lifecycle.yaml)
- [request_completion.yaml](request_completion.yaml)
- [trace_scale_out.yaml](trace_scale_out.yaml)

`cache_scale_in.yaml` 是 125P/536D 的真实前缀谱系缩容门禁；
同一文件的 `analysis` 字段只供已完成 run 的 A/B 对比读取，不参与编排。
该 workload 不进入默认功能回归。
列出或运行此 661-worker 拓扑前设置 `FLEXLB_FT_WORKER_PORT_CAPACITY=700`；
core/functional 列表不会编译未选中的大型 workload。

更多配置及资源语义见 [执行器说明](../../src/scenario/README.md)。
