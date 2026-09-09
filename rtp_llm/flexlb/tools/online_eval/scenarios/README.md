# Python case 配置目录

每份 YAML 只声明环境、profile、Python 变体选择和参数，格式版本为 `schema_version: 2`。
编排与断言位于 [`case_programs/`](../flexlb_test_framework/case_programs)；
公共 action 位于 `scenario/actions/`。旧实现仍保留作契约对照。

先读 [框架设计](../docs/framework-design.md) 了解执行和资源模型；
新增用例按 [如何添加新 case](../docs/adding-cases.md) 操作。

从仓库根目录列出全部可执行实例：

```sh
python3 rtp_llm/flexlb/tools/online_eval/scenario_runner.py \
  --source rtp_llm/flexlb/tools/online_eval/scenarios --list-json
```

| 类别 | YAML 文件 |
|---|---|
| admission | [admission_queue.yaml](admission/admission_queue.yaml), [batcher_placement_admission.yaml](admission/batcher_placement_admission.yaml), [engine_admission_gate.yaml](admission/engine_admission_gate.yaml), [prefill_batch_token_budget.yaml](admission/prefill_batch_token_budget.yaml), [priority_admission.yaml](admission/priority_admission.yaml) |
| balance | [balance_distribution.yaml](balance/balance_distribution.yaml), [balance_overload_transfer.yaml](balance/balance_overload_transfer.yaml) |
| cancel | [cancel_fence_settlement.yaml](cancel/cancel_fence_settlement.yaml), [cancel_lifecycle.yaml](cancel/cancel_lifecycle.yaml) |
| core | [request_completion.yaml](core/request_completion.yaml) |
| elastic | [added_worker_fault.yaml](elastic/added_worker_fault.yaml), [concurrent_mutation.yaml](elastic/concurrent_mutation.yaml), [lifecycle.yaml](elastic/lifecycle.yaml), [pending_drain.yaml](elastic/pending_drain.yaml) |
| engine_fault | [engine_fault_recovery.yaml](engine_fault/engine_fault_recovery.yaml), [engine_rpc_fault.yaml](engine_fault/engine_rpc_fault.yaml) |
| kv | [cache_affinity.yaml](kv/cache_affinity.yaml), [cache_capacity_recovery.yaml](kv/cache_capacity_recovery.yaml), [cache_churn.yaml](kv/cache_churn.yaml), [cache_global_holders.yaml](kv/cache_global_holders.yaml), [cache_local_index.yaml](kv/cache_local_index.yaml) |
| master | [client_fallback_failback.yaml](master/client_fallback_failback.yaml), [master_coldstart.yaml](master/master_coldstart.yaml), [master_dispatch_quota.yaml](master/master_dispatch_quota.yaml), [master_ha_failover.yaml](master/master_ha_failover.yaml), [master_lifecycle.yaml](master/master_lifecycle.yaml) |
| observation | [terminal_cohort.yaml](observation/terminal_cohort.yaml) |
| priority | [priority_preemption.yaml](priority/priority_preemption.yaml), [priority_queue.yaml](priority/priority_queue.yaml) |
| status | [batch_ack_and_execution.yaml](status/batch_ack_and_execution.yaml), [status_protocol.yaml](status/status_protocol.yaml) |

`core` 和 `observation` 包含框架执行/观测场景，不额外计为业务目标场景族。

更多配置及资源语义见 [执行器说明](../flexlb_test_framework/scenario/README.md)。
