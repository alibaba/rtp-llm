# Case 收缩分析（暂不实施）

可以压缩到 **9 个业务 Python 入口文件**，YAML 扩展 P/D 规模、输入长度、负载量与运行形态。
这个数量指业务流程入口，不包含 runner、公共 action、资源管理和框架测试。
本次仍保留 31 个程序文件、183 个变体和 385 个实例，未实施文件合并或实例删减。

## 建议归并

| 目标入口 | 当前程序 | 数量 |
|---|---|---:|
| admission.py | admission_queue、batcher_placement_admission、engine_admission_gate、prefill_batch_token_budget、priority_admission | 5 |
| priority.py | priority_queue、priority_preemption | 2 |
| master.py | client_fallback_failback、master_coldstart、master_dispatch_quota、master_ha_failover、master_lifecycle | 5 |
| balance.py | balance_distribution、balance_overload_transfer | 2 |
| elastic.py | elastic_added_worker_fault、elastic_concurrent_mutation、elastic_lifecycle、elastic_pending_drain | 4 |
| cache.py | cache_affinity、cache_capacity_recovery、cache_churn、cache_global_holders、cache_local_index | 5 |
| cancel.py | cancel_fence_settlement、cancel_lifecycle | 2 |
| fault.py | engine_fault_recovery、engine_rpc_fault | 2 |
| status.py | request_completion、observed_terminal_cohort、batch_ack_and_execution、status_protocol | 4 |

9 个入口合计覆盖当前 31 个程序。每个文件先保留多个小函数与现有公开 ID；
不要改成一个数千行函数。共同构造提成公共 Python helper，YAML 只传声明过的数据。
同样的流程在 2P/2D、4P/8D 下运行，不需要复制 Python 文件。

## 三种收缩的区别

| 做法 | 减少文件 | 减少独立环境启动 | 风险 |
|---|---|---|---|
| 把同类函数搬进 9 个入口，保留全部实例 | 是 | 否 | 最小；可逐实例比较计划和断言不变 |
| 对同一流程用 YAML 参数替代重复 Python 函数 | 是 | 通常否 | 要确认只差数据，不丢分支、阈值或 profile |
| 多个流程组合成同一环境下的大 case | 可 | 是 | 前序故障污染、检查被阻断、计时起点改变、共享缓存影响结果 |

不能仅根据 385 这个数量直接给出可信的缩减后实例数。四个 profile 的投递/决策路径不同，
还存在独立的故障、KV、取消和 HA 契约。应先按“流程相同、只差参数”的等价类统计。
目标可以是 9 个入口，不能由此声称只需运行 9 次即可保持覆盖。

## 大 case 怎样组织

优先组合无破坏性、能共享固定环境的检查，例如同一批请求的业务终态、状态计数和账本回收；
一次采样返回多项稳定 ID 的检查，减少重复 setup。顺序由 Python 定义，YAML 不列动作清单。

弹性加减、kill/restart、取消、KV 填满/淘汰、性能覆盖、Master 选主等流程会改变状态。
合并之前必须给每段增加明确的初始条件、资源登记、清理或重建，并保存分段结果。
当前执行器遇到普通失败会阻断后续步骤；直接拼接多个独立业务段，会让后半段失去覆盖。
因此需要独立子段的异常隔离、清理失败时终止共享环境、后续段重建，以及完整的分段 ID 汇总。
这部分执行器能力尚未实现，不应靠扩大 timeout 或吞掉 FAIL 来模拟。

推荐分两步推进：先归并 9 个文件并证明 385 个计划及检查不变；
随后挑只读/兼容的场景共享启动，以实际运行时间、资源峰值和失败可定位性判断收益。
共享的性能基线仍应串行测量；功能独立实例继续用多 lane 并行。

## 验收标准

- 原有实例与检查 ID 有完整映射；不丢 profile、finding、数值阈值和观察窗口。
- 一段 FAIL/ERROR/TIMEOUT 后，其余独立段仍有明确执行或阻断理由。
- 复用环境前证明请求、配额、KV 和后台线程回到规定初始状态；否则重建。
- 新增 YAML 规模配置编译时即可算出资源上限，运行后保留程序和配置哈希。
- 同一 Java 版本比较收缩前后的实例集合、业务断言和清理结果。
