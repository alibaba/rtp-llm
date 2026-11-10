# Block transfer 恢复框架：第一步

本步对应核对事件表的事件 5（完成证据）和事件 7（执行身份），先建立结果契约、聚合规则和可独立验证的本地登记组件。尚未接入实际 I/O 的安全证明、生产登记入口、查询 RPC 或 frozen 引用回收。

## 执行身份：复用现有 trace

不增加 `task_id` 或新的每任务 ID 生成器。采用整组 worker 同生共死、没有跨实例复用的生命周期约束。

| 路径 | 用于登记的现有 trace | 登记边界（后续接入） |
| --- | --- | --- |
| 普通 Load / Evict | `rpcSubtask()` 生成的数值 trace B；各 rank 共用，worker 内部 `subtask()` 继承 | 每个 worker 的 `executePerRank()` 外层只登记一次；等所有内部 batch 汇总后更新 |
| 普通直接本地执行 | 调用者提供的 `TransferTask::traceId()` | 同一执行入口；独立执行应使用独立 task |
| KVCM 读写 payload | `readTransaction()` / `writeTransaction()` 生成的字符串 T；各 rank 请求共用 | `executePayload()` 中使用 `request.trace_id()`，不使用其中另建的数值 `TransferTask` trace |

普通 trace 已通过 `MemoryOperationRequestPB.transfer_trace_id` 传递。无 MultiRank 引擎时，同一个 B 直接进入本地执行。profiling 开关只影响记录，不影响 trace 的生成和继承。

KVCM 的字符串 T 保存在 `RemoteOperationRequestPB.trace_id`，RPC 和本地分支都保留。payload 的 SDK 数据读写接口目前不接收 T；恢复登记可以由调用 SDK 的外层维护。StorageBackend 的数值 trace 是另一个字段。

旧请求没有普通 trace 时，worker 当前会生成本地 trace；接入恢复时必须检查原始 wire 字段，不能把这个本地值当成协调端可查询的执行身份。本步不改变旧请求的执行行为。

## `buffer_safe` 的产生与传递

`TransferResult.buffer_safe` 默认 `false`，表示未取得“本次执行以后不再访问相关 buffer”的证明。它独立于业务成功、失败和每个 block 的诊断。

实际 I/O 出口后续负责提供证明：仅在 GPU、磁盘及相关后台访问全部结束时显式置 `true`。本步保留所有未接入出口的 `false`，不从成功、错误码或 `NOT_EXECUTED` 推断安全。

现有结果和完成回调直接携带该字段；汇总规则为所有参与结果的逻辑 AND：

- `TransferResult::merge()` 在处理业务成功的提前返回之前合并安全标记。
- `TransferStageState` 等待所有 batch 和提交结束，再发布结果；默认提交结束通知不额外否定安全证明。
- WorkLoop、legacy DEVICE↔DISK 的连续阶段保留此前所有阶段的证明。
- 多 rank 汇总包含成功 rank；当前 RPC 响应尚无安全字段，因此 RPC 成功仍不会产生安全证明。
- `emptyAggregate()` 的 `true` 只是 AND 初值，不能用作已执行 I/O 的证明；正在等待的外部 context 默认仍为 `false`。

安全失败可以是 `buffer_safe=true`，但其业务结果仍为失败。后续恢复仅释放该执行持有的物理引用，不补提交业务结果。

## 本地登记组件

`TransferTaskRegistry` 用同一互斥锁保护登记、完成更新和批量查询。普通传输与 KVCM 各自使用登记表实例；数字 trace 仅转成字符串作为内部索引，不生成新身份。

| 接口 | 行为 |
| --- | --- |
| `begin(trace)` | 首次登记为 RUNNING；拒绝空字符串、数值 0、重复 trace 或容量已满 |
| `finish(trace, false)` | 保留已有状态；不把未知任务加入表 |
| `finish(trace, true)` | 已登记任务变为 BUFFER_SAFE；以后不倒退 |
| `query(epoch, traces)` | 按输入顺序返回状态；缺失或登记表 epoch 不匹配返回 UNKNOWN |

RUNNING 表示尚未取得安全证明，可能是仍在运行，也可能是已返回但安全未知。UNKNOWN 不允许解冻。登记必须先于任何 buffer 访问；同 trace 的重复请求不能再次启动 I/O。

本步不提供删除或 TTL，保留完成证据和重复执行防护。容量默认 4096，满后拒绝新登记。由于未接入生产，当前业务不受此上限影响；生产接入前须设计安全的记录回收和迟到请求防重放协议。

## 后续逐步接入

1. 逐个审查本地 GPU、磁盘和 SDK 出口，明确何时签发 `buffer_safe=true`，以及未取得证明时资源由谁持有。
2. 在上述外层执行入口登记 trace，并由完整执行的汇总结果更新状态；验证重复请求只执行一次。
3. 接入协调端保存 trace、查询 RPC 和全 rank 安全判定。仅 RPC_TIMEOUT / RPC_ERROR 进入恢复候选。
4. 按事件表接入任务独立冻结引用及回收；同任务按 `(pool, block)` 去重，共享 block 的不同任务分别释放引用。

本步测试覆盖结果 AND、业务失败与安全独立、异步发布、内部子任务继承 trace、登记去重、并发容量和 UNKNOWN 查询。它们不代表真实 I/O 已签发安全证明，也不代表自动解冻已完成。
