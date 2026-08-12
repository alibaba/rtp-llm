# Auto-TPM 优先级抢占 Cancel 最终设计方案

## 1. 结论摘要

本方案将过去混在一起的两类结果彻底拆开：

1. **Victim 终态**：一个已经被接纳的请求，被严格更高优先级的 incoming 请求主动取消。它的终态错误码是 `PRIORITY_PREEMPTED (8429)`。
2. **Incoming 准入拒绝**：新请求从未被接纳。Master 必须明确说明它是被更高优先级任务阻塞、排在同优先级任务之后，还是所选路径未能在 admission budget 内提供容量。

Cancel RPC 首版采用**弱 ACK**：`ACCEPTED` 只表示原始 Prefill 已经原子安装优先级抢占意图，并触发当前或未来的 P-to-D 取消。它不表示 Engine 已完成取消，也不表示 slot/KV 可以复用。Master 收到 ACK 后只把 victim 转为 `CANCEL_REQUESTED`，继续保留资源账本；只有原始 Prefill 的 WorkerStatus 明确上报 `CANCELED + 8429` 后，Master 才执行一次资源结算并继续接纳 incoming。

最终系统边界如下：

- Master 负责准入决策、优先级排序、victim 选择、请求生命周期账本和 incoming 拒绝原因。
- 原始 Prefill 是 Prefill/Decode 全阶段的 Cancel 控制 owner。
- Engine 负责真正执行取消；ACK 后以 `CANCELING -> CANCELED` 暴露进度。
- 原始 Prefill WorkerStatus 的类型化 `CANCELED` 是完成凭证；普通 `finished` 不是。
- DashSC 保留 Master 的类型化原因用于诊断；对于显式携带
  `x-dashscope-inner-qos-level` 的 incoming 准入拒绝，按 QoS `<50` / `>=50`
  选择公开 `status_name`。Victim 8429 始终独立映射。

## 2. 设计目标与非目标

### 2.1 目标

- Decode `ACCEPTED_NOT_RUNNING` 和 `RUNNING` 都允许成为严格低优先级 victim。
- 覆盖 Prefill 正在连接 Decode、申请 Decode 资源的 Stage 2 窗口。
- Cancel 始终发往原始 Prefill，而不是 Decode endpoint。
- Cancel 返回明确状态，而不是 `EmptyPB`。
- EnqueueBatch 或 FetchResponse 对 victim 保留精确错误码 8429。
- 收到弱 `ACCEPTED` 后不清理账本；等待 WorkerStatus `CANCELED` 后才结算一次。
- Master 在决策点区分 incoming 的不同 429 原因。
- 保持协议和实现尽量小，不向 Engine 引入一套通用生命周期协议。

### 2.2 非目标

- 本期不处理 request ID 复用。
- 本期不改造 WorkerStatus 的通用 `finished/resource_released` 计算方式；只增加最小的优先级取消进度 `NONE/CANCELING/CANCELED`，并把类型化 `CANCELED + 8429` 作为本流程的完成凭证。
- 不引入 USER/DEADLINE/ADMIN 等通用 Cancel reason；本 RPC 专用于优先级抢占。
- Decode 不暴露面向 Master 的独立 Cancel RPC；Prefill 复用既有 P-to-D `ClientContext::TryCancel()` 触发停止。
- 不建设一套通用 Decode Cancel 服务，也不在 Cancel RPC 内同步等待 Decode terminal/capacity barrier。
- 既有通用 Frontend-to-Master Cancel API 不属于本方案。

## 3. 领域模型

### 3.1 Victim 终态与 incoming 拒绝是两套语义

| 领域 | 含义 | 事实 owner | 结果 |
|---|---|---|---|
| Victim 终态 | 已接纳请求被主动让位 | Master 决策，Engine 执行 | `8429 PRIORITY_PREEMPTED` |
| Incoming 拒绝 | 新请求未被接纳 | Master admission planner | 类型化 `ScheduleFailureReasonPB` |

下列组合都可能发生，必须保持可区分：

- victim 为 8429，incoming 成功；
- 一个或多个 victim 为 8429，但 incoming 后续仍因真实资源不足而失败；
- incoming 因 `HIGHER_PRIORITY_AHEAD` 被拒绝，但没有任何 victim 被取消；
- Cancel 返回 `NOT_FOUND`，Master 刷新并重规划，该目标不能被记成 8429。

### 3.2 Victim 与 admission attempt 状态

Master 侧把 victim 生命周期、终态原因和本次 admission attempt 分开：

```text
Victim
  ACTIVE
    -> CLAIMED(attemptToken)
    -> CANCEL_IN_FLIGHT
        -> PREEMPTED_TERMINAL
        -> NOT_FOUND_STALE
        -> CANCEL_UNKNOWN
  ACTIVE/CLAIMED -> NATURAL_TERMINAL

PREEMPTED_TERMINAL:
  state           = CANCELLED
  terminal_reason = PRIORITY_PREEMPTED
  terminal_code   = 8429
```

```text
AdmissionAttempt
  PLANNED
    -> CLAIMED
    -> CANCELLING
        -> READY_COMMIT
        -> COMMITTED
        -> SIDE_EFFECT_FREE_REPLAN_ONCE
        -> PARTIALLY_APPLIED
        -> CONTROL_FAILED
```

- `CANCELLED` 描述 victim 的生命周期状态；
- `PRIORITY_PREEMPTED` 描述为什么被取消；
- `8429` 是该原因对应的内部错误码；
- `NOT_FOUND_STALE` 在 fresh reconciliation 前不可再次成为 victim，避免反复选中同一个 stale entry；
- `CANCEL_UNKNOWN` 保留 accounting，只允许对同一 Prefill、同一 request ID 做幂等收敛。

### 3.3 Incoming admission 结果

Planner 不再使用 `proposal == null`、自由文本和统一 `NO_AVAILABLE_WORKER` 表达不同结果，而是返回最小类型化结果：

```text
AdmissionResult
  = Admitted(plan)
  | PreemptThenCommit(plan, victims, attemptToken)
  | Retry(controlConflict)
  | Rejected(failureReason)
```

最小业务拒绝原因：

```text
HIGHER_PRIORITY_AHEAD
SAME_PRIORITY_AHEAD
RESOURCE_EXHAUSTED
```

`attemptToken` 绑定 attempt sequence、snapshot/admission version 和 victim 集合。为避免再引入一套资源向量 escrow，Master 采用更简单、保守的完整 provisional reservation：

1. 原子 claim 目标 victim；victim 仍计为 used，但不能再被其他 attempt 选择；
2. 同时为 incoming 建立完整的 provisional reservation。该 reservation 尚不可 dispatch，但从建立起就参与 slot/KV admission accounting，因此并发请求无法抢走本次 incoming 最终需要的容量；
3. victim 首次上报类型化 `CANCELED + 8429` 时，只删除该 victim 的 accounting；incoming 的完整 provisional reservation 始终存在，不会出现 victim 容量先回到全局 free pool、再被其他请求抢走的窗口；
4. 全部必要 victim 结算后，把 incoming reservation 从 provisional 提升为 committed；attempt abort 则删除 incoming reservation。已经真实取消的 victim 保持终态，其容量自然回到 free pool；unknown victim 继续计为 used。

Cancel 等待期间同时计入 victim 和 incoming 是刻意的保守双计数，仅影响该 endpoint 暂时不再接纳其他请求，不会向 Engine 过量 dispatch。它换取了清晰的资源所有权和更小的状态机。

- 每个 victim 的 accounting 只允许由携带当前 `attemptToken` 的类型化 `CANCELED + 8429` 删除一次；
- incoming provisional reservation 从 begin 到 commit/abort 始终由同一个 token 持有；
- 旧 callback 必须携带当前 token 才能结算，不能修改后续 attempt。

版本冲突、victim 消失、Cancel 传输失败、commit 冲突属于控制面结果。它们先触发安全的内部重试；重试用尽后，如果没有一致快照能证明 higher/same，公开结果统一折叠为 `RESOURCE_EXHAUSTED`，内部 detail 和指标仍保留真实控制面原因。

### 3.4 Victim 所有权分层

为避免把“Master 本地队列删除”和“Engine Cancel”混成一个事务，victim 必须先按所有权分层：

| Master 可见状态 | 含义 | 处理方式 |
|---|---|---|
| `MASTER_QUEUED_NOT_DISPATCHED` | 请求仍由 Master 队列独占，确定没有发给 Engine | 走独立 local-yield plan，不发 Cancel |
| `ENGINE_MAY_HAVE_SEEN` | 已开始 dispatch、EnqueueBatch 在途，可能正处于 Stage 2 | 向原始 Prefill 发弱 ACK Cancel，等待类型化完成 |
| `ACCEPTED_NOT_RUNNING` | Decode 已接纳但尚未运行 | 向原始 Prefill发弱 ACK Cancel，等待类型化完成 |
| `RUNNING` | Decode 正在生成 | 向原始 Prefill发弱 ACK Cancel，等待类型化完成 |

同一个 preemption attempt 不能混合 local-yield victim 与 Engine-Cancel victim。前者由 Master 本地事务结算，后者由弱 ACK 加类型化 WorkerStatus 完成事件结算。请求一旦从 Master batch queue 脱离，就必须原子进入 `ENGINE_MAY_HAVE_SEEN`，不能继续按“Engine 一定没看到”做本地释放。

## 4. Cancel 协议

### 4.1 Protobuf

```proto
message CancelRequestPB {
    int64 request_id = 1;
}

enum CancelStatusPB {
    CANCEL_STATUS_UNSPECIFIED = 0;
    CANCEL_STATUS_ACCEPTED = 1;
    CANCEL_STATUS_NOT_FOUND = 2;
}

message CancelResponsePB {
    CancelStatusPB status = 1;
}

enum PriorityPreemptionProgressPB {
    PRIORITY_PREEMPTION_NONE = 0;
    PRIORITY_PREEMPTION_CANCELING = 1;
    PRIORITY_PREEMPTION_CANCELED = 2;
}

message TaskInfoPB {
    // existing fields
    PriorityPreemptionProgressPB priority_preemption_progress = 14;
}

service RpcService {
    rpc Cancel(CancelRequestPB) returns (CancelResponsePB);
}
```

这是新版本的优先级抢占接口，不携带为兼容历史而增加的 reason、phase、release、lifecycle revision 或 terminal error 字段。

`priority_preemption_progress` 只描述 Engine 内部优先级取消进度，不替代 `TaskPhase`，也不使用含混的 `master_preempted` 命名：

- first-cause CAS 赢得优先级抢占后，运行中的 task 上报 `CANCELING`；此时仍占用 capacity，Master 不做资源扣减；
- 原始 Prefill 安装抢占意图后立即上报 `CANCELING` 并允许 Cancel RPC 返回 `ACCEPTED`；此时仍占用 capacity，Master 不做资源扣减；
- P-to-D 与 Prefill 本地执行链真正退出后，原始 Prefill 的 terminal finalizer 上报 `CANCELED`，并携带 `error_info.error_code=8429`；
- Master 合并 Prefill/Decode 进度时只允许单调 `NONE -> CANCELING -> CANCELED`，迟到或乱序 observation 不能把状态回退。

### 4.2 `ACCEPTED` 的弱语义

Engine 返回 `ACCEPTED` 前只保证：

- 优先级抢占赢得了终态竞争；
- 原始 Prefill active context 已锁存 `PRIORITY_PREEMPTED`；
- 当前或未来发布的 P-to-D context 已被触发取消；
- WorkerStatus progress 已进入或将进入 `CANCELING`；
- 重复 Cancel 不会安装第二份冲突意图。

`ACCEPTED` 明确**不保证**：

- 请求已经停止输出；
- 请求已经离开 scheduler/registry；
- slot/KV 已经释放；
- incoming 可以立即使用 victim 容量。

因此 Master 收到 `ACCEPTED` 后只进入 `CANCEL_REQUESTED`，保留 victim accounting 和 incoming provisional claim，等待原始 Prefill WorkerStatus 的类型化 `CANCELED + 8429`。

终态 tombstone 仍用于幂等和 FetchResponse 8429：

- 重复调用命中 `CANCELING` 或已完成的同一优先级取消时，返回幂等 `ACCEPTED`；
- Stage 3 尚未建立 Fetch 时，Engine 可以独立完成取消，未来 FetchResponse 从 terminal mailbox/tombstone 读取 8429；
- active entry 到 terminal tombstone 的切换必须原子完成，不能出现短暂 `NOT_FOUND` 空窗；
- tombstone 至少保留到原 Enqueue/Fetch 已消费 8429，或原请求的 fetch-attach lease/deadline 明确到期；同时覆盖 Master 最大 Cancel deadline、幂等 retry 和 reconciliation 窗口。本期不处理 request ID 复用，因此可按 request ID fencing。

### 4.3 `NOT_FOUND` 的语义

在线性化点上，目标 Prefill 没有可由本次优先级抢占接管的 active request（可能不存在，也可能其他 terminal cause 已经先赢），并且本次调用没有安装 priority-preemption latch 或 CancelIntent。

`NOT_FOUND` 不代表：

- Decode 资源已经释放；
- 请求已经自然成功；
- 该请求应返回 8429；
- incoming 可以立即复用这部分容量。

### 4.4 gRPC 错误

- `request_id <= 0`：返回 `INVALID_ARGUMENT`。
- Engine 内部取消失败：返回 `INTERNAL` 或对应的类型化 gRPC 状态。
- RPC deadline/transport failure：结果未知。Master 保留账本，不能基于这部分容量接纳 incoming。

### 4.5 Prefill-to-Decode 传播

Master-facing Cancel 仍只发给原始 Prefill。首版复用既有 upstream cancellation 机制：Prefill 锁存 priority first-cause 后，对当前 P-to-D `ClientContext` 调用 `TryCancel()`；若 Cancel 早于 context 发布，发布线程看到 latch 后立即 `TryCancel()`。

该传播是异步 stop trigger，不增加同步 P-to-D TerminalAck，也不能被 Master-facing `ACCEPTED` 当作完成证明。原始 Prefill 在下游 RPC 与本地执行链真正结束后，才通过 WorkerStatus 发布 `CANCELED + 8429`。

## 5. Engine 生命周期与 owner

### 5.1 控制 owner

原始 Prefill 是所有阶段的控制 owner。Master 从权威 inflight 记录中解析原始 Prefill，并始终向它发送 Cancel。

Decode 是 Prefill 所持有 P-to-D RPC context 后面的执行实现，不暴露优先级抢占 Cancel RPC。

### 5.2 Cancelable 与 Fetchable 分离

同一个 context 有两个独立可见点：

- **Cancelable**：在开始连接、remote allocate 或其他 prepare 工作前注册。
- **Fetchable**：只有 prepare 成功后才发布给 FetchResponse。

这样既能关闭 Stage 2 空窗，又不会让 FetchResponse 读取半初始化 context。

### 5.3 线性化规则

active map 只负责 discoverability，不能代替 terminal-cause 仲裁。每个 context 必须有唯一的 first-cause 状态：

```text
ACTIVE
  -> PREEMPTING
  -> TERMINAL(PRIORITY_PREEMPTED)

ACTIVE
  -> TERMINAL(NATURAL_SUCCESS | DEADLINE | USER_CANCEL | ENGINE_ERROR)
```

所有终态路径都通过同一个 CAS/受锁状态机：

- Cancel 将 `ACTIVE -> PREEMPTING` 成功：优先级抢占赢，触发 downstream 取消并立即返回弱 `ACCEPTED`；
- CAS 失败且已有其他 terminal cause：不能覆盖原因，返回 `NOT_FOUND`；
- 自然终态先赢：写 terminal cause、发布对应 tombstone 并移除 active；后到 Cancel 返回 `NOT_FOUND`。

Cancel handler 不能等待终态：锁内完成 lookup 与 first-cause CAS，释放锁，触发 downstream 取消并返回 `ACCEPTED`。后续 terminal finalizer 独立完成 8429 tombstone 与 WorkerStatus `CANCELED`。

`PREEMPTING` 是封口状态，不只是一个标记：

- prepare/retry/handoff 不得再发起新的连接、malloc、enqueue 或 cache-transfer 副作用；
- 已经开始的 prepare/handoff 必须被取消并 join；
- FetchResponse 看到 `PREEMPTING` 时不得 attach 或继续操作 live context，只能等待同一个 terminal token，随后从 tombstone 读取 8429；
- 同 request 的重复 Cancel 命中 `PREEMPTING` 时幂等返回 `ACCEPTED`，不能重复安装 latch 或启动第二个 finalizer。

统一 finalizer 必须通过 RAII 覆盖 prepare exception、TTL、shutdown、Fetch exception 等所有退出路径，并最终发布 `CANCELED + 8429` 或真实自然终态。

优先级抢占不再写全局 scheduler `CancelIntentMap`。已注册 context 的 first-cause latch、阶段 checkpoint、线程安全的 stream error 与 P-to-D `TryCancel()` 已覆盖同一职责，避免 Stage 4 未命中 intent 在调度热路径滞留。

### 5.4 P-to-D 取消握手

实现必须覆盖两种顺序：

1. P-to-D `ClientContext` 先发布，随后 Cancel 调用 `TryCancel()`；
2. Cancel 先锁存，随后 context 发布并立即调用 `TryCancel()`。

context 发布与抢占 latch 必须使用同一个无竞态握手，可以使用 mutex 或正确的 atomic shared-pointer 协议。`TryCancel()` 只负责触发停止；完成事实由原始 Prefill active context 的 finalizer 产生，并通过 WorkerStatus `CANCELED + 8429` 暴露。

### 5.5 四阶段行为

| 阶段 | Engine 状态 | Cancel 行为 |
|---|---|---|
| Stage 1：请求尚未进入 Prefill active | 无 cancelable context | 返回 `NOT_FOUND`；Master 刷新并重规划 |
| Stage 2：Prefill 正在连接/申请 Decode 资源 | context 已 cancelable、尚不可 Fetch | 锁存 8429，触发当前或未来 P-to-D `TryCancel()`，立即返回 `ACCEPTED`；执行链结束后上报 `CANCELED` |
| Stage 3：请求处于 Prefill 或 handoff | context/Prefill stream active，Fetch 可能尚未建立 | 锁存 8429、触发取消并返回 `ACCEPTED`；发布 terminal mailbox，未来 Fetch 读取 8429 |
| Stage 4：Decode generating | Fetch 后 context 仍保留在 active registry | 对 P-to-D context `TryCancel()` 并返回 `ACCEPTED`；FetchResponse 最终返回 8429，Prefill WorkerStatus 随后发布 `CANCELED` |

## 6. Master Cancel 流程

```mermaid
sequenceDiagram
    participant M as Master
    participant P as Original Prefill
    participant D as Decode
    participant F as Frontend

    M->>P: Cancel(request_id)
    P->>P: 锁存 priority_preempted
    P->>D: TryCancel current/future P-to-D context
    P->>P: WorkerStatus progress=CANCELING
    P-->>M: CancelResponse(ACCEPTED)
    M->>M: victim=CANCEL_REQUESTED; accounting保持
    D-->>P: 既有P-to-D执行链结束
    P-->>F: 当前或未来 Enqueue/Fetch 返回 ErrorDetails(8429)
    P->>P: finalizer; progress=CANCELED + 8429
    P-->>M: WorkerStatus(CANCELED, 8429)
    M->>M: settleOnce; victim=CANCELLED
    M->>M: 全部victim完成后commit incoming
```

### 6.1 收到 `ACCEPTED` 与 `CANCELED`

收到弱 `ACCEPTED`：

1. victim 保持 `CANCEL_REQUESTED`；
2. Decode confirmed accounting、slot/KV 与 inflight 均不释放；
3. incoming 保持 provisional，不得向 Decode 提交；
4. Coordinator 开始等待原始 Prefill WorkerStatus 的 `CANCELED + 8429`。

每个 victim 首次收到类型化 `CANCELED + 8429` 时，Master 才执行一次带 `attemptToken` 条件的原子 settlement：清理 marker、将 victim 转为 8429 `CANCELLED`、删除该 victim accounting，并写 terminal tombstone。incoming 的完整 provisional reservation 在整个等待期始终存在；全部必要 victim 都完成后，只需把它提升为 committed reservation。

该路径必须等待权威 WorkerStatus 的 `CANCELED + 8429`。首次完成事件负责结算；其后的迟到或重复 WorkerStatus 命中 terminal tombstone 后只能对账，不能重新建立 confirmed entry、覆盖 8429 或重复释放 accounting。

`CLAIMED -> CANCEL_IN_FLIGHT` 必须在 admission lock/CAS 内完成，成功后该 victim 的生命周期结算权立即归 `CancelCoordinator(attemptToken)` 独占，随后才允许发 RPC。网络上的“Cancel 已发出”不是线性化点。

- 若普通 WorkerStatus 先完成自然终态 CAS，claim 失败，Master 不得再发送 Cancel；
- 若 coordinator claim 先赢，在收到 `ACCEPTED`、`NOT_FOUND` 或进入 reconciliation 期间，普通 WorkerStatus handler 只能把 observation 附加到该 token；只有类型化 `CANCELED + 8429` 能完成 victim 和结算 accounting，其他状态不能根据 attribution marker 推导 8429。

marker 只表示“发起过抢占”，不是终态证明。

Cancel 响应与 WorkerStatus 的并发规则如下：

- WorkerStatus 先到且报告普通成功/失败：暂存 observation；等待 Cancel 的线性化结果决定 first cause；
- WorkerStatus 先到且报告类型化 `CANCELED + 8429`：由当前 `attemptToken` 执行唯一 settlement；迟到 ACK 只能更新指标；
- Cancel 返回 `ACCEPTED`：只确认 `CANCEL_REQUESTED`，不结算 accounting；
- Cancel 返回 `NOT_FOUND`：不得用 marker 伪造 8429，转入 stale reconciliation，并按权威 WorkerStatus/全量快照恢复真实自然终态；
- Cancel transport unknown：保持 coordinator ownership 与 accounting，直到幂等 retry 或 reconciliation 收敛。

Master terminal tombstone 检查、claimed-victim 检查与 Decode confirmed upsert 必须位于同一个 admission lock/CAS 边界。否则“旧 WorkerStatus 先查无 tombstone、随后 Cancel 写 tombstone、旧 WorkerStatus 再插入 confirmed”的竞态会复活已经取消的请求。

### 6.2 收到 `NOT_FOUND`

`NOT_FOUND` 的 Engine 事实只作用于对应 victim，不能把其他 victim 伪装成未取消；但它会使依赖完整 victim 集合的 incoming plan 失效：

1. 仅用 `attemptToken` CAS 清理由当前 attempt 写入的 cancel claim 和 attribution marker；
2. 保留该 victim 的资源 accounting；
3. 将其置为不可再次选择的 `NOT_FOUND_STALE`，等待 fresh reconciliation；
4. 当前 incoming attempt 原子转入失败或唯一允许的单 victim replan 分支，并由当前 token 释放尚未 commit 的 provisional entitlement；其他已经发出的 Cancel callback 继续按各自 child token 收敛。

只有同时满足以下条件，才允许一次 side-effect-free replan：

```text
confirmed victim 恰好 1 个
&& 收到显式 NOT_FOUND
&& 两处 attempt marker CAS 清理成功
&& incoming 尚未 promote/offer/commit
&& provisional entitlement 可由当前 token CAS 释放
&& 没有其他已发出的 Cancel 或已产生副作用的 victim
&& replanCount == 0
```

下列动作必须在同一 admission 原子边界完成：

```text
CANCELLING(token) -> REPLAN_PENDING
+ 清理当前 token 的两处 marker
+ victim -> NOT_FOUND_STALE
+ 释放完整 incoming provisional reservation
```

重规划必须使用新快照并排除该 stale victim，且不能把它的容量算作已释放。任一 CAS 或前置条件不满足，原 incoming attempt 进入 `PARTIALLY_APPLIED` 或 `CONTROL_FAILED`，不能沿用原 plan。

### 6.3 RPC 超时或传输失败

Master 无法判断 Cancel 是否生效，因此：

- victim 保持 `CANCEL_REQUESTED/CANCEL_UNKNOWN`；
- 不释放 victim accounting；
- 不基于该容量 reserve incoming；
- 在 Engine terminal tombstone 有效期内，只向同一原始 Prefill、同一 request ID 做有界幂等 retry；
- aggregate attempt 立即释放完整 incoming provisional reservation 和所有尚未发出 Cancel 的 victim claim；
- 每个 `CANCEL_UNKNOWN` victim 保留独立 child token/accounting，供 late result 幂等结算；
- 已经完成类型化取消的 victim 保持真实终态；由于 incoming reservation 被释放，其容量自然回到 free pool，不能被已失败 incoming 使用；
- incoming attempt 进入 `CONTROL_FAILED`，不能切换 victim 或生成新 request ID 继续原计划。

### 6.4 Reconciliation 的确定出口

`NOT_FOUND_STALE` 和 `CANCEL_UNKNOWN` 都是暂态，不能永久占用 accounting，也不能靠超时直接释放容量。收敛规则是：

| 当前状态 | 权威新事实 | 结算结果 |
|---|---|---|
| `CANCEL_UNKNOWN` | 对同一 Prefill/request ID 幂等重试得到 `ACCEPTED` | 保持 `CANCEL_REQUESTED`，继续等待类型化 `CANCELED` |
| `CANCEL_UNKNOWN` | 幂等重试得到 `NOT_FOUND` | 转为 `NOT_FOUND_STALE`，不释放 accounting |
| `CANCEL_UNKNOWN` | WorkerStatus 是普通自然终态 | 按该自然终态结算并释放 accounting，不能改写为 8429；后到的 typed CANCELED 只能命中 terminal tombstone |
| `CANCEL_UNKNOWN` | 暂存的 WorkerStatus 是 `CANCELED + 8429` | 由 child token 执行唯一 settlement；原 incoming 已失败时资源转 free pool |
| `NOT_FOUND_STALE` | 权威终态 observation | 按真实自然终态结算，绝不改写为 8429 |
| `NOT_FOUND_STALE` | fresh snapshot 证明请求仍 active 且 control owner 可解析 | 恢复 `ACTIVE`，清除本次 claim；后续新 attempt 才可重新选择 |
| 任一暂态 | fresh snapshot 证明请求仍 active 但 control owner 不可解析 | 进入 `CONTROL_ORPHANED`，保留 accounting，禁止继续驱逐 |
| 任一暂态 | 多轮全量快照与 owner registry 对账仍无法确定 | 将 endpoint 隔离为 unhealthy，并用完整 worker baseline 重建该 endpoint accounting；重建事务同时关闭 victim lifecycle 与 child token，禁止单独猜测释放这一个请求 |

普通 finished 只负责恢复自然终态事实；只有类型化 `priority_preemption_progress=CANCELED` 且 `error_code=8429` 才能让本次 priority Cancel 结算资源。

### 6.5 Multi-victim 结果矩阵

| 结果组合 | Victim settlement | Incoming 动作 |
|---|---|---|
| ACK 全部 `ACCEPTED`，随后 WorkerStatus 全部 `CANCELED` | 每个 victim 独立、幂等结算为 8429 | promote 完整 incoming provisional reservation 并 commit |
| 单 victim `NOT_FOUND` 且满足无副作用条件 | 标记 `NOT_FOUND_STALE`，保留 accounting | 排除该 victim，fresh snapshot replan 一次 |
| `CANCELED + NOT_FOUND` | 完成者保持 8429；NOT_FOUND 保留 accounting | 原 plan 失效；释放 incoming provisional reservation，不能回滚成功取消 |
| 任一 timeout/transport unknown | 已成功者正常结算；unknown 保留 accounting | 原 plan 进入 `CONTROL_FAILED`；释放 incoming provisional reservation，不得使用 unknown 容量 |
| commit/CAS failure 或 `PARTIALLY_APPLIED` | 已成功者保持真实终态；unknown 保留 accounting | incoming 不得 commit；释放 incoming provisional reservation |
| aggregate deadline 后结果迟到 | callback 仍逐个完成幂等 settlement | 不复活已失败的 incoming attempt |

## 7. WorkerStatus 类型化完成与单次资源结算

最终流程：

```text
CancelResponse ACCEPTED -> CANCEL_REQUESTED，资源保持
WorkerStatus CANCELING   -> 仅更新进度，资源保持
WorkerStatus CANCELED    -> Master settleCapacityOnce(attemptToken)
重复 CANCELED/finished   -> 仅对账，不重复结算
```

Master 对每个 victim 维护唯一资源所有权状态，而不是靠两个回调分别做加减：

```text
HELD_BY_VICTIM
  -> CLAIMED(attemptToken)
  -> CANCELED_SETTLED(attemptToken)   // exactly once
  -> COMMITTED_TO_INCOMING | RELEASED_TO_FREE
```

类型化 WorkerStatus `CANCELED + 8429` 是 settlement 触发点：CAS `CANCEL_REQUESTED(token) -> CANCELED_SETTLED(token)`。后续重复状态或普通 finished 看到资源已由同一 token 结算，只更新状态/指标，**不再减少 running、KV、slot 或 inflight accounting**。

若 WorkerStatus `CANCELED` 因调度先于 CancelResponse 到达，可以直接由 coordinator token 结算；迟到的 `ACCEPTED` 不再改变资源。若响应丢失但已看到类型化 `CANCELED`，同样只结算一次。

`CANCELING` 永远不触发资源扣减；它只表示取消仍在执行。只有 `CANCELED + 8429` 才允许结算本次 priority-preemption 容量。

P/D 分离时，原始 Prefill 是 priority-preemption progress 的权威 producer。Decode 的普通运行/终态状态只做资源账本对账，不产生第二份 priority `CANCELED`。Master 以 request ledger/tombstone 和 `attemptToken` 保证资源结算只发生一次。

因此，优先级 Cancel 提交主链：

- 不接受任意 `finished` 或合成 `resourceReleased=true`；
- 只等待原始 Prefill 的类型化 `CANCELED + 8429`；
- 使用 request ledger/tombstone 保证 exactly once。

Decode 的迟到普通状态仍用于对账，不能覆盖原始 Prefill 的 priority terminal，也不能再次释放 accounting。

### 7.1 Timeout

保留两个独立 deadline：

```text
autoTpmCancelAckTimeoutMs
autoTpmCancelCompletionTimeoutMs
```

首版建议 ACK 50ms、completion 1000ms；completion 必须覆盖多次 WorkerStatus 轮询与 Stage 4 退出。两个阶段各自从零开始计时，ACK 已消耗的时间不能减少 completion budget。

多个 victim 的 Cancel RPC 并发发出并共享 ACK deadline；全部 ACK 后，再为类型化 `CANCELED` 建立独立 completion deadline。等待使用 async completion，不能阻塞 scheduler/gRPC ingress 线程。completion timeout 后 incoming 失败，但 victim 的迟到 `CANCELED` 仍必须幂等清理账本。

## 8. Incoming 拒绝原因设计

### 8.1 事实 owner

只有 Master 拥有 cluster snapshot、incoming priority、队列顺序、victim policy 和资源缺口，因此只有 Master 可以判定 `HIGHER_PRIORITY_AHEAD` 或 `SAME_PRIORITY_AHEAD`。C++ 不做此类推断。

### 8.2 确定性判定

对选定的失败 endpoint 和资源维度：

1. 先验证 hard feasibility。如果请求在空闲兼容 worker 上也永远放不下，返回既有 hard-capacity/参数错误，而不是瞬时 429。
2. 计算资源缺口向量，例如 `(prefill_queue_slot, decode_slot, decode_kv)`。
3. 按既有确定性 victim 顺序，选择能够覆盖完整资源缺口向量的**最小前缀**；不能为了分类而驱逐所有低优请求。
4. 扣除该最小 victim 集合能释放的资源向量。
5. 如果仍有 residual deficit：
   - 有实际贡献的请求 `priority > incoming`：`HIGHER_PRIORITY_AHEAD`；
   - 否则存在对尚未满足资源维度有正贡献、且拥有更早 `admissionSeq` 的同优先级请求：`SAME_PRIORITY_AHEAD`；
   - 都无法解释 residual：`RESOURCE_EXHAUSTED`。

同优先级 FIFO 必须使用 routing 和 failure classification 共享的不可变 `admissionSeq`。不能重新使用优先级阈值、墙上时钟或 request ID 猜顺序。

如果 blocker 的 priority 来源未知，Master 不能声称 higher/same 因果；对外统一折叠为
`8431 RESOURCE_EXHAUSTED + RESOURCE_EXHAUSTED`。这里的 resource 表示：没有可证明的
higher/same 前序阻塞，但所选路径未能在 admission budget 内提供容量，包含 KV/slot/token、
dispatch/engine ACK 和 backpressure，并不只表示 CUDA OOM。`8432` 只用于读取旧 peer 响应，
新 Master 不再生成。

### 8.3 多 endpoint

每个 endpoint 返回类型化 failure。只有 endpoint 对 higher/same 归因一致时，Scheduler 才公开该优先级原因；结果不一致或证据不足时统一返回 `RESOURCE_EXHAUSTED`。禁止在多个 endpoint 的自由文本中任意挑一个原因。

Planner 内部可保留最小事实对象用于可观测性和正确聚合，但不需要全部放入 wire：

```text
AdmissionFailureFacts {
  stage,
  resource_dimension,
  deficit,
  attribution_known,
  blocker_summary
}
```

只有测量事实能证明对应优先级原因时，才能填写 higher/same。owner missing、Cancel unsupported、policy victim cap、commit conflict 等真实 subcause 保留在内部事实和指标中；对外三分类没有 higher/same 证据时折叠为 broadened `RESOURCE_EXHAUSTED`，不声称发生了物理 OOM。

## 9. Schedule 失败协议

保留 Schedule response 已有稳定字段，只增加一个类型化原因：

```proto
enum ScheduleFailureReasonPB {
    SCHEDULE_FAILURE_REASON_UNSPECIFIED = 0;
    HIGHER_PRIORITY_AHEAD = 1;
    SAME_PRIORITY_AHEAD = 2;
    RESOURCE_EXHAUSTED = 3;
}

message FlexlbScheduleResponsePB {
    // existing fields 1-8
    ScheduleFailureReasonPB admission_reject_reason = 9;
}
```

字段职责：

- `code`：稳定的内部错误大类；
- `admission_reject_reason`：Master 在 incoming admission 决策现场产生的类型化原因；
- `error_message`：只用于诊断，禁止解析它判断公开状态；
- Dash 的 `status_code/status_name/status_message` 不进入 Master 协议。

Victim 的 8429 不属于 incoming admission reason：

- Stage 2 在 Schedule 完成前被抢占时，`code=8429` 本身就是完整证明；`admission_reject_reason` 保持 `UNSPECIFIED`；
- Stage 4 通过原 FetchResponse 的 `ErrorDetailsPB.error_code=8429` 返回，不经过 Schedule field 9；
- Dash 对精确 code 8429 固定映射，不要求额外 reason。

建议内部错误码：

| Code | 名称 | 含义 |
|---:|---|---|
| 8429 | `PRIORITY_PREEMPTED` | 已接纳 victim 被取消 |
| 8430 | `PRIORITY_ADMISSION_REJECTED` | incoming 因优先级阻塞被拒绝，reason 区分 higher/same |
| 8431 | `RESOURCE_EXHAUSTED` | 无可证明 higher/same blocker，且路径未在 admission budget 内提供容量 |
| 8432 | `ADMISSION_UNAVAILABLE` | 仅兼容读取旧 peer 响应；新 Master 不生成 |

### 9.1 类型化 reason 端到端传递

`admission_reject_reason` 必须保持类型化，不能在任一层退化为字符串：

```text
Java AdmissionResult
  -> Response.admissionRejectReason
  -> FlexlbScheduleResponsePB.admission_reject_reason
  -> Python MasterClient
  -> FtRuntimeException.admission_reject_reason
  -> BackendRPCServerVisitor 原样透传
  -> Dash mapper(exception_type, admission_reject_reason)
```

约束：

- `FlexlbServiceImpl.toProtoResponse` 和所有 response copy/forward 路径必须保留该字段；
- `MasterClient` 必须从 protobuf enum 构造 Python 类型化 enum，不能把它拼入 message；
- `FtRuntimeException` 增加可选的类型化 `admission_reject_reason` 属性；
- retry、access metric 和 Dash mapper 使用该属性；
- 字段缺失只能走明确 fallback，禁止解析 `error_message` 恢复原因；
- 必须有 Java proto golden test 和 Java Schedule response -> Python exception -> Dash 的合同测试。

## 10. 对外状态映射

Dash mapper 使用内部结果和显式请求 header：

```text
(internal code, typed failure reason, explicit qos header) -> public status
```

Master 的 failure reason 不由 QoS 阈值推断；QoS 只决定 incoming admission rejection
在 Frontend 的公开 `status_name`。

| 内部结果 | HTTP | status_name | status_message | 服务端透明重试 |
|---|---:|---|---|---|
| `8429` | 429 | `Throttling.Aborted` | `Too many requests.` | 禁止 |
| `8430/8431` 且显式 QoS `<50` | 429 | `Throttling.ServiceOverloaded` | `Too many requests.` | 禁止 |
| `8430/8431` 且显式 QoS `>=50` | 429 | `Throttling.ResourceExhausted` | `Too many requests.` | 禁止 |
| `8430 + HIGHER/SAME` 且无有效 QoS header | 429 | `Throttling.ServiceOverloaded` | 稳定原因消息 | 禁止 |
| `8431 + RESOURCE_EXHAUSTED` 且无有效 QoS header | 429 | `Throttling.ResourceExhausted` | 稳定资源消息 | 禁止 |
| 旧 peer `8432 + UNSPECIFIED` | 429 | `Throttling.ServiceOverloaded` | `Too many requests.` | 禁止 |
| 其他非法 code/reason 组合 | 429 | `Throttling.ServiceOverloaded` | `Too many requests.` | 禁止，并上报 protocol error |
| 无健康 worker/路由不可用 | 503 | `ServiceUnavailable` | `Service unavailable.` | 沿用有限 route retry |
| 单请求超过 hard limit | 400/413 | 参数/容量专用名称 | 稳定校验信息 | 禁止 |

约束：

- 8429 与 header、数值优先级无关，永远映射为 `429 / Throttling.Aborted / Too many requests.`。
- 只读取显式 `x-dashscope-inner-qos-level`；边界固定为 `<50` / `>=50`。
- reason 缺失、未知或与 code 不匹配时，禁止从 `error_message` 猜测，固定回落为 `429 / Throttling.ServiceOverloaded / Too many requests.`，并上报 protocol-error 指标。
- 公开 message 不暴露 victim request ID 或内部 owner 信息。

合法 code/reason 组合：

| code | 允许的 admission_reject_reason |
|---:|---|
| 8429 | `UNSPECIFIED`（victim 终态由 code 自描述） |
| 8430 | `HIGHER_PRIORITY_AHEAD`、`SAME_PRIORITY_AHEAD` |
| 8431 | `RESOURCE_EXHAUSTED` |
| 8432 | `UNSPECIFIED`（仅兼容旧 peer） |
| 8400 等既有非 admission 专用错误 | `UNSPECIFIED` |

### 10.1 Retry owner

- Master 内部的 version conflict、victim gone 等控制冲突，只能由 Master 在返回 Schedule response 前有限重试；
- 8429、8430、8431、兼容 8432 和 legacy 8511 一旦返回 Python，Backend 不得通过生成新 request ID 透明重试；
- 客户端是否根据 429 退避重试属于客户端策略，不属于服务端透明重试；
- Cancel transport unknown 只能针对同一 Prefill、同一 request ID 做有界幂等重试，不能释放容量、切换 victim 或生成新请求。

对于流式响应，如果 HTTP headers 或 token 已经发出，HTTP transport 可能保持 200；终止 Dash frame 必须携带 `status_code=429`、`status_name` 和 `status_message`。

### 10.2 实施边界：必须改与明确不改

必须改：

- **Proto**：Master-to-Prefill `CancelRequest/CancelResponse`；TaskInfo 的 `priority_preemption_progress`；Schedule 的 `admission_reject_reason`。
- **C++ Prefill**：请求在 prepare 前可取消、first-cause CAS、`PREEMPTING` gate、原始 Prefill active registry、terminal future/tombstone、8429 返回。
- **C++ Prefill/P-to-D**：复用当前请求链的 `TryCancel()`；原始 Prefill 在执行链退出后发布一次 WorkerStatus `CANCELED + 8429`。Decode 不拥有优先级终态原因。
- **Java Master**：从权威 inflight 解析原 Prefill；允许严格低优先级 `RUNNING`；用 `attemptToken + 完整 incoming provisional reservation` 做唯一资源结算；处理 typed Cancel status 和 typed incoming reason。
- **Python/Dash**：保留 8429 与 typed admission reason，禁止透明重试，按原因映射公开 429。

明确不改：

- 不向 Master 暴露 Decode Cancel RPC；
- 不建设通用 USER/DEADLINE/ADMIN Cancel 框架；
- 不用 QoS 阈值推断 Master 错误原因；Frontend 只按显式 header 选择公开名称；
- 不让 WorkerStatus `finished` 重新进入同步 Cancel 提交链；
- 不做与本请求链无关的 scheduler、async runner、KV allocator 或通用生命周期重构；
- 不把自由文本、mock 行为或 `resourceReleased=true` 合成值当作协议事实。

## 11. 并发不变量

1. Cancel 与自然完成只有一个线性化点。
2. Engine 在锁存 priority-preemption first-cause 并触发取消后即可返回弱 `ACCEPTED`；此时资源仍计为 used。
3. Master 不会在 `NOT_FOUND`、timeout 或 transport failure 时释放 accounting。
4. 所有必需 victim 都收到类型化 `CANCELED + 8429` 后，Master 才使用对应容量接纳 incoming。
5. 只有 priority preemption 赢得终态竞争时，victim 才能得到 8429。
6. Incoming 优先级拒绝永远不能得到 8429。
7. Stage 4 `RUNNING` victim 必须严格低于 incoming priority。
8. 同优先级请求绝不成为 victim；它们遵循 FIFO，可能让 incoming 得到 `SAME_PRIORITY_AHEAD`。
9. Cancel 始终发送到原始 Prefill 控制 owner。
10. WorkerStatus 的首次 `CANCELED + 8429` 是完成与结算依据；迟到或重复状态只能用于对账。
11. Incoming 的完整 provisional reservation 必须在发 Cancel 前建立；因此 victim accounting 删除时，不会出现容量被第三个请求抢走的窗口。
12. CancelCoordinator claim 生效期间，WorkerStatus 只能附加 observation，不能独立结算该 victim。
13. Master-local queued eviction 与 Engine weak-ACK Cancel 是两种 plan，不能混在同一个 attempt。

## 12. 测试方案

### 12.1 C++ UT/合同测试

- unknown request 返回 `NOT_FOUND`，且不安装 priority-preemption latch；
- 非法 request ID 返回 `INVALID_ARGUMENT`；
- Stage 2：卡住 remote allocation，发送 Cancel，验证 Prefill 立即返回 `ACCEPTED`、prepare 停止、WorkerStatus 先为 `CANCELING`，随后 EnqueueBatch 返回 raw 8429 并上报一次 `CANCELED + 8429`；
- Stage 3：在 Fetch 尚未建立时取消，验证立即返回 `ACCEPTED`、发布 8429 mailbox，未来 Fetch 得到 8429，且完成时上报一次 `CANCELED + 8429`；
- Stage 4：建立真实 P-to-D stream，向 Prefill Cancel，验证 `ACCEPTED` 不等待 Decode 退出；Decode 经既有 cancellation 路径停止，FetchResponse 返回 8429，原始 Prefill 再上报一次 `CANCELED + 8429`；
- 仅收到 `ACCEPTED` 或观察到 `CANCELING` 时，Master 必须保留 slot/KV accounting；测试要证明 incoming 尚未 dispatch；
- `PREEMPTING` 后 prepare/retry/handoff 不再产生副作用，已运行任务完成 join；并发 Fetch 只能等待 tombstone，不能 attach live context；
- 重复 Cancel 命中 `PREEMPTING` 时幂等返回 `ACCEPTED`，但不能重复安装 latch、启动 finalizer 或发布 `CANCELED`；
- Cancel 与自然成功/deadline/user cancel/engine error 两种先后顺序的 first-cause CAS barrier test；
- completion token 与 context 析构解耦，不因 Cancel waiter 持有 shared context 而自锁；
- Stage 2 未入 scheduler 的请求也能仅由 context latch 完成取消，不依赖 scheduler intent；
- Cancel 响应丢失后重试，从 active/tombstone 幂等返回 `ACCEPTED`；
- terminal tombstone 在原 Enqueue/Fetch 消费 8429 或请求 lease 到期前不得回收；
- 普通 upstream/user cancellation 仍返回普通 CANCELLED，不能被改成 8429。

### 12.2 Java Master 测试

- Planner 同时包含 `ACCEPTED_NOT_RUNNING` 和 `RUNNING`，且只选严格低优先级请求；
- higher blocker 得到 `HIGHER_PRIORITY_AHEAD`；
- 更早的 equal blocker 得到 `SAME_PRIORITY_AHEAD`；
- 无 higher/same 证据且 admission budget 内未提供容量时得到 `RESOURCE_EXHAUSTED`；
- priority provenance 未知时不能输出 higher/same；
- victim planner 选择覆盖 slot/KV 资源向量的最小确定性前缀，不多取消一个 victim；
- `ACCEPTED` 只进入 `CANCEL_REQUESTED`，不结算资源；WorkerStatus 首次 `CANCELED + 8429` 才以 `attemptToken` 幂等删除 victim accounting；全部成功后提升既有 incoming provisional reservation；
- 任意并发 admission 都不能在 victim settlement 与 incoming commit 之间取得 incoming 已预留的容量；
- 旧 WorkerStatus 在 settlement 后到达，不能复活 victim、覆盖 8429 或重复扣减 accounting；
- WorkerStatus 在 CancelResponse 之前到达时只附加 observation；分别验证后续 `ACCEPTED`、`NOT_FOUND`、transport unknown 三种收敛；
- WorkerStatus `CANCELING` 和 CancelResponse `ACCEPTED` 都不改变资源账本；首次 `CANCELED + 8429` 完成唯一 settlement，迟到重复状态只能更新指标；
- CancelResponse 丢失但先看到 `CANCELED + 8429` 时，Master 保留 child accounting；幂等 retry 命中 Engine active/tombstone 后只结算一次；
- `NOT_FOUND` 仅在完整无副作用条件成立时重规划一次，否则进入 stale/control failure；
- timeout/transport failure 保留 accounting，不能接纳 incoming；
- multi-victim 覆盖全部完成、`CANCELED+NOT_FOUND`、unknown 和 deadline 后迟到结果；全部必要 victim 完成后才 promote incoming claim。
- `NOT_FOUND_STALE` 和 `CANCEL_UNKNOWN` 覆盖每个 reconciliation 出口，包括 active 恢复、自然终态、幂等 `CANCELLED` 和 endpoint rebaseline；
- Master-local queued victim 与 `ENGINE_MAY_HAVE_SEEN/ACCEPTED/RUNNING` victim 使用不同 plan，不能混合结算。

### 12.3 Python/Dash 测试

- 8429 永不透明重试；
- QoS absent/49/50 对 8429 输出完全一致；
- 8430/8431 在显式 QoS 49/50 时分别输出 ServiceOverloaded/ResourceExhausted；
- 无有效 QoS header 时保留 typed higher/same/resource 的既有公开映射；
- unknown reason 不解析 `error_message`，并上报 protocol-error metric；
- legacy `error_msg` JSON 与 standalone `status_code/status_name/status_message` 一致；
- 已输出 token 后，terminal frame 仍保留完整 429 字段。

### 12.4 端到端 smoke

使用两个角色严格分离的真实或严格 gRPC Prefill/Decode：

1. 低优请求经过 Prefill 进入 Decode `RUNNING`；
2. 提交需要该容量的严格高优请求；
3. 验证 Master 只向原始 Prefill 发送 Cancel；
4. 验证 Prefill 经既有 P-to-D cancellation 触发 Decode 取消，并立即向 Master 返回弱 `ACCEPTED`；
5. 验证 victim FetchResponse 返回 raw 8429；
6. 验证 `ACCEPTED/CANCELING` 阶段 Master 保留 victim accounting，incoming 不会提前进入 Decode；
7. 验证原始 Prefill 后续上报一次 `CANCELED + 8429`，Master 此时才 settleOnce 并接纳 incoming；
8. 验证最终 Dash 状态为 `429 / Throttling.Aborted / Too many requests.`。

允许 Decode 直接接受 Cancel，或者直接删除 Decode mock state 的测试，不算通过该 smoke。

## 13. 可观测性

必须提供：

- Cancel request 数，按 endpoint 和 victim stage 分类；
- Cancel ACK 数：`accepted/not_found/timeout/transport_failure`；
- Cancel completion 数与延迟：`canceling/canceled/completion_timeout`；
- Engine Cancel terminal latency histogram；
- Master admission rejection 数，按 typed reason 分类；
- `NOT_FOUND` replan 成功/失败数；
- WorkerStatus 对账发现的 ledger mismatch 数；
- Schedule failure reason 缺失/未知的 protocol-error 数。

内部日志可以携带 request ID，公开响应不能携带。

## 14. 灰度与部署

Cancel 是新接口，不在协议里增加混部兼容分支。

建议顺序：

1. 部署支持弱 `ACCEPTED`、类型化 `CANCELING/CANCELED` 和 Stage 2/4 的 Prefill/Engine；
2. 部署支持 typed Cancel response 和 typed admission reason 的 Master，功能开关保持关闭；
3. 部署 Python/Dash typed mapping 与 8429 no-retry；
4. 运行严格 P/D 端到端 smoke；
5. 按实例配置逐步打开 priority preemption；
6. 观察 Cancel latency、`NOT_FOUND`、admission reason 和对账 mismatch 后扩大流量。

任何目标 Prefill 不支持弱 ACK 或不生产类型化 `CANCELED + 8429` 时，不允许打开该功能。

## 15. 验收标准

以下条件全部满足才算实现完成：

- Cancel 永远发往原始 Prefill；
- Stage 2 与 Stage 4 都能终止；
- `CancelResponse=ACCEPTED` 只证明 priority-preemption first-cause 已锁存并已触发取消，不能作为资源释放凭证；
- Master 收到 `ACCEPTED` 或 `CANCELING` 后保持 accounting，只有原始 Prefill WorkerStatus 的首次 `CANCELED + 8429` 才能清理；
- incoming 的完整 provisional reservation 在 Cancel 前即建立，不存在 victim 结算后容量被并发请求抢走或重复计费的窗口；
- `NOT_FOUND` 不写 intent；只有无副作用条件成立时才触发一次有界重规划；
- `NOT_FOUND_STALE`、`CANCEL_UNKNOWN` 和 CancelResponse 前到达的 WorkerStatus 都有确定、幂等的 reconciliation 出口；
- Stage 4 `RUNNING` 是合法的严格低优先级 victim；
- Victim 终态错误码精确为 8429，且不透明重试；
- Incoming 拒绝原因由 Master 在一致快照上产生；
- Dash 只对 8430/8431 使用显式 QoS `<50` / `>=50`；8429 不受 QoS 影响；
- C++/Java/Python 合同测试和严格 P/D smoke 全部通过。
