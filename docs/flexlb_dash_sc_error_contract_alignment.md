# FlexLB / frontend / dash-sc 错误协议：按业务决策分类

## 1. 状态与修正范围

2026-09-18 修订。FlexLB 负责错误分类，frontend/dash-sc 复用既有消费协议。本文描述当前实现，不作为部署或测试执行记录。

- 保留原有 FRONTEND 角色选择与 `NO_FRONTEND_WORKER(8407)` 定义；本轮不调整服务发现或角色路由。
- 到期不等于 dispatch 失败。排队原因由实际 Prefill／Decode 门限和组批决策写入 queue；到期直接读取，不扫描或重新归因。
- `BatcherContext.waitReason` 保存队列最近一次实际等待原因，不在每个请求上复制状态，也不新增等待原因枚举。交付阶段复用现有生命周期。
- 不改变 Cancel 的完成凭证、资源释放边界和终态唯一性。

## 2. 真实调用职责

| 环节 | 职责 | 失败归属 |
|---|---|---|
| dash-sc proxy → frontend | proxy 经自己的服务发现选择下游 frontend | proxy 发现／连接失败；不是 FlexLB “无 frontend worker” |
| frontend → FlexLB Schedule | 提交推理请求并取得调度结果 | 参数校验、路由、入场、调度／下发结果 |
| DIRECT | 选择推理 worker，返回路由结果 | 路由失败，不存在 FlexLB 队列等待 |
| QUEUE + FIFO/PRIORITY | 入场并等待 Prefill 调度 | 前序阻塞、容量门限或调度未完成 |
| BATCH delivery | FlexLB 通过 EnqueueBatch 下发 | 开始下发后的错误不能重新称为 incoming 入场拒绝 |
| NON_BATCH delivery | FlexLB 返回地址，由 frontend 发给 Engine | 返回路由前受 request-slot/Decode dispatch 门限约束；交付后遵守既有归属 |
| Schedule 成功之后 | Engine 执行／Fetch stream 交付后续结果 | 不再创建第二个 Schedule 终态 |

Frontend 服务发现与角色路由保持原实现；Java 保留 8407 不代表 Python 已有对应映射，不能据此扩展消费者错误协议。

## 3. 公共返回约定

- FlexLB Schedule 业务失败：gRPC OK，`success=false`。
- dash-sc 业务错误帧：native gRPC OK，`finished=true`，`finish_reason=1000`，顶层 `error_message=""`。
- 以下 HTTP 指 dash-sc 参数中的 `status_code`，不是 native gRPC 状态。JSON `error_msg` 与独立 `status_*` 参数一致。
- 有效 QoS：低 1..49，高 50..100；缺失、非法和越界均为无 QoS。内部默认 priority=50 不代表调用方有高 QoS。
- Cancel ACK、客户端断连和真实传输失败不等同于可交付的业务错误帧。

下表记录**当前工作区实现**，不是把期望规范当成已实现结果。表中的 dash-sc 返回指该异常最终进入错误映射时的结果；frontend 存在重试，因此某一次 FlexLB 失败不一定成为上游终态，见 6.1。外部码按 `error_no / HTTP status / status_name` 书写；`status_message` 单列说明。下表列出的参数、容量、取消和超时业务错误使用上述 `finish_reason=1000` 约定，不外推到所有 Engine 内部错误或 native gRPC 传输异常。

## 4. 入场与排队：原因和返回码

### 4.1 场景对照表

| case | FlexLB code / reason | dash-sc：低 QoS 1..49 | dash-sc：高 QoS 50..100 | dash-sc：无／非法 QoS | status_message／内部诊断 |
|---|---|---|---|---|---|
| A1. 准入拒绝决策确认容量阻塞者是可信高优请求 | 8430 / HIGHER_PRIORITY_AHEAD(1) | 5 / 429 / Throttling.ServiceOverloaded | 5 / 429 / Throttling.ResourceExhausted | 5 / 503 / ServiceUnavailable | 有效 QoS：`Too many requests.`；无 QoS：`Service unavailable.`；FlexLB：`higher-priority requests are ahead` |
| A2. 准入拒绝决策确认无高优，但有可信同优阻塞者 | 8430 / SAME_PRIORITY_AHEAD(2) | 5 / 429 / Throttling.ServiceOverloaded | 5 / 429 / Throttling.ResourceExhausted | 5 / 503 / ServiceUnavailable | 同 A1 的外部固定消息；FlexLB：`same-priority requests are ahead` |
| A3. QUEUE outstanding 全局门限耗尽 | 8431 / RESOURCE_EXHAUSTED(3) | 5 / 429 / Throttling.ServiceOverloaded | 5 / 429 / Throttling.ResourceExhausted | 5 / 503 / ServiceUnavailable | 同 A1 的外部固定消息；内部诊断标明 outstanding 门限 |
| A4. PRIORITY delivered-not-accepted 入场门限耗尽 | 8431 / RESOURCE_EXHAUSTED(3) | 5 / 429 / Throttling.ServiceOverloaded | 5 / 429 / Throttling.ResourceExhausted | 5 / 503 / ServiceUnavailable | 同 A1 的外部固定消息；内部 trigger 标明 backpressure 门限 |
| A5. PRIORITY placement 证实 Decode KV／slot 或 Prefill 队列容量不足，分类结果为纯容量原因 | 8431 / RESOURCE_EXHAUSTED(3) | 5 / 429 / Throttling.ServiceOverloaded | 5 / 429 / Throttling.ResourceExhausted | 5 / 503 / ServiceUnavailable | 同 A1 的外部固定消息；有可信高优／同优或未知优先级归因时分别走 A1/A2/A9 |
| A6. NON_BATCH 的实际等待决策记录 Prefill request slots 满 | 8431 / RESOURCE_EXHAUSTED(3) | 5 / 429 / Throttling.ServiceOverloaded | 5 / 429 / Throttling.ResourceExhausted | 5 / 503 / ServiceUnavailable | 同 A1 的外部固定消息；内部 `prefill request slots exhausted` |
| A7. BATCH 的实际等待决策记录 Prefill batch slots 满 | 8431 / RESOURCE_EXHAUSTED(3) | 5 / 429 / Throttling.ServiceOverloaded | 5 / 429 / Throttling.ResourceExhausted | 5 / 503 / ServiceUnavailable | 同 A1 的外部固定消息；内部 `prefill batch slots exhausted` |
| A8. queued Decode reservation 在实际 `tryClaimEngineDispatch` 被 slot gate 拒绝 | 8431 / RESOURCE_EXHAUSTED(3) | 5 / 429 / Throttling.ServiceOverloaded | 5 / 429 / Throttling.ResourceExhausted | 5 / 503 / ServiceUnavailable | 同 A1 的外部固定消息；内部 `decode engine slots exhausted`；派发门限只确认容量，不重新扫描占用者推断优先级 |
| A9. 准入拒绝中相关阻塞者的 priority 缺失／非法，无法可信归因 | 8432 / UNSPECIFIED(0) | 5 / 503 / ServiceUnavailable | 5 / 503 / ServiceUnavailable | 5 / 503 / ServiceUnavailable | 外部固定 `Service unavailable.`；内部 `admission unavailable; blocker priority attribution is unavailable` |
| A10. Master placement 预算耗尽，尚未完成 inflight 交接 | 8431 / RESOURCE_EXHAUSTED(3) | 5 / 429 / Throttling.ServiceOverloaded | 5 / 429 / Throttling.ResourceExhausted | 5 / 503 / ServiceUnavailable | 标准容量消息，trigger=`Master placement did not complete within budget`；覆盖 submit、priority admission 入口、注册前到期及等待 admission mutation 清理的到期终态，不宣称 GPU/KV 不足 |
| A11a. 已记录组批收集窗口等待，预算内未形成交付；同批次成员不构成优先级阻塞 | 8431 / RESOURCE_EXHAUSTED(3) | 5 / 429 / Throttling.ServiceOverloaded | 5 / 429 / Throttling.ResourceExhausted | 5 / 503 / ServiceUnavailable | 标准容量消息，trigger=`batch collection window exceeded request scheduling budget`；指预算内无法交付，不宣称 GPU/KV 不足 |
| A11b. 尚未观察到具体容量／收集等待，预算内未完成交付 | 8431 / RESOURCE_EXHAUSTED(3) | 5 / 429 / Throttling.ServiceOverloaded | 5 / 429 / Throttling.ResourceExhausted | 5 / 503 / ServiceUnavailable | 标准容量消息，trigger=`queue has no recorded wait reason`；不推断未实际观察到的资源门限 |
| A12. 已取得 delivery claim 后到期 | 8511 / UNSPECIFIED(0) | 5 / 503 / ServiceUnavailable | 5 / 503 / ServiceUnavailable | 5 / 503 / ServiceUnavailable | 诊断携带现有 claim 与触发信息；不能把 executor 排队说成 Engine 已见。仍按原取消凭证清理，Schedule 完成后不再发布第二个终态 |
| A13a. FIFO/PRIORITY offer 在队列锁内确认队列满，读取已维护的优先级占用计数 | 8430 / reason 1、2；或 8431 / reason 3；或 8432 / reason 0 | 8430/31：5 / 429；8432：5 / 503 | 8430/31：5 / 429；8432：5 / 503 | 5 / 503 / ServiceUnavailable | 完整外部消息与 status_name 分别见 A1/A2/A5/A9；包括仍占队列 slot 的 pending delivery，不将其当作可抢占 victim |
| A13b. offer／replacement 时队列已停止 | 8510 / UNSPECIFIED(0) | 5 / 503 / ServiceUnavailable | 5 / 503 / ServiceUnavailable | 5 / 503 / ServiceUnavailable | `Prefill queue stopped ...`；不进入容量重试／抢占 |
| A14a-1. Prefill victim 已离开，重新 offer 被拒绝；同步重试耗尽或异步交接终止 | 保留最后一次实际 offer 的 8430/8431/8432 | 同 A13a | 同 A13a | 5 / 503 / ServiceUnavailable | 同步路径使用原重规划预算；异步 Cancel 已完成后的交接直接交付实际拒绝，不启动第二轮 Engine 抢占，也不忽略返回结果等待 deadline |
| A14a-2. Decode reservation 变化后重规划仍未取得准入容量，预算耗尽 | 8431 / RESOURCE_EXHAUSTED(3) | 5 / 429 / Throttling.ServiceOverloaded | 5 / 429 / Throttling.ResourceExhausted | 5 / 503 / ServiceUnavailable | `admission capacity is temporarily exhausted`；reservation 竞争只记日志，不对外产生 8515 |
| A14a-3. 注册失败且 generation 已关闭 | 不发布新的业务错误 | 由已有终态所有者决定 | 由已有终态所有者决定 | 由已有终态所有者决定 | 只回滚本次未交接资源；取消、到期、关闭不是计划竞争 |
| A14a-4. generation 仍开放，但 request_id 已在 inflight | 8406 / UNSPECIFIED(0) | 8 / 400 / InvalidParameter | 8 / 400 / InvalidParameter | 8 / 400 / InvalidParameter | `duplicate request_id: <id>`；不进入重试／抢占 |
| A14b. router 返回 null，或 victim 删除／入队抛出异常 | 8510 / UNSPECIFIED(0) | 5 / 503 / ServiceUnavailable | 5 / 503 / ServiceUnavailable | 5 / 503 / ServiceUnavailable | 对应无决策或异常诊断；不是已查明 GPU 容量不足 |
| A14c. replacement 已删除部分 victim，但 incoming 的 offer 被容量拒绝 | 按实际 offer 的 8430/8431/8432 原因 | 同 A13a | 同 A13a | 5 / 503 / ServiceUnavailable | 原因不再被 `partial failure` 覆盖；已移除 victim 继续完成原终态清理 |
| A14d. Decode 抢占成功，后续 Prefill 未选出或 endpoint 已注销 | 8402 / UNSPECIFIED(0) | 5 / 503 / ServiceUnavailable | 5 / 503 / ServiceUnavailable | 5 / 503 / ServiceUnavailable | Prefill 路由／注册状态诊断透传；释放 incoming Decode reservation，不伪装成已证明的容量不足 |
| A14e. Cancel coordinator 异常完成，或完成回调抛异常 | 8510 / UNSPECIFIED(0) | 5 / 503 / ServiceUnavailable | 5 / 503 / ServiceUnavailable | 5 / 503 / ServiceUnavailable | 调度控制流程异常诊断；与正常 CONTROL_FAILED 表示未能回收入场容量分开 |

shutdown 保留原有各入口行为：PriorityAdmissionScheduler 的停止拒绝仍为 8431，PriorityScheduler 的停止兜底为 8510。本轮不统一该协议，也不引入 gRPC UNAVAILABLE 返回路径。

8431 标准 message 为 `admission capacity is temporarily exhausted`，决策诊断使用 `; trigger=…`，响应出口的补充信息使用 `; context=…`。8430/8432 保持固定 message，不能用重试诊断覆盖标准描述。`Response.error(...)` 统一校验 code/reason 并规范消息，调度链路直接传递失败响应；两个 scheduler 不再分别维护规范消息规则，也不再保留独立失败包装类。响应按请求创建，不共享可变实例。

### 4.2 dash-sc QoS 矩阵

| FlexLB code | QoS | error_no | HTTP | status_name | status_message |
|---|---|---:|---:|---|---|
| 8430 / 8431 | 低：1..49 | 5 | 429 | Throttling.ServiceOverloaded | `Too many requests.` |
| 8430 / 8431 | 高：50..100 | 5 | 429 | Throttling.ResourceExhausted | `Too many requests.` |
| 8430 / 8431 | 无／非法 | 5 | 503 | ServiceUnavailable | `Service unavailable.` |
| 8432 | 任意 | 5 | 503 | ServiceUnavailable | `Service unavailable.` |

### 4.3 分类不能跨越的边界

1. queue 保存最近一次实际排队原因。组批 loop 记录 Prefill batch 容量或收集等待；ready staging 和 route delivery 记录 Prefill request 容量；Decode claim 拒绝记录 Decode engine slot 容量。
2. 超时只读取 queue 状态，不取快照、不遍历前序请求、不调用容量判断或预测器。原因读取是 O(1)；队列移除和资源清理的原有成本不包括在内。
3. 已排队请求的 Decode KV 在 placement 已预留，派发门限不重复 KV／优先级归因。
4. 8430/8432 保留给准入决策的有据拒绝，不在排队到期时根据其他请求的优先级重新构造。
5. 8431 的容量口径包括 Master 在预算内无法完成交付。没有观察到等待原因时，保留 Master 调度预算耗尽诊断，不推断 Prefill 或 Decode 不足。
6. Schedule 已完成不发布第二个终态；已取得 delivery claim 的到期仍返回 8511。
7. 重试开始不清除上一次门限结果；下一次实际决策覆盖它，成功交付或丢失资源所有权后清除。已有队列的新入队不覆盖等待状态；队列清空后的首次入队清除旧状态。
8. Prefill batch 门限使用 WorkerBatcher 配置；Decode dispatch 门限使用当前生效配置。NON_BATCH 同时受 ready staging 和最终 delivery 的 request-slot 门限约束。

### 4.4 placement 与 delivery 到期的证据边界

API 的 QUEUE deadline 由本地 `BalanceContext.startTime + queueTimeoutMs` 创建；不是调用方的生成超时。outstanding／active-admission 门限失败是即时容量拒绝，不存在额外的 placement 等待队列。A10 的 8431 表示 Master 入场服务未在预算内完成，不代表已查明 GPU/KV 不足；仅凭 deadline 仍不能区分锁竞争、执行耗时与 worker 容量问题。已取得的容量拒绝由原决策交付；已关闭的 generation 由原终态所有者交付，不另造计划冲突。已取得 delivery claim 的到期仍走 A12，不能套用 A10。

BATCH 的 `markBatchEnqueueStarted` 在 dispatcher executor 提交前发生，不是 RPC 已发送的证据。到期响应直接使用已有 delivery claim 与触发信息作诊断，不新增派发阶段状态。Engine 可见性、Cancel ACK/终态和资源回收仍由已有生命周期协议决定。

## 5. 抢占与取消

| 场景 | FlexLB／Engine 结果 | QoS | dash-sc error_no | HTTP / status_name | status_message |
|---|---|---|---:|---|---|
| Victim 被高优请求抢占，包括 Master queued／Decode reserved；Engine 可能已见时须先确认清理终态 | 8429 PRIORITY_PREEMPTED / UNSPECIFIED | 有效 1..100 | 10 | 429 / Throttling.Aborted | `Too many requests.` |
| 同上 | 8429 / UNSPECIFIED | 无／非法 | 5 | 503 / ServiceUnavailable | `Service unavailable.` |
| Cancel 早于 active 注册 | Engine TOMBSTONED(3)，控制 ACK | 任意 | — | 无即时业务错误 | 不将 ACK 当作抢占终态 |
| Cancel 已接收、清理中 | ACCEPTED(1) / CANCELING，等待终态 | 任意 | — | 无即时业务错误 | 不提前发布 8429 |
| 用户主动取消，响应仍可交付 | 8504 REQUEST_CANCELLED / UNSPECIFIED | 任意 | 10 | 499 / ClientClosedRequest | 透传取消诊断 |

已接纳 victim 统一返回 8429，不再借用 8400 触发重新调度。Engine 是否见过请求只决定 Cancel 与资源清理流程。Engine 抢占终态为 CANCELED + 8429；Fetch 为 gRPC 8 + ErrorDetailsPB(8429)。FlexLB message 为 `preempted by higher-priority request <id>`。Schedule 已完成时由 Engine stream 交付，不生成第二个 Schedule 终态。客户端已经断开时不能保证交付取消错误帧。

## 6. 其他业务错误、协议错误与兼容边界

| 场景 | FlexLB／frontend code | dash-sc error_no | HTTP / status_name | status_message／说明 |
|---|---|---:|---|---|
| proxy 无法发现下游 frontend | 无 FlexLB code | 5 | 503 / ServiceUnavailable | proxy 自己产生 `forward backend unavailable` |
| 非法请求、重复 request_id、BATCH 缺 generate_input | 8406 MASTER_INVALID_REQUEST | 8 | 400 / InvalidParameter | 校验诊断透传 |
| 没有可用推理 worker | 8400 NO_AVAILABLE_WORKER | 5 | 503 / ServiceUnavailable | 路由诊断透传，不冒充 admission 容量决策 |
| 未选出可用 Prefill worker，且走原路由错误出口 | 8402 MASTER_NO_PREFILL_WORKER | 5 | 503 / ServiceUnavailable | 路由诊断透传；不表示已证明是 KV 或并发容量不足 |
| 未选出可用 Decode worker，且未被 PRIORITY admission 分类替换 | 8403 MASTER_NO_DECODE_WORKER | 5 | 503 / ServiceUnavailable | 路由诊断透传；有入场容量证据时见第 4 节 |
| 未选出可用 Prefill/Decode 融合 worker | 8404 MASTER_NO_PDFUSION_WORKER | 5 | 503 / ServiceUnavailable | 路由诊断透传 |
| 未选出可用 VIT worker | 8405 MASTER_NO_VIT_WORKER | 5 | 503 / ServiceUnavailable | 路由诊断透传 |
| Batch/dispatch、非 deadline 的 Master 转发失败 | 8510 BATCH_DISPATCH_FAILED | 5 | 503 / ServiceUnavailable | 动态诊断透传 |
| Worker 执行失败 | 8513 WORKER_EXECUTION_FAILED | 5 | 503 / ServiceUnavailable | 执行失败诊断透传 |
| 历史 Batch token 容量错误码（保留兼容映射） | 8514 BATCH_TOKEN_CAPACITY_EXCEEDED | 5 | 503 / ServiceUnavailable | 当前组批逻辑只限制追加成员，不据此拒绝单请求 |
| 8429～8432 的非法 code/reason 配对 | typed 协议异常 | 5 | 503 / ServiceUnavailable | 原 dash-sc 已固定 `Service unavailable.` 并记录协议错误 |
| Follower → Master RPC deadline，或 API TimeoutException | 8511 BATCH_SLO_EXPIRED | 5 | 503 / ServiceUnavailable | 暂保留原有超时响应兼容契约，不是新队列出口 |
| 已入队，预算内未完成交付，包括收集窗口／Master 调度等待 | 8431 / RESOURCE_EXHAUSTED | 5 | 按第 4 节 QoS 矩阵 | 直接读取 queue 记录的等待原因，不重新分析其他请求 |
| frontend → FlexLB 实际 gRPC deadline | 无 FlexLB 业务响应；frontend 8204 DEADLINE_EXCEEDED | 13 | 504 / GatewayTimeout | 真正调用超时 |
| 旧队满／旧队列超时 | 8502 / 8503 | 5 / 13 | 503 ServiceUnavailable / 504 GatewayTimeout | 保留解码，不是新队列拒绝方案 |
| Decode reservation 变化后重规划仍未取得准入容量 | 8431 RESOURCE_EXHAUSTED | 5 | 按第 4 节 QoS 矩阵 | 标准容量消息；8515 仅保留历史错误码解码兼容，不再由调度器产生 |
| 旧 batch 构建失败 | 8512 | 5 | 503 / ServiceUnavailable | 保留解码能力，不据此声称当前生产路径必然会发出 |

合法普通业务错误均为 UNSPECIFIED reason，QoS 不改变其映射。DIRECT 的显式过期检查仍保留 8511；实际 DIRECT API 不设置队列等待 deadline。

8511 与 frontend 8204 的外部差异目前仍是兼容边界，不宣称已统一所有传输超时契约，也不因修正队列分类而把所有 8511 改成 504。

此次不修改消费者的通用协议容错：原 MasterClient 会把未知 code 回退为 8400，且未全面校验 success/code 矛盾。不能声称仅修改 FlexLB 就补齐了这类跨版本防御；本轮在 FlexLB 错误响应构造处限制合法 code/reason。原 frontend 对 8429～8432、8511 不重试，对其他错误保留原有重试规则，不扩大禁止重试集合。

### 6.1 重试之后，上游最终看到什么

以下描述 `BackendRPCServerVisitor.stream_with_aux_info` 的现有行为。重试还要求未输出数据、有剩余次数以及可用的 request_id factory；Java 枚举的 `canRetry=false` 不等于 Python 已禁止重试。

| case／错误序列 | 当前 frontend 行为 | 最终 dash-sc 结果 |
|---|---|---|
| 8429、8430、8431、8432 或 8511 | 作为终态错误，不重新发起调度；若它出现在重试后，优先保留这个终态 | 按第 4～6 节对应行输出 |
| 8400、8402～8406、8504、8510、8513、8514 等非上述终态集合的异常 | 满足条件时可能换 request_id 重试 | 不能仅凭第一次 FlexLB code 确定最终错误 |
| 8510 → 后续重试成功 | 不输出第一次失败 | 正常推理结果，无该失败对应的错误帧 |
| 8400 → 8431，QoS=49 | 后续 typed admission 终态覆盖先前错误 | 5 / 429 / Throttling.ServiceOverloaded / `Too many requests.` |
| 8204 → 8430，QoS=50 | 后续 typed admission 终态覆盖先前超时 | 5 / 429 / Throttling.ResourceExhausted / `Too many requests.` |
| 8400 → 8513，重试终止且无 typed admission 终态 | 保留第一次异常 8400，而非最后一次 8513 | 5 / 503 / ServiceUnavailable；透传第一次 8400 的诊断 |
| 第一次是 8204，后续也失败且没有 typed admission 终态 | 最终保留第一次实际 RPC deadline | 13 / 504 / GatewayTimeout；透传第一次超时诊断 |

### 6.2 协议异常与传输异常：当前实现，不当作合法业务场景

| case | 当前处理 | 最终对外结果／限制 |
|---|---|---|
| 8429 reason 非 0；8430 reason 非 1/2；8431 reason 非 3；8432 reason 非 0 | dash-sc typed contract 校验失败，记录 protocol error | 任意 QoS：5 / 503 / ServiceUnavailable / `Service unavailable.` |
| FlexLB 返回未知 code，例如 8407、8999 | MasterClient 回退为 8400；可能参与普通路由重试 | 若该异常最终交付：5 / 503 / ServiceUnavailable，透传原诊断；不是固定的 protocol-error 消息 |
| 普通 code 配了不合法 reason，例如 8400 + HIGHER_PRIORITY_AHEAD | 原 dash-sc 仅对 8429～8432 校验配对 | 若该异常最终交付：仍按普通 code 映射，如 8400 为 5 / 503，透传诊断；不保证 protocol error log |
| success=false 但 code=200 | 原 MasterClient 按 code=200 进入成功解析分支 | 没有统一的协议错误返回；后续依赖地址等字段，不能标为已实现固定 503 |
| success=true 但 code=8406 | 原 MasterClient 按错误 code 处理 | 若最终交付该异常：8 / 400 / InvalidParameter，透传诊断 |
| proxy 已连接 frontend，frontend 调用抛出 native gRPC 错误 | proxy 通过 `context.abort` 传播下游状态 | native gRPC 非 OK；不能保证有 finished/error_no/status 参数错误帧 |
| 客户端断连／取消传输 | 不再具备可靠交付响应的条件 | 不能保证上游收到 8504 对应的 499 业务错误帧 |

offer 拒绝及 delivery gate 原因由决策处记录；队列关系按到期时的状态诊断；A10 已统一为明确 Master placement 阶段的入场失败。消费者通用协议容错仍是单独边界。Master 调度能力不足不等于已查明 GPU/KV 不足。

## 7. 实现职责

| 所有者 | 职责 |
|---|---|
| common `Response.error(...)` | 校验 code/reason，规范公开消息，直接构造失败响应；与队列及抢占策略无关 |
| `AdmissionFailureClassifier` | 根据 Decode 容量快照解释准入拒绝 |
| `BatcherContext.occupiedSlotsByPriority` | 随容量占用与释放维护 1～100 及未知优先级的计数；队列满时直接归因，包含 pending delivery，不扫描请求 |
| `BatcherContext.waitReason` | 保存 Prefill／Decode 实际门限或收集等待的最近决策；超时直接读取 |
| `DecodeEndpoint.tryClaimEngineDispatch` | 复用原有 CLAIMED / CAPACITY_FULL / NOT_OWNED；只管理资源所有权，不构造业务错误 |
| `PriorityScheduler.buildExpirationResponse` | 区分已取得交付权与未交付请求，再选择对应诊断 |
| `PlanCommitter.CommitResult` | 区分提交成功、generation 关闭、带具体错误的拒绝 |

成功入队只增加对应优先级的计数，不调用就绪预测或全队列分类。队列满时读取固定 100 个优先级计数，开销与请求数量无关，不复制快照、不构造请求 envelope；取消、排空与交付完成时同步扣除计数，staging／重试不重复增减。组批 worker 的容量等待也不为诊断反复扫描队列。Decode dispatch 直接使用原有 CLAIMED / CAPACITY_FULL / NOT_OWNED 结果。容量拒绝后不读取版本、不复制 layered view、不遍历 reservation／confirmed task；只记录容量失败，重试成功后清除。优先级归因只保留在准入拒绝决策中。

超时归因只读 queue 的一个 volatile 状态字段，开销 O(1)。loop 及交付门限在实际决策分支写入字符串常量，不做额外队列扫描，也不为每个请求分配失败对象。queue 状态表达最近一次队列决策，不承诺是每个请求独立的历史原因。

`dropHead` 只负责移除请求，成功后在队列锁外通知 handler，不再保存或计算到期原因。

无可行 Decode 抢占方案时沿用直接拒绝；只有 victim 冲突消耗重规划预算。`scheduleAttempt` 记录实际 placement 次数，包括冲突重规划。

Master 转发结果保留原始 Throwable。真实 TimeoutException（含 CompletionException / ExecutionException 包装）及 gRPC DEADLINE_EXCEEDED 映射为超时；诊断字符串不参与错误码选择。

## 8. 回归检查

- 队列满拒绝不遍历请求；高优／同优／未知优先级归因使用容量计数，包含动态降限及 pending delivery。
- 非空队列的新入队不覆盖等待原因；清空后首次入队清除旧原因。
- Prefill batch／request 门限和 Decode slot 门限在决策处更新 queue 状态。
- 重试成功清除容量等待；开始重试不会短暂丢失上一次门限原因。
- 超时不查询队列快照、其他请求优先级、配置或预测器。
- Cancel、抢占交接、资源回滚及终态唯一性沿用原协议。

功能测试：

```bash
./mvnw -pl flexlb-api -am test
```

独立的配置与到期性能回归：

```bash
./mvnw -P sync-performance-regression -pl flexlb-sync -am \
  -Dtest=SchedulingConfigAndExpirationPerformanceTest \
  -Dsurefire.failIfNoSpecifiedTests=false test
```

消费者兼容范围及未统一的传输错误边界见第 6 节；这里的结构调整不修改 frontend/dash-sc 或 Engine 生产代码。


## 9. 失败 PV 诊断

失败时在现有 `pv.log` 增加 `schedulingDiagnostics`，不改变对外 code / message。
成功响应不输出该字段；重新规划、成功入队会清掉旧尝试的诊断。
诊断只保存数值和字符串，不持有队列请求、Endpoint 或规划快照对象。

| 字段 | 含义 |
| --- | --- |
| `cause` | 内部真实原因，如 Prefill 队列容量耗尽、Decode slot 等待、victim reservation 冲突及已释放/计划 victim 数 |
| `capturedAtMs` | 队列／交付失败摘要的保存时间 |
| `prefill` | 队列拒绝、超时或交付失败时的单 worker 摘要：endpoint、queueDepth/capacity/version、ordering、higher/same/lower/unknownPriorityCount、readyCount、pendingCount、waitReason、inflightRequests/Batches |
| `prefill.aheadCountUpperBound` | 前序请求数量的保守上界；不是精确排名。计数包含仍占位的当前请求和后到的同优请求，ready/pending 请求也计入上界 |
| `decode` | 关联 Decode 的 reservedRequests、totalLoad、engineLoad、kvAvailable/Total、hard/expectedKvReserved；地址缺失时省略 endpoint |
| `prefillEndpointCount` / `decodeEndpointCount` | 入场规划失败时直接读取已有候选 Map 的 size，不展开各 endpoint |
| `endpoint` / `plannedVictims` / `freedVictims` | Decode reservation 冲突分支已有的目标和 victim 数量 |

队列诊断持队列锁读取固定 100 个优先级计数，包含已占位的 pending delivery；
不遍历请求、不排序、不重跑预测器。Decode 只读现成计数器，跨计数器为弱一致视图。
入场规划诊断只保留分支已有的原因和数量，不为 PV 遍历或展开集群快照。
超时原因仍然直接读取 queue 的最后等待状态；诊断不会参与错误码归因。
在本路径释放资源前保存摘要；若上游已经移除请求或释放 victim，摘要反映诊断保存时的剩余占用，不倒推历史值。

示例（省略其他 PV 字段）：

```json
{
  "code": 8431,
  "error": "admission capacity is temporarily exhausted",
  "schedulingDiagnostics": {
    "cause": "decode engine slots exhausted; context=request expired",
    "capturedAtMs": 1789720000000,
    "prefill": {
      "queueDepth": 8,
      "higherPriorityCount": 3,
      "samePriorityCount": 2,
      "readyCount": 0,
      "pendingCount": 0,
      "aheadCountUpperBound": 5,
      "waitReason": "decode engine slots exhausted"
    },
    "decode": {
      "reservedRequests": 8,
      "totalLoad": 12,
      "engineLoad": 4,
      "kvAvailable": 512,
      "hardKvReserved": 1024
    }
  }
}
```
