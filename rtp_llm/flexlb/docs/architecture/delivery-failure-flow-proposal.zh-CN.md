# 请求失败处理：现行流程与接口

2026-09-16。已按此结构实现，`fetchAttachTimeoutMs` 保留。本文件替代此前多层 Delivery/Terminal 转发的方案。

## 主流程

```text
completeDelivery / failDeliveryPreparation
    → failRequest：锁内确定失败结果
    → cleanUpRequest：锁外发布后的本地清账
    → finishRequest：锁内检查能否结束跟踪
```

`completeDelivery` 只消费一次回调并区分成功、明确失败、不确定结果。本地准备失败走同一条失败流程。

| 方法 | 职责 |
|---|---|
| `RequestSlot.failRequest(exact, source, detail)` | 在 Slot 锁内确定失败/已有取消首因，关闭派发入口，选定一次响应。返回已有的 `SelectedPublication`，调用方在锁外提交。 |
| `RequestSlot.cleanUpRequest(exact, source)` | 在 Slot 锁外分别尝试两侧清账，汇总异常；回到锁内合并进度并调用 finishRequest。晚到回调仍清理原 exact。 |
| `RequestSlot.finishRequest(event)` | 检查清账、准入、抢占是否已收尾。满足条件后复用既有终态执行器释放定时器、提交终态记录。 |
| `PrefillEndpoint.settleFailedRequest(exact)` | 删除精确排队记录、结算精确成员；重复清理不影响其他成员。 |
| `DecodeEndpoint.settleFailedRequest(reservation, source)` | 按来源尝试合法回滚，并在 Endpoint 锁内判断这个 exact 是否还有容量或协议责任。返回 true 表示无剩余责任，false 表示仍由 Endpoint 跟踪。 |

没有新增 DeliveryFailureAction、LocalSettlement 枚举或另一套响应发布框架。

## 请求结果与资源责任

失败响应先提交，不等待 Decode 或抢占结束。`RequestState.FAILED` 可以与尚未结束的本地跟踪同时存在。

`CleanupProgress cleanupProgress` 只保存清理来源、执行阶段、Prefill/Decode 是否结清，以及已决定的过期清理。原请求复用 Slot 的 `item`，已选定结果复用 `state/detail`，抢占是否结束复用 `PreemptionRegistration.isFinished()`，不再重复保存。Worker 成功不能覆盖已选定失败。已经选定的其他响应不回写。

清理阶段为 `PENDING`（等待执行）、`RUNNING`（正在执行）、`RUN_AGAIN`（执行中又收到推进通知）、`WAITING`（本次执行结束，等待剩余资源结束）。它替代原来的 `attempted/cleaning/resumePending` 三个布尔值。只有处于 `WAITING` 且两侧资源已结清、准入已退出、抢占已结束，才能关闭跟踪。过期决定不可撤销，迟到的 ACTIVE 消息不会取消清理。

抢占的完成标记只在可靠结束依据到达后设置：单独收到 Prefill 结束而 Decode 清账尚未确认时不设置。标记完成后仍保留登记，等 Slot 清理完成才发送原有的抢占结束通知。

- 继续调度、ACK 使用现有活跃请求条件。
- 资源回调使用 `ownsResourceTracking()`，不因结果 FAILED 丢弃精确资源事件。
- request/decision 定时器退出；必要的 inactivity 定时器保留，且本地超时清理不改写已返回的失败。

## Decode 容量规则

Engine 有 P→D 会话清理链，但现有 PREFILL_REJECTED 分类未逐一证明响应返回前全部容量都已释放。因此不能把本地容量预留当成普通日志删除。

| 来源 | Decode 行为 |
|---|---|
| NOT_SENT | 调用既有精确回滚；Engine 确认或协议冲突仍保护容量。 |
| PREFILL_REJECTED | 不凭拒绝释放容量；由 Worker 终态、退休或既有本地过期规则收尾。 |
| TIMED_OUT / UNCERTAIN | 不进入明确失败流程，保留既有确认路径。 |

接口返回的是“这个 exact 是否还有责任”，不把 STALE 直接当成功。检查包括精确请求 owner、抢占 owner、incoming attempt。原 owner 已结算或被替换，且已无这个 exact 的所有者时，返回 true；不操作新的 reservation。Endpoint 实例只拥有一个 generation，跨 generation 不触碰该实例的新身份。

## 四个并发约束

1. **所有结束入口共用关闭条件。** `finishRequest` 是唯一关闭入口，统一检查失败清理进度。清账仍在执行、某一侧仍未完成、准入未退出或抢占未结算时，不允许清空 Slot。Worker、取消、退休和超时都不能绕过它。
2. **进度只前进。** 回调给出的完成事实与清账返回值使用单调合并。Decode 完成先到、false 的旧查询结果后到，不会退回等待。清账进行中遇到准入/抢占退出或到期推进，会登记一次后续处理；当前执行者完成后接续，避免丢掉推进通知。这不是后台定时重试。
3. **物理容量与响应分开。** Prefill 拒绝立即结束用户结果并结算成员；Decode 容量继续受原所有权保护。明确失败不发送 VictimTerminal；可靠协议结束或原有过期规则完成后，才通过已有终态执行器通知抢占。
4. **异常不冒充完成。** 一侧抛异常仍尝试另一侧；未完成责任继续保留。后续确切 Worker/退休事实或 inactivity 清理可完成责任。没有新增针对 RuntimeException 的后台重试；若这些清理仍失败，不宣称 Slot 已结清，需要 Endpoint 退休等可靠收尾依据。

## 旧函数处理

不是在旧链条外再加一层。以下旧入口和转发已删除：

| 原有链条 | 现在的处理 |
|---|---|
| `settleFailedDelivery`、`releaseUnsentDecodeReservation`、`reduceDeliveryFailureAfterReservationRelease` | 失败选定与清账分别由 `failRequest`、`cleanUpRequest` 完成。 |
| `reduceDeferredTerminalFactLocked`、`reduceOrdinaryTerminal`、`reduceWorkerTerminal` | 合并为 `processRequestEnd`，共用准入与抢占等待规则，Worker 证据只影响能否结束等待。 |
| `terminalEffectLocked`、`decideTerminalLocked`、`applyFailureTerminalLocked`、`applyTimeoutTerminalLocked`、`applyWorkerTerminalLocked`、`applyDispatchFailureOrRetirementLocked`、`beginPreemptedRequestTerminalLocked`、`beginExpiredRequestLocked` | 结果映射集中在 `finishRequest(event)`，不再逐层转发。 |
| `beginTerminalizing` 的重载及 `beginExternalTerminalizing` | 合并到 `finishRequest` 的共同关闭实现；公共 Future 仍保留独立的响应发布竞争。 |
| `acknowledgeDeliveredRequest`、`handleUncertainDelivery` | 回调消费与判断放回 `completeDelivery`，成功和已确认的不确定结果共用 ACK。 |
| 三层 Prefill 退休判断、`cleanupPrefillAfterRequestTerminal` | 退休合成 `finishPrefillRetirement`；成员结算条件直接留在 Endpoint 清理处。 |
| `DeferredTerminal.deliveryFailed / deliveryPreparationFailed`、两个 DELIVERY_* 类型 | 删除；派发失败不再存进抢占的 pendingTerminal。 |

`finishRequest(event)` 只负责已有事件到结果的映射；它与直接提供结果的调用共用同一个关闭实现。`event == null` 表示只推进已选定失败的清理，不再选择结果。关闭实现统一检查清理责任，冻结待释放的资源，交给原执行器在锁外处理。

其他事件保留原语义。通用清理执行器对已由失败主流程结清的 Endpoint 不再重复清理。

## 验证内容

- 抢占尚未结束时立即返回失败，后续成功不覆盖结果。
- 清账暂停时 Worker 完成，清账最终异常：Slot 保留未完成责任。
- 旧等待结果晚于 Decode 完成：不会丢通知或恢复等待。
- 请求结果已失败，inactivity 仍自动清理。
- 准入期间立即返回失败；已发生的清理到期不被后来的 Decode ACTIVE 撤销。
- Decode 分配尚未进入 Worker 快照、已经进入快照、抢占持有容量：拒绝都不提前释放容量。
- Endpoint 无旧 owner、重复结算、不同 reservation token：旧义务结束，新请求不受影响。
- 晚到拒绝、部分批次失败、重复与并发结束、锁顺序和现有全量回归。
