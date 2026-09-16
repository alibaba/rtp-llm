# DecodeEndpoint / DecodeState：已实施的接口与功能块

基于 2026-09-16 工作区完成。本文替代此前逐函数分类的候选方案；不包含其他并行进行的 RequestSlot 生命周期改造。

## 分工

- `DecodeEndpoint`：generation 校验与 pin、对外资源入口、许可句柄的幂等结果、锁外通知、监控适配。
- 包内 `DecodeState`：本代唯一资源账本、容量计数、派发许可、抢占事务，以及保护这些状态的 `admissionLock`。
- State 没有 Endpoint/Slot/RPC/Reporter 回调引用。输入为资源身份、需求或 Worker 观测；输出为结果或不可变事实。
- Worker 观测发布与账本校准仍在 State 的同一锁事务内；Endpoint 在返回之后通知请求层和等待者。

## 四组接口

| 功能 | DecodeEndpoint | DecodeState |
|---|---|---|
| 预留 | `reserve(pin, ..., capacity)`；不检查容量的重载；`reserveUnqueued(pin, ...)` | `reserve(..., queued, capacity)`，一次检查并占用资源 |
| 释放 | `release(reservation, reason)` | `release(reservation, reason)`，同一锁内按依据处理 |
| 派发 | `markQueued`、`acquireDispatchPermit`、`dispatch(permit, outcome)` | 同名操作，维护 queued/permit/Engine 所有权及容量 |
| 本地替换 | `replaceQueuedRequests` | 一个事务释放全部准确 victims 并预留 incoming |
| 远端抢占 | `beginPreemption`、`updatePreemption`、`finishPreemption` | 同名操作；等待 Cancel 的期间保留 claims/incoming |
| 校准 | 父类协议要求的 `applyPreparedStatus`、初始化和心跳入口 | `calibrate`、`initialize`、`observeHeartbeat` |

`EngineDispatchPermit.dispatch()` 与 `release()` 是持有许可的调用方使用的便捷操作，统一进入 Endpoint 的 `dispatch`。许可不能传给另一个 Endpoint 使用。

派发保留明确的成功、所有权丢失、generation 退休结果，供现有调用方区分处理，不将这些结果丢弃成 `void`。

### 释放依据

| ReleaseReason | 行为 |
|---|---|
| LOCAL_ROLLBACK | 严格回滚仍属本地的资源；已进入 Engine/抢占协议时报告契约违例 |
| COUNTERPART_FINISHED | 另一角色已结束，只清理本地所有权；不能据此释放 Engine/协议占用 |
| NOT_SENT | 明确未发送；Worker 已确认接收的并发事实优先 |
| EXPIRED | 清理本地跟踪与协议占用，保留抗迟到上报的历史记录，不改写物理 KV 样本 |

结果区分 RELEASED、ENGINE_ACCEPTED、STILL_OWNED、STALE、CONFLICT。`settleFailedRequest` 仅在 Endpoint 做现有 DeliveryResult 到资源操作的适配，State 不依赖请求响应或交付 DTO。

### 抢占进展

`PreemptionUpdate` 是一次输入事实，不是新状态机：取消开始、取消回复、确定 CANCELED、REQUEST_FENCED、活跃或普通完成。终态事实携带准确 reservation。State 在一次锁事务内核验 attempt、token、可接受证据和账本变化。

`finishPreemption(COMMIT)` 必须确认 incoming 身份仍有效、所有 victims 已结算；`ABORT` 放弃 incoming，仅撤销仍允许本地回滚的 claims。

## 函数按完整功能块排列

两份文件均不再采用“public 在前，所有 private 堆到后面”的排列。

State 的顺序：

1. 预留/释放入口 → 身份与容量校验 → 按证据释放 → 该组的记账原语。
2. 排队与许可获取 → 派发/归还 → 许可身份、计数和派发容量计算 → 许可内部类型。
3. 本地替换/远端抢占开始 → 进展处理 → 提交/放弃 → claim 与 KV hold 维护 → 抢占内部类型。
4. 完整观测/初始化/心跳 → 校准循环 → confirmed 更新及事实生成。
5. 退休清空/孤儿清理 → 准确身份收集、历史记录维护。
6. 只读视图/缓存/监控数据捕获。
7. 各组共用的请求查找、身份核验及单条请求记录。

Endpoint 的 pin 获取、State 调用和锁外通知放在同一个操作入口内；许可结果处理及对应类型紧跟派发入口。共享容量通知仍集中维护一份。

## 实际删除与合并

- 删除旧 `tryReservePlacementPinned`、`releaseReservationExact`、`releaseLocalShadowIfExact`、`expireReservationExact` 等对外入口；调用方使用 `reserve/release`，未保留旧名兼容代理。
- 删除零散 Cancel phase、CANCELED/fence、active/finished 的旧对外入口；改为 `updatePreemption`。
- 删除旧 commit/abort 抢占入口，改为 `finishPreemption`；内部也合并锁和事务查找，未仅增加分发表。
- 本地严格回滚与另一角色结束后的保守释放，共用一份身份检查和资源删除代码，差异只保留在证据规则中。
- 未发送请求的释放直接查询 canonical 请求记录，删除重复的 shadow/permit/lifecycle 所有权查询。
- 删除没有独立行为的许可结果工厂、单次计数递减包装、Engine 生命周期字段清理包装及过时 KV total 查询接口。
- `transferToEngineLifecycle` 改为 `dispatch`；`layeredAdmissionView` 改为 `resourceSnapshot`，调用方和测试同步迁移。

## 规模与验证

原 DecodeEndpoint 为 3221 行；实施后 DecodeEndpoint 为 689 行，DecodeState 为 2028 行，两个生产类合计 2717 行，净减少 504 行。该口径包含注释和空行，不把迁移到 State 的代码算作删除，也不包含测试代码。

Endpoint 变成边界层需要少量委托函数，因此不能用 Endpoint 单文件的方法减少数，声称全库方法总数同样减少。真正删除的是重复分支、旧入口及无独立行为的包装。

已验证：

- `flexlb-sync` 919 个现有测试通过，覆盖资源账本、派发许可、抢占、校准、退休及 TTL 锁竞争。
- 新增 `DecodeStateTest` 3 个测试：旧许可不能影响新许可、校准保持身份且不重复计费、跨 Endpoint 许可被拒绝且不消耗原许可。
- `PreemptionPhasesE2ETest` 5 个测试通过，覆盖真实调度器与 mock Engine 的抢占链路。
- 全 Maven reactor 的生产和测试代码编译通过；未宣称运行 API/mock-engine 的全部测试。
