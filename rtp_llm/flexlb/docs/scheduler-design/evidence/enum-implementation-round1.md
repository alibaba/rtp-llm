# 枚举治理第一批：完整 Review 版本

代码已落在当前本地工作区。实现基于本地 `2b2338ce9a86fbbee6b77203b1d57950dbea3346`；设计审查引用的远端 `017648a3` 是另一个版本，本轮未覆盖远端源码。

## Review 入口与顺序

本批已完成实现与自审，以下顺序可直接用于本地 review。

以下路径相对于当前 FlexLB 根目录：

1. `flexlb-sync/src/main/java/org/flexlb/balance/endpoint/DecodeEndpoint.java`：四个释放操作和精确所有权查询；Endpoint 不再依赖 DeliveryResult。
2. `flexlb-sync/src/main/java/org/flexlb/balance/endpoint/DecodeState.java`：身份保护、Engine/协议所有权、锁与容量版本。
3. `flexlb-sync/src/main/java/org/flexlb/balance/scheduler/RequestTerminalCleanup.java`：在终止清理层解释投递结果，确定未发送才释放，Prefill 拒绝只查 Decode 所有权。
4. `flexlb-sync/src/main/java/org/flexlb/balance/endpoint/PrefillEndpoint.java`：删除与 releaseCommittedItem 同实现的 expireCommittedItem。
5. `flexlb-sync/src/test/java/org/flexlb/balance/endpoint/DecodeStateTest.java`、`DecodeEndpointLayeredViewTest.java`、`PrefillEndpointTest.java`、`flexlb-sync/src/test/java/org/flexlb/balance/scheduler/DeliverySettlementTest.java`：关键行为测试。
6. `RouteAdmission.java`、`RequestRegistry.java`、`RequestSlot.java` 及其余测试：调用迁移和取消原因命名统一。

其余测试 diff 是对应 API、stub、verify 和反射方法名迁移。主工作区已有的 pom.xml、缓存测试及无关文档不属于本批代码改动。

```sh
# 本轮全部生产代码，集中查看
git diff -- flexlb-sync/src/main/java/org/flexlb/balance/
# 核心新增测试
git diff -- flexlb-sync/src/test/java/org/flexlb/balance/endpoint/DecodeStateTest.java
```

最终验证：隔离目录中 API 及其依赖模块 1448 个功能用例，0 failure、0 error、1 个已有 skip；Sync 性能测试 3/3 通过。Mock Engine 独立模块未纳入此次功能命令。

API 性能 profile 的 16 个用例未全部通过。改动版第一次 4 项失败，复跑后 1 项失败；同机同配置的未改基线也有相同的 Master P99 场景失败（基线 995 ms、改动版复跑 1145 ms，门槛 250 ms）。首次改动版另有吞吐失败，复跑未重现。三次运行不能证明无性能回退，性能验收仍未通过；门槛和断言均未修改。最终日志为 `functional-final.log`、`sync-performance-final.log`、`api-performance-final.log`、`api-performance-final-repeat.log`、`api-performance-baseline-same-host.log`。

## 实际修改

- 删除 `DecodeEndpoint.ReleaseReason` 的四个值和 `release(reservation, reason)` 入口，连同 DecodeState 中对应入口一起删除，不保留兼容分支。
- 替换为 `rollbackReservation`、`releaseUnsentReservation`、`releaseLocalOnTerminal`、`expireReservation`。终止路径不只是 Prefill 完成，因此最终命名为 `releaseLocalOnTerminal`。
- DecodeEndpoint 不再依赖 DeliveryResult；`RequestTerminalCleanup` 根据确定的投递结果选择释放或查询，Endpoint 只处理资源。删除 DecodeEndpoint 中间入口 `settleFailedRequest`。
- DecodeState 继续在原 admissionLock 内完成身份校验和账本修改；冲突检查及持有查询复用同一请求记录，不新增查询副本、线程、回调或操作选择布尔参数，也不在已持锁时重入查询锁。
- DecodeEndpoint 仅在实际 RELEASED 时发送原有容量变化通知；通知仍在账本锁外。
- PrefillEndpoint 的 `expireCommittedItem` 与 `releaseCommittedItem` 原先都是同一个 `terminalizeCommittedItem` 调用；删除前者，过期路径仍走精确成员清理。迁移全部仓库内 Java 调用、Mockito 验证及 stub。
- RequestSlot 的取消原因统一使用 `hasCancellationReasonLocked`、`requireCancellationReasonLocked` 和局部变量 reason；Throwable 的 cause 不改名，首次取消语义不变。

| API | 保留的行为 |
| --- | --- |
| rollbackReservation | null 非法；Engine/协议已持有资源时抛不变量异常；旧身份不能释放新预留 |
| releaseUnsentReservation | null 非法；Engine 已确认时返回 ENGINE_ACCEPTED，不释放已受理资源 |
| releaseLocalOnTerminal | null 返回 STALE；请求终止只能释放本地 Decode 所有权，Engine/协议资源保留给对应生命周期结算 |
| expireReservation | null 返回 STALE；沿用原过期清理逻辑，包括精确抢占责任结算 |
| hasOwnedResources | 查询精确 reservation 的 Decode 容量或协议义务；不解释投递结果，也不修改账本 |

所有 API 保留原 ReservationReleaseResult 语义；本批不改变取消策略、错误码、期限或重试规则。

## 测试与 review 范围

先在旧 API 上增加 6 个执行用例，再迁移到新 API：空值契约 1 个、四种操作对旧 token/错误 generation 的保护 4 个、Engine 已确认资源的不同处理 1 个。原基础测试方法未删除，原断言保留并更新 API 调用；Prefill 原先两个入口实为相同操作，重复的参数化分支改为验证二次清理的幂等性。

逐生产文件运行对应 UT。迁移期间按红灯验证了旧 Mockito 入口的删除；API 模块曾发现一个遗漏的旧方法引用，修正后重跑全部通过。合并 Prefill 入口时，一个旧 mock 将两个实际相同的入口分别注入不同行为；测试现按第一次失败、第二次真实清理的方式保留并发重试场景，没有放宽最终断言。

自审三轮：从最终 diff 检查职责与锁边界；查找旧 Java API、重复分支和无效导入；核对主工作区与隔离目录 35 个改动 Java 文件逐字一致，`git diff --check` 通过。当前主工作区在本批开始前已有独立编译问题：未跟踪的 EngineLocalViewTest 调用 close()，但 EngineLocalView 没有该方法。本批未覆盖这部分改动。验证在同一 HEAD 的隔离目录 `/tmp/flexlb-enum-worktree/rtp_llm/flexlb` 执行；隔离验证不能声称主工作区全量构建已通过。同机性能基线位于 `/tmp/flexlb-perf-baseline/rtp_llm/flexlb`。

原始日志：`/tmp/flexlb-enum-implementation/`。逐文件记录含 `state.log`、`endpoint.log`、`route-verified.log`、`cleanup.log`、`registry.log`、`prefill-endpoint-final.log`、`settlement-cleanup-green.log`、`state-lock-review.log` 和最终功能/性能日志。

## 尚未实施

本批只完成释放 API 治理、Prefill 重复入口删除和取消原因命名统一；后来完成的 DeliveryOutcome、EngineOwnership 删除及投递状态副本移除见[第二批](enum-implementation-round2.md)。RequestLifecycle（含 CANCELLING）、SlotPhase/DecisionStage 的合并、DeadlineIndex 和 BatchState 是当时尚未实施的目标；这些新增目标现已撤回，不把这些后续迁移记为完成。
