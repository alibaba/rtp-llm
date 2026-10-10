# 枚举治理第二批：投递结果与请求状态副本

设计说明：本记录保留第二批代码与测试证据；下文“若要移除”的七状态 RequestLifecycle、BatchState、DeadlineIndex 前置迁移属于当时方案，已由 [最终方案](../final-design.md) 撤回。保留这些既有状态的必要性仍须按最终方案逐项评审。

本批基于本地 `2b2338ce9a86fbbee6b77203b1d57950dbea3346`，与[第一批](enum-implementation-round1.md)一起构成当前工作区的枚举改动。设计文档引用的远端 `017648a3` 是另一版本；目标状态机尚不能被当作本地已实施的 API。

## 已完成

| 改动 | 原因和保持的行为 |
| --- | --- |
| `DeliveryResult.Status` → `DeliveryOutcome` | `TIMED_OUT` 与 `UNCERTAIN` 对资源处理完全相同，本地生产路径也没有创建 `TIMED_OUT`；合并为 `UNKNOWN`，异常详情仍携带超时原因。`NOT_SENT` 和 `PREFILL_REJECTED` 仍是不同的确定失败，`DELIVERED` 仍是 ACK。旧状态类型和工厂方法已删除，没有兼容别名。 |
| 删除 `RequestSlot.EngineOwnership` | 该字段仅在精确 Decode 事实使 `DecisionStage` 进入 `ACCEPTED` 时同步写入。现在从这项事实和当前请求的 Decode reservation 派生，不新增 endpoint 查询或锁。Prefill-only/PDFUSION 不因 Prefill 完成而声称 Decode 所有权。 |
| 删除 `RequestSlot.deliveryClaimKind` 与 `batchId` 可变副本 | 活跃请求以唯一 `DeliveryClaim` 保存投递模式、关联 ID 和精确 item 身份；迟到回调必须匹配同一 claim。终态提交时冻结紧凑的 `RequestState` 快照，再释放 claim 和 item，公开的历史投递模式及 batch ID 保持可查询。 |
| 清理单一调用方的辅助判断 | `RequestSlot` 直接用 `DeliveryOutcome` 区分确定失败，删除 `DeliveryResult.failed()`；删除已无仓内调用的 `DeliveryClaimKind.isClaimed()`。不增加别名或另一层状态分类。 |

这批没有增加线程、回调、额外资源账本或请求数据复制。终态直接复用原有提交路径生成的 `RequestState`，之后查询复用该不可变对象。投递中的快照仍按原调用创建。

本地生产路径原本没有创建旧 `TIMED_OUT`，所以合并为 UNKNOWN 不改变现有发送处理；异常中的“timeout”等诊断详情保留。旧 `Delivery timed out:` 前缀、`timedOut(Throwable)` 工厂和旧 Status 类型不再保留，若仓外代码直接调用旧公开方法，需要按 API 迁移。这一点与保持现有错误码和资源语义分开记录。

## 当前为什么没有直接删除其余状态

这里的“保留”只表示**本批未删除现有并发保护**，不表示最终设计必须保留这些枚举。需要保住的是下表中的事实或一次性交接；实现形式可以在后续重构中收敛。目标请求阶段见[最终方案](../final-design.md)。

| 现有类型 | 如果现在直接删除，会丢掉什么 | 当前判断 |
| --- | --- | --- |
| `RequestState.Phase` | 查询 API 仍要回答 QUEUED、ACKNOWLEDGED、FAILED 等旧值；只知道请求处于哪个主阶段，无法还原 ACK 和最终结果。 | **保留公开值，内部字段待迁移。** 目标 `RequestStage` 落地后，从投递确认和终态结果投影旧值，不能同时长期写两套请求阶段。 |
| `RequestSlot.SlotPhase` | 终态结果选定后，endpoint 清理仍可能在锁外进行；这段时间第二个终态事件不能再次取得清理权。 | **目标合并。** 当前的 TERMINALIZING/TERMINAL_RECORD 是这道门；目标 `RequestStage.FINALIZING/FINISHED` 若承担同一门禁，就应删除 SlotPhase。 |
| `RequestSlot.DecisionStage` | Prefill 已被看到、Prefill 已结束但 Decode 尚未被看到、Decode 已受理，对应的可见性期限不同。只看 RPC ACK 无法区分。 | **需要保留事实，不承诺保留五值枚举。** 后续核对精确 Engine 事实和时间戳，能派生的值删除；本批不凭猜测改超时窗口。 |
| `RequestSlot.CleanupProgress.Phase` | 线程 A 正在释放资源时，线程 B 收到新的 Worker 事实并请求重检；没有 RUN_AGAIN 或等价信号，A 结束后可能无人再尝试结算。 | **当前保留执行协调。** 它只管清理 runner，不是请求生命周期；若改为单一清理 owner，必须证明重入事件不会丢。 |
| `RequestSlot.PublicationKind` | ACK 与取消都可能在 Future 尚未完成时选中响应；Future 在锁外完成，先解锁再看 `isDone()` 会允许两个结果同时胜出。 | **需要唯一发布权，不一定需要这个 enum。** 后续可让一个精确 publication permit 表达胜者；替换后删除 `publicationWinner`/PublicationKind 副本。 |
| `ExpirationTimer.DeadlineState` | 零延迟任务可能在 slot 存好 timer 句柄前触发；若不记住 FIRED_BEFORE_INSTALL，这次期限会丢失。 | **当前保留 timer 注册协议。** 若改成先登记后可触发的期限索引，再删除该五值状态；不为删 enum 增加额外线程。 |
| `BatchTransaction.Phase` | commit 前可以回滚本地预留；executor 接收后不能再由 scheduler 同步 abort；transport 接收后发送责任再次交接。 | **保留交接边界，五个值可以继续审减。** 不能把 COMMITTED、SUBMITTED、INFLIGHT 合成一个“进行中”后继续沿用原 abort/close 逻辑；也不新增一份 `BatchState`。 |

所以本批真正保留的是几个尚未迁移的门禁和证据，不是七套平行的请求生命周期。`SchedulingWaitReason`、`PlacementState` 没有独立写入需求，不新增字段；等待与选址分别从实际登记和 Prefill/Decode 精确凭证读取。

## 验证与 review

逐生产文件运行对应 UT。删除 EngineOwnership 后 103 个相关用例通过；移除投递字段副本后 114 个相关用例通过。三路独立审查后，新增未发送释放后的终态可移除断言，测试先按原 mock 失败、修正资源事实后通过；Prefill-only 测试改走公开事件入口；补齐 UNKNOWN 诊断详情及未投递、路由、批次三种终态身份断言。最终相关 116 个用例通过。最终全量功能测试 1451 个用例、0 failure、0 error、1 个原有 skip（实际执行 1450 个），覆盖 common、cache、grpc、sync 和 api。中途一次 API Mock Engine gRPC `UNAVAILABLE` 单独复跑及后续全量均通过。Sync 性能 profile 3/3 通过。

API 性能 profile 未通过，不能记为性能验收完成。同机同配置改动版与未改 HEAD 基线均有相同两项失败：8192 请求场景 Master P99 分别为 1101/1100 ms，门槛 250 ms；单 Prefill、双 Decode、目标 10k QPS 场景客户端吞吐分别为 7082.8/6898.6 QPS，门槛 8500 QPS。其余 14 个 API 性能用例通过。这组对照没有显示本批造成这两项失败，但也不足以证明没有性能回退；门槛和断言均未更改。

Review 路径按职责阅读：`delivery/DeliveryOutcome.java`、`DeliveryResult.java` → `scheduler/DefaultBatchDispatcher.java`、`RequestTerminalCleanup.java` → `scheduler/RequestSlot.java`、`DeliveryClaimKind.java` → `RequestSlotTerminalSettlementTest.java`、`DeliverySettlementTest.java`、`RequestLifecycleDeliveryLockContractTest.java`。上述源码在 `flexlb-sync/src/main/java/org/flexlb/balance/`，测试在对应 `flexlb-sync/src/test/java/org/flexlb/balance/scheduler/`。第一批的 Decode 资源 API 与测试继续按第一批文档审查。

验证在 `/tmp/flexlb-enum-worktree/rtp_llm/flexlb` 的同 HEAD 隔离目录运行，因为主工作区已有与本批无关、未跟踪的 `EngineLocalViewTest` 编译错误。主工作区中 `pom.xml`、缓存测试和 `docs/priority-scheduler-delivery-modes.md` 是已有独立改动，本批没有覆盖。已核对 41 个本批改动 Java 文件与隔离目录逐字一致。原始日志在 `/tmp/flexlb-enum-implementation/`，包括 `round2-final-functional.log`、`final-claim-outcome-targeted.log`、`round2-sync-performance.log`、`round2-api-performance.log` 和 `round2-api-performance-baseline.log`。
