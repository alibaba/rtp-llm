# 请求阶段的类时序与状态转移

状态：目标设计，尚未实施。图中的类名取自当前代码，箭头描述迁移后的职责，不表示这些调用现在已经只写控制意图。`RequestStage` 只由精确 `RequestSlot` 写入；图中其他对象保存各自的队列、资源或批次事实。实线箭头是执行/提交，虚线箭头是结果或唤醒。等待中的请求没有常驻线程。

## 1. 注册、选路与放置：SCHEDULING → DELIVERY

```mermaid
sequenceDiagram
    autonumber
    participant API as RequestScheduler
    participant Registry as RequestRegistry
    participant Slot as RequestSlot
    participant Queue as GlobalQueueCoordinator
    participant Router as Router
    participant Admission as RouteAdmission
    participant Decode as DecodeEndpoint
    participant Prefill as PrefillEndpoint
    participant Cleanup as RequestTerminalCleanup

    API->>Registry: register(context)，取得唯一 Future
    Registry->>Slot: 建请求，RequestStage = SCHEDULING
    alt DIRECT
        API->>Router: select(context)
        Router-->>API: 精确 RouteAdmission
        API->>Admission: 提交 DIRECT route
    else QUEUE
        API->>Queue: offer(context, Future)
        Queue->>Queue: 等待/领取全局候选，重读停止意图
        Queue->>Router: select(context)
        Router-->>Queue: 精确 RouteAdmission
        Queue->>Admission: tryEnqueue(context, Future)
    end
    Admission->>Decode: 需要时申请精确 reservation
    alt route 精确提交成功
        Admission->>Slot: 提交 route，检查请求身份/停止意图/期限
        Slot->>Prefill: 提交精确 Prefill route/队列能力
        Prefill-->>Slot: 提交成功
        Slot->>Slot: SCHEDULING → DELIVERY
        Note over Slot,Prefill: 本地队列的领取须看到已提交的 DELIVERY；交接与可领取性必须配套
    else 容量暂时 BLOCKED
        Admission->>Decode: 回滚仍由本次准入持有的 reservation
        Note over Queue,Slot: QUEUE 保持 SCHEDULING 并等待容量；DIRECT 按原失败语义收尾
    else 确定失败或停止先发生
        Admission->>Decode: 回滚仍由本次准入持有的 reservation
        Slot->>Slot: 冻结终态/TerminalAction，SCHEDULING → FINALIZING
        Slot->>Cleanup: 清理并交接必要的发布权
        Cleanup-->>Slot: 本地责任完成
        Slot->>Slot: FINALIZING → FINISHED
    end
```

DIRECT 提交后在同一调用链继续 DELIVERY；QUEUE 在 Prefill 本地队列等待。RouteAdmission 只管理本次尚未交接的 pin/reservation，失败时只回滚自己仍拥有的能力。路由/容量变动引起的重新选择留在 SCHEDULING，不创建新请求阶段。

## 2. 本地投递：DELIVERY → TRACKING

```mermaid
sequenceDiagram
    autonumber
    participant API as RequestScheduler
    participant Batcher as WorkerBatcher
    participant Route as RouteDeliveryStrategy
    participant Batch as BatchDeliveryStrategy
    participant Tx as BatchTransaction
    participant Registry as RequestRegistry
    participant Slot as RequestSlot
    participant Endpoint as 精确 Endpoint
    participant Sender as DefaultBatchDispatcher
    participant Cleanup as RequestTerminalCleanup

    alt DIRECT
        API->>Registry: 继续本次 route 投递
    else QUEUE + NON_BATCH
        Batcher->>Route: 领取/准备本地候选
        Route->>Registry: claimRouteDelivery(exact)
    else QUEUE + BATCH
        Batcher->>Batch: 领取/准备本地候选
        Batch->>Tx: 提交真实批次与成员
        Batch->>Registry: 逐成员 claimBatchDelivery(exact, Tx)
    end
    Registry->>Slot: claimDelivery(exact, kind)
    Slot->>Slot: 原子检查停止意图、期限、请求/placement 身份
    alt 交接被拒绝
        Slot-->>Registry: 无 claim；DELIVERY 处理未发送能力
        Slot->>Slot: 冻结终态/TerminalAction，DELIVERY → FINALIZING
        Slot->>Cleanup: 清理并交接必要的发布权
        Cleanup-->>Slot: 本地责任完成
        Slot->>Slot: FINALIZING → FINISHED
    else 唯一交接成功
        Slot->>Endpoint: transferToEndpoint(exact)
        Slot->>Slot: 建 delivery claim，DELIVERY → TRACKING
        Slot-->>Registry: 返回唯一 claim
        alt DIRECT 或 QUEUE + NON_BATCH
            Registry->>Slot: 发布本次 route 结果
        else QUEUE + BATCH
            Batch->>Tx: 使用已 claim 的成员执行一次发送
            Tx->>Sender: EnqueueBatch(仅已 claim 成员)
            Sender-->>Tx: DELIVERED / NOT_SENT / PREFILL_REJECTED / UNKNOWN
            Tx->>Slot: 回报每个成员的精确结果
        end
    end
```

图中的 DIRECT route 发布由现有 `RequestScheduler`/`RouteAdmission` 调用，QUEUE NON_BATCH 由 `RouteDeliveryStrategy` 调用；BATCH 的真实发送权只归 `BatchTransaction`。`claimDelivery` 是停止意图与外部可见交接的单一仲裁点。交接后若停止意图才到，TRACKING 按远端可能可见处理；只有精确结果证明 `NOT_SENT`，才按未发送收尾。`UNKNOWN` 不回 DELIVERY，不重发。

## 3. 外部事实与退出：TRACKING → FINALIZING → FINISHED

```mermaid
sequenceDiagram
    autonumber
    participant Transport as 投递回调
    participant Worker as WorkerSynchronizer
    participant Endpoint as Prefill/Decode Endpoint
    participant Registry as RequestRegistry
    participant Slot as RequestSlot
    participant Cleanup as RequestTerminalCleanup
    participant Publisher as RequestCompletionPublisher
    participant Future as 原始 Future

    opt route 发布或 Batch 投递回调先到
        Transport->>Registry: 精确 delivery claim + route 确认或 Batch DeliveryOutcome
        Registry->>Slot: 提交本次投递事实
    end
    opt 投递成功已得到可确认的事实
        Slot->>Publisher: 选定一次成功响应并交接发布权
        Publisher-->>Future: 异步完成成功响应
        Note over Slot,Future: RequestStage 仍为 TRACKING；异步完成与后续远端事实无固定先后
    end
    opt Engine/worker 事实先到或随后到
        Worker->>Endpoint: 按 generation 更新资源/Engine 账本
        Endpoint->>Registry: 通知匹配请求的精确事实
        Registry->>Slot: 提交匹配事实
    end
    Slot->>Slot: TRACKING 内合并事实与控制意图，不重发
    alt 需要继续等待 ACK、终态或资源证据
        Slot-->>Registry: 登记唤醒，保持 TRACKING
    else 可以结束本地请求责任
        Slot->>Slot: 原子冻结终态/TerminalAction，TRACKING → FINALIZING
        Slot->>Cleanup: 执行冻结的 TerminalAction
        Cleanup->>Endpoint: 按精确身份结算本地义务
        opt 尚无响应取得发布权
            Cleanup->>Publisher: 提交终态响应的唯一发布许可
        end
        Cleanup-->>Slot: 本地结算/发布权交接完成
        Slot->>Slot: FINALIZING → FINISHED
        opt 本次由终态响应取得发布权
            Publisher-->>Future: 异步完成终态响应，可晚于 FINISHED
        end
    end
```

终态可先于 ACK，此时若成功响应尚未赢得发布权，进入 FINALIZING 时可选定终态响应。FINALIZING 已提交不可中断的终止决策：迟到取消、超时和抢占不能改判；迟到 ACK 只能补所属账本事实或推动未完清理。不能用 `Future.isDone()` 判断发布权：已选响应可能还在异步队列。FINISHED 表示本地义务与发布权按协议交接，远端仍未知的资源继续由原 endpoint generation 的账本负责。

## 4. 超时或客户端取消：控制入口只记录，当前阶段执行

```mermaid
sequenceDiagram
    autonumber
    participant Timer as ExpirationTimer
    participant Client as 客户端取消入口
    participant Registry as RequestRegistry
    participant Slot as RequestSlot
    participant Global as 全局决策线程 + control inbox
    participant Batcher as WorkerBatcher + control inbox
    participant Runner as 共享继续执行器
    participant Cleanup as RequestTerminalCleanup

    alt 精确 timer 到期
        Timer->>Slot: 核对期限身份，记录到期意图
        Timer->>Registry: 通知当前精确 owner
    else 客户端取消
        Client->>Registry: cancelRequest(requestId, expectedBatchId)
        Registry->>Slot: 冻结第一次有效 CancelReason
    end
    alt SCHEDULING 且仍在全局队列
        Registry->>Global: 精确 entry 入 control inbox；changed.signal
        Global->>Global: 先取控制票据，撤 ordered/waiting 索引
        Global->>Slot: 重读意图；可结束时 SCHEDULING → FINALIZING
        Note over Global,Slot: 在途 planner 只持临时 AdmissionHandle；晚到结果关闭能力
    else DELIVERY 且仍在本地队列，未赢得 delivery claim
        Registry->>Batcher: 精确 ScheduledRequest 入 control inbox；stateChanged.signal
        Batcher->>Batcher: 先取控制票据，撤本地队列/结算未发送能力
        Batcher->>Slot: 重读意图；可结束时 DELIVERY → FINALIZING
    else TRACKING，远端可能已看到请求
        Registry-->>Runner: 合并提交精确继续任务
        Runner->>Slot: 重读投递与 Engine 事实
        Runner->>Slot: 保留停止意图，等待精确结果/终态/期限证据
        Runner->>Slot: 条件满足时冻结终态，TRACKING → FINALIZING
    end
    opt 当前阶段已进入 FINALIZING
        Slot-->>Runner: 提交已选清理任务
        Runner->>Cleanup: 执行冻结的 TerminalAction
        Cleanup-->>Slot: 结算与发布权交接完成 → FINISHED
    end
```

timer/cancel 不直接撤队、释放资源、调用 Engine Cancel 或完成 Future。控制票据在队列锁下入 inbox 并 signal；全局决策/WorkerBatcher 睡前检查 inbox，醒来先按精确身份撤队，**不等队尾或容量等待请求被正常排序选中**。阶段交接检查仍未消费的停止意图，新 owner 对旧票据不依赖；投递 claim 的原子检查也直接读取当前时间，避免 timer 延迟触发让过期请求继续发送。`Runner` 只做 TRACKING/FINALIZING 的请求继续，不进入两个排队索引。

## 5. 抢占：incoming 提交意图，victim 主流程结算

```mermaid
sequenceDiagram
    autonumber
    participant Incoming as incoming 的 SCHEDULING/DELIVERY
    participant Evict as EvictionManager/WorkerBatcher
    participant Victim as victim RequestSlot
    participant Runner as victim 当前阶段
    participant Prefill as PrefillEndpoint
    participant Decode as DecodeEndpoint
    participant Cancel as 原 Prefill Cancel 通道
    participant Cleanup as RequestTerminalCleanup

    Incoming->>Evict: 容量不足，选择精确 victim
    Evict->>Victim: 唯一 claim，记录抢占意图
    Victim-->>Runner: 唤醒并重读 victim 阶段
    alt PREFILL_QUEUED
        Runner->>Prefill: DELIVERY 撤精确队列项
        Runner->>Victim: 冻结 PRIORITY_PREEMPTED，DELIVERY → FINALIZING
        Victim->>Cleanup: 执行精确清理
        Cleanup-->>Victim: 本地责任完成
        Victim->>Victim: FINALIZING → FINISHED
        Prefill-->>Incoming: 精确容量释放后唤醒
    else DECODE_RESERVED 且未外部可见
        Runner->>Victim: DELIVERY 阻止 delivery claim
        Runner->>Decode: 撤旧 placement 的精确 reservation
        Runner->>Victim: DELIVERY → SCHEDULING，保留原顺序/绝对期限
        Decode-->>Incoming: 旧 reservation 撤销后唤醒
    else DECODE_ENGINE_OWNED
        Runner->>Cancel: TRACKING 经原 Prefill 通道推进 Cancel 协议
        Cancel-->>Runner: ACK / NOT_FOUND / timeout / fencing 证据
        Runner->>Decode: 核对匹配终态或有效释放证据
        Decode-->>Incoming: 满足准入门槛后唤醒
    end
    Incoming->>Incoming: 重新检查容量和精确身份，再决定是否准入
```

抢占发起方设置 victim 的控制意图，不替 victim 执行撤队、释放或 Engine Cancel。一个 victim 同时只能有一个抢占 claim。ACK、普通 NOT_FOUND 和 RPC 超时都不能单独证明 Decode 容量已释放；incoming 不因 victim 状态变化直接取得资源。普通客户端取消不走 Engine Cancel 协议。

## 实施时逐点核对

| 交接点 | 状态转移 | 必须同一边界确认的事实 |
| --- | --- | --- |
| route 提交 | SCHEDULING → DELIVERY | 请求仍有效、精确 endpoint 能力已交接、本地队列不能抢跑 |
| delivery claim | DELIVERY → TRACKING | 停止意图/当前时间、精确身份和 endpoint 责任一次仲裁 |
| 未发送的 Decode 撤回 | DELIVERY → SCHEDULING | 无外部 delivery claim、旧 reservation 精确撤销、原顺序/期限保留 |
| 投递确认 | TRACKING 内无阶段变化 | 可赢得唯一成功响应发布权，随后继续跟踪 Engine 事实 |
| 终态选择 | SCHEDULING/DELIVERY/TRACKING → FINALIZING | 终态和 TerminalAction 同时冻结；后到控制事件不能中断或改判，不能覆盖已选成功响应 |
| 本地完成 | FINALIZING → FINISHED | 清理义务及必要的终态响应发布权已交接，远端未知账本仍归原 endpoint |

现有代码中 `RequestSlot.cancelRequest`、`ExpirationTimer` 回调和 Prefill 排队抢占仍可能直接触发清理或资源替换；上图是迁移后的目标边界。改代码时应沿这些实际入口逐步迁移，不能在旧写入口旁增加一套可写 RequestStage 并长期双写。
