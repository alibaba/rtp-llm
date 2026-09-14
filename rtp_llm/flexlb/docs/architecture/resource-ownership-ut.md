# 资源所有权重构的 UT 基线

先固定现有行为，再修改所有权转移和终止清理。UT 基线先独立通过，随后用于验证生产代码的资源移交改动。

## 资源断言

`RequestResourceAccountingTest` 使用真实 RequestRegistry、PrefillEndpoint、DecodeEndpoint 和 EndpointEventProjector。
只 mock 配置服务和指标 reporter；不 mock 资源释放、permit 转移或 endpoint 账本。

| 账本 | 检查项 |
| --- | --- |
| 请求 | liveRequestCount |
| Prefill | 本地 inflight、单请求 lease、队列 membership、batch occupancy |
| Decode KV | inflightHardKv、inflightExpectedKv |
| Decode 请求容量 | queuedCount、activeDispatchPermits、acceptedCount、runningCount、engineCapacityUsed |
| 精确身份 | reserved/confirmed 集合、替代 reservation token |
| 可用容量 | 清理后重新取得 Decode permit 和唯一 Prefill slot，并再次归零 |

断言在 fixture teardown 之前执行，不能靠 teardown 清理使测试通过。
DIRECT fixture 显式停在 Prefill committed、尚未取得 delivery claim 的边界，再执行真实 permit/handoff。
它不替代 BATCH transaction 或全局队列测试。

## 新增与加强的场景

| 场景 | 测试及预期 |
| --- | --- |
| 本地取消、future cancel、inactivity、shutdown | RequestResourceAccountingTest：请求、本地 Prefill、Decode KV 与容量归零 |
| admission 尚未结束时取消/到期 | 同上：结束前保留资源，AdmissionHandle.close 后归零 |
| 已取得 dispatch permit、尚未 handoff 时取消 | 同上：permit 和 reservation 都回收；transaction 再回滚不重复释放 |
| 路由响应成功、RPC UNCERTAIN/TIMED_OUT | 同上：保留本地资源与 Engine-facing 容量；client cancel 也不提前回收；inactivity 最终回收 |
| Endpoint 已接管、slot 尚未收到投影 | RequestResourceAccountingTest：本地回滚保留 Engine 容量；精确 worker terminal 再释放，最终可复用 |
| 确定交付失败 | 同上：已移交的容量通过现有精确拒绝事务结算 |
| Decode running 后才收到交付失败 | 同上：running 事实优先，不提前终止；worker terminal 后两端收尾 |
| running 请求 inactivity | 同上：本地 running/容量归零，保留最后物理 KV 采样 |
| 旧 reservation/迟到 delivery 回调 | 同上：不能释放同 ID 的替代 reservation |
| Cancel ACK、CANCEL_UNKNOWN | DecodeEndpointLayeredViewTest：victim running/容量保留；incoming 到期不能释放 victim；victim 自身到期再结算 |
| BATCH 单成员重复过期 | PrefillEndpointTest：只减一个成员；最后成员结束才释放 batch occupancy |

## 回归命令

```sh
./mvnw -B -P 'opensource,!internal' -pl flexlb-sync -am \
  '-Dtest=Request*Test,DecodeEndpoint*Test,PrefillEndpointTest,BatchDeliveryStrategyTest,DirectRequestLifetimeRaceTest,PreemptionRegistrationTest,DecodePreemptionCoordinatorTest' \
  -DfailIfNoTests=false -Dsurefire.failIfNoSpecifiedTests=false test
```

该集合还覆盖既有的 admission publication 竞争、delivery 锁契约、响应发布竞争、endpoint retirement 和精确身份校验。
UT 基线完整运行 flexlb-sync 311 个测试通过，上游 RequestTest 3 个通过。生产改动另补 endpoint 接管先于 slot 投影的真实账本回归（资源账本测试共 15 个参数展开用例）。

上一轮生产改动的 `./mvnw -B -P 'opensource,!internal' -pl flexlb-sync -am test`：
上游模块 236 个测试通过；flexlb-sync 867 个用例中 866 个通过，
`ZkLeaderElectionTest.restartedLeaderRejoinsAsFollowerWithoutSplitBrain` 首次等待稳定状态超时。
单独重跑整个 ZkLeaderElectionTest，3 个用例全部通过。该次全量运行不能记为一次全绿。
BATCH delivery、资源账本、admission、发布竞争、endpoint 与锁契约回归均通过。

## 验证范围

running/accepted 检查针对 Master 的 Decode 账本，物理 KV 检查针对 Engine 上报采样的保留语义。
Java UT 不证明真实 Engine 的 GenerateStream 或 GPU KV 已释放；需要 Engine/mock-engine 集成验证补充。
inactivity 结束本地记账不等于收到 Engine 物理资源释放证明，不能将两者混为一个归零断言。

## 删除重复资源字段后的回归

Slot 仅保留 `item`；删除 `queuedItem`、`prefillAccounting`、
`localDecodeReservation` 以及 TerminalResources 中对应的资源引用。
清理由既有交付阶段、原始终止事件与 endpoint 的精确账本共同决定。
加强 RequestSlotTerminalSettlementTest：有/无抢占两种情形下，
Decode worker terminal 都不能释放 Prefill 的 BATCH 记账，且原响应对象保持不变。

上述相关回归集合：flexlb-sync 324 个用例、上游 RequestTest 3 个用例全部通过。
