# Decode KV Cache 写回实现文档

本文记录[写回设计方案](decode_kvcache_writeback_design.md)的实现进度与验证结果。设计文档描述目标行为和待讨论意见；本文记录当前代码状态、实现选择及验证过程。

## 1. 当前范围与实现计划

- 开发分支：`rym/feat/decode-writeback`。
- 基线分支：`codex/dsv4-block-tree-p2p-region`。
- 基线提交：`ce489746d081e7fb861fa706fc3cb2d933ac5214`。
- 当前仓库：`/data6/renyuanming.rym/rtp-llm`。
- 2026-09-10 先撤回 Slice 1，完成 Slice 0 的期限修复和验证后提交；随后按逐片实施的安排重新开始 **Slice 1**，本轮实现见 §6。
- 当前包含 Slice 0 和 Slice 1 基础设施，尚未接入接收分配、插树和请求收尾触发逻辑；写回开关默认关闭。

### 1.1 五个 Slice 的职责

按照设计文档 §7.8，Phase 1 共分为 **5 个 Slice（0～4）**，每片应能独立编译、验证。这里的 Slice 指实现阶段，与传输协议中的数据切片不是同一个概念。两端先分别实现和验证各自的 scheduler，最后单独接入生产触发点并验证真实两端协作。

| 实现阶段 | 主要工作 | 当前状态 |
| --- | --- | --- |
| Slice 0：Read 路径重构 | scheduler/worker 改为 Read 命名；将首 token、复用统计和 MTP 结果拆入 `PrefillResultStore`，保留等待、取消和终态语义 | 已实现；期限交接修复后目标容器编译及 212 个单测通过，待代码审阅 |
| Slice 1：写回基础设施 | proto、Decode Write sender、Prefill Write receiver、backend 接线、角色分发及配置 | 已实现；复用自审后 251 个 C++ 单测通过，后续空 routes 修复编译及 78 个 worker 用例通过；初版 17 个 Python 单测通过，待 review，详见 §6 |
| Slice 2：Prefill 接收与插树 | `StartWrite` 服务入口、`P2PConnector::handleWrite` / `P2PConnectorPrefill::processWrite` 和 `P2PSchedulerPrefillWrite`；准入与范围协商、接管分配引用的 RAII 句柄、接收汇总、HOST/DISK 后缀冲突拒绝、同锁预检和插树、失败释放 | 尚未实现；由测试发起方驱动，独立验证完整接收流程 |
| Slice 3：Decode 发起与收尾 | `P2PConnector::asyncWrite` / `P2PConnectorDecode::write`、Write caller 和 `P2PSchedulerDecodeWrite`；源块保活上下文、握手及发送、并发与超时管理、CANCEL/QUERY 和资源收尾 | 尚未实现；使用模拟 Prefill 服务验证完整发起流程，不接入请求收尾触发点 |
| Slice 4：生产链路接通 | `KVCacheManager::asyncWriteBack` 与 `StreamCacheResource::tryReleaseKVBlock` 接线；释放原始引用前同步建立源块 hold、传给 Slice 3 上下文，确认 KV 就绪及 routing context 生命周期，连接真实两端 | 尚未实现；端到端集成、引用交接与异常路径测试、正向回归及写回 smoke |

Slice 0 的目标是整理现有正向 Read 路径，为后续写回提供清晰的资源与结果存储边界。当前代码不提供 Decode 到 Prefill 的自动写回。

Slice 1 已提供实际的注册、异步发送、查询和取消能力，并通过真实 RPC/TCP 测试；Slice 2/3 在这些 worker 能力上分别构建本端编排。Slice 4 单独审阅生产触发时机和两端协作，不再把它们与 Decode scheduler 的实现放在同一片。

### 1.2 两侧 scheduler 开工前的接口约定

Slice 2/3 必须基于同一份协议和资源生命周期约定实现；Slice 4 验证这些约定在真实链路中成立。涉及设计文档中尚待讨论的接口时，先明确对应条目，再开始依赖它的实现。

| 约定 | 两侧共同遵守的边界 |
| --- | --- |
| `StartWrite` 握手 | 固定请求字段、接受范围 `{p, k}`、接收端点、拒绝和 `k=0` 语义；接受且 `k>0` 的应答可见时，对应接收任务已经登记 |
| 任务关联与路由 | 固定 unique key、cache keys、plan digest 和 routes 的解释；两侧沿用相同任务标识及 `_wb_` 传输命名空间 |
| 启动失败与停止确认 | 区分 `StartWrite` 握手 RPC 和 rank0 向 worker 下发的 `WRITE_START`；worker START 的失败、超时或缺失响应不能证明停止，必须继续 CANCEL/QUERY。握手响应丢失时，Prefill 接收上下文仍负责其已注册任务和资源收尾 |
| deadline 与状态 | 使用原始绝对传输 deadline；控制 RPC 使用独立超时。`stopped` 与 `write_success` 分别判断，期限到达不能代替物理停止确认，详见 §6.3.1 和 §6.4 |
| 资源所有权 | Prefill 分配句柄接管原始引用；Decode 上下文接管已建立的源块 hold。两侧均须等相关任务确认停止后释放。Slice 3 用测试持有者验证该契约，Slice 4 在生产 free 之前建立 hold 并完成交接 |

各片验证重点：Slice 2 覆盖分配引用账本、部分登记失败、超时仍在途、后缀冲突和零部分发布；Slice 3 覆盖拒绝/`k=0`、范围校验、并发准入、START 响应丢失、取消和迟到回调；Slice 4 覆盖真实任务标识及路由对齐、源块交接、响应后缀复用、失败收尾及正向 Read 回归。

### 1.3 Write 入口命名与实现边界

| 层级 | Decode | Prefill | 实现阶段 |
| --- | --- | --- | --- |
| 广播类型 | `WRITE = 7` | `HANDLE_WRITE = 6` | Slice 1 已实现，枚举数值保持不变 |
| 单 rank connector 入口 | `P2PConnectorDecode::writePerRank` | `P2PConnectorPrefill::processWritePerRank` | Slice 1 已实现，解析协议、分发 START/QUERY/CANCEL、封装响应 |
| worker 启动入口 | `P2PWorkerDecodeWrite::write` | `P2PWorkerPrefillWrite::handleWrite` | Slice 1 已实现，接收内部 route plan，返回 `ErrorInfo` |
| worker 控制入口 | `cancelWrite` / `queryWriteStatus` | `cancelWrite` / `queryWriteStatus` | Slice 1 已实现，取消与查询独立于启动入口 |
| rank0 编排入口 | `P2PConnectorDecode::write` | `P2PConnectorPrefill::processWrite` | 分别在 Slice 3 / Slice 2 实现 |

当前 Decode worker 的 `write()` 执行单 rank 任务；未来 Decode connector 的 `write()` 发起整个写回流程，二者位于不同的类和调用层级。本轮统一既有方法命名，不为尚未实现的 scheduler 增加空入口。

Write 沿用现有 Read 的分层：connector 把 protobuf route 转为 `P2PWorkerRoutePlan`，worker 只接收内部数据。Prefill `handleWrite(request_id, unique_key, deadline_ms, worker_plan)` 对应接收侧 `P2PWorkerDecodeRead::read(...)` 的参数形式，返回 `ErrorInfo`；Decode `write(...)` 使用相同参数形式。两侧 `cancelWrite(key, deadline)` 返回是否接受取消，`queryWriteStatus(key, status)` 返回是否找到任务，状态通过普通 C++ 结构 `WriteTaskStatus` 输出，包含 sealed、任务计数、stopped、write_success 和业务错误。

接口分层对齐不改变等待语义：Read 的 `read()` 包含接收等待；Write 的 `handleWrite()` 完成接收登记即返回，`write()` 完成后台提交即返回。完成汇总和资源收尾由后续 scheduler 通过独立状态接口处理。

## 2. Slice 0 已实现的改动

### 2.1 Read 类命名与路由编码

现有 scheduler 和 worker 只调整命名，未改变内部逻辑：

| 原类名 | 当前类名 |
| --- | --- |
| `P2PConnectorSchedulerDecode` | `P2PSchedulerDecodeRead` |
| `P2PConnectorSchedulerPrefill` | `P2PSchedulerPrefillRead` |
| `P2PConnectorWorkerDecode` | `P2PWorkerDecodeRead` |
| `P2PConnectorWorkerPrefill` | `P2PWorkerPrefillRead` |

`RouteCodec::encodeForPrefill/encodeForDecode` 改为 `encodeForSender/encodeForReceiver`。调用点、头文件引用、测试及 BUILD 配置同步使用新名称。各角色 connector 仍直接持有本角色的 Read 对象，backend 的持有方式恢复为原有 Read 路径。

### 2.2 PrefillResultStore

`PrefillResultStore` 使用单个 map 保存首 token、复用长度和 MTP 结果，独立于 KV 资源 entry。`NormalGenerateStream` 通过 `KVCacheManager` 将结果通知转发给它；`P2PConnectorPrefill` 等待结果并填充 StartLoad 响应。现有 `SideChannelPayloadPB` 协议字段保持不变。

已实现的 API 包括 `registerRequest`、`beginTransfer`、`notify`、`waitAndFill`、`seal`，以及定期清理。`ResourceStore` 在注册资源时只计算一次 hold deadline 和归一化后的业务 deadline，再通过注册回调将相同绝对值传给 ResultStore。`P2PRequestDeadline.h` 集中保留原有的 deadline 归一化和资源持有上限规则；结果早于资源注册到达时，也使用这套规则设置临时期限。

已注册但尚未取走的结果标记为 `resource_owned`，hold 过期由 ResourceStore 统一判定。取走资源时，在 `resource_map_mutex_` 内校验 hold/transfer deadline，调用 `beginTransfer` 更新结果期限，再更新资源 entry 并从资源 map 移除。ResultStore 清扫和 notify 不会按旧 hold deadline 独立终止这类结果；回调失败时资源交接也失败，并统一进入终态。锁顺序固定为 ResourceStore 到 ResultStore，注册和交接回调不能反向调用 ResourceStore。

结果被消费或调用 `seal` 后，仍保留终态记录，防止迟到通知为已完成或已取消的请求重新创建数据。这是对原有终态行为及设计中 `seal` 契约的保留；当前 `notify` 并非无条件覆盖，详见设计意见 R5。

### 2.3 ResourceStore 与结果清理联动

`P2PConnectorResourceStore` 已移除 side channel 数据及等待接口，只管理资源租约。资源释放回调携带 unique key，对已被取走资源的 entry（`request_id == -1`）也会调用，使角色 connector 能够对对应结果执行 `seal`。清理 computed buffer 时仍要求存在有效的 request ID。

两个清理线程均使用现有的 `p2p_resource_store_timeout_check_interval_ms`，调用 `LoopThread` 时统一转换为微秒。ResultStore 清扫只有在等待状态改变时才唤醒等待者；重复扫描终态记录或删除已结束记录不会无条件唤醒。有限且已过期的业务 deadline 不会重新追加一小时，只有缺失或无限期限才使用原配置的 fallback TTL。

`addResource` 保留原有的 Meta 入口。此前为写回预留的 `addResource(key, request_id, deadline, resource)` 重载已撤回；结果存储测试通过现有 `MockMeta` 注册资源，继续覆盖资源取消及过期时对结果等待的影响。

### 2.4 协议与配置边界

Slice 0 提交中的 proto 相比基线仅保留 Read 类名的注释更新，消息、枚举、字段和 RPC 定义不变。该提交的 `CacheStoreConfig` 字段、Python 绑定、类型声明、CLI 参数及 pickle 格式恢复到基线；C++ 配置头文件只保留与 Read 重命名和结果存储拆分对应的注释更新。后续 Slice 1 新增的协议和配置见 §6。

### 2.5 超时控制详解

本次修复由 ResourceStore 确定资源期限，并在注册、取走和释放时将生命周期变化通知 ResultStore。两个 store 仍各自持有 map、锁和清理线程，通过回调连接；当前没有引入共享生命周期对象。

#### 2.5.1 期限和配置来自哪里

| 概念 | 来源 | 控制范围 |
| --- | --- | --- |
| 业务 deadline | 请求的 `routing.deadline_ms`，经公共函数归一化 | 请求整体期限，以及终态记录的主要保留期限 |
| hold deadline | ResourceStore 注册资源时，根据业务 deadline、注册时间和 `p2p_prefill_resource_hold_ms` 计算 | 资源等待 Decode 取走的阶段 |
| transfer deadline | 当前 StartLoad Read 路径的传输期限，再取与资源 entry 中业务 deadline 的较小值 | 本次 KV 发送与首 token 等结果的等待 |
| 清扫间隔 | `p2p_resource_store_timeout_check_interval_ms` | 两个 store 各自执行周期清理的频率 |
| fallback TTL | `p2p_cancelled_keys_ttl_ms` | 业务 deadline 缺失或无限时的兜底期限 |

`P2PConnectorPrefill::init()` 将同一份 scheduler 配置中的清扫间隔、hold 时长和 fallback TTL 传给两个 store，没有新增 ResultStore 专用超时参数。注册资源时，ResourceStore 只取一次当前时间并计算：

```text
hold_deadline = min(
    normalized_business_deadline,
    registration_time + p2p_prefill_resource_hold_ms,
    registration_time + 1 hour
)
```

其中一小时是原有资源持有上限。ResourceStore 在资源锁内调用 `on_request_registered`，将计算好的 hold deadline 和业务 deadline 原样交给 `ResultStore::registerRequest`。因此两边保存的是相同的绝对时间，避免分别读取当前时间、分别计算期限导致偏差。

首 token 的 `notify` 可能早于资源注册到达。此时 ResultStore 使用 `P2PRequestDeadline.h` 中相同的归一化和 hold 计算函数设置临时期限；后续注册成功时，再采用 ResourceStore 提供的期限。已经进入终态的记录不能通过注册恢复。

#### 2.5.2 每个阶段由谁判定过期

| 阶段 | ResultStore 状态 | 过期与清理责任 |
| --- | --- | --- |
| 结果先到，资源尚未注册 | 非终态，`resource_owned = false` | ResultStore 按临时期限判定过期 |
| 资源已注册，尚未取走 | 非终态，`resource_owned = true` | ResourceStore 统一判定 hold 过期，释放回调通知 ResultStore 执行 `seal` |
| 资源交接成功，进入传输 | 非终态，`resource_owned = false` | ResultStore 使用交接时发布的 transfer deadline |
| 结果已消费，或请求已结束 | `terminal = true`，结果数据已清空 | ResultStore 按终态保留期限删除记录 |

`resource_owned` 只是“当前 hold 过期由 ResourceStore 管理”的阶段标记，不代表 ResultStore 持有 KV 资源或新增 KV 引用。在该阶段，ResultStore 的清扫和 `notify` 都不会独立按旧 hold deadline 将结果置为终态。

`waitAndFill` 仍检查调用方传入的等待 deadline 和取消状态；正常 StartLoad 路径在交接成功后才调用它，并传入资源 entry 中的 transfer deadline。

#### 2.5.3 资源取走与期限更新如何交接

原来的竞态窗口是：StartLoad 先从 ResourceStore 取走资源，ResultStore 清扫线程随后按旧 hold deadline 清掉结果，最后 StartLoad 才尝试更新结果期限。此时结果已是终态，KV 发送即使成功，也无法取得首 token；`no_transfer` 同样会受到影响。

现在 `waitAndStealResource` 在持有 `resource_map_mutex_` 时完成以下步骤：

```text
取得资源锁，找到待交接 entry
  -> transfer_deadline = min(StartLoad 传输期限, entry 的业务 deadline)
  -> 校验 hold deadline 和 transfer deadline 均未过期
  -> 调用 on_request_acquired
       -> ResultStore::beginTransfer 取得结果锁
       -> 检查记录存在、非终态且 resource_owned 为 true
       -> 同时更新结果 deadline 和 resource_owned = false
  -> 更新资源 entry 的 deadline
  -> 从资源 map 移除 entry，并返回资源
释放资源锁
```

锁顺序固定为 **ResourceStore -> ResultStore**。注册和交接回调在资源锁内执行，不能反向调用 ResourceStore；释放回调在资源锁外执行。

如果交接成功，返回资源前结果期限已经切换，旧 hold deadline 不再决定该结果的有效性。如果 hold 已过期，即使清扫线程还没来得及删除 entry，取资源也会失败。如果结果已终态或交接回调拒绝，ResourceStore 同样拒绝交接，移除资源并通过释放回调收尾。

正常发送和 `no_transfer` 都经过这次交接。它保证资源和结果使用一致的传输期限；底层传输的停止和 lease 释放仍沿用已有 worker 逻辑。

#### 2.5.4 时间示例

假设业务 deadline 为 `10:05:00`，注册后的 hold deadline 为 `10:01:00`，本次 StartLoad 的 transfer deadline 为 `10:01:04`：

| 时间 | 事件与结果 |
| --- | --- |
| `10:00:59` | StartLoad 成功交接，资源 entry 和结果 entry 的有效期限都切换为 `10:01:04` |
| `10:01:00` | 原 hold deadline 到达，不会据此清掉已进入传输阶段的结果 |
| `10:01:04` | 本次 KV 发送和结果等待使用的期限到达；首 token 等待不会重新获得一段完整超时时间 |
| `10:05:00` | 该请求对应的终态记录保留期限到达，可由清扫线程删除 |

作为另一条独立路径，如果 StartLoad 到 `10:01:01` 才尝试取走仍处于 hold 阶段的资源，就会被拒绝，即使资源尚未被周期清扫删除。

#### 2.5.5 清扫频率与终态保留

`LoopThread` 的间隔参数单位是微秒。两个 store 的 `init()` 都使用 `int64_t(interval_ms) * 1000` 转换，默认清扫间隔因此为 100ms，修复了直接传入毫秒值后实际每 100μs 扫描的问题。

ResultStore 当前仍遍历整个 map，但只有清扫改变了等待者可见的状态时才调用 `notify_all()`，例如未结束的结果被判定过期。重复扫描终态记录、删除已经结束的记录，不会无条件唤醒等待者。注册、交接、结果到达和 `seal` 等事件仍会主动通知。

清扫间隔影响回收调度，不额外延长业务等待时间。`waitAndFill` 直接检查绝对 deadline，每次条件变量等待最多 10ms，以便继续检查取消和期限；这不是操作系统调度延迟的上界。

`normalizeP2PRequestDeadline` 的规则为：

```text
deadline <= 0 或 deadline == INT64_MAX:
    normalized_deadline = now + p2p_cancelled_keys_ttl_ms
其他有限 deadline:
    normalized_deadline = max(deadline, now)
```

因此，有限但已经过期的 deadline 归一化为当前时间，不再追加默认一小时的 TTL。`seal` 会保留已有终态保留期限和本次归一化期限中的较大值，所以不会缩短先前已经确定的未来保留期限；“过期不追加一小时”也不意味着强制覆盖该已有期限。

成功的 `waitAndFill` 会移走结果数据并立即置为终态。StartLoad 成功或失败的收尾再通过 `ResourceStore::markTerminal` 和释放回调执行 `seal`。终态记录在保留期间拒绝迟到通知和重新注册；清扫删除的是这条防重复记录，首 token、复用统计和 MTP 数据仍通过 `StartLoadResponse.payload` 返回。

#### 2.5.6 代码与验证对照

以下文件均位于 `rtp_llm/cpp/cache/connector/p2p/`：

| 文件 | 本次超时修复的职责 |
| --- | --- |
| `P2PRequestDeadline.h` | 共享业务 deadline 归一化和 hold deadline 计算规则 |
| `P2PConnectorResourceStore.h/.cc` | 注册与交接回调、锁内校验和交接、清扫间隔单位转换 |
| `PrefillResultStore.h/.cc` | `resource_owned` 阶段标记、`beginTransfer`、条件唤醒和终态期限处理 |
| `P2PConnectorPrefill.cc` | 传入同源配置、连接生命周期回调、让 KV 发送和结果等待使用资源 entry 的期限 |
| `test/PrefillResultStoreTest.cc` | 清扫初始化、期限同源、交接过程中清扫及终态期限回归 |
| `test/P2PConnectorResourceStoreTest.cc` | 清扫单位以及 hold/transfer 阶段边界回归 |
| `test/P2PConnectorTest.cc` | 正常传输和 `no_transfer` 路径的交接竞态回归 |

交接回归测试在资源取走回调中、`beginTransfer` 之前，以超过旧 hold deadline 的时间执行 ResultStore 清扫，验证旧期限不会提前终止仍由 ResourceStore 管理的结果。另有用例覆盖 hold 过期先发生、终态拒绝交接和有限过期 deadline 不追加 TTL。修复后的容器编译与 212 个单测结果见 §4.5。

## 3. Slice 0 测试覆盖

| 测试目标 | 相关覆盖范围 |
| --- | --- |
| `//rtp_llm/cpp/cache/connector/p2p/test:components_test` | 结果通知时序、单次消费、seal、过期、资源释放联动、MTP 数据，以及现有组件回归 |
| `//rtp_llm/cpp/cache/connector/p2p/test:p2p_connector_worker_test` | 改名后的现有 Read worker 回归 |
| `//rtp_llm/cpp/cache/connector/p2p/test:p2p_connector_test` | 现有 StartLoad 和结果响应路径，包括 notify 先到、资源被取走后 notify、零值首 token 及无需传输路径 |
| `//rtp_llm/cpp/cache/connector/p2p/test:p2p_connector_scheduler_test` | 改名后的现有 Read scheduler 回归 |
| `//rtp_llm/cpp/cache/connector/p2p/test:p2p_connector_worker_decode_lease_test` | 改名后的现有 Decode lease 回归 |
| `//rtp_llm/cpp/cache/connector/p2p/plan/test:route_codec_test` | Sender/receiver 编码接口改名，以及现有 key/路由编码 |

保留原有 11 个 `PrefillResultStoreTest` 用例，并补充实际清理线程初始化、期限同源、交接时清扫、过期先发生、终态拒绝交接和业务期限归一化的回归测试。connector 测试同时覆盖正常传输和 `no_transfer` 的交接竞态。§4.2 的撤回阶段曾移除 10 个 Write worker 用例、2 个 Write 角色分发用例和 2 个写回配置用例，并将配置测试整体恢复到基线；这些用例不属于 Slice 0。重新实施 Slice 1 后的测试覆盖见 §6。

## 4. 验证记录

### 4.1 2026-09-09：撤回前的历史结果

迁移前的源机器仓库位于 `/home/admin/workspace/aop_lab/app_source/rtp-llm`，使用 Bazel 6.4.0、CUDA 12.9 和 H20 GPU（`sm9x`）。当时 Slice 0/1 合并版本的 7 个测试目标及 `normal_engine` 编译通过，共 229 个单测通过，未执行 smoke。

当时编译使用了源机器专属的 `/tmp/decode-writeback-cache.bazelrc`，复用 11 个包定义和 wheel SHA256 与锁文件一致的缓存仓库，并用 `--per_file_copt=external/grpc/.*@-Wno-error=tautological-compare` 局部处理第三方 gRPC 在 `tcp_posix.cc:746` 的既有告警。Python pickle 测试修正后单独复跑通过。

**上述结果仅属于撤回前的版本，不能作为当前仅保留 Slice 0 版本的编译或单测通过结论。**

### 4.2 2026-09-10：迁移与 Slice 1 撤回

迁移包校验通过后，代码在相同基线的干净工作区上应用，最初得到 58 个状态项。随后按用户安排撤回 Slice 1：

- 删除两个 Write worker、公共工具类及对应测试文件。
- 删除 Write 广播分发、角色入口、backend 额外持有逻辑和写回 key 工具。
- 撤回 StartWrite 消息/RPC、Write 操作与状态字段、全部写回配置和绑定。
- 撤回 ResourceStore 的无 Meta 重载，两个相关结果存储用例改用原有 `MockMeta` 入口。
- 保留 Read 重命名、`PrefillResultStore` 拆分及资源终态与结果清理联动。

撤回后的静态检查全部通过：无 Slice 1 符号残留，66 个 BUILD 源文件/头文件引用有效，8 个 Read 文件在替换标识符后与基线一致，协议和配置仅保留对应注释更新，diff 与新增文件的空白检查通过。随后在目标容器内完成编译和单测，结果见下一节。

### 4.3 2026-09-10：Slice 0 容器编译与单测

期限交接修复前，仅保留 Slice 0 的版本已在 `renyuanming.rym_vscode` 容器中验证通过，容器直接挂载当前工作区。修复后的结果见 §4.5。

- 环境：Bazel 6.4.0、Python 3.10.9、GCC 10.2.1、CUDA 13.2。
- GPU：NVIDIA L20D，计算能力 8.9；显式将目标及 host 的 CUDA 架构设置为 8.9。
- 并发：`--jobs=96 --local_cpu_resources=96`，构建期间达到 96 个并发任务。
- 编译：6 个 C++ 单测目标及 `//rtp_llm/cpp/normal_engine:normal_engine` 全部通过，耗时 380.747 秒。
- 单测：6 个目标共 **201 个用例全部通过**，无失败、错误、跳过或禁用项；测试命令耗时 73.708 秒。
- 测试使用 `--local_test_jobs=1 --test_env=CUDA_VISIBLE_DEVICES=0` 串行运行，并关闭测试结果缓存。
- 本轮无须修改实现代码或构建配置，未运行 smoke。

| 测试目标 | 用例数 | 结果 |
| --- | ---: | --- |
| `components_test` | 84 | 通过，包含全部 11 个 `PrefillResultStoreTest` 用例 |
| `p2p_connector_worker_test` | 44 | 通过 |
| `p2p_connector_test` | 20 | 通过 |
| `p2p_connector_scheduler_test` | 28 | 通过 |
| `p2p_connector_worker_decode_lease_test` | 20 | 通过 |
| `route_codec_test` | 5 | 通过 |
| **合计** | **201** | **全部通过** |

依赖准备阶段曾等待 Abseil 的 GitHub 拉取，期间直连探测出现超时；原始拉取最终完成，没有切换镜像或修改固定提交。编译沿用仓库 `cuda13` 配置已有的告警选项，未添加源机器的临时缓存 rc。

### 4.4 实际执行命令与结果位置

以下命令从宿主机执行，编译和测试均在指定容器中运行。结果目录为 `/data6/renyuanming.rym/.cache/decode-writeback-slice0-20260910`。

```bash
slice0_tests=(
  //rtp_llm/cpp/cache/connector/p2p/test:components_test
  //rtp_llm/cpp/cache/connector/p2p/test:p2p_connector_worker_test
  //rtp_llm/cpp/cache/connector/p2p/test:p2p_connector_test
  //rtp_llm/cpp/cache/connector/p2p/test:p2p_connector_scheduler_test
  //rtp_llm/cpp/cache/connector/p2p/test:p2p_connector_worker_decode_lease_test
  //rtp_llm/cpp/cache/connector/p2p/plan/test:route_codec_test
)

docker exec --workdir /data6/renyuanming.rym/rtp-llm \
  renyuanming.rym_vscode bazelisk build --config=cuda13 \
  --action_env=TF_CUDA_COMPUTE_CAPABILITIES=8.9 \
  --host_action_env=TF_CUDA_COMPUTE_CAPABILITIES=8.9 \
  --jobs=96 --local_cpu_resources=96 --color=no --curses=no \
  --build_event_json_file=/data6/renyuanming.rym/.cache/decode-writeback-slice0-20260910/build-events.json \
  "${slice0_tests[@]}" //rtp_llm/cpp/normal_engine:normal_engine

docker exec --workdir /data6/renyuanming.rym/rtp-llm \
  renyuanming.rym_vscode bazelisk test --config=cuda13 \
  --action_env=TF_CUDA_COMPUTE_CAPABILITIES=8.9 \
  --host_action_env=TF_CUDA_COMPUTE_CAPABILITIES=8.9 \
  --jobs=96 --local_cpu_resources=96 --color=no --curses=no \
  --local_test_jobs=1 --test_env=CUDA_VISIBLE_DEVICES=0 \
  --test_output=errors --test_timeout=600 --nocache_test_results \
  --build_event_json_file=/data6/renyuanming.rym/.cache/decode-writeback-slice0-20260910/test-events.json \
  "${slice0_tests[@]}"
```

构建和测试事件分别记录在结果目录的 `build-events.json`、`test-events.json`。各测试目标的原始日志和 XML 位于仓库 `bazel-testlogs/rtp_llm/cpp/cache/connector/p2p/` 下的对应目录，用例数从 XML 汇总。

### 4.5 2026-09-10：deadline 同源与交接竞态修复

本轮修复三条审阅意见：清扫间隔误用微秒、资源取走与结果期限更新之间的竞态，以及已过期业务 deadline 额外延长终态 TTL。实现采用 ResourceStore 的注册、交接和释放回调统一生命周期；没有新增共享 KV 资源对象，正向 StartLoad 响应协议保持不变。

| 修改文件（相对 `rtp_llm/cpp/cache/connector/p2p/`） | 变更 |
| --- | --- |
| `P2PRequestDeadline.h`、`BUILD` | 新增公共期限计算规则，并登记头文件 |
| `P2PConnectorResourceStore.h`、`.cc` | 毫秒转微秒；一次计算注册期限并下发；锁内完成 hold 校验及传输交接；释放回调使用同源的业务期限 |
| `PrefillResultStore.h`、`.cc` | 引入 `beginTransfer` 与 `resource_owned`；hold 阶段跟随 ResourceStore；修正清扫单位、唤醒条件和终态 TTL |
| `P2PConnectorPrefill.cc` | 接入生命周期回调，删除重复计算及取走后的二次注册；发送和结果等待使用资源 entry 的传输期限 |
| `test/PrefillResultStoreTest.cc` | 保留原有 11 个用例，新增 9 个期限、清扫和交接回归用例 |
| `test/P2PConnectorResourceStoreTest.cc` | 新增清扫单位用例；更新 hold 到 transfer 的期限断言 |
| `test/P2PConnectorTest.cc` | 新增交接期间清扫用例，同时验证传输与 `no_transfer` 响应 |

在 `renyuanming.rym_vscode` 中沿用 §4.4 的 CUDA 13、计算能力 8.9 和 `--jobs=96 --local_cpu_resources=96` 参数完成验证：

- 6 个单测目标及 `normal_engine` 编译通过，耗时 50.592 秒。
- 6 个单测目标共 **212 个用例全部通过**，无失败或跳过，耗时 72.464 秒。
- 用例数：components 94、connector 21、scheduler 28、worker 44、decode lease 20、route codec 5。
- 新增用例共 11 个；ResultStore 测试共 20 个。交接清扫测试显式推进清扫时间，覆盖期限更新前的交错顺序；初始化测试实际启动清理线程并核对传入的微秒间隔。
- 结果目录：`/data6/renyuanming.rym/.cache/decode-writeback-slice0-deadlines-20260910`，包含 `build-events.json` 和 `test-events.json`。测试关闭缓存，未运行 smoke。

## 5. 当前审阅重点与后续设计契约

当前 Slice 0 重点审阅 `PrefillResultStore` 的状态转换、deadline 交接、资源取消与结果等待的联动，以及迟到 notify 和 seal 的竞争。设计意见 R5 仍保留在设计文档中，当前实现采用终态记录防止迟到通知重新创建数据。

R1 的物理传输停止与资源释放、R2 的 KV 就绪边界、R3 的纯探测接口及同锁发布实现、R4 的跨 rank 完成汇总，以及 R6 的混合 group 范围，属于后续写回阶段需要确认或落实的契约。Slice 1 先提供 worker 本地任务状态，跨 rank 汇总和资源持有者仍需在 Slice 2/3 落实；设计文档中的待讨论 comments 继续保留。

### 5.1 Slice 2 已明确、尚未实现的两项契约

- **接收分配引用（设计 R7，本轮 comment 4）**：`mallocForExternalInsert` 返回 owning RAII 句柄，直接接管 malloc 已增加的原始引用，接收端不再调用 `incrKVCacheRef` 叠加引用。成功插树后归还分配引用，只保留树的 CACHE 引用；失败及重复结果未采纳的新块均按同一规则释放，释放前须确认没有在途传输。分配接口负责部分失败回滚。
- **HOST/DISK 后缀冲突（设计 R8，本轮 comment 5）**：p=p_dev 不代表后缀不存在。Phase 1 对已有低层或忙碌 FULL 节点冲突放弃写回，不提供节点升级接口。准入先检查一次；settle 在树锁内完整检查前缀和全部后缀后才发布，防止普通 insert 提前停止而留下部分更新。

上述两项已明确设计契约，接收分配与插树尚未实现。后续 Slice 2 必须补分配引用账本、失败回滚、HOST 后缀拒绝、传输中降级和部分发布防护测试。§4.5 的 212 个通过用例仅验证 Slice 0，不作为这两项未实现能力的验证结果。

## 6. Slice 1：写回基础设施

### 6.1 本轮范围与代码位置

本轮以已提交的 Slice 0 为起点，参考迁移包中曾撤回的 Slice 1 实现，接入协议、两个轻量 Write worker、backend 持有、角色分发和基础配置。Slice 0 的 ResourceStore/ResultStore 期限交接保持现有实现。

| 位置 | 当前实现 |
| --- | --- |
| `model_rpc/proto/model_rpc_service.proto` | 声明 `StartWrite` 请求/响应及 RPC；增加两个 Write 广播类型、操作枚举和结果状态字段 |
| `p2p/P2PWorkerDecodeWrite.{h,cc}` | `write()` 按显式 route 构造一次性 `SendRequest`，跟踪发送任务及完成回调 |
| `p2p/P2PWorkerPrefillWrite.{h,cc}` | `handleWrite()` 将内部 route 的显式落点注册到已有 receiver，返回注册错误；查询和取消使用独立入口 |
| `p2p/P2PWriteWorkerUtil.{h,cc}` | 共用内部路由校验、传输单元构建、`WriteTaskGroup` 状态管理和 `WriteTaskStatus` 状态结构 |
| `p2p/P2PTransferLease.h` | 由原 `DecodeTargetWriteLease` 改名，Read 与两侧 Write 共用任务计数及停止判定 |
| `p2p/P2PBroadcastClient.{h,cc}` | 提供专用 Write CANCEL/QUERY 广播，独立判断全部停止和全部写回成功 |
| `p2p/P2PConnectorDecode.{h,cc}` | 开关开启时将 backend sender 交给 Write worker；`writePerRank` 负责协议解析、操作分发和状态响应 |
| `p2p/P2PConnectorPrefill.{h,cc}` | 开关开启时将 backend receiver 交给 Write worker；`processWritePerRank` 负责协议解析、操作分发和状态响应 |
| `p2p/P2PConnector.cc` | 将 `WRITE` / `HANDLE_WRITE` 分发到对应角色；错误角色和关闭的 Write 路径返回失败 |
| `p2p/P2PKeyUtil.h` | 增加 `makeWriteBackRouteLayerKey`，使用 `_wb_` 标记并包含 plan digest |
| `config/ConfigModules.{h,cc}`、`pybind/ConfigInit.cc`、Python 类型声明和 CLI 参数 | 写回开关、超时及并发配置的声明、绑定、序列化和诊断输出 |

以上 C++ 路径相对 `rtp_llm/cpp/`。BUILD 和对应单测同步更新。当前只声明 `StartWrite` RPC，`RemoteRpcServiceImpl` 尚未实现处理入口，调用仍由生成的默认实现返回未实现；真实请求尚不会自动发起写回。

协议解析直接放在两侧角色 connector 的 START 分支。公共响应封装集中在现有 `P2PConnector`：`setP2PResponse()` 将 `ErrorInfo` 转成 protobuf 错误码和错误信息，省略错误参数时写入成功状态；`fillWriteResponse()` 写入 lease 计数、stopped、write_success，并复用前者填写错误字段。总入口、Decode 和 Prefill 共用这些静态方法，不再各保留一份文件内实现，也不新建 RPC util。两侧保留必要的局部解析重复；worker 共用的任务状态和传输单元构建仍保留在 `P2PWriteWorkerUtil` 中。

### 6.2 路由与 backend 契约

Decode Write worker 持有 sender，Prefill Write worker 持有 receiver，两侧复用已有 backend 创建和内存注册流程。开关关闭时不创建 Write worker，原有 Read worker 继续持有其所需的一半 backend。

每个 Write route 的 `layer_blocks` 显式携带 layer、cache tag、cache key 和本侧 block id。发送侧还使用 `peer_index` 查找 `peer_workers` 中的 Prefill 传输端点；接收侧只需要落点，不要求对端地址。先校验整份请求、构造全部传输单元，再开始注册或发送，避免最后一条路由非法时前面的任务已经启动。

connector 在解码前校验原始 partition/slice 字段，避免 `RouteCodec` 的默认值归一化掩盖非法输入；同时检查 key/block 数量对应且非空、key 无重复、数值可由内部类型表示，并解析发送侧 peer 端点。worker 校验有效 key/deadline、route id 唯一、layer/tag 归属、内部 buffer 非空、block id 有效、转换后地址非空及发送端点有效。Phase 1 仅接受 partition `{1, 0}`、无 slice 的整块路由，内部 route 同样校验该约束；接收方另检查整份请求是否重复使用同一个 `(layer, tag, block_id)`，拒绝重叠落点。两层均在全部校验通过后才启动任何传输任务。

两侧 layout 对称断言、写回方向 plan digest 比较、接受范围协商，以及不同 group 的有效键集仍由后续 scheduler 负责。worker 将 digest 编入传输 key 以隔离 rendezvous，但不把这一点视为完成了 layout 校验。

### 6.3 注册、完成与取消状态

两个 Write 广播类型共用 `write_operation`，由角色 connector 分发到 worker 的启动、取消和查询方法，再把内部状态封装为响应：

| 操作 | 行为 |
| --- | --- |
| `WRITE_START` | 接收侧登记落点，发送侧向有界后台队列提交整个请求；返回当前状态，不等待 D2H 拷贝或传输完成 |
| `WRITE_QUERY` | 查询已有任务组；未知 key 返回错误 |
| `WRITE_CANCEL` | 请求取消并返回当前状态；取消先于 START 到达时也记录终态，阻止迟到 START 重新打开同一个 key |

`lease_status.sealed` 表示不会继续向任务组添加任务；`started_ops` / `finished_ops` 统计 transfer task 数量，当前每个 route 的每个 layer/tag 对应一个任务，并非物理块数量。发送侧的 `started_ops` 包含已登记但仍在队列中的 PENDING 任务，不表示 backend 已开始处理。`stopped` 要求任务列表已封闭，且所有任务的 `done()` 都为真。`write_success` 还要求所有任务成功、任务组未取消且没有注册或入队错误。

`error_code` 反映操作或注册是否发生错误；单个数据传输失败时，调用方必须结合 `stopped` 和 `write_success` 判断结果，不能仅以 `error_code == NONE_ERROR` 判定写回成功。空 routes 表示本 rank 无任务：Decode 完成校验和去重登记后直接 seal 并返回 stopped/success，不提交发送闭包，不受发送队列是否已满影响；任务组仍保留供 QUERY 和防重复。重复 START 返回错误，不替换原任务及落点。

任务组织采用与 Read 相同的 group + lease 结构。原 `DecodeTargetWriteLease` 改名为 `P2PTransferLease`，保留原有登记、完成回调、seal 及停止判断接口，新增完成计数快照更新接口；Decode Read 的 `ReadTaskGroup` 与两侧 Write 的 `WriteTaskGroup` 均持有该类型。lease 只负责 `seal`、登记/完成计数和 `isStopped()`，不持有 KV 块引用，也不判断业务成功。

Write 登记每个 task 时调用 `onTransferStarted()`，登记或提交结束后调用 `seal()`；Read 查询及 Write 查询/清扫统计已完成任务总数后，共用 `updateFinishedOps(done_now)` 单调更新 lease。该方法通过原子比较更新保留最大完成数，重复或较旧的快照不会重复累加或使计数回退，调用方不再维护 `finish_counted`。当前任务组统一使用快照更新，不同时对相同任务调用 `onTransferFinished()`。group 保存取消和业务错误，全部停止后缓存 `terminal_success`；释放 task 描述后，lease 保留最终计数，成功或失败结果仍可查询。Write 的未知 QUERY、取消终态和在途任务保留期限沿用 §6.3.1/§6.4，不随结构对齐改变。

简化时序：

```text
Prefill WRITE_START -> 注册 recv task -> sealed=true, stopped=false -> 返回
Decode  WRITE_START -> 登记全部 PENDING task -> 非阻塞入队 -> sealed=true -> 返回
后台发送线程        -> 检查原 deadline / task 取消状态 -> startTransfer -> sender.send
双方 WRITE_QUERY   -> 仍在传输时 stopped=false
backend 完成       -> task.notifyDone
双方 WRITE_QUERY   -> stopped=true，write_success 反映全部任务结果
清扫                -> 释放已停止的任务描述，暂留终态结果供查询和防重复
```

部分注册失败时取消已经注册的任务，并从 receiver 的 rendezvous store 移除落点。正在传输的 task 仍由 worker 的任务组跟踪，直至 backend 报告完成。

#### 6.3.1 START RPC 失败后的停止确认

`LocalRpcServer::ExecuteFunction` 会把 connector 返回的 `false` 转成 gRPC `INTERNAL`。Prefill worker 在部分注册失败时返回 `ErrorInfo` 并保留可查询的任务状态；connector 查询该状态、填充本地 response 后返回 `false`，客户端仍拿不到其中的正常状态。START 的网络错误、超时或响应缺失同样只能视为状态未知。

后续 Slice 2/3 scheduler 必须遵守以下契约：

1. START 失败后保留源块和接收落点，对所有可能收到 START 的 rank 执行 `WRITE_CANCEL`，再按需重复 CANCEL/QUERY。不能依据失败的 START 响应释放资源。
2. 使用 `P2PBroadcastClient::controlWrite(key, type, operation, transfer_deadline_ms, control_timeout_ms)`。该入口仅接受两个 Write 类型及 CANCEL/QUERY 操作，向 client 配置的所有 worker 广播；所有控制 RPC 成功后才读取状态，任一 RPC 失败则结果无有效停止确认。
3. `WriteStatusResult::allStopped()` 要求非空结果、每个 rank 均有 lease 状态、sealed/stopped 为真且任务计数一致。业务错误仍可携带有效停止确认。`allSucceeded()` 进一步要求所有 rank 的 `write_success` 为真且无业务错误。已有通用 `Result::success()` 不能代替这两种判断。
4. 未知 key 的 QUERY 返回错误，不表示没有在途任务，因为 QUERY 可能先于 START 到达。CANCEL 会留下终态记录，阻止迟到 START；只有收到有效停止确认后，才能据此归还资源。
5. 控制 RPC 使用独立的 `control_timeout_ms`，允许在传输 deadline 过期后继续确认停止；payload 保留原传输 deadline，不延长已有任务期限。后续 scheduler 应从现有 `p2p_cancel_broadcast_timeout_ms` 等统一控制配置取值。

```text
START -> 已有部分任务开始 -> 注册失败 / 响应丢失 -> 客户端状态未知，继续持有资源
CANCEL / QUERY -> stopped=false -> 继续持有资源
backend 完成回调 -> CANCEL / QUERY -> 全部 rank stopped=true -> 允许释放
```

本片补齐控制接口与真实 RPC 回归测试，跨 rank 的重试、资源持有和收尾编排仍属于 Slice 2/3。

#### 6.3.2 有界后台发送与排队取消

Decode Write worker 在 `init()` 中创建独立 `autil::ThreadPool`，默认 4 个线程、最多 10000 个排队请求；每个请求作为一个队列任务，后台线程逐个调用其传输单元的 `sender->send()`。线程数和队列容量可由初始化参数指定，测试使用 1 个线程、1 个队列槽位验证边界；这不是 `p2p_writeback_max_inflight` 的业务准入，后者仍由 Slice 3 实现。选择 `ThreadPool` 是为了使用严格的队列容量上限及 `STOP_AFTER_QUEUE_EMPTY` 排空语义。

所有 transfer task 在任务组对外可见前完成登记。入队显式使用 `isBlocked=false`、`executeWhenFail=false`：队列满或线程池已停止时，取消该请求全部 PENDING task，封闭任务组并返回失败；不等待队列空位，也不在 RPC 线程回退执行发送。失败后同样通过上述控制协议确认停止。

后台每次发送前检查原始绝对 deadline，并以 `task->startTransfer()` 与取消操作互斥。排队时已取消或过期的 task 不调用 backend，即使队列闭包仍持有地址描述，也不会读取 KV 数据。已进入 TRANSFERRING 的任务继续等 backend 回调，CANCEL 和清扫均不能提前报告 stopped。sender 抛出异常时当前 task 记为失败，并取消本组其余任务。析构时先取消各任务组，再等待后台队列排空，保证排队闭包不会在 worker 销毁后发起新传输。

### 6.4 超时与资源生命周期边界

Write worker 接收的是 scheduler 下发的绝对 deadline，不在收到广播时重新增加一段超时。清扫间隔沿用 `p2p_resource_store_timeout_check_interval_ms`，通过 `P2PConnectorWorkerConfig::create` 从相同 CacheStoreConfig 赋值，调用 `LoopThread` 时统一乘 1000 转为微秒。相比迁移版，删除了 Write worker 写死的 10ms 清扫间隔。

deadline 到达或收到 CANCEL 后，只调用 task 的 `cancel()`。PENDING 任务可以立即结束；TRANSFERRING 任务须等待完成回调，不调用 `forceCancel()`，也不因期限到达就报告 stopped。即使超过终态 TTL，只要物理任务仍未完成，任务组也继续保留。已停止任务的描述可以提前释放，但终态状态保留到任务 deadline 加 `p2p_cancelled_keys_ttl_ms` 后再清扫，防止迟到 START 覆盖原任务。

**worker 中的地址是借用地址，不持有 KV 块的分配引用或 connector hold。** 后续 scheduler/接收上下文必须持有资源直到相关 rank 报告 stopped；业务超时只停止业务等待并禁止插树，不能直接归还仍被传输访问的块。当前状态协议为 R1/R4 提供 worker 层基础，尚未实现跨 rank 完成汇总、StartWrite 返回后的 owning 上下文或 settle。

### 6.5 配置与兼容性

| 配置 | 默认值 | Slice 1 中的作用 |
| --- | --- | --- |
| `p2p_writeback_enable` | `false` | 控制 Write worker 创建、backend 持有和广播入口是否可用 |
| `p2p_writeback_timeout_ms` | `5000` | 完成配置绑定和序列化，后续 Slice 2/3 接入 deadline 计算 |
| `p2p_writeback_max_inflight` | `4` | 完成配置绑定和序列化，后续 Slice 3 接入并发准入 |

三项配置均提供对应的 CLI 参数和大写环境变量。CacheStoreConfig 的 pickle 输出从 29 项扩为 32 项，并兼容原有 20、23、26、29 项状态；旧状态恢复时写回默认关闭。新增 round-trip、旧格式默认值和 CLI/环境变量优先级测试；既有 pickle 测试改为使用真实 pickle round-trip，或通过 `__new__` 创建未初始化实例后调用 `__setstate__`。

提前写回和 HOST 目标配置没有在本片开放，仍按后续方案范围处理。

### 6.6 测试与验证记录

#### 2026-09-10 初版验证

新增 Write worker 用例覆盖多层/多 key 复制、真实 TCP 收发且两端物理 block id 不同、注册先于发送返回、配置同源、实际清扫线程、异常路由无部分副作用、重叠落点、重复 START、取消先到、部分注册失败、传输中取消、超时后继续跟踪及迟到失败完成。角色 connector 用例覆盖默认关闭与正确角色分发；配置用例覆盖 pickle 兼容和 CLI/环境变量绑定。

- 容器：`renyuanming.rym_vscode`；CUDA 13、计算能力 8.9，`--jobs=96 --local_cpu_resources=96`。
- 编译：6 个 C++ 单测目标、`normal_engine`、`model_rpc_server` 及 `//rtp_llm/libs:libth_transformer_config_so` 通过，耗时 80.894 秒。
- C++ 单测：6 个目标共 **230 个用例通过**，无失败或跳过；沿用 §4.4 的目标及 GPU 0 串行、关闭结果缓存的参数。用例数为 components 94、connector 23、scheduler 28、worker 60、decode lease 20、route codec 5；其中本片新增 16 个 Write worker 用例和 2 个角色分发用例。
- 首轮 C++ 回归耗时 77.875 秒，只有新增真实 TCP 用例失败：测试使用了 CPU 数组，但已有 TCP KV backend 的拷贝路径要求 CUDA 缓冲区。将该夹具改为初始化 runtime、分配真实 CUDA tensor 并同步源数据就绪后，单独重编并复跑 worker 目标，60 个用例全部通过；重编加测试耗时 47.395 秒，测试进程耗时 14.9 秒。其他 5 个已通过目标的代码没有变化。
- Python 配置单测：在容器中执行 `python3 -m unittest rtp_llm.test.engine_config_test -v`，**17 个用例通过**；新增 3 个用例，分别覆盖写回 pickle round-trip、四种旧状态格式和默认值/环境变量/CLI 优先级。
- 环境差异：直接构建 `//rtp_llm/test:engine_config_test` 时，CUDA 13 SDK 依赖引用不存在的 `@pip_gpu_cuda13_torch//flash_attn` Bazel 包，分析失败。因此配置扩展单独构建，Python 单测使用容器现有环境运行，无需改动仓库依赖配置或安装依赖。
- 结果目录：`/data6/renyuanming.rym/.cache/decode-writeback-slice1-20260910`。`build-events.json` 记录 Python Bazel 目标的依赖分析失败，`build-cpp-events.json` 记录成功编译，`test-events.json` 记录首轮 C++ 回归，`test-worker-events.json` 记录修正夹具后的 worker 复跑，`python-test.log` 记录 Python 配置测试。本轮未运行 smoke，也未执行 RDMA 硬件传输测试。

#### 2026-09-11 Review 修复验证

针对 START 失败状态丢失及发送 START 同步阻塞的问题，新增 10 个 C++ 用例：

- 通过真实 gRPC stub 和 `LocalRpcServer -> KVCacheManager -> P2PConnector -> Prefill Write worker` 验证部分注册失败返回 INTERNAL 且无正常 response；正在传输的任务必须等回调后才能确认停止，业务错误不妨碍有效停止确认。
- START 处理后响应超时丢失、未知 QUERY、CANCEL 抢先于 START、传输期限过期后的独立控制 RPC、缺失状态、部分 rank RPC 失败，以及成功传输的停止/成功判定。
- sender 同步阻塞时 START 仍返回；队列满时拒绝且不回退同步发送；排队取消和排队期限到达均不调用 backend；sender 异常取消剩余单元。

编译与回归在 `renyuanming.rym_vscode` 中运行，沿用 CUDA 13、计算能力 8.9 和 `--jobs=96 --local_cpu_resources=96`；测试仍使用 GPU 0、单目标串行并关闭结果缓存。

- 6 个 C++ 单测目标、`normal_engine` 和 `model_rpc_server` 编译通过。
- 6 个单测目标共 **240 个用例全部通过**，无失败或跳过：components 94、connector 23、scheduler 28、worker 70、decode lease 20、route codec 5。编译与回归合计耗时 106.411 秒。
- 收紧 START 响应丢失用例，额外断言超时前已经注册两个接收任务且 QUERY 返回有效状态；重编及复跑 worker 目标后 **70 个用例全部通过**，合计耗时 55.575 秒。
- 本次修改不涉及 Python 配置；沿用初版 17 个 Python 配置用例的验证记录，本轮未重跑 Python、smoke 或 RDMA 硬件测试。
- 结果目录：`/data6/renyuanming.rym/rtp-llm/.cache/decode-writeback-slice1-review-20260911`。`build-events.json` 记录初次编译，`test-events.json` 记录线程池边界用例发现的问题，`test-final-events.json` 记录最终实现的编译和 240 个 C++ 用例通过。收紧 START 响应丢失用例的断言后，worker 复跑记录在 `test-worker-final-events.json`。

#### 2026-09-11 命名统一与实现计划调整

Decode 单 rank connector 入口统一为 `writePerRank()`，对应 worker 入口统一为 `write()`；调用点、既有单测及设计文档同步命名，执行逻辑保持不变。新的五片计划及各层入口边界见 §1；Slice 2/3 分别独立验证 Prefill/Decode scheduler，生产触发和真实两端接通划入 Slice 4。

本轮在同一容器中使用 96 CPU 线程，connector/worker 单测目标、`normal_engine` 和 `model_rpc_server` 编译通过；复跑 **93 个 C++ 用例全部通过**（connector 23、worker 70），无失败或跳过，编译及测试合计 81.435 秒。结果记录为上述目录中的 `test-write-entry-rename-events.json`。未新增单测或代码注释；本轮未运行 Python、smoke 或 RDMA 硬件测试。

#### 2026-09-11 Write worker 与 Read 分层对齐

两侧 Write worker 启动接口改为接收 `request_id`、unique key、绝对 deadline 和 `P2PWorkerRoutePlan`，返回 `ErrorInfo`；Prefill 入口使用 `handleWrite()`。START/QUERY/CANCEL 分发、protobuf 校验与状态响应封装移到角色 connector，worker 通过独立 `cancelWrite()` / `queryWriteStatus()` 管理任务。该轮曾将协议转换单独拆为 RPC util，随后按 §6.1 收回角色 connector；worker 的共用实现保留。

原有协议、取消、异步发送及真实 TCP 用例改走实际角色 connector 入口，真实 gRPC 用例继续经过 `LocalRpcServer`。新增 4 个直接使用内部 route plan 的 worker 用例，覆盖注册与完整复制、未知查询与取消终态、非法内部路由无传输副作用，以及部分注册失败后必须独立查询停止状态。扩充非法协议字段用例，覆盖 partition 默认值、非法 slice 枚举及 layer/block 数值越界，确保转换过程不会吞掉原有校验。

- 容器 `renyuanming.rym_vscode`，CUDA 13、计算能力 8.9；`--jobs=96 --local_cpu_resources=96`，测试 GPU 0、单目标串行、关闭结果缓存。
- 6 个 C++ 单测目标、`normal_engine` 和 `model_rpc_server` 编译通过；**244 个用例全部通过**，无失败或跳过：components 94、connector 23、scheduler 28、worker 74、decode lease 20、route codec 5。编译与回归合计 109.545 秒。
- 结果记录为上述目录中的 `test-worker-layering-events.json`。本轮未运行 Python、smoke 或 RDMA 硬件测试。

#### 2026-09-11 RPC util 收敛

删除 `P2PWriteRpcUtil.h/.cc` 及 BUILD 引用，将 protobuf 路由校验和转换直接放回两侧角色 connector 的 START 分支。Decode 解析发送端点，Prefill 只解析接收落点；Write 状态由文件内函数封装，错误字段复用已有 `setP2PResponse()`。worker 接口、共用任务状态和传输单元构建保持现有实现。

本轮为代码搬移与错误响应函数复用，未新增单测。容器 `renyuanming.rym_vscode` 中使用 CUDA 13、计算能力 8.9 和 96 CPU 线程，connector/worker 单测目标、`normal_engine` 和 `model_rpc_server` 编译通过。复跑 **97 个 C++ 用例全部通过**（connector 23、worker 74），无失败或跳过，包含非法协议、真实 RPC/TCP、取消和异步发送回归；编译与测试合计 79.816 秒。

结果记录为上述目录中的 `test-rpc-util-consolidation-events.json`。本轮未运行 Python、smoke 或 RDMA 硬件测试。

#### 2026-09-11 Task group 与 lease 对齐

将现有 `DecodeTargetWriteLease.h` 改名为 `P2PTransferLease.h`，类型和引用同步调整；lease 的计数、封闭及停止判断代码保持原样。Decode Read 继续使用 `ReadTaskGroup + lease`；Decode/Prefill Write 使用 `WriteTaskGroup + lease`，登记任务时增加 started 计数，登记或提交结束后 seal，查询和清扫时按 `done_now - finish_counted` 增量更新 finished 计数。

Write group 移除独立的 `sealed` 字段和整份终态快照。lease 统一保存封闭状态及最终计数，group 单独缓存 `terminal_success`；成功与失败终态都能在 task 描述释放后继续查询。任务取消、后台发送、原始 deadline、未知 QUERY 和防止迟到 START 的行为保持不变。

新增 3 个用例：两侧部分完成时重复 QUERY/清扫不重复计数，完成清理及迟到取消后成功结果仍保持；取消后清理任务描述，最终计数和失败结果仍保持；任务全部完成但尚未 seal 时，不得报告 stopped。既有 5 个独立 lease 用例沿用并更新类型名，Read 的 lease 回归一并复跑。

- 容器 `renyuanming.rym_vscode`，CUDA 13、计算能力 8.9；`--jobs=96 --local_cpu_resources=96`，GPU 0、单目标串行、关闭测试结果缓存。
- 6 个 C++ 单测目标、`normal_engine` 和 `model_rpc_server` 编译通过；**247 个用例全部通过**，无失败或跳过：components 94、connector 23、scheduler 28、worker 77、decode lease 20、route codec 5。编译与回归合计 111.361 秒。
- 结果记录为上述目录中的 `test-shared-transfer-lease-events.json`。本轮未运行 Python、smoke 或 RDMA 硬件测试。

### 6.7 2026-09-11 复用与镜像结构自审

本轮对照 Read 的 connector、广播、worker 和 lease 实现，收敛以下重复逻辑。§6.6 保留各次验证时的实现记录，当前结构以本节及 §6.1/§6.3 为准。

| 发现 | 修改与行为 |
| --- | --- |
| 普通广播、Read CANCEL/QUERY 与 Write 控制广播重复创建相同的 RPC lambda；两种查询重复等待所有 RPC 完成 | 在现有 `P2PBroadcastClient` 内增加 `broadcastRpc()` 和 `broadcastRpcAndWait()`，共同处理 RPC 提交、等待及传输层成功检查。普通广播和 Read CANCEL 继续异步返回，Read QUERY 和 Write CANCEL/QUERY 等待本次控制响应 |
| Write 控制入口校验超时范围，Read 入口却直接把 `int64_t` 转成 `int`；0 还可能落入底层无限等待语义 | 公共提交入口统一拒绝非正数和超过 `INT_MAX` 的毫秒超时，不发送 RPC。控制请求的等待期限与 payload 中原有传输 deadline 仍然分开 |
| 总 connector 和两侧角色 connector 重复封装错误响应，两侧另有相同的 Write 状态封装 | 合并到 `P2PConnector::setP2PResponse()` / `fillWriteResponse()`。前者只更新错误字段，保留已写入的状态；后者先清空 P2P 响应，再完整填写 Write 状态与错误 |
| Read lease 查询和 Write group 都维护 `finish_counted`，重复做完成数量差值和逐次累加 | 移到共用 lease 的 `updateFinishedOps()`，调用方只统计当前已完成任务数；移除两处额外计数成员 |
| 两侧 Write 清扫无论是否发生超时取消，都连续扫描两次 task 状态 | 使用首次状态快照；仅在本轮执行超时取消后再次检查。两侧流程保持镜像，Prefill 另负责从 receiver 移除落点 |

已有的复用继续使用：内部路由使用 `P2PWorkerRoutePlan` / `RouteCodec`，地址转换使用 `LayerCacheBufferUtil::buildKeyBlockInfos()`，写回 rendezvous key 复用 `P2PKeyUtil::makeRouteLayerKey()` 的编码规则，实际传输复用 `TransferTask`、sender/receiver 接口及 TCP/RDMA backend。两个 Write worker 的 cancel、query 和 cleanup 保持对应顺序，接收侧额外移除 rendezvous 落点；这部分短流程直接镜像保留，不再增加基类或任务注册表类。

以下差异有调用时序或生命周期依据，暂不强行合并：

| 差异 | 原因与后续约束 |
| --- | --- |
| Read 接收入口等待完成，Write 接收入口注册后返回 | 正向 scheduler 已独立发起远端 StartLoad 和本地 READ，发送不依赖 READ 的完成响应。写回必须先由 Prefill 返回接受范围和端点，Decode 才能构造发送任务；`handleWrite()` 等待传输完成会阻塞这个握手 |
| Read sender 按计算完成的 layer/tag 发送，Write sender 按已完成 KV 的显式 routes 发送 | 两者复用 buffer 转换和 backend，但前者还维护逐层计算事件、发送结果与节流，不能把该调度流程直接套到写回。源 KV 就绪证明仍在后续 Slice 3/4 落实 |
| Write 使用有界 `autil::ThreadPool`，不直接套用 Read 的线程池及失败时同步执行路径 | TCP `send()` 在返回前包含 D2H 拷贝和同步，Write START 需要后台提交。队列满时必须拒绝，析构时必须排空已入队闭包，具体语义见 §6.3.2 |
| Write 区分停止确认与业务成功，保留独立状态解析 | `Result::success()` 合并 RPC 与业务错误，不能证明物理传输停止。START 返回错误或超时后必须 CANCEL/QUERY；只在所有相关 rank 有有效 stopped 结果后归还资源 |
| Write 保留未知 QUERY 不确定性、取消终态及未停止任务 | 不能套用 Read lease map 的固定硬 TTL 或把未知 key 当作安全释放依据。Write 超时只发起取消，在途 task 继续保留至 backend 完成；后续 scheduler 必须持有实际 KV 引用。设计 R1/R4 仍保留 |
| 两侧 connector 的 START 协议校验仍保留局部重复 | 按已确认的分层，protobuf 处理留在角色 connector。Write 在 `RouteCodec` 默认值归一化前严格校验原始字段，Decode 还解析发送端点；不改变原 Read 的输入兼容策略，也不恢复单独的 RPC util |

本轮新增 4 个回归用例：Read/Write 控制超时非法时不发 RPC、普通广播超时超过 `int` 范围时拒绝、重复/过时 lease 快照不重复计数、并发快照保持最大完成数。已有真实 gRPC/TCP、部分注册失败、排队取消、在途超时和 Read lease 用例一并复跑。

- 容器 `renyuanming.rym_vscode`，CUDA 13、计算能力 8.9；`--jobs=96 --local_cpu_resources=96`，GPU 0、单目标串行、关闭测试结果缓存。
- 6 个 C++ 单测目标、`normal_engine` 和 `model_rpc_server` 编译通过；**251 个用例全部通过**：components 96、connector 23、scheduler 28、worker 77、decode lease 22、route codec 5。编译与回归合计 108.341 秒。
- 结果记录为 `.cache/decode-writeback-slice1-review-20260911/test-mirror-reuse-review-events.json`。本轮未运行 Python、smoke 或 RDMA 硬件测试。

### 6.8 2026-09-11 空 routes 与路由注释修复

Decode Write 原先会将空 routes 包装成空闭包提交发送队列，队列已满时因此返回失败，与本 rank 无任务直接成功的契约不符。现在在同一把任务表锁内完成去重登记；无传输单元时直接 seal 并返回，保留可查询状态和原始 deadline。非空任务继续使用已有后台队列。

新增 `EmptyRoutesSucceedWithFullQueueAndKeepQueryableTerminalState`：用一个阻塞的发送请求占用唯一线程，再用另一个请求填满唯一队列槽位，验证空 routes START 成功、任务计数为 0、sealed/stopped/write_success 为真且队列占用不变；重复 START 仍被拒绝，清扫后 QUERY 仍返回成功终态，backend 不增加发送调用。

同步修正 `P2PWorkerRoute.h` 的原有注释：`layer_buffers` 在 Read Decode 接收侧以及 Write 两侧均显式提供，Read Prefill 发送侧使用逐层产出的本地投影；目的端点供 Read Prefill / Write Decode 发送侧使用；按 tag 统计 route 数量的节流用途明确为 Read Prefill。

在 `renyuanming.rym_vscode` 中使用 CUDA 13、计算能力 8.9 和 96 CPU 线程，worker 单测目标、`normal_engine` 和 `model_rpc_server` 编译通过。worker 目标 **78 个用例全部通过**，含上述新增用例及既有 Read/Write、真实 RPC/TCP 回归，编译与测试共 57.307 秒。结果记录为 `.cache/decode-writeback-slice1-review-20260911/test-empty-routes-events.json`；本轮未重跑其他五个 C++ 目标、Python、smoke 或 RDMA 硬件测试。
