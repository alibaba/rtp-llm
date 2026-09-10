# Decode KV Cache 写回实现文档

本文记录[写回设计方案](decode_kvcache_writeback_design.md)的实现进度与验证结果。设计文档描述目标行为和待讨论意见；本文记录当前代码状态、实现选择及验证过程。

## 1. 当前范围与实现计划

- 开发分支：`rym/feat/decode-writeback`。
- 基线分支：`codex/dsv4-block-tree-p2p-region`。
- 基线提交：`ce489746d081e7fb861fa706fc3cb2d933ac5214`。
- 当前仓库：`/data6/renyuanming.rym/rtp-llm`。
- 2026-09-10 按逐个 Slice 实施的安排撤回 Slice 1，当前工作区只保留 **Slice 0**。
- 当前没有新增写回协议、Write worker、写回配置或请求收尾触发逻辑。

### 1.1 四个 Slice 的职责

按照设计文档 §7.8，Phase 1 共分为 **4 个 Slice（0～3）**，每片应能独立编译、验证。这里的 Slice 指实现阶段，与传输协议中的数据切片不是同一个概念。

| 实现阶段 | 主要工作 | 当前状态 |
| --- | --- | --- |
| Slice 0：Read 路径重构 | scheduler/worker 改为 Read 命名；将首 token、复用统计和 MTP 结果拆入 `PrefillResultStore`，保留等待、取消和终态语义 | 已实现；期限交接修复后目标容器编译及 212 个单测通过，待代码审阅 |
| Slice 1：写回基础设施 | proto、Decode Write sender、Prefill Write receiver、backend 接线、角色分发及配置 | 已撤回，后续单独实施 |
| Slice 2：Prefill 接收与插树 | `StartWrite` 服务入口和 Prefill Write scheduler；纯探测、范围协商、接管分配引用的 RAII 句柄、接收汇总、HOST/DISK 后缀冲突拒绝、同锁预检和插树 | 尚未实现 |
| Slice 3：Decode 发起与链路接通 | Write caller/scheduler、源块保活、握手及发送、并发和超时管理、请求收尾触发及端到端验证 | 尚未实现 |

Slice 0 的目标是整理现有正向 Read 路径，为后续写回提供清晰的资源与结果存储边界。当前代码不提供 Decode 到 Prefill 的自动写回。

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

当前 proto 相比基线仅保留 Read 类名的注释更新，消息、枚举、字段和 RPC 定义不变。`CacheStoreConfig` 字段、Python 绑定、类型声明、CLI 参数及 pickle 格式恢复到基线；C++ 配置头文件只保留与 Read 重命名和结果存储拆分对应的注释更新。

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

保留原有 11 个 `PrefillResultStoreTest` 用例，并补充实际清理线程初始化、期限同源、交接时清扫、过期先发生、终态拒绝交接和业务期限归一化的回归测试。connector 测试同时覆盖正常传输和 `no_transfer` 的交接竞态。此前新增的 10 个 Write worker 用例、2 个 Write 角色分发用例和 2 个写回配置用例均已撤回。配置测试文件整体恢复到基线，包括之前在 Slice 1 验证过程中修正的 pickle 测试写法；这些配置测试不属于当前 Slice 0 的改动范围。

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

R1 的物理传输停止与资源释放、R2 的 KV 就绪边界、R3 的纯探测接口及同锁发布实现、R4 的跨 rank 完成汇总，以及 R6 的混合 group 范围，属于后续写回阶段需要确认或落实的契约。当前先完成 Slice 0 的审阅与验证。

### 5.1 Slice 2 已明确、尚未实现的两项契约

- **接收分配引用（设计 R7，本轮 comment 4）**：`mallocForExternalInsert` 返回 owning RAII 句柄，直接接管 malloc 已增加的原始引用，接收端不再调用 `incrKVCacheRef` 叠加引用。成功插树后归还分配引用，只保留树的 CACHE 引用；失败及重复结果未采纳的新块均按同一规则释放，释放前须确认没有在途传输。分配接口负责部分失败回滚。
- **HOST/DISK 后缀冲突（设计 R8，本轮 comment 5）**：p=p_dev 不代表后缀不存在。Phase 1 对已有低层或忙碌 FULL 节点冲突放弃写回，不提供节点升级接口。准入先检查一次；settle 在树锁内完整检查前缀和全部后缀后才发布，防止普通 insert 提前停止而留下部分更新。

上述两项本轮只修订设计文档和本实现计划，尚无写回接收代码。后续 Slice 2 必须补分配引用账本、失败回滚、HOST 后缀拒绝、传输中降级和部分发布防护测试。§4.5 的 212 个通过用例仅验证当前 Slice 0，不作为这两项未实现能力的验证结果。
