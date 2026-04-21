# Mooncake P2P KVCache Transfer 后端接入设计（prepare/finish 控制面）

最后更新：2026-04-21  
适用分支：`feature/p2p_connector-3`  
范围：`rtp_llm/cpp/cache/connector/p2p/transfer/*` 及其上层 `P2PConnectorWorker` 的后端选择逻辑

## 1. 目标与非目标

### 1.1 目标（Goals）

1. 在不改变上层接口语义的前提下，引入 Mooncake 作为 P2P transfer 的一种后端实现。
   - **固定边界**：不修改 [`rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h) 与 [`rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h) 的接口语义，复杂度收敛在 transfer 后端内部。
2. 新增轻量控制面（Control Plane），复用现有 `TcpTransferService` 的服务模式与部署方式，但 RPC 请求变为仅传元数据的两类调用：
   - `prepare(unique_key)`
   - `finish(unique_key, status)`
   控制面 **不再传 block 内容**。
3. 扩展 transfer backend 选择能力：
   - 扩展 [`rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendConfig.h`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendConfig.h) 与 [`rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.cc`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.cc)，新增 `kMooncake`。
   - **保留** legacy 的 `cache_store_rdma_mode`，默认行为保持不变；**不引入** `kAuto`。
   - `P2PConnectorConfig / P2PConnectorWorker` 读取“显式 backend 选择”，并提供兼容逻辑：
     - `cache_store_mooncake_mode -> kMooncake`
     - `cache_store_rdma_mode -> kBarexRdma`
     - 否则 `kTcp`
     - 后端选择实现尽量用 **switch 风格**表达。
4. 线程与参数治理：
   - 继续复用现有 messager/arpc 线程配置（`messager_io_thread_count` / `messager_worker_thread_count` / `cache_store_tcp_anet_rpc_thread_num` / `cache_store_tcp_anet_rpc_queue_num`），**不新增**一套 Mooncake 控制面线程参数。
5. 先做 Mooncake adapter 抽象，便于条件编译与 mock：
   - `init`
   - `registerLocalMemory`
   - `openSegment`
   - `allocateBatchID`
   - `submitTransfer`
   - `getTransferStatus`
6. 接收端（receiver）：
   - 维护 `unique_key -> remote descriptor` 索引。
   - `recv()` 注册 task，`regMem()` 做内存注册。
   - `prepare()` 返回 `segment name` 与目标地址列表（target addr list）。
7. 发送端（sender）：
   - `send()` 内部流程：先 `prepare`，再 `openSegment`，把 source blocks 与 target addrs 一一配对，批量 `WRITE`，结束后 `finish`。
8. 语义对齐：
   - timeout/cancel/done 语义与现有 `TransferTask` 行为一致。
9. 完整交付：
   - 补 BUILD、配置透传与测试计划，保证 **无 Mooncake 依赖** 的构建不受影响。

### 1.2 非目标（Non-goals）

1. 不调整 `P2PConnectorWorkerPrefill/Decode` 的对外行为与上层 RPC 语义（deadline/return_before/steal_before 的策略保持一致）。
2. 不做 Mooncake 性能优化细节（如 pipeline depth、batch 合并策略、分片策略等），这些属于后续迭代。
3. 不在开源默认构建中强制引入 Mooncake 三方依赖或工具链。

## 2. 现状（基于当前开源分支代码）

### 2.1 Transfer 后端与选择方式

1. 后端枚举与工厂：
   - 枚举定义在 [`rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.h`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.h)：当前仅 `kTcp`、`kBarexRdma`。
   - 工厂实现 [`rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.cc`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.cc)：开源构建中 `kBarexRdma` 会直接抛异常（`"not supported in this build"`），仅 TCP 可用。
2. `P2PConnectorWorker` 选择后端方式：
   - [`rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.cc`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.cc) 当前仅用 `cache_store_rdma_mode` 的 bool 进行二选一（`?:`），没有“显式 backend”概念。
3. 后端配置结构：
   - [`rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendConfig.h`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendConfig.h) 当前包含 `cache_store_rdma_mode`、messager 线程数、listen_port、TCP anet 线程等。

### 2.2 TCP 模式：控制面与数据面耦合

1. TCP RPC 协议在 [`rtp_llm/cpp/cache/connector/p2p/transfer/tcp/proto/tcp_service.proto`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/tcp/proto/tcp_service.proto)。
2. 该协议把 block content 直接嵌在 `TcpLayerBlockTransferRequest` 中（inline bytes）。
3. Decode 侧 RPC handler 为 [`rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpTransferService.h`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpTransferService.h) / [`rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpTransferService.cc`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpTransferService.cc)：
   - 通过 wait-check 轮询把 incoming RPC 与预先注册的 recv task 匹配，然后在 worker thread pool 中做 copy。

### 2.3 任务语义：TransferTask

`TransferTask` 定义在 [`rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h) / [`rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.cc`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.cc)：

1. `deadline_ms` 超时优先级最高：`notifyDone()` 时若 deadline 已过，统一对调用方视为 TIMEOUT。
2. `cancel()` 的语义：
   - PENDING 阶段：立即 done 且返回 CANCELLED。
   - TRANSFERRING 阶段：仅记录取消意图，等待 `notifyDone()` 决定最终返回 CANCELLED。
3. `startTransfer()`：原子地从 PENDING 迁移到 TRANSFERRING；若此前已 done（例如 cancel fast-fail），返回 false。

### 2.4 P2P 上层 deadline/steal/cancel 策略

1. Decode 侧注册 recv task 的 deadline 对齐策略在 [`rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerDecode.cc`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerDecode.cc)：
   - `recv_req.deadline_ms = D - p2p_read_return_before_deadline_ms`
   - 在 `D - p2p_read_steal_before_deadline_ms` 时刻执行 `receiver_->stealTask(key)`，阻止后续 sender 再匹配到新的传输。
2. Cancel 行为：
   - `cancelRead()` 会对任务组中每个 `IKVCacheRecvTask` 调用 `cancel()`（同文件）。
   - Prefill `cancelSend()` 为 best-effort，主要用于终止上层 dispatch/等待（[`rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerPrefill.cc`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerPrefill.cc)）。

## 3. 总体方案概览

### 3.1 架构分层

1. 上层保持不变：
   - `P2PConnectorWorkerPrefill` 仍调用 `IKVCacheSender::send()`
   - `P2PConnectorWorkerDecode` 仍调用 `IKVCacheReceiver::recv()/stealTask()/getTask()`
2. Transfer 后端新增 Mooncake 实现：
   - `MooncakeKVCacheSender` 实现 `IKVCacheSender`
   - `MooncakeKVCacheReceiver` 实现 `IKVCacheReceiver`
3. 控制面 RPC：
   - 新增 `MooncakeTransferService`（服务模式对齐 `TcpTransferService`，但只承载 `prepare/finish` 元数据）。
4. 数据面（Data Plane）：
   - sender 端通过 Mooncake adapter 发起批量 WRITE 到 receiver 的目标地址列表。

### 3.2 时序（推荐）

#### Decode 侧（receiver）

1. `regMem(block_info)`：把本机接收 buffer 注册到 Mooncake（registerLocalMemory）。
2. `recv(unique_key, block_info, deadline)`：
   - 创建并注册 `TransferTask`。
   - 构建 `unique_key -> RemoteDescriptor` 并缓存（包含 `segment_name`、`target_addr_list` 以及必要的校验信息）。
3. `prepare(unique_key)`（RPC 入站）：
   - 校验任务仍存在且未被 steal/cancel。
   - 调用 `task->startTransfer()` 进入 TRANSFERRING。
   - 返回 `segment_name + target_addr_list`。
4. `finish(unique_key, status)`（RPC 入站）：
   - `task->notifyDone(...)`，并清理 descriptor。

#### Prefill 侧（sender）

1. `regMem(block_info)`：把本机源 buffer 注册到 Mooncake（registerLocalMemory）。
2. `send(request)`：
   - `prepare(unique_key)` 获取 `segment_name + target_addr_list`
   - `openSegment(segment_name)`
   - `allocateBatchID()`
   - 将 `request.block_info` 展平成稳定顺序的 source blocks，并与 `target_addr_list` 一一配对
   - `submitTransfer(WRITE, pairs, batch_id)`
   - 轮询 `getTransferStatus(batch_id)`，直到成功/失败/超时/取消
   - `finish(unique_key, status)` best-effort 通知对端完成
   - callback 返回 `TransferErrorCode`

#### 关键约束

1. `prepare/finish` 为控制面，仅传元数据，**不承载 block 内容**。
2. `deadline_ms` 与现有 `TransferTask`、P2P 上层 return_deadline 策略对齐：超时优先级最高，晚到 finish 视为 TIMEOUT。

## 4. 控制面设计（prepare/finish）

### 4.1 RPC 接口草案（proto）

建议新增 `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/proto/mooncake_service.proto`（命名可调整），服务形态对齐 `tcp_service.proto`：

1. `PrepareRequest { unique_key, deadline_ms }`
2. `PrepareResponse { error_code, error_message, segment_name, repeated target_addr }`
3. `FinishRequest { unique_key, status, error_message? }`
4. `FinishResponse { error_code, error_message }`

说明：

1. `deadline_ms` 为绝对时间戳（ms），与当前 TCP 协议一致，便于复用现有 controller timeout 逻辑。
2. `target_addr` 的类型需要与 Mooncake 的 remote descriptor 对齐（可能是 `uint64` 地址、或结构体：`{addr, len, key, block_index}`）。为避免 sender/receiver 的“顺序对不齐”风险，建议在 `target_addr` 中带上足够的标识信息（详见 7.1 风险）。

### 4.2 与现有线程/服务模式的复用

1. 复用现有 `TcpServer` / arpc 线程配置字段：
   - `messager_io_thread_count`
   - `messager_worker_thread_count`
   - `cache_store_tcp_anet_rpc_thread_num`
   - `cache_store_tcp_anet_rpc_queue_num`
2. 不新增 `mooncake_*_thread_num` 一类参数。
3. 复用现有 “listen_port” 体系：建议 Mooncake 控制面与 TCP 控制面共享 `cache_store_listen_port`（通过 service_id 区分服务），或沿用现有 cache_store server 的 service 注册方式。

## 5. Mooncake Adapter 抽象（便于条件编译与 mock）

### 5.1 适配层接口（建议）

新增一个纯虚接口（示例命名）：`IMooncakeAdapter`，包含：

1. `bool init(const TransferBackendConfig& cfg, MetricsReporterPtr metrics)`
2. `bool registerLocalMemory(const BlockInfo& block, uint64_t aligned_size)`
3. `StatusOr<SegmentHandle> openSegment(const std::string& segment_name)`
4. `BatchID allocateBatchID()`
5. `bool submitTransfer(const SegmentHandle& seg, BatchID batch_id, const std::vector<WritePair>& pairs)`
6. `TransferStatus getTransferStatus(BatchID batch_id)`

说明：

1. `WritePair` 至少包含 `{src_ptr, dst_addr, len}`，并可以扩展 device 信息。
2. `TransferStatus` 要能表达 `DONE/FAILED/CANCELLED/IN_PROGRESS` 与错误码映射。

### 5.2 条件编译与依赖隔离

1. 仅在启用构建开关（例如 `RTP_LLM_WITH_MOONCAKE`）时编译真实 Mooncake 适配实现与依赖。
2. 默认构建提供 stub 实现（不引入 Mooncake 头文件）：
   - `createTransferBackend(kMooncake, ...)` 返回空或抛出 runtime_error，并在日志明确“not supported in this build”，对齐当前 `kBarexRdma` 的行为。
3. 单元测试使用 mock adapter，验证时序与错误传播，不依赖 Mooncake。

## 6. Receiver 设计细节（MooncakeKVCacheReceiver）

### 6.1 核心数据结构

1. `TransferTaskStore`：沿用现有（见 [`TransferTask.h`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h)）。
2. `RemoteDescriptorIndex`：新增 `unique_key -> RemoteDescriptor` 映射。
   - `RemoteDescriptor` 至少包含：
     - `segment_name`
     - `target_addr_list`
     - 与 `RecvRequest.block_info` 的一致性校验信息（例如总 block 数、每块 size 的摘要）
     - 可选：`create_time_ms` 用于超时清理

### 6.2 recv()/stealTask()/getTask() 语义保持

1. `recv(request)`：
   - `task_store_->addTask(unique_key, block_info, deadline_ms)` 返回 `TransferTask`（失败返回空，行为与 TCP 一致）。
   - 构建并写入 `RemoteDescriptorIndex[unique_key]`。
2. `stealTask(unique_key)`：
   - 必须从 `task_store_` 移除 task。
   - 建议同时移除 `RemoteDescriptorIndex`，确保 sender 之后 `prepare` 失败且不会泄漏。
3. `getTask(unique_key)`：
   - 只查询，不转移所有权，与现有一致。

### 6.3 prepare(unique_key)

1. 查 `task_store_->getTask(unique_key)` 与 `RemoteDescriptorIndex`：
   - 不存在：返回 `CONTEXT_TIMEOUT` 或 `TASK_CANCELLED`（需定义清晰映射规则，建议与 TCP 类似）。
2. 调用 `task->startTransfer()`：
   - 返回 false：说明 task 已 done 或在 PENDING 阶段被 cancel fast-fail，应返回 `TASK_CANCELLED`。
3. 返回 `segment_name + target_addr_list`。

### 6.4 finish(unique_key, status)

1. 查找 task（可用 `getTask` 或者“descriptor 内持有 task ptr”）。
2. `task->notifyDone(...)`：
   - 成功：`OK`
   - 失败：映射到 `UNKNOWN/BUFFER_MISMATCH/RPC_FAILED/...`（具体见 8.2）
   - 取消：映射到 `CANCELLED`
3. 清理 `RemoteDescriptorIndex[unique_key]`（必须）。

## 7. Sender 设计细节（MooncakeKVCacheSender）

### 7.1 send(request) 的内部状态机

1. 输入：`SendRequest {ip, port, unique_key, block_info, deadline_ms}`（保持不变）。
2. prepare：
   - 向对端控制面发起 `prepare(unique_key, deadline_ms)`。
3. 数据面：
   - `openSegment(segment_name)`
   - 将 `block_info` 展平成稳定顺序（建议：cache_key 升序 + blocks 顺序；或明确定义“协议顺序”，见 7.2）。
   - 与 `target_addr_list` 做一一配对，形成 write pairs。
   - `submitTransfer(WRITE, ...)`
   - `getTransferStatus(batch_id)` 轮询至完成或超时。
4. finish：
   - best-effort 调用 `finish(unique_key, status)`，用于驱动 receiver `notifyDone`。
5. callback：
   - 将结果映射为 `TransferErrorCode`，与上层 `P2PConnectorWorkerPrefill` 的 error handling 兼容。

### 7.2 目标地址列表的对齐要求（必须在协议中写死）

为避免 sender/receiver 对 “block 展平顺序” 理解不同导致写错地址，建议协议层满足以下之一：

1. **强约束顺序**：`target_addr_list` 按照 `RecvRequest.block_info` 的确定性顺序生成，并把排序规则写入文档与代码注释（例如：cache_key 升序、每个 key 的 blocks 按原 vector 顺序）。
2. **显式标识**：`target_addr` 条目携带 `(cache_key, block_index, len)`，sender 端用键值匹配而非依赖顺序。

建议选项 2，可显著降低故障概率（代价是轻微元数据体积上升）。

## 8. 语义对齐：timeout/cancel/done

### 8.1 deadline 对齐策略

1. 现有策略（必须保持）：
   - Decode recv task deadline 使用 `D - return_before_ms`（见 [`P2PConnectorWorkerDecode.cc`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerDecode.cc)）。
   - Prefill send 使用 `return_deadline_ms = D - return_before_ms` 作为 transfer 层 deadline（见 [`P2PConnectorWorkerPrefill.cc`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerPrefill.cc)）。
2. Mooncake backend 规则：
   - `prepare/finish` 的 controller 超时应基于 `deadline_ms - now`（最小值为 1ms）；
   - 数据面轮询必须在 `deadline_ms` 前结束（或明确在 finish 中上报超时），避免 receiver 侧出现“物理完成但语义超时”的不可解释现象。
3. receiver `TransferTask::notifyDone()` 内部仍以 `deadline_ms` 为准：finish 晚到直接 TIMEOUT。

### 8.2 cancel 对齐策略

1. Decode `cancelRead()` 会对 task 调用 `cancel()`：
   - 若尚未 prepare（PENDING）：立刻 done=CANCELLED，prepare 应失败（`startTransfer() == false`）。
   - 若已 prepare（TRANSFERRING）：仅记录 `cancel_requested_`，finish 到达时会返回 CANCELLED（与现有 `TransferTask` 规则一致）。
2. Prefill `cancelSend()` 为 best-effort：
   - Mooncake sender 若观察到上层 cancel 标志，应尽量：
     - 停止后续 submit/poll；
     - 调用 `finish(unique_key, CANCELLED)` best-effort，帮助对端更快收敛状态。

## 9. 配置与兼容策略

### 9.1 新增“显式 backend 选择”

建议新增一个显式字段（示例命名）：

1. `cache_store_backend`：枚举或字符串（`tcp` / `barex_rdma` / `mooncake`）。
2. 保留现有 legacy：
   - `cache_store_mooncake_mode`（bool，兼容字段）
   - `cache_store_rdma_mode`（bool，已有）

### 9.2 兼容逻辑（要求 switch 风格）

后端选择优先级：

1. 若显式 `cache_store_backend` 有值：按其解析并 `switch` 到 `TransferBackend::{kTcp,kBarexRdma,kMooncake}`。
2. 否则走兼容模式：
   - `cache_store_mooncake_mode == true` 选 `kMooncake`
   - `cache_store_rdma_mode == true` 选 `kBarexRdma`
   - 默认 `kTcp`

注意事项：

1. 不引入 `kAuto`，避免行为漂移。
2. 默认行为保持不变：在“无显式 backend”且 `cache_store_rdma_mode=false` 的情况下仍为 TCP。

## 10. 类/文件改动建议（仅建议，不在本文档中改代码）

### 10.1 Transfer 层

1. 扩展 enum：
   - [`rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.h`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.h)
     - `TransferBackend::kMooncake`
2. 扩展工厂：
   - [`rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.cc`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.cc)
     - `switch(backend)` 新增 `case kMooncake` 分支
     - 默认构建未启用 Mooncake 时，行为对齐 `kBarexRdma`：runtime_error 或返回空并打 error log
3. 新增目录（建议）：
   - `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/`
     - `MooncakeKVCacheSender.{h,cc}`
     - `MooncakeKVCacheReceiver.{h,cc}`
     - `MooncakeTransferService.{h,cc}` + `proto/`
     - `IMooncakeAdapter.h` + `RealMooncakeAdapter.cc` + `MockMooncakeAdapter`（test）
4. BUILD：
   - 修改 [`rtp_llm/cpp/cache/connector/p2p/transfer/BUILD`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/BUILD)
     - 增加 mooncake backend target
     - 用 `select()` 或 arch_select alias 把 Mooncake 依赖隔离在可选构建路径内

### 10.2 P2P 上层

1. `P2PConnectorWorker` 的后端选择逻辑：
   - [`rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.cc`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.cc)
     - 从“基于 bool 的二选一”升级为“显式 backend + 兼容映射”的 `switch` 风格实现
2. 配置透传：
   - `CacheStoreConfig`/启动参数透传需要新增 `cache_store_backend` 或 `cache_store_mooncake_mode`（见 [`rtp_llm/cpp/config/ConfigModules.h`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/config/ConfigModules.h)）。

## 11. 测试计划（建议）

### 11.1 单元测试（不依赖 Mooncake）

1. 后端选择测试：
   - 覆盖显式 backend 与兼容字段的优先级与默认行为。
2. Mooncake sender/receiver 行为测试（mock adapter）：
   - prepare/finish 正常路径：sender send() -> receiver task done=OK。
   - cancel 路径：
     - PENDING cancel 后 prepare 必须失败。
     - TRANSFERRING cancel 后 finish 必须导致 task=CANCELLED。
   - timeout 路径：
     - finish 晚到时 receiver task 必须返回 TIMEOUT（对齐 `TransferTask::notifyDone()` 规则）。
3. unique_key 冲突与并发：
   - 与 [`TransferTaskStore::addTask`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.cc) 的“不允许重复 key”语义对齐。

### 11.2 集成测试（可选，按构建开关）

1. `ENABLE_MOONCAKE` 打开时：
   - 加一套类似 [`rtp_llm/cpp/cache/connector/p2p/transfer/tcp/test/TcpTransferServiceTest.cc`](/data0/qiongshi.gb/RTP-LLM/github-opensource/rtp_llm/cpp/cache/connector/p2p/transfer/tcp/test/TcpTransferServiceTest.cc) 的 service-level 测试，验证 prepare/finish 的匹配与超时行为。
2. `ENABLE_MOONCAKE` 关闭时：
   - 保证 `bazel test //...` 不因 Mooncake 头文件/库缺失而失败。

### 11.3 构建验证矩阵（必须）

1. 默认开源构建（无 Mooncake）：编译与测试全绿，且 `TransferBackend::kMooncake` 分支不可达或明确报错（行为可预期）。
2. 启用 Mooncake 构建：可编译、可链接、基本单测通过。

## 12. 风险与未决项

### 12.1 实现风险（高优先级）

1. **target_addr_list 与 block_info 展平顺序不一致** 导致写错地址，破坏显存数据完整性。
2. **unique_key -> descriptor/task 的生命周期管理**：
   - sender 异常退出或未调用 finish 时，receiver 侧 descriptor 泄漏与 task 悬挂，需要有超时清理策略。
3. **语义对齐细节**：
   - prepare 触发 startTransfer 的时机、stealTask 与 prepare 的竞态、finish 晚到的 TIMEOUT 优先级，容易出现边界 bug（需用单测锁死行为）。

### 12.2 未决项（需要与 Mooncake 能力对齐后定稿）

1. `target_addr` 的真实类型与可序列化表示（`uint64` 还是结构体描述符）。
2. `openSegment/segment_name` 的命名规则与复用策略（是否一 key 一 segment，还是复用大 segment + offset）。
3. GPU/CPU 内存注册粒度、对齐要求（`aligned_size` 的定义与最佳实践）。
4. `getTransferStatus` 轮询 vs 回调：对 CPU 开销与 latency 的影响取舍。

