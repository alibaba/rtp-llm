# Mooncake P2P Transfer Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在开源分支上把 Mooncake P2P transfer 从“配置与 stub”推进到“有控制面 proto/service、receiver descriptor 状态机、sender 数据面骨架”的可继续集成状态。

**Architecture:** 保持 `IKVCacheSender` / `IKVCacheReceiver` 接口语义不变，在 `transfer/mooncake` 内部新增轻量控制面和 descriptor 索引。控制面复用现有 `TcpServer` / `TcpClient` / arpc 服务模式，数据面继续通过 adapter 抽象隔离真实 Mooncake TransferEngine 依赖。

**Tech Stack:** C++17, protobuf/arpc, Bazel, Mooncake adapter abstraction, existing RTP transfer/task primitives.

---

### Task 1: Mooncake 控制面 Proto 与 Service 骨架

**Files:**
- Create: `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/proto/mooncake_service.proto`
- Create: `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTransferService.h`
- Create: `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTransferService.cc`
- Modify: `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/BUILD`

- [ ] 定义 `prepare(unique_key, deadline_ms)` / `finish(unique_key, success, error_code, error_message)` 的 proto，请求不传 block payload，只返回 descriptor 元数据。
- [ ] 实现 `MooncakeTransferService` 基础骨架：挂到 `TcpServer`，提供 `prepare` 与 `finish` 两个 RPC，并把 receiver 状态访问抽象成接口，不在 service 内写死具体索引结构。
- [ ] 为 proto 和 service 增加 Bazel 目标，保证默认构建下可编译，不引入真实 Mooncake 三方库。

### Task 2: Receiver Descriptor 状态机

**Files:**
- Create: `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeRemoteDescriptor.h`
- Modify: `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeKVCacheReceiver.h`
- Modify: `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeKVCacheReceiver.cc`
- Test: `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeBackendStubTest.cc`

- [ ] 在 receiver 内新增 `unique_key -> RemoteDescriptor` 索引，descriptor 至少包含 `segment_name`、`target_addr_list`、`cache_key/block_index/len` 映射信息。
- [ ] 让 `recv()` 创建 task 时同步注册 descriptor；`stealTask()` / `finish()` / 失败路径清理 descriptor，避免悬挂。
- [ ] 提供给 `MooncakeTransferService` 调用的 `prepareDescriptor()` / `finishTransfer()` 入口，并把 `TransferTask::startTransfer()` / `notifyDone()` 语义对齐进去。
- [ ] 为 descriptor 生命周期补单测：recv 后可 prepare，steal 后 prepare 失败，finish 后清理索引。

### Task 3: Sender 数据面骨架与 Adapter 扩展

**Files:**
- Modify: `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTransferEngineAdapter.h`
- Modify: `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeKVCacheSender.h`
- Modify: `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeKVCacheSender.cc`
- Test: `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeBackendStubTest.cc`

- [ ] 在 adapter 抽象里补 `openSegment`、`allocateBatchID`、`submitTransfer`、`getTransferStatus` 所需的数据结构，支持按 `(cache_key, block_index, len)` 构造 WRITE 请求。
- [ ] 在 sender 内实现控制面调用骨架：先 `prepare` 拉 descriptor，再做 block 展平和映射校验，再调 adapter；暂时允许最终 `send()` 返回 stub 错误，但内部路径要把 descriptor 对齐和错误传播逻辑写完整。
- [ ] 为 sender 增加轻量测试：控制面返回 descriptor 时能正确校验 block 数量/长度，不匹配时返回 `BUFFER_MISMATCH` 或 `UNKNOWN`。

### Task 4: 集成与验证

**Files:**
- Modify: `rtp_llm/cpp/cache/connector/p2p/transfer/BUILD`
- Modify: `rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.cc`
- Test: `rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeBackendStubTest.cc`

- [ ] 把 Mooncake proto/service/backend stub 纳入 transfer BUILD 图，但默认 factory 仍只在 `kMooncake` 分支返回 unsupported 或未完全实现错误，不影响 TCP-only 路径。
- [ ] 跑现有 TCP 基线测试和 Mooncake stub 测试，确认新代码没有破坏已通过的 Bazel 目标。
- [ ] 收口剩余缺口，明确下一阶段要接的真实 `/data0/qiongshi.gb/mooncake/Mooncake` TE 依赖点。
