# GLM53 Flash host KV cache：传输链路修复复核

本轮针对 host MLA 双层存储改动复核 P2P RDMA、RemoteConnector，以及相邻的块地址转换和内存复制路径。用户提供的两条评论分别为 P1、P2，没有独立的 P0 评论。

## 修复内容

### P1：P2P backend 缺少完整 HBM 块的注册

调用链为 `P2PConnectorWorker::init()` → `LayerBlockConverterImpl::getAllBuffers()` → sender/receiver `regMem()`。它与 `BlockPool::regUserMr()` 使用的 CacheStore 注册链独立，不能用后一条链的成功来证明前一条链已注册。

[LayerBlockConverterImpl.h](../rtp_llm/cpp/cache/connector/p2p/LayerBlockConverterImpl.h) 现在枚举：

- 原有 KV/scale tensors，包括 host overflow arena；内存类型取 tensor 的实际 device。
- typed attention regions，包括独立的 `INDEXER_KV`；相同地址和长度的 DEFAULT 别名只注册一次。
- `mla_host_cache_by_layer` 中 HBM 的完整块前缀。注册长度为 `hbm_tokens × 每 token 字节数`，不包含 attention 私有 resident slots 和层尾 padding。

因此 MLA 逻辑块 ID 位于 HBM 边界两侧时，地址转换返回的完整物理块均落在对应注册范围中。KDA 原有状态 tensor 继续按完整物理块注册，不改写其 SSM/conv 布局。

### P2：RemoteConnector 把 host 地址标为 GPU

[GroupPolicy.cc](../rtp_llm/cpp/cache/connector/remote_connector/GroupPolicy.cc) 的 `genBlockBuffers()` 现在按每个 `BlockInfo::is_cuda` 生成 CPU 或 GPU IOV，保留原地址、长度和 ignore 属性。同一次调用中允许 HBM 块与 host 块共存。

### 继续复核发现并修复的问题

1. **MR 拆分对齐使用了整个 arena 大小。** 当 arena 超过 backend 的单次注册上限时，原实现要求 `max_reg_mem_size >= arena_size`，无法按较小片段注册。现在从 allocator 的块转换接口取得物理 KV/scale 块大小作为对齐值；backend 可以拆分 arena，并保证一块不会跨两个 MR。不能使用 kernel page 大小替代物理块大小，因为一个 allocator block 可能包含多个 kernel pages。单次注册上限由 backend/配置决定，并非所有环境固定为 64 MiB。
2. **RDMA 接收端注册失败仍继续初始化。** [P2PConnectorWorker.cc](../rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.cc) 现在对 RDMA receiver 注册失败返回 false，避免带着未注册地址继续启动。TCP 保留原有处理。

## 验证与边界

[GroupPolicyTest.cc](../rtp_llm/cpp/cache/connector/remote_connector/test/GroupPolicyTest.cc) 新增两项回归：

- CPU/GPU 混合块生成正确类型、地址和长度的 RemoteConnector IOV。
- 使用真实 pinned CPU tensor 和 CUDA tensor，检查 host/HBM 完整块、KDA 和 typed indexer 的注册枚举、别名去重、物理块对齐，以及 HBM/host 边界两侧每块恰好被一个注册范围覆盖；resident slots 不纳入该注册范围。

第二项使用模拟 allocator 提供物理布局，验证真实 converter 的输出；没有调用 NIC 注册或执行网络传输。P2P worker 现有测试使用 mock backend，也不构成真实 RDMA 验证。

CUDA13/SM10x 两个验证目标全部通过：`remote_connector/test:group_policy_test` 25 项，`p2p/test:p2p_connector_worker_test` 27 项，共 52 项，Bazel 总耗时 79.853 秒。P2P worker 测试补齐了 CUDA13 PyTorch 的 `libtorch_nvshmem.so` 条件链接依赖。构建沿用设计文档中的本地 wheel repository override 和 RDMA 指标宏兼容参数，没有修改共享 internal source。日志：`/tmp/glm53_review_transfer_verified_test.log`。

另沿实际调用链检查了物理块与 kernel page 的区别、HBM 前缀长度、KDA 原有布局，以及 MemoryConnector 根据 `is_cuda` 选择 host copy/device copy 的路径。上述修复后，本轮改动范围内未发现额外可确认的 P0/P1；这不是整个仓库不存在 bug 的保证。

当前分支存在以下既有能力边界，不能把本次组件修复描述为完整 GLM53 P/D 支持：

- `KVCacheConnectorCoordinator::initP2PConnectorInternal()` 仍被 `#if 0` 禁用，注释要求先接入 scheduler async load；本轮未启用它。
- 枚举 typed region 并注册内存，不等于补齐 typed region 的传输协议。现有 P2P payload 路径和 RemoteConnector 分组路径的完整混合模型支持，仍需独立验证。
- whole-state linear request cache 与 RemoteConnector 的组合在 coordinator 中已有禁用逻辑。
- 本轮未运行真实 NIC RDMA、远端缓存服务或完整模型 P/D 请求，未新增端到端精度、吞吐或 TPOT 结论。
