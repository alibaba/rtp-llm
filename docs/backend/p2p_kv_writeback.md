# Decode→Prefill P2P KV Cache Writeback

在 PD 分离（Decode 入口）部署中，Prefill 仅计算首轮 prompt 的 KV Cache，Decode 后续生成的新增 token 对应的 KV 仅存在于 Decode 侧。多轮对话或共享后缀请求再次路由到 Prefill 时，这些生成段的 KV 需要重新计算。

**P2P KV Cache Writeback** 在 Decode 请求正常结束（`StreamState::FINISHED`）准备释放 KV 前，将超过 prompt 边界、凑满整块（或在 Prefill 启用 Sharded CP 时凑满整 CP 单元）的新增 KV 块异步写回原 Prefill 实例并发布进 `BlockTreeCache`，供后续请求直接复用。

---

## 开启条件与配置参数

Prefill 与 Decode 双方均需开启 `--p2p_writeback_enable 1`（且 Prefill 启用 `--reuse_cache 1`）。支持以下服务端启动参数（对应同名环境变量）：

| CLI 参数 | 环境变量 | 默认值 | 适用角色 | 说明 |
|---|---|---|---|---|
| `--p2p_writeback_enable` | `P2P_WRITEBACK_ENABLE` | `false` | Prefill / Decode | 是否启用 Decode→Prefill KV 写回。 |
| `--p2p_writeback_require_device_prefix` | `P2P_WRITEBACK_REQUIRE_DEVICE_PREFIX` | `true` | Prefill | 为 `true` 时要求 prompt 前缀已在 Prefill DEVICE 显存中，否则拒绝写回；为 `false` 时允许从本地 HOST/DISK 补载缺失前缀后再合并写回。 |
| `--load_cache_timeout_ms` | `LOAD_CACHE_TIMEOUT_MS` | `5000` | Prefill / Decode | 复用正向 P2P 通信超时配置，作为单次写回请求的端到端超时预算（毫秒），独立于生成请求本身的超时。 |
| `--p2p_writeback_max_kv_ratio` | `P2P_WRITEBACK_MAX_KV_RATIO` | `0.3` | Prefill | Prefill 准入水位：仅当写回持有的块数低于各独立显存池「空闲块 + 可回收缓存块」总量的该比例时才接受写回，避免挤占正常推理显存。Decode 侧忽略此参数。 |

### 请求级覆盖

请求级 `GenerateConfigPB.enable_p2p_writeback`（`google.protobuf.BoolValue`）支持按请求开关写回：
- 未设置时沿用服务端 `--p2p_writeback_enable`；
- 显式设为 `false` 时跳过该请求的写回；
- 显式设为 `true` 时仅在服务端已启用写回的前提下生效。

---

## 部署端口与网络要求

与正向 P2P Read 方向对称，Writeback 由 **Decode 各 worker 向 Prefill 各 worker** 发起数据推送（TCP 或 Barex RDMA）：

- **数据面端口**：Prefill 开启 `--p2p_writeback_enable 1` 后，每个 Prefill worker 的接收后端同样监听在 `cache_store_listen_port + 1`（默认端口布局为 `START_PORT + rank_id × worker_info_port_num + 3`）。升级部署时需确保网络策略允许 Decode 访问 Prefill worker 的该入站端口。
- **控制面端口**：Decode `writebackWorkerIndex == 0`（PP0/TP0）通过 gRPC 调用 Prefill 的 `StartWriteBack` / `CancelWriteBack`；Prefill 内部各非零 TP/PP rank 在完成接收 buffer 注册后通过 `HANDLE_WRITE_READY` 回报给 Prefill PP0/TP0。

---

## 端到端两阶段协议

```text
Decode Executors (all TP/PP ranks)
  ├─ PP0/TP0: setKVCacheWriteProgress(step, cuda_event)
  └─ non-owner ranks: publishWritebackStep(step, cuda_event)

Decode Stream / KVCacheManager (PP0/TP0)
  1. tryReleaseKVBlock(): 同步本地 kv_cache_write_event_，确认 GPU 写完后再提交写回或释放块
  2. P2PSchedulerDecodeWrite::asyncWrite(): 筛选后缀候选块并引用住源 KV（holdSourceResource）
  3. gRPC StartWriteBack(token_ids, cache_keys, decode_workers, decode_pipeline_layout, plan_digest, writeback_kv_step)
        │
        ▼
Prefill P2PSchedulerPrefillWrite (PP0/TP0)
  4. admitWriteBackDecodeCache()（malloc_mutex_ 内原子检查前缀、容量水位、分配目标块及可选 HOST/DISK 补载）
  5. Phase 1（先注册接收）:
       广播 HANDLE_WRITE -> 所有 Prefill worker 注册接收 buffer -> 上报 HANDLE_WRITE_READY
  6. Phase 2（再通知发送）:
       广播 WRITE(routes, writeback_kv_step) -> 所有 Decode worker
         └─ 非 owner 源 rank 经 WritebackStepTracker::wait(writeback_kv_step) 等待本地 GPU 完成目标 step
         └─ 执行 RDMA / TCP 传输到 Prefill 目标块
  7. beginCommit() 仲裁通过后 commitWriteBackDecodeCache() 入树发布
  8. drainWrite() 确认目标接收侧物理停止，释放目标 KV 引用，返回 StartWriteBackResponse(stopped=true)
        │
        ▼
Decode P2PConnectorAsyncWriteContext
  9. 收到 stopped=true 后释放源 KV；若失败/超时，走双边取消与 lease 排空后再释放源 KV
```

### 关键设计不变量

1. **跨 TP/PP Rank 的 GPU 完成栅栏**
   - RDMA NIC DMA 与 TCP `NoBlockCopy` 均不在模型计算 CUDA stream 上执行。
   - Decode 各 TP/PP rank 在每个非跳过解码步推进单调 `writeback_kv_step` 并记录对应 CUDA event：
     - allocator owner（PP0/TP0）在 `tryReleaseKVBlock()`（含同步释放和异步 bookkeeping）前先等待本地 `kv_cache_write_event_`；
     - 其他非 owner 源 rank 在 `P2PWorkerDecodeWrite::write()` 入发送队列前，经 `WritebackStepTracker::wait(writeback_kv_step)` 等待本地 GPU 达到该 step，保证不会发送未落盘的 KV（含 MTP / DSpARK / DFlash 等 block-draft 投机解码场景），且不会被后续新 batch 阻塞。
2. **Prefill 准入原子性**
   - `admitWriteBackDecodeCache()` 在 `malloc_mutex_` 锁内完成前缀匹配、容量水位校验与目标块分配，不与正常请求的 `initMalloc` / `incrMalloc` 交错。
   - 若目标后缀已完全命中 Prefill DEVICE 缓存，则跳过 worker 传输直接成功返回（记入 `no_transfer`）。
   - 若 `--p2p_writeback_require_device_prefix=false` 且部分前缀在 HOST/DISK，本地补载与网络接收并发执行，二者均完成后才统一提交。
3. **非对称拓扑与 Sharded CP 对齐**
   - 支持 **Prefill / Decode 非对称 TP**（KV head scatter/merge）、**非对称 PP**（基于 `decode_pipeline_layout` 按全局层精确映射）以及 **Prefill Sharded CP**（`cp_size > 1`；其中 MHA 分片 CP 仅支持 `ALL_GATHER` 非融合/非 overlap 路径及非量化 BF16/FP16 KV，不含 `ALLTOALL` 和 `ALL_GATHER_WITH_OVERLAP`，MLA 走 DSv4/DSv4.1 独立路径）。
   - 当 Prefill 启用 Sharded CP 时，写回起止范围按 `cp_unit = block_size * cp_size` 向下对齐并按 canonical key 发布；对于 **SWA**（Sliding Window Attention）和 **LINEAR**（如 GDN/SSM 状态层）分组，Decode 侧会按对齐回退量保留额外尾部块，避免端点状态提前回收。
   - 若 Decode 生成后凑不满新的完整块或完整 CP 单元，或 KV head 分片几何不兼容，直接跳过写回而不影响主推理链路。

---

## 失败排空、滞留保护与停机顺序

写回严格遵守「**物理传输未确认停止，绝不提前释放 KV 引用**」原则，避免 RDMA/TCP 传输中发生 UAF：

- **双边停止确认**：超时、取消或部分 rank 失败时，Decode 后台 `P2PConnectorAsyncWriteContextChecker` 会先向 Prefill 发送 `CancelWriteBack` 并轮询至 `stopped=true`，同时向本侧 Decode worker 广播 `CANCEL_WRITE` 并通过 `QUERY_LEASE_STATUS` 确认全部 rank 满足 `sealed && started_ops == finished_ops` 后，才释放源 KV 引用。
- **Prefill 目标块排空**：Prefill 在发生取消、失败或部分注册失败时，同样通过 `CANCEL_HANDLE_WRITE` + `QUERY_LEASE_STATUS` 确认全部接收 rank 物理停止后才释放目标块。
- **滞留 KV 观测（与正向 P2P Read 对齐）**：若因对端失联或传输卡死导致 KV 长时间无法释放，滞留块按 `(cache group, block ID)` 去重并按物理块字节数（含 KV scale，跨 PP stage 汇总）计入滞留占比，上报 `rtp_llm_p2p_writeback_retained_*` 指标；存在滞留时每 30 秒打印 `writeback retained KV` 告警日志，全部释放后打印恢复日志（不自动 `abort` 线上进程）。
- **请求表容量上限**：Prefill 写回请求状态表最多保留 1,048,576 条活跃及终态记录（终态记录保留 1 小时防重放）；超限时返回 `OUTPUT_QUEUE_FULL`（不谎报 `stopped=true`）。
- **停机顺序**：服务退出时先调用 `stopP2PWriteback()` 停止新写回并等待在途 `handleWrite()` 与异步上下文排空，再释放 GIL 关闭 gRPC server，最后销毁推理引擎与缓存管理器。

---

## 首次复用计费（Billing）

- Prefill 将写回块提交进 `BlockTreeCache` 时会把对应节点标记为未计费（`is_accounted = false`）；后续 Prefill 请求 `match()` 命中这些块时记录未计费索引，并在加载完成后累加计算 `writeback_first_use_len`。
- 节点的 `is_accounted` 仅在**首个正常 Prefill 请求完成并执行 `insert()`（非写回插入）时**翻转为 `true`。因此串行多轮场景下同一块只计一次首次复用；在该次正常 `insert()` 完成前的并发窗口内，多个同时 `match()` 命中同一写回块的请求都会看到 `is_accounted == false` 并各自计入 `writeback_first_use_len`。
- 该长度通过 `SideChannelPayloadPB.writeback_first_use_len` 回传给 Decode，最终暴露在响应 `AuxInfoPB.writeback_first_use_len` 以及监控指标 `rtp_llm_writeback_reuse_len` 中（同时从普通 `total_reuse_len` 中扣除）。
- 若命中请求在加载阶段失败并清理重试，`clearCacheReuseState()` 会重置本请求的计费状态，避免本请求重复计费。

---

## 状态观测与监控指标

### 1. 请求与延迟指标（`side=prefill|decode`, `submitted`, `reason`）

- QPS 计数：
  - `rtp_llm_p2p_writeback_success_qps`：写回成功 QPS
  - `rtp_llm_p2p_writeback_rejected_qps`：被 Prefill 准入拒绝 QPS（`reason=prefix_evicted_or_demoted` 或 `kv_capacity`）
  - `rtp_llm_p2p_writeback_skipped_qps`：Decode 侧跳过写回 QPS（如无新增整块、请求关闭等）
  - `rtp_llm_p2p_writeback_failed_qps`：其他错误/超时失败 QPS
  - `rtp_llm_p2p_writeback_no_transfer_qps`：Prefill 已全量命中、无需数据传输 QPS
- 耗时与数据量（Gauge）：
  - `rtp_llm_p2p_writeback_cost_time_us`：单次写回总耗时
  - `rtp_llm_p2p_writeback_hold_time_us`：Prefill 目标 KV 持有耗时
  - `rtp_llm_p2p_writeback_drain_time_us`：停止排空耗时
  - `rtp_llm_p2p_writeback_planned_bytes`：规划传输字节数
  - `rtp_llm_p2p_writeback_committed_token_count`：本次成功提交发布到 Prefill 的 token 数
  - `rtp_llm_p2p_writeback_stage_cost_time_us` / `rtp_llm_p2p_writeback_transferred_bytes`（按 `side`、`stage`、`success` 分维度统计）

### 2. 在途与滞留监控（`side=prefill|decode`）

- `rtp_llm_p2p_writeback_inflight_context_count`：当前在途写回上下文数
- `rtp_llm_p2p_writeback_oldest_inflight_ms`：最老在途写回耗时（ms）
- `rtp_llm_p2p_writeback_retained_context_count`：等待物理停写的滞留上下文数
- `rtp_llm_p2p_writeback_retained_kv_bytes`：滞留 KV 字节数
- `rtp_llm_p2p_writeback_retained_kv_percent`：滞留 KV 占总显存池容量百分比
- `rtp_llm_p2p_writeback_oldest_kv_hold_ms`：最老滞留 KV 持有时间（ms）

### 3. 复用与计费指标

- `rtp_llm_writeback_reuse_len`：请求首次命中写回 KV 的 token 长度。

---

## 测试与真机验收

### 1. 单元测试与集成测试（Bazel）

- 调度、准入、取消、滞留与协议边界：`//rtp_llm/cpp/cache/connector/p2p/test:p2p_connector_scheduler_test`
- Worker、lease 快照、gRPC 丢响应/超时与 GPU TCP 拷贝：`//rtp_llm/cpp/cache/connector/p2p/test:p2p_connector_worker_test`
- 端到端写回（含非对称 PP、CP、TP、SWA/LINEAR/Hybrid、GPU fence、TP2 rank step）：`//rtp_llm/cpp/cache/connector/p2p/test:decode_writeback_integration_test`
- CP Sharded Attention 算子数值精度：`//rtp_llm/models_py/modules/factory/attention/cuda_cp_impl/test:test_allgather_cp_impl`

### 2. H20 Smoke Test（`rtp_llm/test/smoke/suites_h20_oss.bzl`）

基于 `Qwen2.5-0.5B-Instruct`（BF16、TCP、贪婪解码）验证冷热输出逐 token 一致与 Prefill 复用命中，覆盖 3 组 TP 拓扑：
- `p2p_tp2_to_tp2_bf16_decode_writeback`
- `p2p_tp2_to_tp1_bf16_decode_writeback`
- `p2p_tp1_to_tp2_bf16_decode_writeback`

### 3. 真机端到端验收工具（`rtp_llm/test/smoke/writeback_acceptance.py`）

针对已拉起的真实 PD 集群，提供 4 种验收子命令（配套单测 `//rtp_llm/test/smoke:writeback_acceptance_test`）：
- `v4-cp`：校验 Prefill/Decode CP 参数、写回发布、`cp_unit` 对齐、冷热输出一致与首次复用计费；
- `asymmetric-tp`：验证非对称 TP 拓扑下的写回、复用与输出一致；
- `multiturn`：多轮对话连续写回与复用验收；
- `billing`：专测写回首次复用计费（`writeback_first_use_len`）与重复命中不重复计费。
