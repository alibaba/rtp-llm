# Kimi K3 Page-RR Prefill TP 到 Decode DP Cache 传输设计

## 1. 目标

在不引入新传输协议或中间聚合 cache 的前提下，正确、稳定地支持以下生产拓扑：

- Page-RR Prefill TP8 → Decode TP1/DP16/KTP16
- Page-RR Prefill TP16 → Decode TP1/DP16/KTP16

首个交付同时覆盖：

- 无推测目标模型
- Eagle3
- Kimi K3 独立 MTP
- BF16 MLA cache
- `KIMI_K3_MLA_FP8=1`
- cold request 和 `reuse_cache=1` 的 prefix hit

P8 → DP8/KTP8 是可运行的 Kimi K3 全模型参考拓扑。现有
`example/k3/kimi_k3_full_model_two_host_pd_smoke.sh` 通过 KTP 和 FP8 降低每个 Decode rank
的显存压力，默认使用 Prefill TP8/EP8 和 Decode TP1/DP8/KTP8/EP8。该 smoke 当前的
Prefill cache 默认是 replicated，不能原样代替 Page-RR 新功能验收；它作为可运行的
全模型回归参考，Page-RR 验收在此基础上启用 source cache sharding 并使用兼容的 page geometry。

## 2. 非目标

本次不做：

- 新的 protobuf cache-layout 字段或版本协议
- Prefill rank0 或其他中间节点的全量 cache 聚合
- Decode KTP rank 间的 cache 广播或 shadow cache
- CacheStore、Connector 或 Transfer Backend 对 Kimi K3 模型语义的感知
- 单 page 重试、传输中换源或部分 cache 降级运行
- 修改 Master 的 Decode 调度算法
- 修改 Eagle3/MTP 在 GENERATE 阶段传递动态状态的方式

## 3. 现有架构与约束

### 3.1 请求归属

Master 在 Prefill 执行前选定一个 Decode 服务实例。路由结果是 `RoleAddr` 而非数字 `dp_rank`。Prefill rank0 与该 Decode 实例建立 RemoteGenerate stream；该进程的静态 `parallelism_config.dp_rank` 就是请求的 Decode owner。

Decode TP1/DP16 中，每个 DP owner 拥有该请求的完整 cache。KTP16 仅切分投影计算，不改变请求 cache 所有权、物理布局或传输目标。

### 3.2 现有协议能力

ALLOCATE 请求已携带实现本功能所需的信息：

- rank-ordered Prefill `peer_addrs`
- `prefill_cp_size`，在本功能中表示 Prefill Page-RR owner 数 `N`
- Prefill physical/kernel page size
- Prefill attention TP
- cache/state dtype
- MLA FP8 format、Q scale 和 KV scale

因此不需要新增协议字段。`peer_addrs[i]` 必须稳定表示 Prefill rank `i`。

### 3.3 现有实现缺口

当前 Decode 逻辑将 Page-RR 和 Projection-KTP fan-in 作为互斥模式，导致组合拓扑不能同时表达：

- FULL MLA 需要按 page owner 选源；
- LINEAR KDA 需要从全部 Prefill ranks 聚合 head segments；
- Eagle3 SWA 只需要一个完整副本；
- MTP FULL MLA 需要和 Target MLA 一样按 page owner 选源。

此外，Page-RR 启动校验目前只接受 TP2/4/8 和 BASE cache，Page-RR prefix reuse 也显式要求 BF16 raw cache，因此尚不能完整支持 TP16 和 K3 plain FP8 MLA cache。

### 3.4 与现有 Page-RR → replicated Decode 路径的关系

本功能是对现有路径的直接扩展，不创建并行架构。以下形式保持一致：

- Decode 先分配 replicated 目标 cache；
- Decode 拿到 rank-ordered 的全部 Prefill peers；
- Decode load planner 生成 source key 和 destination address；
- FULL page 使用 `global_page % N` 选择 Prefill owner；
- CacheStore/Connector 执行规划好的传输；
- 必要 cache 全部完成后，请求才进入 Decode。

差异仅在 KDA 目标布局。现有 equal-attention-TP 路径中，Decode rank `i` 只拥有
对应的 KDA head shard，因此 KDA 使用 `P_i -> D_i` rank affinity。新路径的 Decode
attention TP 为1，每个 DP owner 拥有完整96-head KDA，因此需要将全部 Prefill head
segments 写入同一 Decode owner 的不同目标偏移。KTP 不改变这一 cache 契约。

## 4. 总体设计

保持现有 PD 数据面，将 cache 来源选择收敛到 Decode load planner：

```text
Master
  └─ 选择一个 Decode DP owner
       └─ Decode owner 分配完整本地 cache
            └─ 从 N 个 Prefill Page-RR ranks 拉取必要物理区间
                 ├─ Target MLA：按 page owner
                 ├─ MTP MLA：按 page owner
                 ├─ KDA：全 peer head-segment fan-in
                 └─ Eagle3 SWA：单个确定副本
```

关键是将两个正交事实分开：

```text
prefill_cp_size > 1   => 源 cache 使用 Page-RR
decode attention TP=1 => 目标 cache 是 DP-owner-local 完整副本
```

Decode KTP 不得再作为 cache 布局选择条件。

## 5. 组件设计

### 5.1 拓扑与格式校验

`DecodeRpcServer::prepareGenerateContext()` 在资源分配前校验：

- Prefill Page-RR shard 数是已支持的2/4/8或新增的16，生产验收只覆盖8和16；
- `peer_addrs.size() == prefill_cp_size`；
- 新增的 replicated Decode-owner 分支要求 Decode attention TP 为1；保留现有 equal-attention-TP 等已支持分支；
- 生产目标的 Decode KTP 为1或16，但 KTP 不影响 cache 布局，也不额外收紧旧拓扑的 KTP 配置；
- KDA 总 head 数可被 source shard 数整除；
- Prefill/Decode page geometry 兼容；
- MLA FP8 format 和 fixed scales 完全一致。

校验应使用明确的 Page-RR source 和 replicated Decode cache 概念，不使用 `projection_ktp` 代替二者。

### 5.2 RemoteLoad 请求

`constructRemoteLoadRequestForMla()` 对 Page-RR 请求总是将全部 `N` 个 Prefill peers 交给选中的 Decode owner worker。

不在请求级设置统一的 `partition_count=N`，因为一个请求内的 FULL、LINEAR 和 SWA group 有不同分区契约。真正的分区参数由 load planner 在 group 级传入 `convertIndexToBuffer()`。

### 5.3 Group-level source policy

Decode `loadCache()` 内按下列 source policy 理解现有 group。这是对已有条件的概念整理，
不要求新的运行时子系统；只有在能明显减少歧义时才实现成局部 helper/enum：

| Policy | Cache group | 来源规则 | 目标分区 |
|---|---|---|---|
| `PAGE_OWNER` | Target FULL MLA、MTP FULL MLA | page `g` 来自 `peer[g % N]` | `partition_count=1` |
| `ALL_PEER_PARTITION` | KDA LINEAR | 每个 peer 提供一个 head segment | `partition_count=N`, `partition_id=peer_index` |
| `SINGLE_REPLICA` | Eagle3 SWA | 只从 `peer[decode_dp_rank % N]` 读完整副本 | `partition_count=1` |

KDA 拥有96个 head：

- P8 中每个 Prefill rank 提供12个 head；
- P16 中每个 Prefill rank 提供6个 head。

`SINGLE_REPLICA` 使用 `decode_dp_rank % N` 而非固定 Prefill rank0，使 P8 → DP16 中 DP ranks 8—15 确定性地回映射到 P0—P7，避免单源带宽热点。

现有以下机制继续复用：

- `convertIndexToBuffer()` 物理地址映射；
- LINEAR segment key；
- MTP local-to-physical group 映射；
- request-scoped CacheStore key；
- 通过有效 cache key 数量排除 allocation padding。

### 5.4 FP8 Page-RR prefix reuse

K3 MLA FP8 是 plain E4M3 cache，每 token 为576字节，fixed scale 位于配置元数据中，不是独立的 scale block。因此 PD 网络层仅复制原始 FP8 bytes。

`MlaPageRRCacheAdapter` 保持 dtype-agnostic，只负责：

1. 从 rank-local pages pack prefix；
2. TP AllGather；
3. 恢复 request-major 逻辑 token 顺序。

attention wrapper 读到 canonical prefix 后解释 cache 精度：

```text
canonical E4M3 prefix
  -> cast to BF16
  -> in-place multiply by mla_fp8_kv_scale
  -> existing prefix merge/projection/attention path
```

使用非1 scale 进行测试，确保实现不会遗漏解量化。BF16 路径不增加 cast 或 scale 操作。

Page-RR geometry 保留 TP2/4/8 并增加 TP16，接受且仅接受以下两种 cache 精度契约：

- BF16 compute + BASE cache；
- BF16 model compute + `mla_fp8_compute=true` + FP8 cache。

INT8、旧混合 FP8 layout 和 format/scale 不一致仍然 fail fast。Target MLA 和独立 MTP MLA 使用 FP8 Page-RR；Eagle3 draft SWA 保持其现有 BF16 replicated 精度策略；KDA 保持现有 FP32/BF16 state 格式。

## 6. 请求时序

### 6.1 路由与分配

1. Frontend 请求 Master 选择 Prefill 和 Decode RoleAddr。
2. Frontend 将路由结果交给 Prefill。
3. Prefill rank0 连接选中的 Decode owner，发送 ALLOCATE。
4. Decode 校验拓扑、geometry 和 FP8 契约。
5. Decode 为 Target MLA、KDA 和启用的 draft cache 分配完整本地 blocks。
6. 分配全部成功后，Decode 才回复 ALLOCATE 成功。

### 6.2 Prefill 计算与 LOAD 重叠（复用现有机制）

Prefill 在可以执行后尽早发送 LOAD。Decode 按 group policy 将 CacheStore load 注册到确切的目标物理地址。

如果 source key 尚未发布，load 保留在现有 `wait_tasks_` 中。Prefill 对某层 cache 的 CUDA 写入完成后，cleanup/checker thread 确认 event 完成，再将 `LayerCacheBuffer` 发布到 CacheStore。Transfer Backend 随后将原始物理字节写入 Decode 目标地址。

这保留了 Prefill 计算与 PD 传输的重叠，也保证不会读取尚未写完的 cache。

Prefix reuse 中的 Prefill TP AllGather 只服务 attention 计算；PD 仍从每个 Page-RR owner 传输原始本地 FP8/BF16 pages，不传输 AllGather 产生的 BF16 临时展开结果。

上述 LOAD-before-publish、`wait_tasks_`、CUDA event 发布门控、chunk/layer-wise publication 与
计算/传输重叠均已由基线实现。本次不修改这些状态机；只要求新 load plan
生成与现有 publisher 完全一致的 keys 和 byte ranges。FP8 prefix reuse 的解量化修改
属于 Prefill attention 计算路径，不是对传输重叠机制的修改。

### 6.3 转入 Decode

Decode 等待以下所有必要内容：

- 全部有效 Target MLA pages；
- 全部 KDA head segments；
- Eagle3 SWA 完整副本或全部 MTP MLA pages；
- 所有已投递 load task 的成功结果。

全部成功后，Decode 回复 LOAD 完成。Prefill 确认没有传输仍在读取源 blocks 后释放其 PD cache，然后通过 GENERATE 发送首 token、position 和 Eagle3/MTP 动态运行状态。Decode 只在此时创建并 enqueue GenerateStream。

CacheStore 负责静态 cache 内容，GENERATE gRPC 负责不能从 cache 重建的动态推理状态。

## 7. 错误处理与资源生命周期

本节定义的事务、取消和释放语义由现有 PD/CacheStore 实现提供。本次不新建
错误处理或资源生命周期状态机，只补充新拓扑和 group mapping 的 fail-fast 校验，
并通过现有失败/取消路径验证这些新计划。

### 7.1 Fail-fast

拓扑、peer 数、page geometry、cache spec、KDA head 可分性和 FP8 format/scales 错误在 ALLOCATE 资源分配前返回 `INVALID_ARGUMENT`。Prefill 不开始计算或传输。新校验只为 Page-RR → replicated Decode owner 添加合法组合，不使已有 TP2/4/8 或 equal-attention-TP 路径回归。

Load plan 生成期间检查：

- owner/partition 索引范围；
- 有效 cache key 和 block 数；
- 被选 block 不是 `NULL_BLOCK_IDX`；
- `convertIndexToBuffer()` 返回的区间数量、大小和地址；
- MTP local group 到 physical group 的映射；
- padding block 没有进入传输计划。

### 7.2 整体失败语义

任意一个必要 LoadContext 超时、连接失败、key 发布失败或目标写入失败，整个请求的 LOAD 失败。Decode 不使用部分 cache 创建 GenerateStream。

保留现有的连接建立重试。首版不新增单 page/数据传输重试或 Eagle3 运行中换源。
这保持一个单一、可推理的失败契约，并避免将新重试状态机混入首个交付。

### 7.3 取消和释放

请求取消时：

1. Decode gRPC cancellation 通过现有 `cancel_check_func` 传给 CacheStore load；
2. TransferTask 进入 `CANCELLED`；
3. Decode 等待所有 receiver/load tasks 退出；
4. 确认无后台任务继续写入后，才释放并复用目标 blocks；
5. Prefill 同样等待 CUDA event 和已发布传输引用退出，再释放源 blocks。

如果在某个 peer 的计划或投递后才发现后续错误，停止投递新任务，但必须等待已投递任务结束。`request_id/request_key` 继续作为请求隔离边界。

### 7.4 可观测性

保留请求级摘要日志：

- request ID；
- source shard 数；
- Decode DP rank；
- 各 source policy 生成的任务数；
- FP8 开关；
- 最终成功、失败或取消状态。

不添加默认逐 page 日志，避免长上下文请求产生大量日志开销。

## 8. 测试与验收

### 8.1 Load planner 单元测试

直接构造 cache config、block table 和 peer list，覆盖：

- N=8、16；
- Target/MTP FULL page `g` 只由 `g % N` owner 传输；
- KDA P8 的8个12-head segments 和 P16 的16个6-head segments；
- KDA 所有 segments 无重叠、无空洞，恰好覆盖96 heads；
- Eagle3 `decode_dp_rank % N`，特别是 P8 → DP16 中 ranks 8—15 的回映射；
- KTP1 与 KTP16 生成完全相同的 cache 计划；
- 非整 stripe 尾部不传输 padding blocks；
- FP8 MLA 物理范围恰好是 `page_tokens * 576` 字节；
- MTP local group 正确映射到 Decode physical group。

### 8.2 Page-RR FP8 prefix GPU 测试

验证 pack、AllGather、restore 和 fixed-scale 解量化，覆盖：

- 空 prefix；
- 小于、等于和刚越过一个 physical page；
- TP8 的1023/1024/1025 token prefix；
- TP16 的2047/2048/2049 token prefix；
- 不同 prefix 长度的多请求 batch；
- 多个完整 Page-RR stripes；
- 非1 `kv_scale`，例如0.5；
- 恢复 BF16 结果与按逻辑 token 顺序直接解量化的 reference 一致；
- BF16 现有路径不变。

### 8.3 CacheStore 集成测试

每个 Prefill peer 填充可区分数据，执行真实 load 计划，验证：

- Target/MTP pages 进入对应 Decode 全量页；
- KDA segments 重建完整96-head state；
- Eagle3 只传一份完整副本；
- FP8 raw bytes 完全一致；
- LOAD 早于 publish 时正确等待；
- 两个并发请求的目标地址和 request key 不互相污染；
- peer、geometry、FP8 format/scales、NULL block、timeout 和 cancellation 失败语义；
- 取消后所有任务退出才允许 block 重新分配。

### 8.4 真实部署验收

首先保留 P8 → DP8/KTP8 的现有全模型 smoke 作为回归参考。该脚本：

- 默认 `SP_TYPE=eagle3`；
- 使用 `SP_TYPE=mtp` 单独选择 Kimi K3 MTP，两者不在同一次启动中同时开启；
- 默认 `KIMI_K3_MLA_FP8=1` 并启用 FP8 target weights/collective GEMM；
- 要求 draft checkpoint，当前不提供无推测模式；
- 默认 Prefill cache 为 replicated，因此需扩展一个 Page-RR profile 而不是把原脚本结果当作 Page-RR 验收。

脚本中“KTP with MTP is rejected”的说明已落后于当前分支；现有
`validate_projection_ktp_sp_type()` 已接受 `mtp`，且 draft model 会被固定为 KTP1。实现时应同步修正
该 smoke 说明，并用 `SP_TYPE=mtp` 实际验证，不仅依赖启动校验。

生产验收拓扑：

- P8 → DP16/KTP16；
- P16 → DP16/KTP16。

每种拓扑覆盖以下6种模式，共12个部署配置：

| Speculative mode | BF16 | `KIMI_K3_MLA_FP8=1` |
|---|---:|---:|
| 无推测 | 必测 | 必测 |
| Eagle3 | 必测 | 必测 |
| MTP | 必测 | 必测 |

每个部署内执行：

1. cold request；
2. seed prefix；
3. 命中该 prefix 的新请求；
4. 跨越至少两个 Page-RR stripes 的长 prefix；
5. 分属不同 Decode owners 的并发请求。

验收条件：

- 输出与对应非 PD/reference 路径一致，FP8 使用现有数值容差；
- reuse 统计非零且命中长度正确；
- Decode 开始前必要 cache 完整到达；
- padding pages 不进入 PD 传输；
- P8 → DP16 的 Eagle3 单副本流量按 DP owner 均匀回映射到8个 Prefill ranks；
- KTP1/KTP16 不改变 cache 内容和来源映射；
- 无 hang，超时或取消后无后台写入和 block 提前复用。

## 9. 实现原则

- 优先修正现有条件和 group-level 计划，不新建传输子系统。
- source policy 是对现有 Decode planner 分支的概念整理；优先用小范围 helper 表达，
  不引入新类层次或运行时子系统。
- 根据 cache spec/group type 选择 policy，不根据 KTP 选择 cache 布局。
- FP8 解量化属于 attention 语义，不进入 Page-RR adapter 或 Transfer Backend。
- 任何部分 cache 失败都不允许 Decode 继续，确保正确性优先于局部可用性。
- 实现不依赖 P8 特例，所有映射使用协议传入的 source shard 数 `N`。
- 新功能是对现有拓扑矩阵的扩展；不删除或收紧原有 Page-RR/equal-TP 合法路径。
