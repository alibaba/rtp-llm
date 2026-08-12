# DSV4 Mega Wuda / CUDA Extension 重构方案

更新日期：2026-08-12

本文描述算子仓库的后续重构方案，不代表 RTP 已接入 Mega。当前 RTP `dsv4-mega` 分支与
`upstream/feat/dsv4_on_dev@bbf66a5f8` 的代码完全一致；旧旁路 backend 和框架实验改动已撤回。

## 1. 目标

Wuda 是算子优化和 TP1/TP2 演进源，CUDA Extension 是 RTP-LLM 的稳定交付层。重构目标是：

- 最新优化只在 Wuda 维护一份算法实现；
- TP1 和 TP2 共用真正相同的计算 core，但拥有独立 route/sync/launch policy；
- CUDA Extension 只迁入经过正确性和性能验证的稳定实例；
- host-only 重构不改变 device code；
- device helper 抽取不引入寄存器、spill、shared memory 或 latency 回退。

当前 TP1 首次接入不要求完成 TP2 的两个后置通信：FlashMLA output redistribution 和
O-proj all-reduce。

## 2. 为什么先重构 Wuda，再迁 CUDA Extension

最新 Wuda `main@d9e53a0` 已新增 WQ_B query-ready、RMSNorm/RoPE route、MQA PDL 和 TopK
join，但这些代码仍把一部分 TP1/TP2 参数和调度放在同一实现中。CUDA Extension
`dsv4_megakernel@b2f07a8` 还是较早快照。

如果现在直接复制：

- 会把尚未稳定的 `world/rank/peer` 分支复制到交付仓库；
- Wuda 和 CUDA Extension 随后需要重复做同一轮重构；
- TP1 可能承担 system-scope fence、peer 参数或 TP2 grid 调优；
- 两边很快再次实现漂移。

正确顺序是：

1. 在 Wuda 拆清 TP1/TP2 policy；
2. 固定 TP1 correctness、SASS 和 latency 基线；
3. 迁移稳定的 TP1 specialization 到 CUDA Extension；
4. TP2 完整后再迁其 transport policy 和生产入口。

## 3. 最新 Wuda 需要先拆出的边界

### 3.1 Query 计算 core

共享部分：

```text
per-head RMSNorm
RoPE
query output layout
per-head completion publication
```

建议形成编译期 core，不读取 `world/rank/peer`。

### 3.2 Route policy

```text
LocalRoute<TP1>
  输出写本卡完整 [B, 128, 512]
  GPU-scope ordering
  无 peer pointer

SymmetricTpDpRoute<TP2>
  输出按 request owner 写 symmetric peer buffer
  system-scope ordering
  rank/world/row_split 仅存在于该实例
```

### 3.3 Completion policy

```text
LocalGpuJoin<TP1>
  等待本卡 query-ready

PeerSystemJoin<TP2>
  等待两个 rank generation
```

TopK 只调用 policy 的 `wait()`，不要在通用 hot path 中循环 `rms_world`。

### 3.4 Launch policy

WQ_B 的 cluster 数和空闲 SM 策略必须按实际 geometry 单独调优。当前 59-cluster 选择明确来自
B300 TP2/DP2 overlap，不能只因为启用了 query PDL 就应用到 TP1。

建议至少有：

```text
WqbLaunchPolicyFullRankTp1
WqbLaunchPolicyTp2Dp2
```

两个实例分别做 batch matrix 性能门禁。

## 4. CUDA Extension 当前抽象问题

`csrc/kernels/dsv4_mega` 的 bring-up 结构以单 op 自包含为主：一个 `.cu/.cuh` 同时包含 tensor
校验、TMA descriptor、launch dispatch、workspace、device primitive、kernel 和 binding。

这种结构便于从 Wuda 快速迁移和单算子调优，但造成：

- FP4/FP8 MQA 重复 host/TMA 代码；
- CSA/HCA 重复少量 SM100 primitive；
- MAIN compressor 在不同 MQA dtype 中复制；
- Indexer 文件名仍带 FP4，但实现已同时服务 FP4/FP8；
- op contract 依赖 binding 文件中的手工前向声明。

融合 kernel 的 tile、warp role、pipeline depth 和 shared-memory layout 本身是性能设计，不应为
减少行数而统一。

## 5. 应该抽取什么

### 5.1 MQA host/TMA 工具

抽取 host-only inline helper：

- device SM 数查询；
- swizzle 转换；
- 2D/3D TMA descriptor 构造；
- paged cache geometry 基础校验。

建议位置：

```text
dsv4_mega/common/mqa_host_utils.h
dsv4_mega/common/tma_host_utils.h
```

FP4 packed layout 修正必须是显式 policy，不能通过 runtime dtype 猜测。

### 5.2 MAIN compressor device core

FP4/FP8 MQA 共享：

- `MainCompressorArgs`；
- 8-row state aggregation；
- BF16 boundary 和 RMSNorm；
- RoPE；
- MODEL1 FP8 body、scale tail 和 page write。

建议抽为：

```text
dsv4_mega/compressor/main_compressor.cuh
```

`run_main_compressor_row` 保持 `__device__ __forceinline__`。FP4/FP8 kernel 只负责分配 tail
warpgroup、barrier id 和调用 core。不能改变 reduction order、BF16 rounding、scale clamp、
RoPE pair layout 或 page tail offset。

### 5.3 Cache layout 常量

`common/cache_layout.cuh` 只保存编译期字节常量和 `page + offset` 地址计算 helper。

禁止：

- allocator 对象；
- 虚函数；
- runtime dtype 分派；
- 隐式持有 framework tensor。

### 5.4 Indexer 命名中立化

当前 `idx_comp_fp4.cuh` / `idx_post_fp4.*` 已同时包含 FP4 和 FP8 路径。建议改为：

```text
indexer/indexer_compressor.cuh
indexer/indexer_postprocess.cuh
indexer/indexer_postprocess.cu
```

公共输入加载、RoPE 和 row addressing 保持一份；FP4 Hadamard/MXFP4 与 FP8 E4M3/folded
weight epilogue 继续是两个编译期函数。

### 5.5 薄 SM100 primitive 层

只抽逐字一致的 inline PTX wrapper：

- lane/cluster rank；
- elect 和 cluster fence；
- TMEM alloc/dealloc；
- commit 和 load fence；
- 完全相同调用约定的 TMA load。

不要统一不同 barrier 类型，不抽 pipeline state machine、`SharedStorage`、warp role 或完整
kernel body。

### 5.6 Public contract

增加 `dsv4_mega_ops.h`，集中声明各 op init 和稳定 C++ contract。pybind 仍集中在一个 module，
不要为每个算子拆一个 `.so`。

## 6. 暂时不要抽取什么

- 不统一 CSA/HCA front 完整 kernel；
- 不统一 CSA/HCA WQ_B 完整 kernel；
- 不把 FP4/FP8 MQA 做成一个大型 runtime dtype kernel；
- 不引入跨 op 的大型 `SharedStorage` 或 pipeline framework；
- 不用继承和虚函数描述 page geometry；
- 不在 TP1 kernel 中保留未完成的 TP2 runtime branch；
- 不在重构提交中顺便改变 tile、scheduler 或数值顺序。

## 7. 建议目录

```text
dsv4_mega/
  dsv4_mega_ops.h
  bindings_dsv4_mega.cc
  common/
    cache_layout.cuh
    mqa_host_utils.h
    sm100_cluster_primitives.cuh
  compressor/
    main_compressor.cuh
  communication/
    local_query_route.cuh
    symmetric_tpdp_query_route.cuh
  indexer/
    indexer_compressor.cuh
    indexer_postprocess.cuh
    indexer_postprocess.cu
  front/
    front_csa_kernel.cuh
    front_csa_op.cu
    front_hca_kernel.cuh
    front_hca_op.cu
  wqb/
    wqb_csa_kernel.cuh
    wqb_csa_op.cu
    wqb_hca_kernel.cuh
    wqb_hca_op.cu
  mqa/
    mqa_fp8_kernel.cuh
    mqa_fp8_op.cu
    mqa_fp4_kernel.cuh
    mqa_fp4_op.cu
  topk/
    topk_v2_kernel.cuh
    topk_v2_op.cu
```

先抽内容，再做机械目录移动，避免一份提交同时包含逻辑变化和大量 rename。

## 8. 提交顺序

Wuda：

1. `refactor(dsv4): split local and symmetric query route policies`；
2. `refactor(dsv4): specialize TP1 and TP2 query joins`；
3. `refactor(dsv4): isolate WQ_B launch policies`；
4. `refactor(dsv4): share MQA host and TMA helpers`；
5. `refactor(dsv4): share MAIN compressor device core`；
6. `refactor(dsv4): neutralize Indexer FP4-era names`；
7. `refactor(dsv4): share identical SM100 primitives`。

CUDA Extension：

1. 迁入验证后的 TP1 WQ_B/query route/MQA/TopK 快照；
2. 接入现有 RTP cache ABI 和 Python wrapper；
3. 拆 kernel implementation 与 op wrapper；
4. 最后做目录移动和 build source path 更新；
5. TP2 两处后置通信完成后再迁 TP2 policy。

## 9. 每个提交的性能门禁

- `sm_100a` / `sm_103a` clean build；
- DSV4 mega CUDA 定向测试；
- WQ_B standalone `--fullrank`，不定义 TP2；
- FP8 MQA 对比 `deep_gemm.fp8_paged_mqa_logits`；
- standalone 与 fused MAIN compressor 字节/容差对齐；
- ptxas registers、spill、static/dynamic smem 对比；
- hot kernel SASS 对比；
- M=1/4/16/32/128 latency；
- query route 单独 latency 和 WQ_B/MQA overlap exposed latency；
- 最终 normal FP8 与 Mega FP8 整段 attention CUDA graph A/B。

host-only 重构要求 device 指标完全不变。device core 抽取如出现寄存器、spill、smem 或稳定态
latency 回退，必须恢复后再合入。
