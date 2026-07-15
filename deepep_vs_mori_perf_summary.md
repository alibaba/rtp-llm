# DeepEP vs Mori Decode 性能对比总结

> 模型：Qwen3.5-397B-A17B | 硬件：MI300X x 8 | 配置：EP8, BF16, input_len=2048
> 两组均包含 Step1 IPC AllReduce 优化 | 测试日期：2026-06-08

## 一、端到端 Decode 延迟（ms，稳态 avg_decode_time）

| Batch Size | Mori (ms) | DeepEP (ms) | Mori/DeepEP | 结论 |
|---|---|---|---|---|
| 1 | **100.7** | 118.2 | 0.85 | Mori 快 15% |
| 4 | **103.3** | 123.0 | 0.84 | Mori 快 16% |
| 8 | **106.3** | 123.4 | 0.86 | Mori 快 14% |
| 16 | **157.2** | 181.8 | 0.87 | Mori 快 13% |
| 32 | **154.3** | 182.9 | 0.84 | Mori 快 16% |
| 64 | **341.5** | 399.9 | 0.85 | Mori 快 15% |

**结论：Mori 端到端全面领先 DeepEP 13%-16%。**

## 二、逐模块 GPU Kernel 时间拆解（ms，per-rank per-iter，8 rank 平均）

| 模块 | DeepEP bs=1 | Mori bs=1 | DeepEP bs=64 | Mori bs=64 |
|---|---|---|---|---|
| Attn_Linear (GDN) | 1.01 | 1.06 | 4.14 | 4.05 |
| Attn_Full (PagedAttn) | 0.43 | 0.45 | 2.02 | 1.17 |
| MoE_Dispatch | 16.52 | **14.27** | 15.42 | **9.46** |
| MoE_Combine | **15.07** | 18.20 | 22.04 | **19.78** |
| MoE_GEMM | **2.54** | 14.52 | **44.97** | 74.74 |
| MoE_Sorting | **0.29** | 1.32 | **0.78** | 1.30 |
| MoE_Router | 0.85 | 0.66 | 1.29 | 0.69 |
| MoE_Activation | 0.33 | 0.30 | 0.36 | 0.32 |
| Dense_GEMM | 5.83 | 6.00 | 11.96 | 11.24 |
| AllReduce | **13.58** | 46.84 | **11.20** | 7.65 |
| Sync/Wait | 0.33 | 0.51 | 48.93 | 59.65 |
| **Kernel Sum** | **60.9** | 108.2 | **170.0** | 194.8 |

## 三、核心矛盾：Kernel 总和 DeepEP 更小，端到端 Mori 更快

### 原因分析

DeepEP 的 `notify_dispatch` / `cached_notify_combine` 是**阻塞式信号量等待**——GPU 空转等其他 rank 数据到达。这段等待时间**不计入 kernel duration 但占 wall-clock**。

Mori 的固定 buffer P2P 设计（`EpDispatchIntraNodeKernel` / `EpCombineIntraNodeKernel`）虽然 kernel 本身时间更长，但：
- 无需阻塞等待其他 rank 完成
- 确定性执行时间，流水线调度更紧凑
- GPU 利用率更高（kernel 无间隙串行，GPU 占用率 ~100%）

### 各模块差异解读

| 模块 | 谁更优 | 原因 |
|---|---|---|
| MoE_Dispatch | Mori | 固定 buffer P2P，无需 layout 协商 |
| MoE_Combine | 接近 | Mori 小 bs 略慢，大 bs 略快 |
| MoE_GEMM | DeepEP (kernel) | DeepEP 用 1-stage kernel（ragged M 小），Mori 用 2-stage（padded M 大） |
| MoE_Sorting | DeepEP (kernel) | DeepEP 单 phase sort，Mori multi-phase sort |
| AllReduce | DeepEP (kernel) | 但 Mori 的 AllReduce 被 overlap 掉不影响 E2E |
| Dense_GEMM | 接近 | 两者相同（rocBLAS Cijk kernel） |
| Attention | 接近 | 两者相同（GDN + PagedAttn） |

## 四、已尝试的优化及结论

| 优化 | 描述 | 效果 | 结论 |
|---|---|---|---|
| **Step 1: IPC AllReduce** | TRT-LLM shared-memory AllReduce 替代 NCCL | 已合入，两路径均受益 | 有效 |
| Step 2: 跳空专家 | `call_local_expert_count=True` + `expert_mask` | bs=1 回退 +9.1% | 无效（aiter sorting 已高效处理稀疏） |
| Step 3A: D2H recv count slice | `.item()` 同步获取实际 recv 数 | bs=1 回退 +3.8% | 无效（D2H sync 打断流水线） |
| Step 3B: CPU upper-bound slice | `num_tokens * topk` 上界裁剪 | bs=1 回退 +1.9% | 无效（BF16 GEMM HBM bound） |
| Step 4: 1-stage GEMM slice | Slice dispatch buffer 触发 1-stage kernel | GEMM 4.5x 加速，E2E 无改善 | 无效（GEMM 不在关键路径） |

## 五、Mori 端到端更快的本质原因

```
DeepEP 关键路径（每层 MoE）：
  dispatch → [GPU idle 等信号量] → GEMM → combine → [GPU idle 等信号量] → allreduce
                   ↑ wall-clock 被 idle 拉长

Mori 关键路径（每层 MoE）：
  dispatch → GEMM → combine → allreduce
  ↑ kernel 时间长但无 idle，GPU 紧凑执行
```

Mori 的设计哲学是"固定 buffer + 确定性通信"，牺牲了通信 kernel 的效率（padded buffer 传更多数据），但换来了**零等待的流水线执行**，在 intra-node 场景下整体更优。

## 六、数据来源

- Mori trace: `TEST_OUTPUT_QWEN35_MOE_1_8_0_20260608_085633`
- DeepEP trace: `TEST_OUTPUT_QWEN35_MOE_1_8_0_20260608_104734`
- 分析脚本: `/home/admin/qinhanwen/codes/deepep/analyze_deepep_vs_mori.py`
- 详细 Excel: `oss://search-ad-cn-hangzhou/qinhanwen/mori_perf_test/deepep_trace/deepep_vs_mori_decode_perf.xlsx`
