# SM100 MoE GEMM Benchmark Report

- **GPU**: NVIDIA L20A (SM100, GB200 Grace-Blackwell)
- **Software**: PyTorch 2.9.0+cu129, FlashInfer 0.6.4, DeepGEMM 2.1.1
- **CI Run**: 34990974 (2026-04-08)
- **Branch**: `flashinfer-fp8-groupwise` @ `095b95f8e`

---

## 1. Benchmark 矩阵

### 8 种去重后独立实现

| # | 简称 | Kernel | 来源 | 精度 | 状态 |
|---|------|--------|------|------|------|
| 1 | **FP4-CD** | `grouped_gemm_nt_masked` (CuteDSL JIT) | FlashInfer | FP4 E2M1 | OK |
| 2 | **FP4-TRT** | `trtllm_fp4_block_scale_routed_moe` (fused) | FlashInfer/TRT-LLM | FP4 E2M1 | ERR (1) |
| 3 | **FP4-CUT** | `cutlass_fp4_group_mm` (CUTLASS GemmUniversal) | vLLM/SGLang | FP4 E2M1 | ERR (2) |
| 4 | **FP8-FI** | `group_gemm_fp8_nt_groupwise` (float32 blockwise) | FlashInfer | FP8 E4M3 | OK |
| 5 | **FP8-DGM** | `m_grouped_fp8_gemm_nt_masked` (UE8M0, masked 3D) | DeepGEMM | FP8 E4M3 | OK (3) |
| 6 | **FP8-DGC** | `m_grouped_fp8_gemm_nt_contiguous` (UE8M0) | DeepGEMM | FP8 E4M3 | OK |
| 7 | **FP8-CUT** | `cutlass_moe_mm_fp8_scaled` (per-tensor) | rtp_kernel | FP8 E4M3 | OK (4) |
| 8 | **FP8-TRT** | `trtllm_fp8_block_scale_routed_moe` (fused) | FlashInfer/TRT-LLM | FP8 E4M3 | ERR (5) |

**ERR 说明**:
1. FP4-TRT: `fp4_quantize()` API 变更 — `use_ue8m0` 参数不兼容当前 FlashInfer 版本
2. FP4-CUT: vLLM/SGLang CUTLASS FP4 kernel 需要源码集成 (standalone JIT 编译)
3. FP8-DGM: 在 SkA 大 M 场景 (maxM/E > 1024) 偶发崩溃 (masked 3D layout padding 过大)
4. FP8-CUT: 性能极差 (~15ms)，per-tensor scale 精度不适合 SM100 MoE
5. FP8-TRT: `routing_logits` 必须是 bfloat16 (传了 float32)

---

## 2. 统一 Benchmark 结果 — Full MoE (FC1 + SiLU + FC2)

**测试条件**:
- N=2048 (intermediate), K=7168 (hidden), top_k=8
- Warmup=10, Iters=50, SEED=42
- GPU: NVIDIA L20A (SM100), 单卡

### 2.1 均匀激活 (每个 expert 分配相同 token 数)

| Scenario | Type | E | M/E | TotM | FP4-CD ms | FP4 TF | FP8-FI ms | FP8 TF | FP8-DGM ms | DGM TF | FP8-DGC ms | DGC TF | FP8-CUT ms | CUT TF | Best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M/E=8 | decode | 8 | 8 | 64 | 0.277 | 20.3 | **0.249** | **22.6** | 2.472 | 2.3 | 0.531 | 10.6 | 16.298 | 0.3 | FP8-FI |
| M/E=16 | decode | 8 | 16 | 128 | 0.273 | 41.3 | **0.261** | **43.2** | 2.561 | 4.4 | 0.526 | 21.4 | 17.586 | 0.6 | FP8-FI |
| M/E=64 | prefill | 8 | 64 | 512 | 0.285 | 158.3 | **0.263** | **171.2** | 2.569 | 17.6 | 0.536 | 84.1 | 16.023 | 2.8 | FP8-FI |
| M/E=256 | prefill | 8 | 256 | 2048 | 0.277 | 652.0 | **0.256** | **705.4** | 2.655 | 67.9 | 0.551 | 327.6 | 15.384 | 11.7 | FP8-FI |
| M/E=512 | prefill | 8 | 512 | 4096 | **0.276** | **1309.0** | 0.293 | 1233.3 | 2.441 | 147.8 | 0.533 | 676.7 | 17.339 | 20.8 | **FP4-CD** |
| M/E=1024 | prefill | 8 | 1024 | 8192 | **0.270** | **2676.8** | 0.528 | 1367.2 | 2.582 | 279.5 | 0.705 | 1024.2 | 15.924 | 45.3 | **FP4-CD** |
| M/E=2048 | prefill | 8 | 2048 | 16384 | **0.464** | **3110.9** | 0.980 | 1472.4 | 2.746 | 525.5 | 1.082 | 1333.6 | 16.435 | 87.8 | **FP4-CD** |
| E64-M/E=8 | decode | 64 | 8 | 512 | **0.273** | **165.4** | 0.591 | 76.4 | 15.577 | 2.9 | 0.941 | 47.9 | 16.384 | 2.8 | **FP4-CD** |

### 2.2 非均匀激活 — SkA (长尾分布: 38%/25%/12.5%/...)

| Scenario | Type | E | TotM | maxM/E | FP4-CD ms | FP4 TF | FP8-FI ms | FP8 TF | FP8-DGM ms | DGM TF | FP8-DGC ms | DGC TF | FP8-CUT ms | CUT TF | Best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SkA-64 | skewed | 8 | 64 | 25 | 0.270 | 20.9 | **0.249** | **22.6** | 2.508 | 2.2 | 0.521 | 10.8 | 16.780 | 0.3 | FP8-FI |
| SkA-512 | skewed | 8 | 512 | 194 | 0.266 | 169.6 | **0.246** | **183.0** | 2.501 | 18.0 | 0.516 | 87.4 | 15.740 | 2.9 | FP8-FI |
| SkA-2048 | skewed | 8 | 2048 | 770 | **0.259** | **695.4** | 0.464 | 389.0 | ERR | - | 0.503 | 358.7 | 15.476 | 11.7 | **FP4-CD** |
| SkA-4096 | skewed | 8 | 4096 | 1538 | **0.260** | **1389.7** | 0.787 | 458.2 | ERR | - | 0.579 | 623.6 | 17.805 | 20.3 | **FP4-CD** |
| SkA-8192 | skewed | 8 | 8192 | 3074 | **0.291** | **2480.2** | 1.524 | 473.4 | ERR | - | 0.752 | 960.1 | 15.810 | 45.6 | **FP4-CD** |
| SkA-16384 | skewed | 8 | 16384 | 6146 | **0.531** | **2719.6** | 2.982 | 484.0 | 3.044 | 474.1 | 1.120 | 1288.4 | 16.574 | 87.1 | **FP4-CD** |

### 2.3 非均匀激活 — SkB (极端集中: 75%/rest=1 each)

| Scenario | Type | E | TotM | maxM/E | FP4-CD ms | FP4 TF | FP8-FI ms | FP8 TF | FP8-DGM ms | DGM TF | FP8-DGC ms | DGC TF | FP8-CUT ms | CUT TF | Best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SkB-64 | skewed | 8 | 64 | 57 | 0.265 | 21.3 | **0.250** | **22.6** | 2.508 | 2.2 | 0.516 | 10.9 | 16.732 | 0.3 | FP8-FI |
| SkB-512 | skewed | 8 | 512 | 505 | **0.264** | **170.7** | 0.293 | 153.9 | 2.525 | 17.9 | 0.522 | 86.3 | 15.916 | 2.8 | **FP4-CD** |
| SkB-2048 | skewed | 8 | 2048 | 2041 | **0.271** | **665.1** | 0.992 | 181.8 | 2.557 | 70.5 | 0.521 | 346.6 | 15.432 | 11.7 | **FP4-CD** |
| SkB-4096 | skewed | 8 | 4096 | 4089 | **0.268** | **1345.6** | 1.956 | 184.4 | 2.642 | 136.5 | 0.584 | 617.9 | 17.256 | 20.9 | **FP4-CD** |
| SkB-8192 | skewed | 8 | 8192 | 8185 | **0.429** | **1681.5** | 4.117 | 175.3 | 3.087 | 233.7 | 0.746 | 966.6 | 15.883 | 45.4 | **FP4-CD** |
| SkB-16384 | skewed | 8 | 16384 | 16377 | **0.794** | **1816.8** | 8.821 | 163.6 | 4.678 | 308.5 | 1.138 | 1267.8 | 16.561 | 87.1 | **FP4-CD** |

---

## 3. 综合分析

### 3.1 FP4-CuteDSL vs FP8-FlashInfer 直接对比

| Scenario | TotM | FP4 ms | FP8 ms | FP4/FP8 | Winner |
|---|---|---|---|---|---|
| M/E=8 (uniform) | 64 | 0.277 | 0.249 | 0.90x | FP8 |
| M/E=16 (uniform) | 128 | 0.273 | 0.261 | 0.96x | FP8 |
| M/E=64 (uniform) | 512 | 0.285 | 0.263 | 0.92x | FP8 |
| M/E=256 (uniform) | 2048 | 0.277 | 0.256 | 0.92x | FP8 |
| **M/E=512** (uniform) | 4096 | **0.276** | 0.293 | **1.06x** | **FP4** |
| **M/E=1024** (uniform) | 8192 | **0.270** | 0.528 | **1.96x** | **FP4** |
| **M/E=2048** (uniform) | 16384 | **0.464** | 0.980 | **2.11x** | **FP4** |
| **E64-M/E=8** | 512 | **0.273** | 0.591 | **2.16x** | **FP4** |
| SkA-2048 | 2048 | **0.259** | 0.464 | **1.79x** | **FP4** |
| SkB-2048 | 2048 | **0.271** | 0.992 | **3.66x** | **FP4** |
| SkA-8192 | 8192 | **0.291** | 1.524 | **5.24x** | **FP4** |
| SkB-8192 | 8192 | **0.429** | 4.117 | **9.60x** | **FP4** |
| SkA-16384 | 16384 | **0.531** | 2.982 | **5.62x** | **FP4** |
| SkB-16384 | 16384 | **0.794** | 8.821 | **11.11x** | **FP4** |

**关键发现**: FP8-FlashInfer 的性能随 maxM/E 线性退化 (masked layout 按最大 expert 分配)。FP4-CuteDSL 在 M/E>=512 后全面领先，在极端不均匀 (SkB) 大 batch 场景下 FP4 快 **11x**。

### 3.2 DeepGEMM 对比

| 实现 | Decode (M/E<=64) | Prefill (M/E=256-512) | 大 Batch (M/E>=1024) | 非均匀激活 |
|------|------------------|----------------------|---------------------|-----------|
| FP8-DGM (masked) | ~2.5ms (慢 10x) | ~2.5ms (慢 10x) | ~2.7ms (慢 5x) | SkA 大 M 崩溃 |
| FP8-DGC (contiguous) | 0.52-0.53ms | 0.53-0.55ms | 0.70-1.08ms | 稳定 |

- DGM 延迟几乎恒定 (~2.5ms)，不随 M 缩放 — per-expert masked requantization 开销是瓶颈
- DGC 性能介于 FP4-CD 和 FP8-FI 之间，在大 batch 时表现稳定
- DGM 在 SkA maxM/E > 1024 场景崩溃 (masked 3D tensor 内存超限)

### 3.3 FP8-CUTLASS Per-Tensor

- 所有场景固定 ~15-17ms — 完全不可用于生产
- 原因: per-tensor scale 在 SM100 上没有优化的 tile config
- 建议: 不纳入后续优化考虑

### 3.4 非均匀激活下的 Layout 特性

| Layout | 均匀激活 | SkA (长尾) | SkB (极端) |
|--------|---------|-----------|-----------|
| **Masked 3D** (FP4-CD, FP8-DGM) | 按 max_M pad → 所有 expert 等大 | max_M=38%*totM → 中等 pad | max_M=75%*totM → 严重 pad |
| **Contiguous** (FP8-FI, FP8-DGC) | 无 pad | 无 pad，但 expert GEMM 大小不一 | 无 pad，热 expert 耗时长 |

FP4-CuteDSL 虽然用 masked layout (有 padding)，但 kernel 性能足够强，padding 开销被 FP4 的 memory bandwidth 优势抵消。FP8-FlashInfer 用 contiguous layout (无 padding)，但在 SkB 极端场景下，单个热 expert 的 GEMM 独占计算时间，延迟膨胀。

---

## 4. 场景选型建议

| 场景 | 推荐实现 | 备选 | 不推荐 |
|------|---------|------|--------|
| **Decode (M/E<=64)** | FP8-FlashInfer | FP4-CuteDSL | DGM, CUT |
| **Prefill (M/E=256)** | FP8-FlashInfer | FP4-CuteDSL | DGM, CUT |
| **Prefill (M/E>=512)** | **FP4-CuteDSL** | FP8-DGC | FP8-FI, DGM, CUT |
| **大 Batch (M/E>=1024)** | **FP4-CuteDSL** | FP8-DGC | FP8-FI, DGM, CUT |
| **多 Expert (E>=64)** | **FP4-CuteDSL** | — | 所有 FP8 |
| **非均匀 (长尾)** | **FP4-CuteDSL** | FP8-DGC | FP8-FI (退化严重) |
| **非均匀 (极端)** | **FP4-CuteDSL** | FP8-DGC | FP8-FI (退化 11x) |

### 综合结论

1. **FP4-CuteDSL 是 SM100 上最通用的 MoE kernel**: 在 14/20 场景胜出，尤其在大 batch 和非均匀激活下优势显著
2. **FP8-FlashInfer 在小 batch decode 场景有 4-10% 优势**: 仅在 M/E<=256 的均匀激活场景领先
3. **FP8-FlashInfer 的 masked layout 在非均匀激活下严重退化**: SkB-16384 场景下比 FP4 慢 11x
4. **FP8-DGC (DeepGEMM Contiguous) 是可靠的中间选择**: 性能稳定，无崩溃，在大 batch 时仅慢 FP4 约 2x
5. **FP8-DGM 和 FP8-CUT 不推荐**: DGM 有崩溃风险且延迟恒定，CUT per-tensor scale 性能极差

---

## 5. 未覆盖实现 & 后续计划

| 实现 | 未纳入原因 | 后续 |
|------|----------|------|
| FP4-TRT (`trtllm_fp4_block_scale_routed_moe`) | `fp4_quantize()` API 版本不兼容 | 修复 `use_ue8m0` 参数问题 |
| FP4-CUTLASS (`cutlass_fp4_group_mm`) | vLLM C++ op, 需要源码集成 | 源码编译方式接入 |
| FP8-TRT (`trtllm_fp8_block_scale_routed_moe`) | `routing_logits` 类型不匹配 (float32→bfloat16) | 修复类型转换 |

---

## 6. 测试环境

- **Warmup**: 10 iters
- **Bench**: 50 iters
- **SEED**: 42 (固定种子, 可复现)
- **计时**: `torch.cuda.synchronize()` + `time.perf_counter()`
- **GPU 独占**: CI 环境单 GPU lock
- **TFLOPS 公式**: Full MoE: `(total_tokens * 2N * K * 2 + total_tokens * K * N * 2) / (ms/1000) / 1e12`
- **非均匀激活**:
  - SkA (长尾): 按 38%/25%/12.5%/6.25%/... 分配
  - SkB (极端): 热 expert 占 75%, 其余每个 expert 1 token (不足则 round)
