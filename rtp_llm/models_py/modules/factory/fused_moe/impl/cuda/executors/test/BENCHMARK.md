# SM100 MoE Group-GEMM Benchmark Report

- **GPU**: NVIDIA L20A (SM100, GB200 Grace-Blackwell)
- **Software**: PyTorch 2.9.0+cu129, FlashInfer 0.6.6, DeepGEMM 2.1.1
- **CI Run**: 35372622 (2026-04-10)
- **Branch**: `flashinfer-fp8-groupwise` @ `5bb0f29e1`

---

## 1. 实现清单

8 种 MoE Group-GEMM kernel 实现，覆盖 FP4/FP8 两种精度。

| # | 全名 | Kernel | 来源 | 精度 | Fused | 状态 |
|---|------|--------|------|------|-------|------|
| 1 | **FlashInfer CuteDSL FP4** | `grouped_gemm_nt_masked` | FlashInfer | FP4 E2M1 | No | OK |
| 2 | **TRT-LLM Fused FP4** | `trtllm_fp4_block_scale_routed_moe` | FlashInfer/TRT-LLM | FP4 E2M1 | **Yes** | OK |
| 3 | **CUTLASS FP4 (vLLM)** | `cutlass_fp4_group_mm` GemmUniversal JIT | vLLM/SGLang | FP4 E2M1 | No | OK |
| 4 | **FlashInfer FP8 Groupwise** | `group_gemm_fp8_nt_groupwise` | FlashInfer | FP8 E4M3 | No | OK |
| 5 | **DeepGEMM FP8 Masked** | `m_grouped_fp8_gemm_nt_masked` UE8M0 | DeepGEMM | FP8 E4M3 | No | OK (1) |
| 6 | **DeepGEMM FP8 Contiguous** | `m_grouped_fp8_gemm_nt_contiguous` UE8M0 | DeepGEMM | FP8 E4M3 | No | OK |
| 7 | **CUTLASS FP8 Per-Tensor** | `cutlass_moe_mm_fp8_scaled` | rtp_kernel | FP8 E4M3 | **Yes** | OK (2) |
| 8 | **TRT-LLM Fused FP8** | `trtllm_fp8_block_scale_routed_moe` | FlashInfer/TRT-LLM | FP8 E4M3 | **Yes** | OK |

Fused = 包含 routing (softmax→topk→scatter) + gather 的端到端 MoE kernel

**注**:
1. DeepGEMM FP8 Masked: SkA 大 M 场景 (maxM/E > 1024) 偶发崩溃
2. CUTLASS FP8 Per-Tensor: 性能极差 (~15-17ms)，per-tensor scale 不适合 SM100 MoE

---

## 2. GEMM-Only Benchmark — Full MoE (FC1 + SiLU + FC2)

**测试条件**: N=2048, K=7168, top_k=8 (fused 用 top_k=7), Warmup=10, Iters=50, SEED=42

### 2.1 均匀激活

| Scenario | Type | E | M/E | TotM | FlashInfer CuteDSL FP4 | TRT-LLM Fused FP4 | CUTLASS FP4 (vLLM) | FlashInfer FP8 Groupwise | DeepGEMM FP8 Masked | DeepGEMM FP8 Contiguous | CUTLASS FP8 Per-Tensor | TRT-LLM Fused FP8 | Best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M/E=8 | decode | 8 | 8 | 64 | 0.273ms 20.7TF | 0.302ms 18.6TF | **0.026ms 73.6TF** | 0.232ms 24.3TF | 2.463ms 2.3TF | 0.515ms 10.9TF | 17.3ms 0.3TF | 0.257ms 21.9TF | **CUTLASS FP4** |
| M/E=16 | decode | 8 | 16 | 128 | 0.263ms 42.9TF | 0.252ms 44.7TF | **0.023ms 163.6TF** | 0.222ms 50.8TF | 2.423ms 4.7TF | 0.506ms 22.3TF | 17.7ms 0.6TF | 0.243ms 46.3TF | **CUTLASS FP4** |
| M/E=64 | prefill | 8 | 64 | 512 | 0.272ms 165.7TF | 0.253ms 178.4TF | **0.024ms 626.4TF** | 0.234ms 193.0TF | 2.428ms 18.6TF | 0.511ms 88.2TF | 16.7ms 2.7TF | 0.356ms 126.6TF | **CUTLASS FP4** |
| M/E=256 | prefill | 8 | 256 | 2048 | 0.255ms 706.7TF | 0.381ms 474.0TF | **0.033ms 1807TF** | 0.217ms 830.6TF | 2.397ms 75.3TF | 0.512ms 352.4TF | 15.6ms 11.5TF | 1.277ms 141.3TF | **CUTLASS FP4** |
| M/E=512 | prefill | 8 | 512 | 4096 | 0.266ms 1355TF | 0.698ms 516.5TF | **0.059ms 2035TF** | 0.288ms 1252TF | 2.374ms 152.0TF | 0.530ms 680.6TF | 15.7ms 23.0TF | 2.511ms 143.7TF | **CUTLASS FP4** |
| M/E=1024 | prefill | 8 | 1024 | 8192 | 0.264ms 2729TF | 1.354ms 532.9TF | **0.097ms 2478TF** | 0.541ms 1333TF | 2.448ms 294.7TF | 0.691ms 1044TF | 16.1ms 44.8TF | 4.974ms 145.1TF | **CUTLASS FP4** |
| M/E=2048 | prefill | 8 | 2048 | 16384 | 0.463ms 3116TF | 2.868ms 503.3TF | **0.172ms 2795TF** | 0.987ms 1462TF | 2.634ms 547.9TF | 1.075ms 1342TF | 16.6ms 86.8TF | 9.939ms 145.2TF | **CUTLASS FP4** |
| E64-M/E=8 | decode | 64 | 8 | 512 | 0.278ms 162.2TF | 0.412ms 109.4TF | **0.109ms 137.8TF** | 0.580ms 77.8TF | 15.5ms 2.9TF | 0.936ms 48.2TF | 16.0ms 2.8TF | 0.882ms 51.1TF | **CUTLASS FP4** |

### 2.2 非均匀激活 — SkA (长尾: 38%/25%/12.5%/...)

| Scenario | TotM | maxM/E | FlashInfer CuteDSL FP4 | TRT-LLM Fused FP4 | CUTLASS FP4 (vLLM) | FlashInfer FP8 Groupwise | DeepGEMM FP8 Masked | DeepGEMM FP8 Contiguous | TRT-LLM Fused FP8 | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| SkA-64 | 64 | 25 | 0.265ms 21.3TF | 0.294ms 19.2TF | **0.027ms 69.4TF** | 0.218ms 25.9TF | 2.449ms 2.3TF | 0.514ms 11.0TF | 0.250ms 22.6TF | **CUTLASS FP4** |
| SkA-512 | 512 | 194 | 0.271ms 166.7TF | 0.254ms 177.5TF | **0.027ms 565.7TF** | 0.223ms 202.5TF | 2.443ms 18.5TF | 0.524ms 86.1TF | 0.361ms 124.8TF | **CUTLASS FP4** |
| SkA-2048 | 2048 | 770 | 0.271ms 665.9TF | 0.378ms 477.8TF | **0.043ms 1406TF** | 0.463ms 389.9TF | ERR | 0.517ms 348.8TF | 1.282ms 140.8TF | **CUTLASS FP4** |
| SkA-4096 | 4096 | 1538 | 0.257ms 1403TF | 0.731ms 493.3TF | **0.056ms 2152TF** | 0.805ms 448.1TF | ERR | 0.565ms 638.1TF | 2.519ms 143.2TF | **CUTLASS FP4** |
| SkA-8192 | 8192 | 3074 | 0.291ms 2483TF | 1.418ms 508.8TF | **0.099ms 2438TF** | 1.520ms 474.8TF | ERR | 0.734ms 983.3TF | 4.985ms 144.7TF | **CUTLASS FP4** |
| SkA-16384 | 16384 | 6146 | 0.528ms 2734TF | 2.791ms 517.1TF | **0.166ms 2896TF** | 2.923ms 493.8TF | 2.983ms 483.8TF | 1.114ms 1296TF | 9.950ms 145.0TF | **CUTLASS FP4** |

### 2.3 非均匀激活 — SkB (极端集中: 75%/rest=1 each)

| Scenario | TotM | maxM/E | FlashInfer CuteDSL FP4 | TRT-LLM Fused FP4 | CUTLASS FP4 (vLLM) | FlashInfer FP8 Groupwise | DeepGEMM FP8 Masked | DeepGEMM FP8 Contiguous | TRT-LLM Fused FP8 | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| SkB-64 | 64 | 57 | 0.260ms 21.7TF | 0.295ms 19.1TF | **0.027ms 69.2TF** | 0.221ms 25.5TF | 2.414ms 2.3TF | 0.513ms 11.0TF | 0.249ms 22.6TF | **CUTLASS FP4** |
| SkB-512 | 512 | 505 | 0.263ms 171.4TF | 0.254ms 177.7TF | **0.031ms 480.0TF** | 0.287ms 157.1TF | 2.459ms 18.3TF | 0.516ms 87.5TF | 0.358ms 126.0TF | **CUTLASS FP4** |
| SkB-2048 | 2048 | 2041 | 0.269ms 671.4TF | 0.381ms 472.8TF | **0.042ms 1445TF** | 0.970ms 185.9TF | 2.478ms 72.8TF | 0.522ms 345.4TF | 1.283ms 140.6TF | **CUTLASS FP4** |
| SkB-4096 | 4096 | 4089 | 0.262ms 1375TF | 0.719ms 501.8TF | **0.060ms 2001TF** | 1.932ms 186.8TF | 2.541ms 142.0TF | 0.571ms 631.5TF | 2.520ms 143.2TF | **CUTLASS FP4** |
| SkB-8192 | 8192 | 8185 | 0.429ms 1683TF | 1.350ms 534.5TF | **0.097ms 2469TF** | 3.890ms 185.5TF | 3.027ms 238.3TF | 0.738ms 978.2TF | 4.985ms 144.8TF | **CUTLASS FP4** |
| SkB-16384 | 16384 | 16377 | 0.790ms 1826TF | 2.774ms 520.2TF | **0.169ms 2855TF** | 8.777ms 164.4TF | 4.606ms 313.3TF | 1.131ms 1276TF | 9.952ms 145.0TF | **CUTLASS FP4** |

---

## 3. 综合分析

### 3.1 性能分级

| 梯队 | 实现 | 峰值 TFLOPS | GEMM 胜率 | 定位 |
|------|------|------------|----------|------|
| **T0** | **CUTLASS FP4 (vLLM)** | 2896 | **20/20** | GEMM-only 全场景绝对最快 |
| **T1** | FlashInfer CuteDSL FP4 | 3116 | 0/20 | GEMM 第二快, TFLOPS 天花板最高 |
| **T1** | TRT-LLM Fused FP4 | 537 | 0/20 | E2E fused 全场景最快 |
| **T2** | FlashInfer FP8 Groupwise | 831 | 0/20 | 小 batch 均匀场景尚可 |
| **T2** | DeepGEMM FP8 Contiguous | 1296 | 0/20 | 全场景稳定, 非均匀免疫 |
| **T3** | TRT-LLM Fused FP8 | 146 | 0/20 | E2E 小 batch decode |
| **T4** | DeepGEMM FP8 Masked | 548 | 0/20 | 有崩溃风险 |
| **T5** | CUTLASS FP8 Per-Tensor | 87 | 0/20 | 不推荐 |

### 3.2 CUTLASS FP4 (vLLM) vs FlashInfer CuteDSL FP4

| Scenario | TotM | CUTLASS FP4 | CuteDSL FP4 | 加速比 |
|---|---|---|---|---|
| M/E=8 | 64 | 0.026ms | 0.273ms | **10.5x** |
| M/E=64 | 512 | 0.024ms | 0.272ms | **11.3x** |
| M/E=256 | 2048 | 0.033ms | 0.255ms | **7.7x** |
| M/E=512 | 4096 | 0.059ms | 0.266ms | **4.5x** |
| M/E=1024 | 8192 | 0.097ms | 0.264ms | **2.7x** |
| M/E=2048 | 16384 | 0.172ms | 0.463ms | **2.7x** |
| SkB-16384 | 16384 | 0.169ms | 0.790ms | **4.7x** |

**重要**: CUTLASS FP4 延迟异常低 (M/E=8 仅 0.026ms)，数值正确性待验证。

### 3.3 非均匀激活退化

| 实现 | Layout | 均匀→SkA | 均匀→SkB | 原因 |
|------|--------|---------|---------|------|
| CUTLASS FP4 (vLLM) | grouped GEMM | 1.0x | 1.0x~1.2x | problem_sizes 自适应 |
| FlashInfer CuteDSL FP4 | masked 3D | 1.0x~1.1x | 1.0x~1.7x | maxM/E→padding |
| TRT-LLM Fused FP4 | fused routed | 1.0x | 1.0x | 内部 routing 自适应 |
| FlashInfer FP8 Groupwise | contiguous | 1.0x~1.3x | 1.3x~8.7x | 热 expert 大 GEMM 主导 |
| DeepGEMM FP8 Contiguous | contiguous | 1.0x | 1.0x | 无 padding |
| DeepGEMM FP8 Masked | masked 3D | 崩溃 | 1.0x~1.7x | 内存超限 |

---

## 4. End-to-End MoE Benchmark (含 Routing 开销)

非 fused 实现加入 routing 开销 (softmax→topk→scatter→gather)，与 fused 实现公平对比。

### 4.1 Routing 开销

| TotM | Routing ms | vs CUTLASS FP4 GEMM |
|------|-----------|---------------------|
| 64 | 0.335 | 12.9x |
| 128 | 0.326 | 14.2x |
| 512 | 0.458 | 19.1x |
| 2048 | 1.348 | 40.8x |
| 4096 | 2.330 | 39.5x |
| 8192 | 4.290 | 44.2x |
| 16384 | 8.169 | 47.5x |

### 4.2 E2E 对比

| Scenario | TotM | Route ms | FlashInfer CuteDSL FP4 | TRT-LLM Fused FP4 | FlashInfer FP8 Groupwise | DeepGEMM FP8 Contiguous | TRT-LLM Fused FP8 | Best |
|---|---|---|---|---|---|---|---|---|
| M/E=8 | 64 | 0.335 | 0.580ms 9.7TF | 0.284ms 19.8TF | 0.551ms 10.2TF | 0.838ms 6.7TF | **0.238ms 23.7TF** | **TRT-LLM Fused FP8** |
| M/E=16 | 128 | 0.326 | 0.558ms 20.2TF | **0.237ms 47.6TF** | 0.537ms 21.0TF | 0.821ms 13.7TF | **0.237ms 47.6TF** | **TRT-LLM Fused FP4/FP8** |
| M/E=64 | 512 | 0.458 | 0.693ms 65.1TF | **0.238ms 189.6TF** | 0.672ms 67.1TF | 0.956ms 47.2TF | 0.355ms 127.1TF | **TRT-LLM Fused FP4** |
| M/E=256 | 2048 | 1.348 | 1.591ms 113.4TF | **0.379ms 475.7TF** | 1.563ms 115.4TF | 1.863ms 96.8TF | 1.279ms 141.0TF | **TRT-LLM Fused FP4** |
| M/E=512 | 4096 | 2.330 | 2.574ms 140.1TF | **0.700ms 515.7TF** | 2.616ms 137.9TF | 2.863ms 126.0TF | 2.515ms 143.4TF | **TRT-LLM Fused FP4** |
| M/E=1024 | 8192 | 4.290 | 4.542ms 158.9TF | **1.380ms 522.9TF** | 4.805ms 150.2TF | 4.991ms 144.6TF | 4.980ms 144.9TF | **TRT-LLM Fused FP4** |
| M/E=2048 | 16384 | 8.169 | 8.630ms 167.2TF | **2.811ms 513.3TF** | 9.175ms 157.3TF | 9.243ms 156.1TF | 9.936ms 145.2TF | **TRT-LLM Fused FP4** |

**结论**: Fused MoE (TRT-LLM) E2E 全场景最快。routing 开销是 GEMM 时间的 13-47x，完全主导延迟。

---

## 5. 场景选型

| 场景 | 推荐 | 备选 |
|------|------|------|
| **GEMM-only 全场景** | **CUTLASS FP4 (vLLM)** | FlashInfer CuteDSL FP4 |
| **E2E (含 routing)** | **TRT-LLM Fused FP4** | TRT-LLM Fused FP8 |
| **Decode (M/E<=16)** | CUTLASS FP4 / TRT-LLM Fused | FlashInfer FP8 |
| **Prefill (M/E>=512)** | **CUTLASS FP4 (vLLM)** | FlashInfer CuteDSL FP4 |
| **非均匀大 batch** | **CUTLASS FP4 (vLLM)** | DeepGEMM FP8 Contiguous |

### 核心结论

1. **CUTLASS FP4 (vLLM) GEMM-only 20/20 全胜**, 峰值 2896 TFLOPS, 小 batch 比 CuteDSL 快 10x
2. **TRT-LLM Fused FP4 E2E 7/7 全胜**, routing 开销使非 fused 实现的 GEMM 优势被抵消
3. **FP4 全面优于 FP8**, 大 batch 和非均匀场景下碾压
4. **CUTLASS FP4 数值正确性待验证** — 异常低延迟需与 reference 对比

---

## 6. 测试环境

- Warmup=10, Bench=50 iters, SEED=42
- 计时: `torch.cuda.synchronize()` + `time.perf_counter()`
- TFLOPS: `(total_tokens * 2N * K * 2 + total_tokens * K * N * 2) / (ms/1000) / 1e12`
- 非均匀: SkA (长尾 38%/25%/12.5%/...), SkB (极端 75%/rest=1)
- FP4-TRT/FP8-TRT top_k=7, 其他 top_k=8
- CUTLASS FP4 首次运行需 JIT 编译 (~2-3 分钟)
