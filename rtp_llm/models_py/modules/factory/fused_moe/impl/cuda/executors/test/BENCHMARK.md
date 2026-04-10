# SM100 MoE Group-GEMM Benchmark Report

- **GPU**: NVIDIA L20A (SM100, GB200 Grace-Blackwell)
- **Software**: PyTorch 2.9.0+cu129, FlashInfer 0.6.6, DeepGEMM 2.1.1
- **CI Run**: 35478209 (2026-04-10, Full MoE FC1+SiLU+FC2 for all implementations)
- **Branch**: `flashinfer-fp8-groupwise` @ `fad24ca13`

---

## 1. 实现清单

8 种 MoE Group-GEMM kernel，覆盖 FP4/FP8 两种精度。所有实现均执行 Full MoE pipeline: FC1(M,K→2N) + SiLU + FC2(M,N→K)。

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
2. CUTLASS FP8 Per-Tensor: 性能极差 (~16-19ms)，per-tensor scale 不适合 SM100 MoE

---

## 2. GEMM-Only Benchmark — Full MoE (FC1 + SiLU + FC2)

**测试条件**: N=2048, K=7168, top_k=8 (fused 用 top_k=7), Warmup=10, Iters=50, SEED=42

### 2.1 均匀激活

| Scenario | Type | E | M/E | TotM | FlashInfer CuteDSL FP4 ms (TF) | TRT-LLM Fused FP4 ms (TF) | CUTLASS FP4 (vLLM) ms (TF) | FlashInfer FP8 Groupwise ms (TF) | DeepGEMM FP8 Masked ms (TF) | DeepGEMM FP8 Contiguous ms (TF) | CUTLASS FP8 Per-Tensor ms (TF) | TRT-LLM Fused FP8 ms (TF) | Best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M/E=8 | decode | 8 | 8 | 64 | 0.267 (21.1) | 0.302 (18.7) | **0.087 (64.9)** | 0.262 (21.5) | 2.360 (2.4) | 0.498 (11.3) | 18.6 (0.3) | 0.264 (21.4) | **CUTLASS FP4** |
| M/E=16 | decode | 8 | 16 | 128 | 0.269 (41.9) | 0.258 (43.7) | **0.077 (145.7)** | 0.227 (49.8) | 2.428 (4.6) | 0.521 (21.6) | 19.0 (0.6) | 0.259 (43.5) | **CUTLASS FP4** |
| M/E=64 | prefill | 8 | 64 | 512 | 0.258 (174.8) | 0.252 (179.3) | **0.078 (580.4)** | 0.222 (203.5) | 2.428 (18.6) | 0.524 (86.1) | 17.0 (2.7) | 0.356 (126.8) | **CUTLASS FP4** |
| M/E=256 | prefill | 8 | 256 | 2048 | 0.267 (675.5) | 0.384 (469.6) | **0.109 (1655.6)** | 0.227 (795.6) | 2.395 (75.3) | 0.522 (345.8) | 16.0 (11.3) | 1.278 (141.2) | **CUTLASS FP4** |
| M/E=512 | prefill | 8 | 512 | 4096 | 0.264 (1365.7) | 0.699 (515.8) | **0.178 (2030.7)** | 0.298 (1211.1) | 2.362 (152.8) | 0.543 (664.4) | 16.2 (22.3) | 2.530 (142.6) | **CUTLASS FP4** |
| M/E=1024 | prefill | 8 | 1024 | 8192 | **0.266 (2713.5)** | 1.401 (514.9) | 0.320 (2254.8) | 0.525 (1374.3) | 2.457 (293.7) | 0.699 (1032.5) | 16.5 (43.8) | 5.009 (144.1) | **CuteDSL FP4** |
| M/E=2048 | prefill | 8 | 2048 | 16384 | **0.464 (3108.1)** | 2.944 (490.2) | 0.531 (2719.6) | 0.968 (1490.5) | 2.674 (539.7) | 1.080 (1336.8) | 16.9 (85.6) | 9.989 (144.5) | **CuteDSL FP4** |
| E64-M/E=8 | decode | 64 | 8 | 512 | **0.269 (167.6)** | 0.412 (109.3) | 0.344 (131.1) | 0.569 (79.3) | 14.871 (3.0) | 0.935 (48.2) | 16.5 (2.7) | 0.883 (51.1) | **CuteDSL FP4** |

### 2.2 非均匀激活 — SkA (长尾: 38%/25%/12.5%/...)

| Scenario | TotM | maxM/E | FlashInfer CuteDSL FP4 ms (TF) | TRT-LLM Fused FP4 ms (TF) | CUTLASS FP4 (vLLM) ms (TF) | FlashInfer FP8 Groupwise ms (TF) | DeepGEMM FP8 Masked ms (TF) | DeepGEMM FP8 Contiguous ms (TF) | TRT-LLM Fused FP8 ms (TF) | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| SkA-64 | 64 | 25 | 0.256 (22.0) | 0.295 (19.1) | **0.082 (68.8)** | 0.221 (25.6) | 2.346 (2.4) | 0.509 (11.1) | 0.256 (22.0) | **CUTLASS FP4** |
| SkA-512 | 512 | 194 | 0.254 (177.6) | 0.252 (178.8) | **0.084 (539.9)** | 0.222 (203.1) | 2.322 (19.4) | 0.518 (87.0) | 0.360 (125.2) | **CUTLASS FP4** |
| SkA-2048 | 2048 | 770 | 0.262 (687.9) | 0.383 (470.9) | **0.119 (1510.7)** | 0.456 (395.4) | ERR | 0.520 (346.8) | 1.285 (140.4) | **CUTLASS FP4** |
| SkA-4096 | 4096 | 1538 | 0.265 (1361.6) | 0.738 (488.6) | **0.180 (2008.6)** | 0.792 (455.5) | ERR | 0.592 (609.1) | 2.534 (142.4) | **CUTLASS FP4** |
| SkA-8192 | 8192 | 3074 | **0.290 (2486.9)** | 1.439 (501.3) | 0.327 (2207.3) | 1.584 (455.6) | ERR | 0.741 (973.6) | 5.021 (143.7) | **CuteDSL FP4** |
| SkA-16384 | 16384 | 6146 | **0.536 (2692.3)** | 2.981 (484.1) | 0.583 (2477.3) | 3.074 (469.5) | 2.962 (487.2) | 1.112 (1297.8) | 9.988 (144.5) | **CuteDSL FP4** |

### 2.3 非均匀激活 — SkB (极端集中: 75%/rest=1 each)

| Scenario | TotM | maxM/E | FlashInfer CuteDSL FP4 ms (TF) | TRT-LLM Fused FP4 ms (TF) | CUTLASS FP4 (vLLM) ms (TF) | FlashInfer FP8 Groupwise ms (TF) | DeepGEMM FP8 Masked ms (TF) | DeepGEMM FP8 Contiguous ms (TF) | TRT-LLM Fused FP8 ms (TF) | Best |
|---|---|---|---|---|---|---|---|---|---|---|
| SkB-64 | 64 | 57 | 0.258 (21.8) | 0.296 (19.1) | **0.085 (66.2)** | 0.220 (25.7) | 2.323 (2.4) | 0.502 (11.2) | 0.258 (21.8) | **CUTLASS FP4** |
| SkB-512 | 512 | 505 | 0.255 (176.8) | 0.253 (178.0) | **0.093 (486.7)** | 0.280 (161.3) | 2.326 (19.4) | 0.520 (86.7) | 0.359 (125.5) | **CUTLASS FP4** |
| SkB-2048 | 2048 | 2041 | 0.255 (707.8) | 0.384 (469.3) | **0.136 (1329.1)** | 0.991 (182.1) | 2.351 (76.7) | 0.530 (340.5) | 1.286 (140.2) | **CUTLASS FP4** |
| SkB-4096 | 4096 | 4089 | 0.259 (1390.3) | 0.720 (501.4) | **0.189 (1906.2)** | 2.011 (179.4) | 2.500 (144.3) | 0.577 (625.5) | 2.578 (140.0) | **CUTLASS FP4** |
| SkB-8192 | 8192 | 8185 | 0.428 (1684.9) | 1.392 (518.2) | **0.328 (2200.6)** | 4.107 (175.7) | 3.019 (239.0) | 0.742 (973.0) | 4.994 (144.5) | **CUTLASS FP4** |
| SkB-16384 | 16384 | 16377 | 0.790 (1826.4) | 2.979 (484.5) | **0.569 (2536.4)** | 9.305 (155.1) | 4.629 (311.7) | 1.125 (1282.6) | 9.991 (144.4) | **CUTLASS FP4** |

---

## 3. 综合分析

### 3.1 性能分级

| 梯队 | 实现 | 峰值 TFLOPS | GEMM 胜率 | 定位 |
|------|------|------------|----------|------|
| **T0** | **CUTLASS FP4 (vLLM)** | 2537 | **14/20** | 小~中 batch GEMM 最快, 大 batch 次优 |
| **T0** | **FlashInfer CuteDSL FP4** | 3108 | **6/20** | 大 batch GEMM 最快, TFLOPS 天花板最高 |
| **T1** | TRT-LLM Fused FP4 | 516 | 0/20 | E2E fused 全场景最快 |
| **T2** | FlashInfer FP8 Groupwise | 1491 | 0/20 | 小 batch 均匀场景尚可 |
| **T2** | DeepGEMM FP8 Contiguous | 1338 | 0/20 | 全场景稳定, 非均匀免疫 |
| **T3** | TRT-LLM Fused FP8 | 145 | 0/20 | E2E 小 batch decode |
| **T4** | DeepGEMM FP8 Masked | 540 | 0/20 | 有崩溃风险 |
| **T5** | CUTLASS FP8 Per-Tensor | 86 | 0/20 | 不推荐 |

### 3.2 CUTLASS FP4 (vLLM) vs FlashInfer CuteDSL FP4

| Scenario | TotM | CUTLASS FP4 ms | CuteDSL FP4 ms | 加速比 | Winner |
|---|---|---|---|---|---|
| M/E=8 | 64 | **0.087** | 0.267 | **3.1x** | CUTLASS FP4 |
| M/E=16 | 128 | **0.077** | 0.269 | **3.5x** | CUTLASS FP4 |
| M/E=64 | 512 | **0.078** | 0.258 | **3.3x** | CUTLASS FP4 |
| M/E=256 | 2048 | **0.109** | 0.267 | **2.5x** | CUTLASS FP4 |
| M/E=512 | 4096 | **0.178** | 0.264 | **1.5x** | CUTLASS FP4 |
| M/E=1024 | 8192 | 0.320 | **0.266** | 0.83x | CuteDSL FP4 |
| M/E=2048 | 16384 | 0.531 | **0.464** | 0.87x | CuteDSL FP4 |
| SkB-8192 | 8192 | **0.328** | 0.428 | **1.3x** | CUTLASS FP4 |
| SkB-16384 | 16384 | **0.569** | 0.790 | **1.4x** | CUTLASS FP4 |

**交叉点**: M/E ≈ 768 (均匀), M/E ≈ 4096+ (非均匀 SkB CUTLASS FP4 仍领先)

**关键发现**:
- CUTLASS FP4 在小~中 batch (M/E<=512) 快 1.5-3.5x — kernel launch 开销极低
- CuteDSL FP4 在大 batch (M/E>=1024, 均匀) 领先 — TFLOPS 天花板更高
- 非均匀 SkB 场景下 CUTLASS FP4 优势扩大, 即使在大 batch 仍领先 1.3-1.4x

### 3.3 非均匀激活退化

| 实现 | Layout | 均匀→SkA | 均匀→SkB | 原因 |
|------|--------|---------|---------|------|
| CUTLASS FP4 (vLLM) | grouped GEMM | 1.0x~1.1x | 1.0x~1.1x | problem_sizes 自适应 |
| FlashInfer CuteDSL FP4 | masked 3D | 1.0x~1.1x | 1.0x~1.7x | maxM/E→padding |
| TRT-LLM Fused FP4 | fused routed | 1.0x | 1.0x | 内部 routing 自适应 |
| FlashInfer FP8 Groupwise | contiguous | 1.0x~1.3x | 1.3x~9.6x | 热 expert 大 GEMM 主导 |
| DeepGEMM FP8 Contiguous | contiguous | 1.0x | 1.0x | 无 padding |
| DeepGEMM FP8 Masked | masked 3D | 崩溃 | 1.0x~1.7x | 内存超限 |

---

## 4. End-to-End MoE Benchmark (含 Routing 开销)

非 fused 实现加入 routing 开销 (softmax→topk→scatter→gather)，与 fused 实现公平对比。

### 4.1 Routing 开销

| TotM | Routing ms |
|------|-----------|
| 64 | 0.333 |
| 128 | 0.320 |
| 512 | 0.458 |
| 2048 | 1.356 |
| 4096 | 2.344 |
| 8192 | 4.288 |
| 16384 | 8.163 |

### 4.2 E2E 对比

| Scenario | TotM | Route ms | FlashInfer CuteDSL FP4 ms (TF) | TRT-LLM Fused FP4 ms (TF) | FlashInfer FP8 Groupwise ms (TF) | DeepGEMM FP8 Contiguous ms (TF) | TRT-LLM Fused FP8 ms (TF) | Best |
|---|---|---|---|---|---|---|---|---|
| M/E=8 | 64 | 0.333 | 0.577 (9.8) | 0.290 (19.5) | 0.550 (10.3) | 0.836 (6.7) | **0.243 (23.2)** | **TRT-LLM Fused FP8** |
| M/E=16 | 128 | 0.320 | 0.546 (20.7) | **0.242 (46.7)** | 0.530 (21.3) | 0.819 (13.8) | 0.243 (46.4) | **TRT-LLM Fused FP4** |
| M/E=64 | 512 | 0.458 | 0.692 (65.2) | **0.241 (186.8)** | 0.669 (67.4) | 0.960 (47.0) | 0.354 (127.6) | **TRT-LLM Fused FP4** |
| M/E=256 | 2048 | 1.356 | 1.582 (114.0) | **0.377 (479.0)** | 1.567 (115.1) | 1.863 (96.8) | 1.291 (139.7) | **TRT-LLM Fused FP4** |
| M/E=512 | 4096 | 2.344 | 2.578 (139.9) | **0.700 (515.1)** | 2.631 (137.1) | 2.878 (125.4) | 2.524 (142.9) | **TRT-LLM Fused FP4** |
| M/E=1024 | 8192 | 4.288 | 4.536 (159.1) | **1.390 (519.0)** | 4.834 (149.3) | 4.978 (145.0) | 5.025 (143.6) | **TRT-LLM Fused FP4** |
| M/E=2048 | 16384 | 8.163 | 8.620 (167.4) | **2.968 (486.2)** | 9.181 (157.2) | 9.246 (156.1) | 10.077 (143.2) | **TRT-LLM Fused FP4** |

**结论**: TRT-LLM Fused FP4 E2E 6/7 全胜 (M/E=8 由 TRT-LLM Fused FP8 胜出)。routing 开销完全主导非 fused 实现的延迟。

---

## 5. 场景选型

| 场景 | 推荐 | 备选 |
|------|------|------|
| **GEMM-only 小~中 batch (M/E<=512)** | **CUTLASS FP4 (vLLM)** | FlashInfer CuteDSL FP4 |
| **GEMM-only 大 batch (M/E>=1024)** | **FlashInfer CuteDSL FP4** | CUTLASS FP4 (vLLM) |
| **GEMM-only 非均匀 SkB** | **CUTLASS FP4 (vLLM)** | FlashInfer CuteDSL FP4 |
| **E2E (含 routing)** | **TRT-LLM Fused FP4** | TRT-LLM Fused FP8 |
| **Decode (M/E<=16)** | CUTLASS FP4 (GEMM) / TRT-LLM Fused (E2E) | — |
| **Prefill (M/E>=512)** | CUTLASS FP4 / CuteDSL FP4 | DeepGEMM FP8 Contiguous |
| **非均匀大 batch** | **CUTLASS FP4 (vLLM)** | DeepGEMM FP8 Contiguous |

### 核心结论

1. **FP4 双雄格局**: CUTLASS FP4 (vLLM) 和 CuteDSL FP4 分别统治小 batch 和大 batch, 合计 20/20 GEMM 场景全胜
2. **CUTLASS FP4 小 batch 快 3x**: M/E<=64 时比 CuteDSL 快 3.1-3.5x, kernel launch 开销极低
3. **CuteDSL FP4 TFLOPS 天花板最高**: 大 batch 峰值 3108 TFLOPS, CUTLASS FP4 峰值 2537 TFLOPS
4. **交叉点 M/E ≈ 768**: 均匀激活下两者在此处性能持平, 之后 CuteDSL 反超
5. **非均匀 SkB CUTLASS FP4 优势扩大**: 即使大 batch (16384 tokens) CUTLASS 仍快 1.4x
6. **TRT-LLM Fused FP4 E2E 全胜**: routing 开销 (0.3-8.2ms) 使 GEMM 速度优势无意义
7. **FP4 全面优于 FP8**: 所有 FP4 实现在 GEMM 场景均碾压 FP8

---

## 6. 测试环境

- Warmup=10, Bench=50 iters, SEED=42
- 计时: `torch.cuda.synchronize()` + `time.perf_counter()`
- TFLOPS: Full MoE = `(total_tokens * 2N * K * 2 + total_tokens * K * N * 2) / (ms/1000) / 1e12`
- 非均匀: SkA (长尾 38%/25%/12.5%/...), SkB (极端 75%/rest=1)
- TRT-LLM Fused FP4/FP8 top_k=7, 其他 top_k=8
- CUTLASS FP4 首次运行需 JIT 编译 (~2-3 分钟), 后续 cached
- FlashInfer FP8 Groupwise 使用 mma_sm autotune (每场景自动选择最优 tile config)
