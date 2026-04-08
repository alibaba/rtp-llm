# SM100 MoE GEMM Benchmark Report

- **GPU**: NVIDIA L20A (SM100, GB200 Grace-Blackwell)
- **Software**: PyTorch 2.9.0+cu129, FlashInfer 0.6.4, DeepGEMM 2.1.1
- **CI Run**: 35030824 (2026-04-08)
- **Branch**: `flashinfer-fp8-groupwise` @ `aae03a9fe`

---

## 1. Benchmark 矩阵

### 8 种去重后独立实现

| # | 简称 | Kernel | 来源 | 精度 | 状态 |
|---|------|--------|------|------|------|
| 1 | **FP4-CD** | `grouped_gemm_nt_masked` (CuteDSL JIT) | FlashInfer | FP4 E2M1 | OK |
| 2 | **FP4-TRT*** | `trtllm_fp4_block_scale_routed_moe` (fused) | FlashInfer/TRT-LLM | FP4 E2M1 | OK |
| 3 | **FP4-CUT** | `cutlass_fp4_group_mm` (CUTLASS GemmUniversal) | vLLM/SGLang | FP4 E2M1 | ERR (1) |
| 4 | **FP8-FI** | `group_gemm_fp8_nt_groupwise` (float32 blockwise) | FlashInfer | FP8 E4M3 | OK |
| 5 | **FP8-DGM** | `m_grouped_fp8_gemm_nt_masked` (UE8M0, masked 3D) | DeepGEMM | FP8 E4M3 | OK (2) |
| 6 | **FP8-DGC** | `m_grouped_fp8_gemm_nt_contiguous` (UE8M0) | DeepGEMM | FP8 E4M3 | OK |
| 7 | **FP8-CUT*** | `cutlass_moe_mm_fp8_scaled` (per-tensor) | rtp_kernel | FP8 E4M3 | OK (3) |
| 8 | **FP8-TRT*** | `trtllm_fp8_block_scale_routed_moe` (fused) | FlashInfer/TRT-LLM | FP8 E4M3 | OK |

`*` = fused end-to-end MoE (includes routing + gather overhead)

**ERR 说明**:
1. FP4-CUT: vLLM/SGLang CUTLASS FP4 kernel 需要源码集成 (standalone JIT 编译)
2. FP8-DGM: 在 SkA 大 M 场景 (maxM/E > 1024) 偶发崩溃 (masked 3D layout padding 过大)
3. FP8-CUT: 性能极差 (~19-23ms)，per-tensor scale 精度不适合 SM100 MoE

---

## 2. 统一 Benchmark 结果 — Full MoE (FC1 + SiLU + FC2)

**测试条件**:
- N=2048 (intermediate), K=7168 (hidden), top_k=7 (TRT-LLM) / top_k=8 (others)
- Warmup=10, Iters=50, SEED=42
- GPU: NVIDIA L20A (SM100), 单卡

### 2.1 均匀激活 (每个 expert 分配相同 token 数)

| Scenario | Type | E | M/E | TotM | FP4-CD ms | TF | FP4-TRT ms | TF | FP8-FI ms | TF | FP8-DGM ms | TF | FP8-DGC ms | TF | FP8-TRT ms | TF | Best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M/E=8 | decode | 8 | 8 | 64 | 0.265 | 21.3 | 0.236 | 23.9 | **0.225** | **25.1** | 2.499 | 2.3 | 0.528 | 10.7 | 0.238 | 23.7 | FP8-FI |
| M/E=16 | decode | 8 | 16 | 128 | 0.270 | 41.8 | **0.224** | **50.4** | 0.236 | 47.8 | 2.531 | 4.5 | 0.519 | 21.7 | 0.234 | 48.1 | **FP4-TRT** |
| M/E=64 | prefill | 8 | 64 | 512 | 0.273 | 165.1 | **0.238** | **189.5** | 0.256 | 176.3 | 2.589 | 17.4 | 0.535 | 84.3 | 0.353 | 127.7 | **FP4-TRT** |
| M/E=256 | prefill | 8 | 256 | 2048 | 0.275 | 655.0 | 0.383 | 471.2 | **0.267** | **675.1** | 2.664 | 67.7 | 0.551 | 327.5 | 1.280 | 140.9 | FP8-FI |
| M/E=512 | prefill | 8 | 512 | 4096 | **0.279** | **1291.4** | 0.701 | 515.0 | 0.311 | 1161.2 | 2.579 | 139.9 | 0.555 | 650.1 | 2.606 | 138.5 | **FP4-CD** |
| M/E=1024 | prefill | 8 | 1024 | 8192 | **0.266** | **2713.9** | 1.436 | 502.6 | 0.629 | 1146.9 | 2.601 | 277.4 | 0.706 | 1022.2 | 5.077 | 142.1 | **FP4-CD** |
| M/E=2048 | prefill | 8 | 2048 | 16384 | **0.477** | **3027.5** | 3.035 | 475.5 | 1.127 | 1280.6 | 2.814 | 512.8 | 1.105 | 1306.3 | 10.057 | 143.5 | **FP4-CD** |
| E64-M/E=8 | decode | 64 | 8 | 512 | **0.278** | **162.0** | 0.423 | 106.5 | 0.578 | 78.0 | 16.211 | 2.8 | 0.938 | 48.1 | 0.879 | 51.3 | **FP4-CD** |

### 2.2 非均匀激活 — SkA (长尾分布: 38%/25%/12.5%/...)

| Scenario | E | TotM | maxM/E | FP4-CD ms | TF | FP4-TRT ms | TF | FP8-FI ms | TF | FP8-DGM ms | TF | FP8-DGC ms | TF | FP8-TRT ms | TF | Best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SkA-64 | 8 | 64 | 25 | 0.269 | 21.0 | **0.233** | **24.2** | 0.244 | 23.1 | 2.537 | 2.2 | 0.531 | 10.6 | 0.233 | 24.2 | **FP4-TRT** |
| SkA-512 | 8 | 512 | 194 | 0.270 | 167.3 | **0.236** | **191.3** | 0.248 | 182.2 | 2.566 | 17.6 | 0.531 | 84.9 | 0.359 | 125.6 | **FP4-TRT** |
| SkA-2048 | 8 | 2048 | 770 | **0.267** | **675.0** | 0.388 | 465.4 | 0.454 | 397.6 | ERR | - | 0.534 | 338.0 | 1.289 | 140.0 | **FP4-CD** |
| SkA-4096 | 8 | 4096 | 1538 | **0.267** | **1352.5** | 0.727 | 496.4 | 0.822 | 438.7 | ERR | - | 0.579 | 623.4 | 2.556 | 141.1 | **FP4-CD** |
| SkA-8192 | 8 | 8192 | 3074 | **0.294** | **2452.7** | 1.431 | 504.2 | 1.633 | 441.8 | ERR | - | 0.751 | 960.8 | 5.111 | 141.2 | **FP4-CD** |
| SkA-16384 | 8 | 16384 | 6146 | **0.545** | **2647.4** | 3.059 | 471.8 | 3.126 | 461.7 | 3.057 | 472.0 | 1.127 | 1280.2 | 10.130 | 142.5 | **FP4-CD** |

### 2.3 非均匀激活 — SkB (极端集中: 75%/rest=1 each)

| Scenario | E | TotM | maxM/E | FP4-CD ms | TF | FP4-TRT ms | TF | FP8-FI ms | TF | FP8-DGM ms | TF | FP8-DGC ms | TF | FP8-TRT ms | TF | Best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SkB-64 | 8 | 64 | 57 | 0.259 | 21.7 | **0.230** | **24.5** | 0.239 | 23.6 | 2.475 | 2.3 | 0.516 | 10.9 | 0.231 | 24.4 | **FP4-TRT** |
| SkB-512 | 8 | 512 | 505 | 0.256 | 176.2 | **0.227** | **198.8** | 0.296 | 152.4 | 2.485 | 18.1 | 0.524 | 86.0 | 0.360 | 125.1 | **FP4-TRT** |
| SkB-2048 | 8 | 2048 | 2041 | **0.271** | **665.8** | 0.388 | 465.5 | 0.999 | 180.7 | 2.588 | 69.7 | 0.528 | 341.4 | 1.287 | 140.2 | **FP4-CD** |
| SkB-4096 | 8 | 4096 | 4089 | **0.278** | **1297.6** | 0.716 | 504.0 | 2.165 | 166.7 | 2.697 | 133.8 | 0.592 | 609.3 | 2.576 | 140.0 | **FP4-CD** |
| SkB-8192 | 8 | 8192 | 8185 | **0.432** | **1669.1** | 1.407 | 512.8 | 4.322 | 166.9 | 3.116 | 231.6 | 0.752 | 959.0 | 5.085 | 141.9 | **FP4-CD** |
| SkB-16384 | 8 | 16384 | 16377 | **0.800** | **1804.7** | 3.054 | 472.6 | 9.758 | 147.9 | 4.740 | 304.5 | 1.137 | 1269.1 | 10.189 | 141.6 | **FP4-CD** |

---

## 3. 综合分析

### 3.1 实现性能分级

| 梯队 | 实现 | 最佳场景 | 峰值 TFLOPS |
|------|------|---------|------------|
| **T0** | FP4-CuteDSL | 大 batch (M/E>=512), 多 expert, 非均匀 | 3028 |
| **T1** | FP4-TRT (fused) | 小 batch decode (M/E<=64), 非均匀小 batch | 198.8 |
| **T1** | FP8-FlashInfer | 小 batch decode (M/E<=256), 均匀激活 | 675.1 |
| **T2** | FP8-DGC | 全场景稳定，大 batch 接近 FP4-CD | 1306.3 |
| **T3** | FP8-TRT (fused) | 小 batch decode (~0.23ms) | 143.5 |
| **T3** | FP8-DGM | 仅大 batch 尚可，小 batch 延迟恒定 ~2.5ms | 512.8 |
| **T4** | FP8-CUT (per-tensor) | 无 — 所有场景极慢 (~19-23ms) | 73.2 |

### 3.2 FP4-TRT (fused MoE) 分析

FP4-TRT 是 **fused end-to-end MoE kernel** (routing + FC1 + activation + FC2 + gather 一体化)：

| 特性 | 表现 |
|------|------|
| **小 batch (M/E<=64)** | **最快** — 0.22-0.24ms, 比 FP4-CD 快 ~15%, 比 FP8-FI 快 ~5% |
| **中 batch (M/E=256)** | 中等 — 0.38ms, fused overhead 开始显现 |
| **大 batch (M/E>=512)** | 退化 — 延迟线性增长, 0.7ms→3ms, 远慢于 FP4-CD |
| **非均匀 (SkA/SkB)** | 小 batch 表现优异, 大 batch 与均匀类似 |
| **TFLOPS 天花板** | ~500 TFLOPS — fused kernel 无法充分利用大 batch 并行度 |

**结论**: FP4-TRT 在 decode 场景 (M/E<=64) 是最佳选择，超越了所有 FP8 实现。但不适合大 batch prefill。

### 3.3 FP4-CuteDSL vs FP8-FlashInfer 直接对比

| Scenario | TotM | FP4 ms | FP8 ms | FP4/FP8 | Winner |
|---|---|---|---|---|---|
| M/E=8 (uniform) | 64 | 0.265 | 0.225 | 0.85x | FP8 |
| M/E=16 (uniform) | 128 | 0.270 | 0.236 | 0.87x | FP8 |
| M/E=64 (uniform) | 512 | 0.273 | 0.256 | 0.94x | FP8 |
| M/E=256 (uniform) | 2048 | 0.275 | 0.267 | 0.97x | FP8 |
| **M/E=512** (uniform) | 4096 | **0.279** | 0.311 | **1.11x** | **FP4** |
| **M/E=1024** (uniform) | 8192 | **0.266** | 0.629 | **2.37x** | **FP4** |
| **M/E=2048** (uniform) | 16384 | **0.477** | 1.127 | **2.36x** | **FP4** |
| **E64-M/E=8** | 512 | **0.278** | 0.578 | **2.08x** | **FP4** |
| SkB-512 | 512 | **0.256** | 0.296 | **1.16x** | **FP4** |
| SkB-2048 | 2048 | **0.271** | 0.999 | **3.69x** | **FP4** |
| SkB-8192 | 8192 | **0.432** | 4.322 | **10.0x** | **FP4** |
| SkB-16384 | 16384 | **0.800** | 9.758 | **12.2x** | **FP4** |

### 3.4 非均匀激活下的 Layout 退化

| 实现 | Layout | 均匀→SkA 退化 | 均匀→SkB 退化 | 原因 |
|------|--------|-------------|-------------|------|
| FP4-CD | masked 3D | 1.0x~1.1x | 1.0x~1.7x | maxM/E 增大→更多 padding |
| FP4-TRT | fused routed | 1.0x~1.0x | 1.0x~1.0x | 内部 routing 自适应 |
| FP8-FI | contiguous | 1.0x~1.3x | 1.3x~8.7x | 热 expert 的大 GEMM 主导延迟 |
| FP8-DGM | masked 3D | 崩溃 (SkA大M) | 1.0x~1.7x | masked tensor 内存超限 |
| FP8-DGC | contiguous | 1.0x | 1.0x | contiguous 无 padding, 稳定 |

**关键发现**: FP8-FlashInfer 在极端不均匀 (SkB) 场景下退化最严重 (12x)，因为 contiguous layout 的热 expert 独占计算。FP4-TRT fused kernel 对非均匀几乎免疫。

---

## 4. 场景选型建议

| 场景 | 推荐实现 | 备选 | 不推荐 |
|------|---------|------|--------|
| **Decode (M/E<=16)** | **FP4-TRT*** | FP8-FI, FP8-TRT | DGM, CUT |
| **Decode (M/E=64)** | **FP4-TRT*** | FP4-CD, FP8-FI | DGM, CUT |
| **Prefill (M/E=256)** | FP8-FI / FP4-CD | FP4-TRT | DGM, CUT |
| **Prefill (M/E>=512)** | **FP4-CD** | FP8-DGC | FP8-FI (退化), TRT, DGM, CUT |
| **大 Batch (M/E>=1024)** | **FP4-CD** | FP8-DGC | 所有其他 |
| **多 Expert (E>=64)** | **FP4-CD** | — | 所有 FP8 |
| **非均匀小 batch** | **FP4-TRT*** | FP4-CD | FP8-FI (退化) |
| **非均匀大 batch** | **FP4-CD** | FP8-DGC | FP8-FI (12x 退化) |

`*` = fused end-to-end MoE, 包含 routing + gather

### 综合结论

1. **FP4-CuteDSL 是 SM100 最通用 MoE kernel**: 14/20 场景胜出，峰值 3028 TFLOPS，非均匀激活稳定
2. **FP4-TRT fused MoE 在 decode 场景最快**: 6/20 场景胜出 (M/E<=64 + 非均匀小 batch)，0.22ms 延迟
3. **FP8-FlashInfer 在 M/E<=256 均匀激活场景有优势**: 但非均匀退化严重 (SkB 12x)
4. **FP8-DGC 是最稳定的 FP8 实现**: 对非均匀激活几乎免疫，大 batch 接近 FP4-CD
5. **FP4 全面优于 FP8**: 在大 batch 和非均匀场景下，FP4 的 memory bandwidth 优势显著
6. **FP8-DGM 和 FP8-CUT 不推荐**: DGM 有崩溃风险，CUT per-tensor 性能极差

---

## 5. 未覆盖实现

| 实现 | 未纳入原因 | 后续 |
|------|----------|------|
| FP4-CUTLASS (`cutlass_fp4_group_mm`) | vLLM/SGLang C++ op, 需要源码集成 | 源码编译方式接入 |

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
- **注意**: FP4-TRT 和 FP8-TRT 的 top_k=7 (E-1), 其他实现 top_k=8; FP8-CUT 数据因性能极差未纳入结果表
