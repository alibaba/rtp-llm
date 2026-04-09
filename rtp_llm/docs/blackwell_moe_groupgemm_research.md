# Blackwell (B200/B300/GB200) MoE GroupGEMM 技术调研报告

> 调研日期: 2026-04-05
> 调研范围: CUTLASS 4.x, FlashInfer, vLLM, SGLang, TensorRT-LLM
> 目标硬件: SM100 (B200), SM103 (B300), SM120 (GB200 GeForce)

---

## 1. 背景与目标

RTP-LLM 当前 MoE 内核栈覆盖 SM70–SM90（CUTLASS 2.x/3.x + DeepGEMM），SM100 路径仅通过 DeepGEMM 的 UE8M0 FP8 block-scale 路径（masked/contiguous）支持。本次调研旨在摸底业界各框架在 Blackwell GPU 上的 MoE GroupGEMM 实现方案，为 RTP-LLM 引入最优 SM100 内核提供技术选型依据。

---

## 2. Blackwell 硬件特性与 MoE 相关优化

### 2.1 关键硬件特性

| 特性 | 说明 | MoE 收益 |
|------|------|---------|
| **tcgen05 MMA** | 新一代 Tensor Core 指令，支持 2SM 协同 MMA | Tile M 从 128→256，prefill 吞吐翻倍 |
| **TMEM (Tensor Memory)** | SM 内部专属片上存储，独立于 SMEM | MMA 累加器直接存在 TMEM，减少 register pressure |
| **TMA 2.0** | 支持 on-device descriptor 更新、multicast across cluster | 权重矩阵（固定 shape）用 TMA，token 矩阵（变长）用 CPASYNC |
| **CPASYNC** | `cp.async` 直接从 GMEM 加载到 SMEM，无需 TMA descriptor | 避免 decode 阶段频繁更新 TMA descriptor 的开销 |
| **CLC (Cluster Launch Control)** | 硬件辅助的 cluster 级 pipeline 调度 | Scheduler warp 可前瞻多个 wave，提高调度效率 |
| **PDL (Programmatic Dependent Launch)** | `cudaLaunchKernelEx` 串联相依 kernel，无需 host barrier | FC1→FC2 之间 launch 延迟降低 |
| **UE8M0 Scale** | Unsigned E8M0 格式的 block quantization scale | FP8 per-block 量化精度与性能的新平衡点 |
| **NVFP4 / MXFP4** | 4-bit 浮点（E2M1）+ block scale | 模型存储减半，MoE 权重带宽瓶颈缓解 |

### 2.2 MoE 场景的核心挑战

**Prefill 阶段**: token 数多（128–4096+），每个 expert 分到的 token 较多，需要最大化计算吞吐。关键优化: 2SM MMA、cluster multicast、大 tile。

**Decode 阶段**: token 数少（1–64），每个 expert 只分到几个 token，N 维度极小且不均匀。关键挑战:
- TMA descriptor 需要按 expert 更新 → 开销大 → 用 CPASYNC 替代
- Tile N 需要动态缩小（16 甚至更小）→ 小 tile 配置
- Kernel launch overhead → 用 PDL 减少

---

## 3. 各框架 Blackwell MoE GEMM 方案详解

### 3.1 CUTLASS (Example 92: Blackwell MoE GEMM)

**代码位置**: `/dev/shm/liukan.lk/cutlass/examples/92_blackwell_moe_gemm/`

CUTLASS 提供了 6 种 Blackwell MoE GEMM 变体:

| 变体 | 文件 | ElementA/B | 加载策略 | 特点 |
|------|------|-----------|---------|------|
| Regular | `_regular.cu` | FP8 E4M3 | TMA + CPASYNC | A=TMA, B=CPASYNC, Cluster<1,1,1> |
| Grouped | `_grouped.cu` | FP8 E4M3 | TMA + CPASYNC | PtrArray, decode 专用, Tile <128,16,16> |
| FP4 Regular | `_fp4_regular.cu` | NVFP4 (E2M1) | TMA + CPASYNC | BlockScaled, Tile <128,64,256> |
| FP4 Grouped | `_fp4_grouped.cu` | NVFP4 (E2M1) | TMA + CPASYNC | PtrArray BlockScaled |
| RC-Grouped | `_rcgrouped.cu` | FP8 E4M3 | TMA (A) + TMA (B) | Ragged-Contiguous, 2SM 支持, dynamic cluster |
| BS-RC-Grouped | `_blockscaled_rcgrouped.cu` | MXFP8 | TMA + TMA | MX-format block scale, 2SM |

**关键架构决策**:

1. **MoEProblemShape vs GroupProblemShape**:
   - MoE: M 和 K 固定（权重矩阵 shape 相同），N 变化（`tokens_per_expert[]`）
   - Group: M、N、K 都可能变化

2. **RC (Ragged Contiguous) Layout**: 权重矩阵 A 连续存放（单一 flat buffer），每个 expert 的偏移为 `i * M * K`，用 batched TMA 访问。Token 矩阵 B 用 PtrArray（每个 expert 独立指针），因为 N 维度不同。

3. **CPASYNC 路径 (Decode 核心)**:
   ```
   在 MoE 模型的 decode 阶段，不同 expert 的 token 数差异很大，
   需要频繁更新 TMA descriptor。使用 CPASYNC 加载 activation (B) 矩阵
   避免了更新 TMA descriptor 的开销。
   ```
   限制: CPASYNC 不支持 cluster multicast，只能用 `Cluster<1,1,1>`。

4. **Warp 专特化**:
   - Warp 0: Scheduler (CLC pipeline)
   - Warp 1-4: TensorMap Updater (异步 TMA descriptor 更新)
   - Warp 5: MMA (tcgen05 issue)
   - Warp 6: TMA Load (mainloop)
   - Warp 7: Epilogue Load
   - Warp 8+: Epilogue Store

### 3.2 FlashInfer

**代码位置**: `/dev/shm/liukan.lk/flashinfer/`

FlashInfer 提供 4 种 MoE kernel 后端:

#### A. CuteDSL Backend (NVFP4, SM100)

Python DSL JIT 编译的 CUTLASS kernel，使用 `tcgen05.mma.kind.block_scale`。

**FC1 Kernel** (`blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py`):
- 功能: Gather (token permutation) + GEMM + SwiGLU，三步融合
- A 加载: LDGSTS (带 gather 的 GMEM→SMEM)，warp 4-7
- B 加载: TMA with multicast
- MMA tile: 128×N (1CTA) 或 256×N (2CTA)
- N 选项: 64, 128, 192, 256
- 输出可选 NVFP4 量化 + scale factor 生成

**FC2 Kernel** (`blockscaled_contiguous_grouped_gemm_finalize_fusion.py`):
- 功能: GEMM + Finalize (router weight scale + scatter)，两步融合
- A/B 加载: TMA
- 支持 `use_blkred` 分块 reduce

#### B. TRT-LLM Gen Backend (SM100, 多种 dtype)

从 TensorRT-LLM 移植的预编译 kernel:
- `trtllm_bf16_moe` — BF16
- `trtllm_fp8_block_scale_moe` / `_routed_moe` — FP8 block-scale (DeepSeek 模式)
- `trtllm_fp4_block_scale_moe` / `_routed_moe` — NVFP4
- `trtllm_mxint4_block_scale_moe` — W4A8

支持 auto-tuner，按 token 数 bucket 选择最优 tactic。

#### C. CUTLASS Backend (SM100/SM90/SM89, FP4)

JIT 编译的 CUTLASS FP4 kernel:
- SM100: `gen_cutlass_fused_moe_sm100_module`
- SM103: `gen_cutlass_fused_moe_sm103_module`
- SM120/121: `gen_cutlass_fused_moe_sm120_module`

#### D. FP8 GroupGEMM (SM100)

独立的 GroupGEMM kernel（非 fused MoE）:
- `group_gemm_fp8_nt_groupwise` — FP8 groupwise quantization
- `group_gemm_mxfp4_nt_groupwise` — MXFP4

Tile 配置:
- FP8: 128×128×128 (1SM) 或 256×128×128 (2SM, `mma_sm=2`)
- MXFP4: tile_n ∈ {64, 128, 192, 256}, tile_k ∈ {128, 256}

#### Tuning Configs (预调优数据)

**B200** (`v0_1_trtllm_fused_moe_NVIDIA_B200.py`):
- Tactic 5 = decode 最优 (小 batch)
- Tactic 3 = 中等 prefill
- Tactic 1 = 大 prefill
- 转折点: ~1024-2048 tokens

**GB200** (`v0_1_trtllm_fused_moe_NVIDIA_GB200.py`):
- Token 范围支持到 65536
- 128 experts, hidden=7168

### 3.3 vLLM

**代码位置**: `/dev/shm/liukan.lk/vllm/`

#### SM100 FP8 GroupGEMM (最成熟的生产实现)

**文件**: `csrc/libtorch_stable/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu`

3 种 tile 配置 + 运行时启发式选择:

| Config | TileShape | ClusterShape | Schedule | 选择条件 |
|--------|-----------|-------------|----------|---------|
| Default | `<128,256,128>` | `<1,1,1>` | `KernelPtrArrayTmaWarpSpecialized1SmSm100` | 通用 |
| M64 | `<128,16,128>` | `<1,1,1>` | 同上, **swap_ab=true** | M ≤ 64 (decode) |
| N8192 | `<128,256,128>` | `<2,1,1>` | `KernelPtrArrayTmaWarpSpecialized2SmSm100` | N ≥ 8192 (大 prefill) |

**运行时 dispatch**:
```cpp
if (m <= 64)        → Config_M64   (swap_ab, 小 tile)
else if (n >= 8192) → Config_N8192 (2SM, cluster 2×1×1)
else                → Config_Default
```

编译守卫: `#if defined ENABLE_CUTLASS_MOE_SM100 && ENABLE_CUTLASS_MOE_SM100`

#### MXFP8 SM100 GroupGEMM

**文件**: `csrc/moe/mxfp8_moe/cutlass_mxfp8_grouped_mm.cu`

- ElementA/B: `mx_float8_t<float_e4m3_t>` (MX-format block-scaled FP8)
- Scale: `float_e8m0_t` (UE8M0)
- OpClass: `OpClassBlockScaledTensorOp`
- Tile: `<128,128,128>`, Cluster: `<1,4,1>` preferred / `<1,2,1>` fallback
- Schedule: `KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100`

从 SGLang 移植而来。

#### NVFP4 GroupGEMM

通过 `sgl_kernel.cutlass_fp4_group_mm` 调用，支持 SM100 和 SM120。
Tile: `<128,128,128>`。

#### 小 batch 回退

`TritonOrCutlassExperts` 在 SM100 上 M ≤ 8 时自动回退到 Triton kernel。

### 3.4 SGLang

**代码位置**: `/dev/shm/liukan.lk/sglang/`

SGLang 使用 `MoeRunner` 抽象 + 可插拔后端:

| 后端 | 量化 | SM 支持 |
|------|------|--------|
| Triton (默认) | FP8/INT8/INT4 | 通用 |
| DeepGEMM | FP8 block-scale | SM90 (H100) |
| FlashInfer TRT-LLM | FP8/MXFP8/NVFP4/BF16 | SM90+ |
| CUTLASS (sgl_kernel) | FP8/MXFP8/NVFP4 | SM90/SM100 |

SM100 特有:
- MXFP8: `es_sm100_mxfp8_blockscaled_grouped_mm` (expert specialization)
- NVFP4: `nvfp4_blockwise_moe.cuh` 显式 dispatch SM100/SM103/SM120

### 3.5 TensorRT-LLM

**代码位置**: `/dev/shm/liukan.lk/TensorRT-LLM/`

5 种 MoE kernel 路径:

| 路径 | GPU | 量化 | 特点 |
|------|-----|------|------|
| **CutlassFusedMoE** | SM80+ | BF16/FP8/NVFP4/W4A8/MXFP4 | 经典 CUTLASS, 最广泛 |
| **TRTLLMGenFusedMoE** | SM100 only | FP8-BlockScale/NVFP4/混合精度 | 全融合 (routing+permute+FC1+act+FC2), PDL |
| **DeepGemmFusedMoE** | SM100 | FP8-BlockScale | DeepGEMM 库 |
| **CuteDslFusedMoE** | SM100 | NVFP4 only | Python JIT tcgen05 kernel, gather+SwiGLU fusion |
| **TritonFusedMoE** | SM90 only | BF16/FP8 | Triton 后端 |

**TRTLLMGen 独有特性**:
- 全融合: routing→permute→FC1→activation→FC2→finalize 一条流水线
- `tileTokensDim` 自动调整: 8/16/32/64/128/256，根据 avg_tokens_per_expert 选择
- 6 种 routing 方式: Default, Renormalize, DeepSeekV3, Llama4, MiniMax2, RenormalizeNaive
- `do_finalize=False` 模式: 延迟 scatter-reduce 以 overlap AlltoAll

**CuteDSL 独有特性**:
- FC1 gather fusion: token permutation 融入 LDGSTS，省去独立 permute kernel
- FC2 finalize fusion: scatter + weight scale 融入 epilogue
- DWDP (Data+Weight Double-Pipelining): 多 B 权重预取
- Autotunable tile: 128 / 256

**TRT-LLM 独有 vs FlashInfer**: FlashInfer 已经移植了 TrtllmGen 的预编译 binary，但**没有**移植 CuteDSL kernels。CuteDSL 的 gather+SwiGLU fusion 是 TRT-LLM 独有的高性能路径。

---

## 4. 性能数据参考

### 4.1 已知 Benchmark 工具

| 项目 | Benchmark 文件 | 测量内容 |
|------|--------------|---------|
| FlashInfer | `bench_cutlass_fused_moe.py` | CUTLASS fused MoE (DeepSeek config) |
| FlashInfer | `bench_moe_deepseek.py` | CuteDSL vs CUTLASS vs TRT-LLM |
| FlashInfer | `bench_groupwise_grouped_gemm_fp8_blackwell.py` | FP8 GroupGEMM TFLOPs |
| FlashInfer | `bench_groupwise_grouped_gemm_mxfp4_blackwell.py` | MXFP4 全 tile 配置扫描 |
| FlashInfer | `bench_deepgemm_blackwell.py` | DeepGEMM TFLOPs |
| vLLM | `benchmark_cutlass_moe_fp8.py` | CUTLASS FP8 vs Triton |
| vLLM | `benchmark_cutlass_moe_nvfp4.py` | NVFP4 CUTLASS vs Triton FP8 |
| SGLang | `benchmark_deepgemm_fp8_gemm_blackwell.py` | DeepGEMM vs FlashInfer (B200) |
| SGLang | `bench_nvfp4_blockwise_moe.py` | NVFP4 JIT vs AOT |

### 4.2 各框架预调优数据推断

根据 FlashInfer 的 tuning config 分析:

| 阶段 | Token 数 | 最优 Tactic | 推断含义 |
|------|---------|------------|---------|
| Decode | 1-8 | Tactic 5 | 极小 tile，低 launch overhead |
| Decode | 16-64 | Tactic 5 | 同上 |
| Decode | 128 | Tactic 5/6 | 过渡区间 |
| Prefill | 256-512 | Tactic 3 | 中等 tile，1SM |
| Prefill | 1024-2048 | Tactic 3/1 | 开始切 2SM |
| Prefill | 4096+ | Tactic 1 | 大 tile, 2SM + cluster multicast |

> 注: 具体 TFLOPs 数据需要在 B200 硬件上实际运行 benchmark 获取。

---

## 5. 最佳 Kernel 选型推荐

### 5.1 按阶段 × 数据类型矩阵

| 阶段 | FP8 Block-Scale | MXFP8 | NVFP4 |
|------|----------------|-------|-------|
| **Prefill** | CUTLASS SM100 RC-Grouped 2SM (256×256×64) + dynamic cluster | CUTLASS SM100 blockscaled RC-Grouped (128×256×128, 2SM) | CuteDSL gather+SwiGLU fused (256×128, 2CTA) |
| **Decode** | DeepGEMM masked (已有) 或 CPASYNC (128×16×128) | CUTLASS SM100 CPASYNC (128×16×128) | TrtllmGen FP4 (tactic 5, auto tileN) |
| **来源** | CUTLASS ex92 + vLLM | CUTLASS ex92 + vLLM MXFP8 | FlashInfer/TRT-LLM |

### 5.2 移植优先级

| 优先级 | 内容 | 来源 | 理由 |
|--------|------|------|------|
| P0 | FP8 SM100 PtrArray GroupGEMM (3-config 启发式) | vLLM `grouped_mm_c3x_sm100.cu` | 最成熟的生产实现，直接可用 |
| P0 | NVFP4 CuteDSL 路径验证 | 已有 (FlashInfer) | 现有路径，验证 2CTA 配置 |
| P1 | MXFP8 SM100 GroupGEMM | vLLM `cutlass_mxfp8_grouped_mm.cu` | 新量化格式，移植工作量适中 |
| P1 | FP8 CPASYNC Decode 路径 | CUTLASS ex92 `_grouped.cu` | Decode 极小 N 场景性能关键 |
| P2 | CuteDSL gather+SwiGLU fusion | TRT-LLM CuteDSL | 最高性能但 JIT 编译，集成复杂 |
| P2 | TrtllmGen full fused pipeline | FlashInfer/TRT-LLM | 完全融合但需要大量辅助 kernel |

### 5.3 与 RTP-LLM 现有架构的集成路径

RTP-LLM 现有两条 MoE 执行路径:

**路径 A: C++ CutlassMoeFCRunner** (`CppMoeExecutor`)
- 用于非量化 (fp16/bf16) MoE
- 需要修改 `moe_gemm_kernels_template.h` 中的 `dispatchToArch` 添加 SM100 分支
- 工作量大，但覆盖面广

**路径 B: Python Executor** (DeepGEMM / FlashInfer / CUTLASS binding)
- FP8 per-tensor: `CutlassExpertsFp8` → `rtp_kernel.fp8_group_gemm`
- FP8 per-block: `DeepGemmMaskedExecutor` → `deep_gemm`
- NVFP4: `TrtllmFp4Executor` → `flashinfer.fused_moe`
- 更灵活，新增 kernel 只需新建 executor + strategy

**推荐**: 优先走路径 B，新建 SM100 专属 Python executor，调用:
- `rtp_kernel.fp8_group_gemm` (FP8, 需确认 SM100 编译支持)
- 新建 MXFP8 C++ binding (移植 vLLM 的 `cutlass_mxfp8_grouped_mm`)
- 现有 FlashInfer (NVFP4)

---

## 6. 实施建议

### 6.1 Sprint 1: 快速验证 (1-2 周)

1. 确认 `rtp_kernel.fp8_group_gemm` 在 SM100 上是否可编译运行
2. 添加 B200 默认 tile config JSON 文件
3. 验证现有 NVFP4 路径 (TrtllmFp4Executor / CutedslFp4Executor) 在 SM100 上性能

### 6.2 Sprint 2: FP8 + MXFP8 Kernel 集成 (2-3 周)

1. 若 `rtp_kernel` 不支持 SM100: 移植 vLLM `grouped_mm_c3x_sm100.cu` 为新的 C++ binding
2. 移植 vLLM `cutlass_mxfp8_grouped_mm.cu` 为新的 MXFP8 binding
3. 新建 Python executor + strategy 类

### 6.3 Sprint 3: UT + Benchmark (1 周)

1. 编写 SM100 executor 功能 UT
2. 编写 MoE GroupGEMM 性能 benchmark

### 6.4 Sprint 4: 高级优化 (可选, 2-4 周)

1. C++ `CutlassMoeFCRunner` SM100 dispatch (路径 A)
2. CuteDSL gather+SwiGLU fusion 集成
3. SM120 支持

---

## 7. 参考代码路径索引

### 本地已 clone 仓库

| 项目 | 路径 |
|------|------|
| CUTLASS | `/dev/shm/liukan.lk/cutlass/` |
| FlashInfer | `/dev/shm/liukan.lk/flashinfer/` |
| vLLM | `/dev/shm/liukan.lk/vllm/` |
| SGLang | `/dev/shm/liukan.lk/sglang/` |
| TensorRT-LLM | `/dev/shm/liukan.lk/TensorRT-LLM/` |

### 关键文件索引

#### CUTLASS Blackwell MoE
- `cutlass/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_rcgrouped.cu` — RC-Grouped FP8 (prefill, 2SM)
- `cutlass/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_grouped.cu` — CPASYNC (decode)
- `cutlass/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_blockscaled_rcgrouped.cu` — MXFP8 RC-Grouped
- `cutlass/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_fp4_grouped.cu` — NVFP4
- `cutlass/examples/90_sm103_fp4_ultra_grouped_gemm/` — SM103 (B300) FP4 Ultra
- `cutlass/include/cutlass/gemm/group_array_problem_shape.hpp` — MoEProblemShape 定义
- `cutlass/include/cutlass/gemm/collective/sm100_mma_mixed_tma_cpasync_warpspecialized.hpp` — CPASYNC collective
- `cutlass/include/cutlass/gemm/collective/sm100_mma_array_warpspecialized_rcggemm.hpp` — RC-GEMM collective

#### FlashInfer
- `flashinfer/fused_moe/cute_dsl/blackwell/blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py` — CuteDSL FC1
- `flashinfer/fused_moe/cute_dsl/blackwell/blockscaled_contiguous_grouped_gemm_finalize_fusion.py` — CuteDSL FC2
- `flashinfer/fused_moe/core.py` — MoE Python API (所有后端入口)
- `flashinfer/tuning_configs/v0_1_trtllm_fused_moe_NVIDIA_B200.py` — B200 tuning config
- `flashinfer/tuning_configs/v0_1_trtllm_fused_moe_NVIDIA_GB200.py` — GB200 tuning config
- `flashinfer/include/flashinfer/gemm/group_gemm_fp8_groupwise_sm100.cuh` — FP8 GroupGEMM SM100
- `flashinfer/include/flashinfer/gemm/group_gemm_mxfp4_groupwise_sm100.cuh` — MXFP4 GroupGEMM SM100

#### vLLM
- `vllm/csrc/libtorch_stable/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu` — **SM100 FP8 GroupGEMM (最佳参考)**
- `vllm/csrc/libtorch_stable/quantization/w8a8/cutlass/moe/grouped_mm_c3x.cuh` — 共享模板
- `vllm/csrc/libtorch_stable/quantization/w8a8/cutlass/moe/get_group_starts.cuh` — PtrArray 填充 kernel
- `vllm/csrc/moe/mxfp8_moe/cutlass_mxfp8_grouped_mm.cu` — **MXFP8 SM100 GroupGEMM (最佳参考)**
- `vllm/csrc/moe/mxfp8_moe/cutlass_mxfp8_grouped_mm_traits.cuh` — MXFP8 traits
- `vllm/vllm/model_executor/layers/fused_moe/cutlass_moe.py` — Python dispatch 层

#### SGLang
- `sglang/python/sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh` — NVFP4 SM100/SM120 dispatch
- `sglang/python/sglang/srt/layers/moe/cutlass_moe.py` — CUTLASS MoE dispatch

#### TensorRT-LLM
- `TensorRT-LLM/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h` — TrtllmGen 全融合 runner
- `TensorRT-LLM/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.h` — 辅助 kernel
- `TensorRT-LLM/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/` — CuteDSL kernels
- `TensorRT-LLM/tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py` — CuteDSL Python 后端
- `TensorRT-LLM/tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py` — TrtllmGen Python 后端

#### RTP-LLM 现有 MoE 代码
- `rtp_llm/cpp/cuda/cutlass/cutlass_kernels/moe_gemm/moe_gemm_kernels_template.h` — C++ dispatch (需添加 SM100)
- `rtp_llm/models_py/modules/factory/fused_moe/defs/type.py` — ExecutorType / RouterType 枚举
- `rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/executors/` — 所有 executor 实现
- `rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/strategy/` — 所有策略类
- `rtp_llm/models_py/kernels/cuda/fp8_kernel/fp8_kernel.py` — FP8 GroupGEMM Python wrapper
- `rtp_llm/models_py/kernels/cuda/fp8_kernel/cutlass_groupgemm/` — Tile config JSON files

---

## 附录: 术语表

| 术语 | 全称 | 说明 |
|------|------|------|
| SM100 | Streaming Multiprocessor 100 | Blackwell (B200) 的 compute capability |
| SM103 | Streaming Multiprocessor 103 | B300 的 compute capability |
| SM120 | Streaming Multiprocessor 120 | GB200 GeForce 变体 |
| TMA | Tensor Memory Accelerator | 硬件 DMA 单元，通过 descriptor 描述 tensor layout |
| CPASYNC | Copy Async | `cp.async` 指令，GMEM→SMEM 异步拷贝，无需 descriptor |
| TMEM | Tensor Memory | SM100 新增的片上存储，专供 MMA 使用 |
| CLC | Cluster Launch Control | SM100 的硬件 cluster 级调度 |
| PDL | Programmatic Dependent Launch | CUDA API，kernel 间无 host 同步的串联启动 |
| UE8M0 | Unsigned E8M0 | 8-bit 浮点格式，仅 exponent 无 mantissa，用作 block scale |
| NVFP4 | NVIDIA FP4 | E2M1 格式 4-bit 浮点 + E4M3 block scale (per 16 elements) |
| MXFP4/MXFP8 | Microscaling FP4/FP8 | OCP MX 标准的 block-scaled 格式 (per 32 elements) |
| RC-GEMM | Ragged Contiguous GEMM | 权重连续存储、token PtrArray 的 GroupGEMM 变体 |
| CuteDSL | CUTLASS Python DSL | CUTLASS 4.x 的 Python 代码生成框架 |
