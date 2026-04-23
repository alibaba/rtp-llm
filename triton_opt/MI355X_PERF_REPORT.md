# MI355X chunk-GDN Triton Kernel 性能报告

日期: 2026-04-23
GPU: AMD Instinct MI355X (gfx950, CDNA4, 304 CUs, 160KB LDS/CU)
环境: PyTorch 2.13.0.dev+rocm7.2, Triton 3.6.0
Shape: B=1, T=65536, Hg=8, H=32, DK=DV=128, BT=64 (Qwen3.5-397B TP2, 64K prefill)
生产实测 shape: B=1, T=67174 (基本一致)

## 1. 端到端性能

| Pipeline | 优化前 (us) | 优化后 (us) | 加速比 |
|----------|------------|------------|--------|
| old (3-kernel kkt+solve+recompute) | 5655 | 4336 | 1.30x |
| new (fused_kkt_solve + exp2) | 5196 | **3946** | **1.32x** |
| new vs 原始 old | — | — | **1.43x** |

精度: PASS (mean_rel_err=5.5e-06, max_abs_diff=3.9e-03)

## 2. 优化内容

仅改 Triton launch 配置参数，零算法改动:

### fwd_h (state recurrence, 占总时间 55%)

| 参数 | 改前 | 改后 | 原因 |
|------|------|------|------|
| BV | 32 | **16** | blocks 128→256，CU 利用率 42%→84% |

- VGPR: 108 → 56, occupancy: 4→9 waves/SIMD
- LDS: 36KB → 34KB
- 耗时: 3095 → **2185 us** (-29%)

### fwd_o (output kernel, 占总时间 15%)

| 参数 | 改前 | 改后 | 原因 |
|------|------|------|------|
| BV | 64 | **128** | 减少 V 维循环次数 |
| BK | 128 | **64** | K 维拆两次迭代，降低 tile 大小 |
| num_warps | 4 | **1** | 消除 warp 间同步开销 |

- VGPR: 108 → 216, blocks: 65536 → 32768 (仍然充足)
- 耗时: 917 → **589 us** (-36%)

### fused_kkt_solve

MI355X 与 MI308X 一致: BK=64, warps=1 最优，无需改动。

## 3. 每阶段耗时分解 (优化后)

| 阶段 | 耗时 (us) | 占比 | Blocks | VGPR | LDS |
|------|----------|------|--------|------|-----|
| cumsum | 57 | 1.4% | 262144 | 4 | 0 |
| fused_kkt_solve | 475 | 12.0% | 32768 | 72 | 1KB |
| recompute_w_u | 642 | 16.3% | 32768 | 88 | 16KB |
| **fwd_h** | **2185** | **55.4%** | 256 | 56 | 34KB |
| fwd_o | 589 | 14.9% | 32768 | 216 | 32KB |
| **总计** | **3946** | 100% | | | |

## 4. Config Sweep 数据

### fused_kkt_solve

| BK | warps | 耗时 (us) |
|----|-------|----------|
| 32 | 1 | 513 |
| **64** | **1** | **475** |
| 64 | 2 | 948 |
| 64 | 4 | 1767 |
| 64 | 8 | 3763 |

### fwd_h (BV x warps)

| BV | warps | 耗时 (us) | blocks |
|----|-------|----------|--------|
| **16** | **4** | **2136** | 256 |
| 16 | 1 | 2809 | 256 |
| 16 | 8 | 2666 | 256 |
| 32 | 2 | 2342 | 128 |
| 32 | 4 | 3107 | 128 |
| 64 | 1 | 10470 | 64 |
| 64 | 4 | 2603 | 64 |

### fwd_o (BV x BK x warps)

| BV | BK | warps | 耗时 (us) | blocks |
|----|-----|-------|----------|--------|
| **128** | **64** | **1** | **580** | 32768 |
| 128 | 128 | 1 | 655 | 32768 |
| 64 | 64 | 1 | 717 | 65536 |
| 64 | 128 | 4 | 948 | 65536 |
| 32 | 64 | 1 | 1214 | 131072 |

### MFMA shape (matrix_instr_nonkdim)

| Kernel | mfma=16 | mfma=32 | 结论 |
|--------|---------|---------|------|
| fwd_h | **2115** | 2412 | Triton 已默认 16，无额外收益 |
| fwd_o | 590 | 581 | 差异可忽略 |
| recompute_w_u | **650** | 723 | 已默认 16 |

## 5. rocprof PMC 瓶颈分析

### 指令分布

| Kernel | VALU | LDS | FLAT | SMEM |
|--------|------|-----|------|------|
| fwd_h | 119.6M | 2.1M | 6.3M | 3.1M |
| fwd_o | 50.1M | 1.6M | 6.1M | 1.6M |
| recompute_w_u | 212.5M | 4.2M | 13.9M | 0 |

### LDS Bank Conflict (头号瓶颈)

| Kernel | LDS 指令 | Bank Conflict Cycles | 冲突/指令 |
|--------|---------|---------------------|----------|
| fwd_h | 2.1M | **134M** | 64 |
| fwd_o | 1.6M | **126M** | 77 |
| recompute_w_u | 4.2M | 67M | 16 |

根源: Triton `tl.dot` 经 LDS staging 给 MFMA，layout 无法用户侧控制。
Triton 已使用最优 MFMA 16x16，配置调优已触顶。

## 6. 进一步优化路线

| 路线 | 预估收益 | 难度 | 说明 |
|------|---------|------|------|
| waves_per_eu=2 (fwd_h) | ~50us (2%) | 低 | 微小 |
| recompute_w_u + fwd_h 融合 | ~200-400us | 中 | 省 w/u 的 GMEM round-trip |
| FlyDSL megakernel | 1.5-3x | 高 | state 留 LDS 不落 GMEM，persistent pipeline |

fwd_h 占 55% 且受限于 1024 chunks 串行依赖，Triton 配置调优已到天花板。
下一步收益需走 kernel fusion 或 FlyDSL megakernel 路线。
