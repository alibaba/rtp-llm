# GLM5 Mega MoE Symmetric Buffer 显存分析

## 部署场景

- **Prefill**: 8 卡，EP=8，CP=8，max_seq_len=256K
- **Decode**: 8 卡，EP=8，max_batch_size=8

## GLM5 模型参数

| 参数 | 值 |
|------|---|
| hidden_size (H) | 6144 |
| moe_intermediate_size (I) | 2048 |
| num_experts (E) | 256 |
| top_k (K) | 8 |
| EP size (R) | 8 |
| experts_per_rank (E_local) | 32 |

## Buffer 大小计算公式

### 核心参数 `max_tokens_per_rank` (T)

每个 rank 在 MoE dispatch 阶段最多发出的 token 数。经 `kLCMCandidateBlockM=384` 对齐后传入 DeepGEMM。

**Prefill (CP=8)**:
```
cp_bound = align_up(262144, cp_size*2) / cp_size = 262144 / 8 = 32768
T_aligned = align(32768, 384) = 86 × 384 = 33024
```

**Decode (max_batch=8)**:
```
max_tokens_per_rank = max_batch_size × tokens_per_step = 8 × 1 = 8
T_aligned = align(8, 384) = 384
```

### Pool Tokens (P) — L1/L2 激活缓冲区容量

Pool 按 worst-case 分配：所有 rank 的所有 token 的 top-k 路由都命中本地专家。

```
P = align(R × T × min(K, E_local) + E_local × (kMaxBlockM - 1), 384)
  = align(R × T × 8 + 32 × 191, 384)
```

### Padded SF Tokens

Scale factor 需要按 block_m 对齐，取所有 candidate block_m 中最大值（block_m=8 时最大）：
```
padded_sf = (P / 8) × 128 = P × 16
```

### 总 Buffer 布局

从低地址到高地址：
```
[Workspace] → [Input区] → [L1区] → [L2区] → [Combine区]
```

各区组成：
- **Workspace**: barrier + expert count + dispatch indices + combine metadata
- **Input区**: input_tokens(FP8) + input_sf + topk_idx(int64) + topk_weights(float)
- **L1区**: l1_tokens(FP8, P×H) + l1_sf(padded_sf × H/32) + l1_topk_weights(P×4)
- **L2区**: l2_tokens(FP8, P×I) + l2_sf(padded_sf × I/32)
- **Combine区**: combine_tokens(BF16, K×T×H×2)

## Prefill 计算 (T=33024)

```
P = align(8 × 33024 × 8 + 6112, 384) = 2,120,064
padded_sf = 2,120,064 × 16 = 33,921,024
```

| 组成部分 | 公式 | 大小 |
|----------|------|------|
| Workspace | 元数据 + dispatch pulling 索引 | ~285 MiB |
| Input tokens | T × H = 33024 × 6144 | 194 MiB |
| Input SF | T × (H/32) = 33024 × 192 | 6 MiB |
| Input topk_idx | T × K × 8 = 33024 × 64 | 2 MiB |
| Input topk_weights | T × K × 4 = 33024 × 32 | 1 MiB |
| **L1 tokens** | **P × H = 2,120,064 × 6144** | **12.1 GiB** |
| **L1 SF** | **padded_sf × (H/32) = 33,921,024 × 192** | **6.1 GiB** |
| L1 topk_weights | P × 4 | 8 MiB |
| **L2 tokens** | **P × I = 2,120,064 × 2048** | **4.0 GiB** |
| **L2 SF** | **padded_sf × (I/32) = 33,921,024 × 64** | **2.0 GiB** |
| **Combine** | **K × T × H × 2 = 8 × 33024 × 12288** | **3.0 GiB** |

### Prefill 总计：~27.7 GiB / GPU

主要开销：L1 pool (12.1 GiB) + L1 SF (6.1 GiB) + L2 pool (4.0 GiB)

## Decode 计算 (T=384)

```
P = align(8 × 384 × 8 + 6112, 384) = 30,720
padded_sf = 30,720 × 16 = 491,520
```

| 组成部分 | 公式 | 大小 |
|----------|------|------|
| Workspace | 元数据 | ~3.4 MiB |
| Input buffers | T × (H + H/32 + K×12) | ~2.4 MiB |
| L1 tokens | P × H = 30720 × 6144 | 180 MiB |
| L1 SF | padded_sf × (H/32) = 491520 × 192 | 90 MiB |
| L2 tokens | P × I = 30720 × 2048 | 60 MiB |
| L2 SF | padded_sf × (I/32) = 491520 × 64 | 30 MiB |
| Combine | K × T × H × 2 = 8 × 384 × 12288 | 36 MiB |

### Decode 总计：~402 MiB / GPU

## 对比总结

| 场景 | max_tokens_per_rank | Pool tokens | 总 symm buffer |
|------|--------------------:|------------:|---------------:|
| Prefill (CP=8) | 33,024 | 2,120,064 | **~27.7 GiB** |
| Decode (batch=8) | 384 | 30,720 | **~402 MiB** |
| ⚠️ 无调整 (raw 256K) | 262,272 | 16,791,552 | **~218 GiB** |

## 修复方案

### 问题

`fused_moe_wrapper.py:81` 直接使用 raw `max_seq_len`，不区分 CP 和 decode role：

```python
max_seq_len = getattr(config, "max_seq_len", 0)
max_tokens_per_rank = max(8192, max_seq_len) if max_seq_len > 0 else 8192
```

**不修复则 256K 场景下会尝试分配 ~218 GiB/GPU 的 symm buffer，必然 OOM。**

### 修复内容

在 `fused_moe_wrapper.py` 中引入 DSv4 的 `resolve_moe_max_tokens_per_rank` 逻辑，
在 `__init__` 阶段一次性完成 role 检测 + CP 调整 + token budget 解析：

1. 从 `parallelism_config.role_type` 检测 decode role
2. 从 `prefill_cp_config.is_enabled()` 检测 CP size
3. 调用 `resolve_moe_max_tokens_per_rank()` 统一解析
4. `generic_moe.py` 将已有的 `max_generate_batch_size` 透传给 wrapper

### 涉及文件

- `glm5_mega_moe/fused_moe_wrapper.py` — 主要修改
- `model_desc/generic_moe.py` — 透传 `max_generate_batch_size`

## DSv4 vs GLM5 传参方式对比

### 相同点

| 方面 | 说明 |
|------|------|
| CP 检测 | 均通过 `prefill_cp_config.is_enabled()` 检测，cp_size 取 `tp_size` |
| CP bound 计算 | `cp_padded_tokens_per_rank_bound(max_seq_len, cp_size)` |
| Decode role 计算 | `max_generate_batch_size × tokens_per_batch` |
| Chunked MoE | 均通过 `resolve_moe_max_tokens_per_rank` 内部 `chunked_moe_enabled()` 处理 |

### 关键不同点

| 方面 | DSv4 | GLM5 |
|------|------|------|
| **解析阶段** | **两阶段**：`__init__` 做 CP 调整，`initialize()` 做 role+speculative 调整 | **一阶段**：`__init__` 一次性完成全部解析 |
| **Buffer 分配时机** | `initialize()` resolve 后再 materialize（延迟构造） | `__init__` 末尾 `setup_weights_from_fp4` 时立即分配 |
| **Decode role 来源** | `init_resource.is_decode_role`（C++ engine 在 `initialize()` 回调时设置） | `parallelism_config.role_type == RoleType.DECODE`（构造时直接读） |
| **Speculative 支持** | `is_speculative` + `gen_num_per_cycle`（MTP 场景 tokens_per_batch > 1） | 暂未传入（默认 tokens_per_batch=1） |
| **二次修正** | `initialize()` 可再次修正 | 一次性确定，后续不可变 |
| **架构层级** | 独立的 `DeepSeekV4Model` descriptor 管理全生命周期 | 复用 `GenericMoeLayer` → `MegaMoeFusedWrapper`，无 model-level `initialize()` hook |

### DSv4 两阶段设计原因

```
Phase 1: __init__() — 只有 parallelism_config 可用
  → CP adjustment（确定性的，不依赖 engine 状态）
  → 此时 is_decode_role 未知，先按 prefill 最大预算

Phase 2: initialize(init_resource) — engine 启动后回调
  → 获得 is_decode_role, is_speculative, max_context_batch_size
  → 再次 resolve，可能大幅缩小 budget（如 decode → batch_size）
  → DSv4 此时才真正 materialize 模型（延迟构造）
```

### GLM5 一阶段可行的原因

GLM5 的 `MegaMoeFusedWrapper` 在 `__init__` 时就必须分配 symm buffer
（`setup_weights_from_fp4` → `_setup_buffer_and_warmup` 同步执行）。
而 `parallelism_config.role_type` 在构造时已经确定——与 DSv4 不同
（DSv4 的 role 由 engine 在 `initialize()` 时通知）。

### 待补充的 gap

1. **Speculative decode**：GLM5 未来若支持 MTP + mega kernel，decode 模式下
   `tokens_per_batch` 应为 `gen_num_per_cycle + 1`，需传入 `config.gen_num_per_cycle`。
2. **role_type 可靠性**：若存在 `role_type` 构造时未设置的 edge case（如 non-PD 部署），
   当前代码 fallback 到 `is_decode_role=False`（按 prefill 预算分配），是安全的 over-allocation。

## 关键代码路径

- Buffer 分配入口: `glm5_mega_moe/mega_buf.py:46` → `get_or_create_mega_buf()`
- Wrapper 初始化: `glm5_mega_moe/fused_moe_wrapper.py:56-131`
- 调用方透传: `model_desc/generic_moe.py:92` → `MegaMoeFusedWrapper(..., max_generate_batch_size=...)`
- DeepGEMM 大小计算 (C++): `DeepGEMM/csrc/apis/mega.hpp:19` → `get_symm_buffer_size_for_mega_moe()`
- Pool 容量公式 (CUDA): `DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh:18` → `get_num_max_pool_tokens()`
- DSv4 参考实现: `dsv4/moe/moe_layer.py:92` → `resolve_moe_max_tokens_per_rank()`
- DSv4 两阶段解析: `deepseek_v4_model.py:323-349` (phase1) + `deepseek_v4_model.py:460-493` (phase2)
