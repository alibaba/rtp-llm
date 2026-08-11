要支持 `kimi_k3_mla_swa_eagle3`，建议复用当前 `MtpExecutor` 的 speculative verify 框架，但不要把它硬塞成普通 MTP 模型。它需要一条明确的 EAGLE-3 feature handoff 路径。

整体数据流应当是：

```text
Kimi-K3 target
  ├─ layer 1 hidden
  ├─ layer 45 hidden
  └─ layer 89 hidden
          │
          ▼
 concat [T, 3H] → draft.fc → [T, H]
          │
          ▼
token embedding + fused target feature
          │
          ▼
EAGLE-3 MLA+SWA draft layer
          │
          ▼
draft norm + own lm_head
          │
          ├─ 连续提出 N 个 token
          ▼
Kimi-K3 target 一次验证 N+1 个位置
          │
          ├─ 接受前缀
          └─ 回滚 draft SWA cache
```

## 1. 当前 RTP-LLM 已经有什么

当前代码已经有一些能复用的基础：

- `SP_TYPE_EAGLE3` 类型；
- propose model 创建流程；
- `MtpExecutor` 多步 draft、target verify、rejection sampling；
- draft model 独立 cache layout；
- speculative block 分配和回滚；
- Qwen EAGLE-3 的部分权重字段；
- `all_hidden_states` 和 `need_all_hidden_states` 基础字段。

相关入口：

- [model_factory.py](/data0/xinfei.sxf/work/kimi/RTP-LLM/github-opensource/rtp_llm/model_factory.py:120)
- [RtpLLMOp.cc](/data0/xinfei.sxf/work/kimi/RTP-LLM/github-opensource/rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.cc:40)
- [MtpExecutor.cc](/data0/xinfei.sxf/work/kimi/RTP-LLM/github-opensource/rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc:1061)
- [qwen_v3_moe.py](/data0/xinfei.sxf/work/kimi/RTP-LLM/github-opensource/rtp_llm/models/qwen_v3_moe.py:128)

但是目前这些不是 Kimi-K3 EAGLE-3 的完整实现。尤其是：

1. target 没有导出指定的三层 hidden；
2. `MtpExecutor` 主要按单个 `last_hidden_states` 传递；
3. 没有 Kimi EAGLE-3 draft model 的配置和权重加载；
4. 没有 draft 专用 RoPE MLA+SWA cache；
5. 当前 `merged_eagle3_hidden` 参数没有形成完整生产数据链，正常路径传的是空 tensor。

所以不能只注册一下模型名就完成支持。

---

# 2. 配置和模型注册

建议新增：

```text
rtp_llm/models/kimi_k3/kimi_k3_eagle3.py
rtp_llm/models/kimi_k3/kimi_k3_eagle3_weight.py
rtp_llm/models_py/model_desc/kimi_k3_eagle3.py
```

模型类型建议明确区分：

```python
model_type = "kimi_k3_mla_swa_eagle3"
is_eagle3 = True
is_mtp = False
num_layers = 1
```

需要从 checkpoint 读取并校验：

```python
hidden_size = 7168
num_aux_hidden_states = 3
eagle_aux_hidden_state_layer_ids = [1, 45, 89]

num_attention_heads = 96
q_lora_rank = 1536
kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128

use_sliding_window = True
sliding_window = 2048
mla_use_nope = False
mla_use_output_gate = True
rope_theta = 1e7
```

不要从 target Kimi-K3 配置覆盖这些字段。target 是 NoPE MLA/KDA hybrid，而 draft checkpoint 的 attention 是 RoPE MLA+SWA。

同时需要注册 checkpoint architecture，例如根据实际 `config.json`：

```python
register_model(
    "kimi_k3_mla_swa_eagle3",
    KimiK3MlaSwaEagle3,
    ["Eagle3DeepseekV2SWAForCausalLM"],
)
```

---

# 3. Draft 权重加载

checkpoint 至少包括：

```text
embed_tokens.weight
fc.weight

layers.0.hidden_norm.weight
layers.0.input_layernorm.weight

layers.0.self_attn.q_a_proj.weight
layers.0.self_attn.q_a_layernorm.weight
layers.0.self_attn.q_b_proj.weight

layers.0.self_attn.kv_a_proj_with_mqa.weight
layers.0.self_attn.kv_a_layernorm.weight
layers.0.self_attn.kv_b_proj.weight
layers.0.self_attn.g_proj.weight
layers.0.self_attn.o_proj.weight

layers.0.post_attention_layernorm.weight
layers.0.mlp.gate_proj.weight
layers.0.mlp.up_proj.weight
layers.0.mlp.down_proj.weight

norm.weight
lm_head.weight
```

这里有三个容易踩坑的点。

### Draft 有自己的 LM head

不能像某些 MTP 一样默认共享 target head：

```text
draft hidden → draft.norm → draft.lm_head
```

### Attention 输入宽度是 `2H`

这些权重的输入维度不是 target layer 的 `H`：

```text
q_a_proj:  2H → q_lora
kv_a_proj: 2H → kv_lora + rope_dim
g_proj:    2H → heads × v_dim
```

通用 `KimiK3MLA` 默认接收 `H`，不能原样复用。

### token 163840

standalone 实现对 token ID `163840` 添加零 embedding row，见 [kimi_k3_mla_swa_eagle3.py](</home/xinfei.sxf/work/MAL_test_codes/model/kimi_k3_mla_swa_eagle3.py:261>)。

生产实现有两种办法：

- embedding 物理扩到 `vocab_size + 1`，最后一行全零；
- embedding kernel 对 `id == vocab_size` 特判返回零。

建议后者，避免改变 draft vocab 和 logits shape。`lm_head` 仍然只输出 checkpoint 的 draft vocab。

---

# 4. Target 导出三层 hidden

这是最核心的改动。

EAGLE-3 要的是 target decoder 的：

```text
layer 1 output
layer 45 output
layer 89 output
```

不是：

- 最终 layernorm 后 hidden；
- logits；
- 任意三层的 normalized input；
- block residual；
- target 最后一层 hidden 重复三份。

需要先确认 checkpoint 训练时的 layer ID 语义。通常应定义为：

```python
hidden_states[layer_id]
= 完成该 decoder layer 后的输出
```

Kimi target forward 位于：

- [kimi_k3.py](/data0/xinfei.sxf/work/kimi/RTP-LLM/github-opensource/rtp_llm/models_py/model_desc/kimi_k3.py:5335)
- decoder layer 输出位于 [kimi_k3.py](/data0/xinfei.sxf/work/kimi/RTP-LLM/github-opensource/rtp_llm/models_py/model_desc/kimi_k3.py:5142)

建议在 `KimiK3Model.forward()` 中增加专用的 feature collector：

```python
eagle_aux = []

for layer_idx, layer in enumerate(self.layers):
    hidden_states, block_residual = layer(...)

    if layer_idx + 1 in self.eagle_aux_layer_ids:
        eagle_aux.append(hidden_states)
```

然后：

```python
aux = torch.cat(eagle_aux, dim=-1)  # [T, 3H]
```

有两种放置 `fc` 的方案。

### 方案 A：target 只导出 `[T,3H]`

```text
target: concat(h1,h45,h89)
draft:  fc(3H→H)
```

优点是严格符合 checkpoint 模块归属；缺点是 target→draft 临时 buffer 约为 `3H`。

### 方案 B：在 target forward 末尾调用 draft fc

```text
target: concat + draft.fc → [T,H]
draft: 只读取 fused feature
```

能减少持久 handoff buffer，但让 target model 引用了 draft 权重，模型边界不干净。

我建议先做方案 A，正确性稳定后再把 `fc` 融合到 feature capture 中。对于 decode/verify，token 数很小，`3H` buffer 的瞬时成本一般可接受；长 prefill 则需要逐请求只保留需要的行。

---

# 5. 不要无条件保存整段三层 hidden

如果 prefill 长度 82K：

```text
82K × 3 × 7168 × 2 bytes ≈ 3.5 GiB
```

这还没算 target 中间激活和 TP 副本。

实际上 speculative decode 初始只需要每个请求最后有效位置对应的三层特征。因此生产实现应该支持两种 capture 模式：

```text
普通 prefill：
每个选定层只 index-select lm_output_indexes
最终得到 [B, 3H]

target verify：
需要每个 verify 位置的 feature
得到 [B × (N+1), 3H]
```

建议在 `PyModelOutputs` 或 ModelBase 增加专用字段/接口：

```cpp
virtual torch::Tensor getEagle3TargetFeatures(int64_t num_tokens);
```

不要继续复用名字含糊的：

```cpp
getMtpTargetHiddenStates()
```

因为 EAGLE-3 返回的是多层 feature，不是一个 MTP final hidden。

接口语义应固定：

```text
shape: [num_rows, num_aux * hidden_size]
layout: request-major
[r0_step0, r0_step1, ..., r1_step0, ...]
dtype: target model dtype
device: CUDA
```

这和 verify 请求排列必须完全一致，否则会出现“运行不报错但接受率接近零”的隐蔽问题。

---

# 6. Draft 首步和递归步必须分开

standalone 模型中有两种不同输入语义。

## 首次/teacher-forced输入

```python
target_feature = fc(concat(h1, h45, h89))
token_emb = embed(input_token)

x = concat(
    input_layernorm(token_emb),
    hidden_norm(target_feature),
)
```

## 后续 speculative step

后续不再重新使用 `fc(target aux)`，而是：

```python
prev_hidden = previous draft pre-norm hidden
token_emb = embed(previous sampled token)

x = concat(
    input_layernorm(token_emb),
    hidden_norm(prev_hidden),
)
```

见 standalone 的：

- prefill：[kimi_k3_mla_swa_eagle3.py](</home/xinfei.sxf/work/MAL_test_codes/model/kimi_k3_mla_swa_eagle3.py:296>)
- decode step：[kimi_k3_mla_swa_eagle3.py](</home/xinfei.sxf/work/MAL_test_codes/model/kimi_k3_mla_swa_eagle3.py:326>)

因此 draft forward 最好显式接受模式：

```python
class Eagle3Inputs:
    token_ids
    target_aux_features   # 仅首步定义
    previous_draft_hidden # 递归步定义
    position_ids
```

不要只塞进通用 `input_hiddens` 后靠 shape 猜测。

可以在 C++ 输入里加枚举：

```cpp
enum Eagle3InputMode {
    TARGET_AUX = 0,
    DRAFT_RECURRENT = 1,
};
```

draft Python forward：

```python
if input_mode == TARGET_AUX:
    prev_hidden = fc(target_aux_features)
else:
    prev_hidden = input_hiddens

emb = embed(token_ids)
x = cat(input_norm(emb), hidden_norm(prev_hidden), dim=-1)
hidden = draft_layer(x, residual=prev_hidden)
logits = lm_head(norm(hidden))
return logits, hidden
```

这里返回给下一步的必须是 `feed_forward()` 输出的 pre-final-norm hidden，不能返回 `norm(hidden)`。

---

# 7. Draft Attention 不能直接用当前 KimiK3MLA

当前主模型 `KimiK3MLA`：

- 输入 `H`；
- 强制 NoPE；
- compressed latent paged cache；
- full attention；
- target output gate；
- 面向 hybrid KDA/MLA 层。

见 [kimi_k3.py](/data0/xinfei.sxf/work/kimi/RTP-LLM/github-opensource/rtp_llm/models_py/model_desc/kimi_k3.py:2881)。

EAGLE-3 draft 要求：

- 输入 `2H`；
- 64 维 interleaved RoPE；
- `rope_theta=1e7`；
- causal SWA 2048；
- output gate 从 `2H` 输入投影；
- q/k head dim 192，v dim 128。

因此建议新增 `KimiK3Eagle3MLA`，但复用底层 FMHA/paged-cache primitive。

拆分为：

```text
KimiK3Eagle3MLA
  ├─ 自己的 q_a / kv_a 投影，输入 2H
  ├─ 自己的 RoPE
  ├─ 复用 MLA cache write
  ├─ 复用 FMHA kernel
  ├─ 设置 sliding_window=2048
  ├─ 自己的 g_proj(2H)
  └─ 自己的 o_proj
```

第一阶段可以先实现 Torch/reference path，对齐 standalone；第二阶段再接 FlashInfer MLA/SWA kernel。

必须验证：

```text
prefill full recompute
≈ chunked prefill
≈ token-by-token decode
```

---

# 8. Draft KV cache 设计

Draft cache 必须独立于 target：

```text
target:
  group KDA state
  group NoPE MLA latent cache

draft:
  group EAGLE3 RoPE MLA+SWA cache
```

不要把 draft layer挂到 target MLA group，因为二者：

- RoPE 语义不同；
- cache 内容不同；
- logical length 独立；
- speculative token可能被拒绝；
- target verify 与 draft proposal 的提交时机不同。

可以复用当前 `mtp_sub_configs[0]` 作为“propose model sub-config”，虽然名字叫 MTP。更长期建议重命名成：

```text
speculative_sub_configs
```

draft cache 配置要表达：

```text
model_id = 1
num_layers = 1
group_type = SWA/MLA
window_size = 2048
kernel_tokens_per_block = ...
```

对于 SWA cache，回滚时通常不用搬移/swap block：

1. 保存 committed sequence length；
2. draft 临时写入 N 个位置；
3. target 返回 accepted length `A`；
4. draft logical length回退到 `committed + A`；
5. 尾部未接受槽位随后覆盖。

但要确保环形窗口不会让 tentative token 覆盖仍然有效的 committed token。最稳妥的第一版：

- speculation 期间不复用即将绕回的 committed slot；
- 或保留 `window + max_spec_tokens` 的物理容量；
- verify 完成后再推进 committed cursor。

## 8.1 已确定的三 Group、三 Pool 方案

Kimi-K3 直接复用 DSV4 的 independent-pool 设计，不增加 logical group 到 physical pool 的二级映射。配置固定为：

```text
independent-pool = true
```

模型只划分三个 KV cache group，每个 group 由 `KVCacheManager` 创建并管理自己的独立 `BlockPool`：

```text
KVCacheManager
├── group 0：FULL
│   └── pool 0：主模型 MLA/FULL attention KV cache
├── group 1：LINEAR
│   └── pool 1：主模型 KDA/linear attention cache/state
└── group 2：MTP
    └── pool 2：MTP/EAGLE-3 draft model KV cache
```

对应关系是一对一的：

```text
group_id 0 (FULL)   → BlockPool 0
group_id 1 (LINEAR) → BlockPool 1
group_id 2 (MTP)    → BlockPool 2
```

不需要新增 `physical_pool_id`、`group_to_physical_pool_id` 或共享同类 group 的逻辑。group id 本身就是 cache 类型、block table 和 pool 的唯一索引。

### 8.1.1 Group 定义

三个 group 分别聚合对应类型的 layer：

```text
FULL group:
  model_id = 0
  layers = 主模型所有 MLA/FULL attention layers

LINEAR group:
  model_id = 0
  layers = 主模型所有 KDA/linear attention layers

MTP group:
  model_id = 1
  layers = MTP/EAGLE-3 draft model layers
```

每个 group 维护自己的：

```text
layer mapping
block table
block size / tokens per block
block ID namespace
容量和回收策略
```

FULL、LINEAR、MTP 之间不共享 block，也不需要跨 pool 借用。

### 8.1.2 分配语义

分配层直接复用现有 independent-pool 实现，不新增 allocator 机制：

- `HybridPoolKVCacheAllocator::doInit()` 已经为每个 group 创建一个独立 `BlockPool`；
- `HybridKVCacheAllocator::initMalloc()` 和 `incrMalloc()` 已经逐 group 完成分配；
- 因此新增 MTP group 后，现有路径会自然形成 FULL、LINEAR、MTP 三个独立 pool。

分配采用简单的 all-or-nothing 语义：

```text
依次为 FULL、LINEAR、MTP group 分配
→ 三个 group 全部分配成功：本次分配成功
→ 任意一个 group 分配失败：本次整体分配失败
```

不做 MTP pool 不足时的 speculative fallback，也不增加跨 pool 的业务事务、状态快照或复杂回滚逻辑。

现有 allocator 在整体失败前释放本次调用中已经新申请的 block，并恢复原 block table；这只是分配函数已有的失败清理，用于避免 block 泄漏，不改变“任意 group 失败则整体失败”的对外语义，也不需要为 Kimi-K3 新写一套实现。

所以分配部分的代码改动应为零。需要修改的是模型/cache 配置生成：把 Kimi-K3 的 cache specs 固定组织为三个 group，并打开 `independent-pool=true`。

### 8.1.3 PD 与 cache-store

PD/cache-store 元数据按 group 传递：

```text
model_id
group_id
group type（FULL / LINEAR / MTP）
block_id
layer mapping
valid coverage
```

Prefill 和 Decode 两端必须使用完全相同的三个 group 定义、顺序、layer mapping、block size 和 tokens per block。即使两端 TP 不同，也不能出现 group 顺序或覆盖范围不一致，否则 producer 发布的 buffer 与 decode 请求的 buffer 无法对应，最终仍可能表现为 `CACHE_STORE_LOAD_BUFFER_TIMEOUT`。

### 8.1.4 指标

指标部分不需要开发。直接沿用当前已有的 per-pool 指标；在 FULL、LINEAR、MTP 三个 group 分别创建独立 `BlockPool` 后，现有指标会自然按三个 pool 上报，包括：

```text
total_blocks
free_blocks
used_blocks
allocation_failures
```

不新增 MTP 专属 pool 指标，也不修改现有指标注册和上报代码。任一 pool 分配失败继续沿用当前请求分配失败的观测方式。

---

# 9. 修改 MtpExecutor 为通用 Feature Draft Executor

当前 `MtpExecutor` 已经能完成：

```text
draft N步
→ target verify
→ rejection sampling
→ 更新 stream
→ cache rollback/继续
```

`draftModelDecode()` 在 [MtpExecutor.cc](/data0/xinfei.sxf/work/kimi/RTP-LLM/github-opensource/rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc:1847)。

建议不要复制整个 executor，而是抽象以下策略：

```cpp
class DraftFeatureAdapter {
public:
    prepareFirstDraftInput(target_output, model_input);
    prepareNextDraftInput(draft_output, sampled_token, model_input);
    preparePostVerifyInput(target_verify_output, accepted_len, model_input);
};
```

实现两类 adapter：

```text
MtpFeatureAdapter:
    target final hidden → MTP fusion

Eagle3FeatureAdapter:
    target aux [h1,h45,h89] → fc
    recursive step使用previous draft hidden
```

短期若希望改动小，可以直接在 `MtpExecutor` 内按：

```cpp
sp_type_ == SP_TYPE_EAGLE3
```

分支，但建议只分支 feature preparation，不分叉 verify/sampler/cache 主流程。

需要特别处理三处：

### Draft cycle 开始

当前请求的 target aux features作为第一个 `prev_hidden`。

### 每个 draft step 后

把：

```cpp
draft_output.all_hidden_states
```

或专用 `draft_recurrent_hidden` 传给下一步。

### Verify 后

对于下一轮 speculative cycle，要从 target verify 的“最后一个接受位置”选择对应三层 feature：

```text
row = request_offset + accepted_len - 1
```

不能总取 verify 最后一行，因为发生拒绝时最后几行对应未接受的 speculative token。

---

# 10. Target verify 的 feature 行选择

假设提出 5 个 token，target verify 输出：

```text
request 0:
v0, v1, v2, v3, v4, bonus
```

若只接受前 2 个，那么下一轮 EAGLE-3 所需 target feature应选择：

```text
feature(v2对应的最后已提交位置)
```

具体下标要和当前 RTP 的 `accept_len` 定义统一。必须写一个明确的 helper：

```cpp
selected_row =
    request_verify_offset
    + accepted_token_count_adjusted;
```

这里最容易产生 off-by-one。建议用小例子覆盖：

- 接受 0 个 draft token；
- 接受 1 个；
- 全部接受；
- EOS 出现在 draft 中；
- bonus token 被接受；
- batch 内各请求接受长度不同。

---

# 11. PD 分离

如果 target decode 与 draft 在同一个 decode worker/GPU 进程中，target aux feature只是本地 CUDA tensor，不需要经 RDMA 传输。

PD 流程应当是：

```text
Prefill worker:
  计算 target prompt KV
  通过 CacheStore 发送 target KV

Decode worker:
  load target KV
  对最后 prompt token/首个 decode token取得 target aux feature
  本地运行 EAGLE-3 draft
```

但有一个问题：decode worker只加载 KV，并不会天然拥有 prefill 最后 token 在 layer 1/45/89 的 hidden。

有两个方案。

### 方案 1：PD 额外传 EAGLE aux feature

Prefill 端发布：

```text
[B, 3H] last committed aux feature
```

随请求 metadata/RDMA buffer传给 decode。

优点：不额外重算 target。

缺点：要扩展 CacheStore/request protocol。

### 方案 2：decode 端重新执行最后一个 prompt token

加载 target KV 后，以最后 prompt token做一次 target decode，生成三层 aux feature。

风险是 KV position和重复写入语义复杂，容易多写一个位置。

建议方案 1。新增独立 buffer 类型，例如：

```text
EAGLE3_AUX_FEATURE
```

并且：

- producer coverage长度必须精确；
- request ID与 feature对应；
- decode load完成后再启动 draft；
- feature load timeout不要复用 KV buffer的模糊错误码。

这也与你前面遇到的 `LoadBufferTimeout` 直接相关：如果把 aux feature纳入 PD buffer，producer 发布范围和 decode 请求范围不一致，会再次出现 900 秒超时。

---

# 12. 并行策略

你的 target Kimi-K3 常见是 TP8，而 draft只有一层。

第一版最稳妥：

```text
draft TP = target TP
draft EP = 1
```

原因：

- 复用现有 communicator；
- target aux feature无需跨新 group；
- draft MLA head可按 TP shard；
- draft dense FFN按 TP shard；
- lm_head vocab parallel。

但是 `fc: 3H → H` 的切分要特别设计：

- 输入 target hidden 如果各 rank持有完整 `H`，fc做 column/row parallel；
- 如果 hidden 是 TP token-SP/local shard，则先统一 feature layout；
- 三层 feature必须按 `[h1_full, h45_full, h89_full]` concat，不能把 rank shard次序与 layer次序混在一起。

后续优化可以让 draft TP1，但这需要：

```text
target TP8 → gather aux feature到一个 rank
draft只在rank0运行
draft token广播给所有target ranks
```

这会引入 gather、广播和单卡大权重/LM head问题。第一版不建议。

---

# 13. 正确性验证顺序

建议按以下顺序推进。

## 第一级：单层数学对齐

从 standalone 导出固定输入：

```text
aux [B,S,3H]
input_ids
positions
initial cache
```

逐项对比：

```text
fc output
attention input
q
compressed kv
k_nope/k_rope/v
attention context
sigmoid gate
o_proj
FFN
pre-norm hidden
logits
```

目标：

```text
FP32: 1e-6级
BF16: argmax/token一致，误差分层记录
```

## 第二级：cache 对齐

比较：

```text
full prefill
chunked prefill
逐 token decode
prefill + decode
S > 2048
position > 2048
```

尤其验证：

```text
RoPE 使用绝对 position
SWA 只限制 causal key范围
rope_theta必须为1e7
```

## 第三级：speculative 状态机

固定 target logits，测试：

- 全接收；
- 第一个拒绝；
- 中间拒绝；
- 全拒绝；
- EOS；
- batch 不同 accept length；
- cache rollback 后下一轮 logits一致。

## 第四级：真实 MAL

先测：

```text
spec=1
spec=2
spec=5
```

记录：

```text
平均接受长度
accepted-token ratio
draft耗时
target verify耗时
端到端tokens/s
```

如果 `spec=1` 都与 standalone draft logits不一致，不要先查 sampler，先查：

1. aux layer ID；
2. hidden 是 layer input 还是 output；
3. token shift；
4. position ID；
5. RoPE base；
6. pre-norm hidden 是否正确；
7. lm_head 是否误共享。

---

# 14. 推荐实施阶段

### Phase 1：单机 correctness

- 新增 Kimi EAGLE-3 config/weight/model；
- Torch attention；
- target导出 `[B,3H]`；
- draft独立 cache；
- linear speculation；
- CUDA graph关闭；
- TP先做 1。

### Phase 2：接 RTP kernel

- paged MLA cache；
- RoPE+SWA FlashInfer；
- TP8；
- vocab-parallel lm_head；
- CUDA graph；
- batch decode。

### Phase 3：PD

- aux feature PD buffer；
- coverage和request metadata；
- timeout/fallback；
- prefill TP8、decode TP8；
- cache load + EAGLE feature load联合测试。

### Phase 4：性能优化

- target capture只保存选定行；
- 三层 concat+fc融合；
- feature buffer复用；
- draft/target verify stream overlap；
- 减少每步 host bookkeeping；
- 可选 tree draft。

## 最终建议

最关键的设计决定有三个：

1. 把 EAGLE-3 feature定义成独立 ABI：`[rows, 3H]`、request-major、明确 layer-output 语义。
2. Draft 使用独立 RoPE MLA+SWA cache group，不能借用 Kimi target 的 NoPE MLA/KDA cache。
3. 复用 `MtpExecutor` 的 verify/sampler/rollback，但把 hidden handoff抽象成 MTP/EAGLE-3 两种 adapter。

如果直接从可落地的最小版本开始，我会优先实现：

```text
TP1 + 非PD + linear spec + reference attention
```

先做到和 `MAL_test_codes` 的逐 tensor/logits 对齐，再接入 TP8、paged cache和 PD。

---

# 15. Kimi3-EAGLE TP8 PD Smoke 方案

新增 smoke：

```text
kimi_k3_4layer_tp8_pd_eagle3_sm100
```

该 smoke 直接基于现有 `kimi_k3_4layer_tp8_pd_sm100` 复制和扩展。保留基线已有的 4-layer target checkpoint、TP8、PD endpoint、SM100 资源、KDA/MLA/MoE backend、请求客户端、超时和进程清理逻辑，仅增加 EAGLE-3 propose model 配置和专项断言。

## 15.1 启动配置

Prefill 和 Decode 两个 role 都设置：

```bash
export MODEL_TYPE=kimi_k3
export SP_TYPE=eagle3
export SP_MODEL_TYPE=kimi_k3_mla_swa_eagle3
export SP_CHECKPOINT_PATH=/home/xinfei.sxf/work/iter_0019457_hf_full
export GEN_NUM_PER_CIRCLE=3
```

`CHECKPOINT_PATH`、TP/EP/DP、端口和其余 Kimi3 环境变量全部继承 `kimi_k3_4layer_tp8_pd_sm100`，避免新 smoke 与基线产生无关差异。

首版配置固定为：

```text
target model：Kimi-K3 4 layers
EAGLE-3 model：1 draft layer
target TP：8
draft TP：8
PD：开启
CUDA Graph：关闭
GEN_NUM_PER_CIRCLE：3
temperature：0
```

Kimi3-EAGLE 模型配置必须将：

```python
config.hybrid_attention_config.enable_independent_kv_cache_pools = True
```

当前该字段由模型代码设置，不为 smoke 新增临时环境变量。

## 15.2 Cache layout 检查

Prefill 和 Decode 必须生成相同的三 group、三 pool 布局：

```text
group 0：FULL   → pool 0
group 1：LINEAR → pool 1
group 2：MTP    → pool 2
```

4-layer target 加单层 draft 时，层号约定为：

```text
target global layer：0～3
EAGLE local layer：0
EAGLE global layer：4
EAGLE pool local layer：0
```

启动阶段至少检查：

```text
HybridPoolKVCacheAllocator init success, group pools=3
FULL、LINEAR、MTP 三个 pool 均成功初始化
Prefill/Decode 的 group 顺序、layer mapping、block size 和 tokens per block 一致
```

不为 smoke 开发新的 allocator 或指标逻辑。分配继续使用现有 all-or-nothing 语义，观测继续使用现有 per-pool 指标。

## 15.3 Smoke query

发送一个确定性、能够触发多轮 proposal/verify 的请求：

```json
{
  "prompt": "Briefly explain why the sky appears blue during the day.",
  "temperature": 0,
  "max_tokens": 32,
  "stream": false
}
```

请求发往 Decode 对外服务端口，API 形式沿用基线 smoke，不单独实现客户端。

## 15.4 验收条件

Smoke 必须同时满足：

1. Prefill 和 Decode 均启动成功；
2. 请求返回成功且生成非空文本；
3. EAGLE draft 至少执行两轮 proposal/verify；
4. 三个独立 pool 均成功初始化；
5. 日志能够确认 `SP_TYPE=eagle3`、`GEN_NUM_PER_CIRCLE=3`；
6. 日志中没有以下错误：

```text
CACHE_STORE_LOAD_BUFFER_TIMEOUT
load kv cache failed
missing EAGLE3_AUX_FEATURE
invalid global layer id
group mapping mismatch
CUDA illegal memory access
NCCL timeout
```

7. Smoke 结束后 Prefill、Decode 及其 TP rank 没有残留进程。

首版不设置接受率或性能阈值。该 smoke 的目标是覆盖 TP8 PD 下 EAGLE-3 的完整数据链、三 pool 初始化和一次真实请求；接受率和吞吐性能放到独立测试中验证。

## 15.5 后续增强

基础 smoke 稳定后再增加非阻塞覆盖：

```text
GEN_NUM_PER_CIRCLE=5
CUDA Graph 开启
batch=2
重复 prompt/cache reuse
不同 accepted length
```

这些场景不进入第一版，避免同时引入 EAGLE correctness、CUDA Graph 和 batch 状态管理三类变量。
