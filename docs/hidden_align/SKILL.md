---
name: precision-alignment
description: "RTP-LLM 与 vLLM 逐层 hidden state 精度对齐：dump hidden states、FP8/FP4/BF16 多种量化配置对比、cosine/RelL2/MaxDiff 指标计算。Use when: 精度对齐、hidden state 对比、vllm 对比、per-layer diff、cosine similarity、FP8 精度、FP4 精度、量化精度验证、数值对齐、隐藏层对比、inference 精度"
related_skills: [test-execution, local-build-and-serve, smoke-nondeterminism-analysis]
---

# RTP-LLM 与 vLLM 逐层精度对齐

## 概述

对比 RTP-LLM 和 vLLM 推理引擎在相同模型、相同 prompt 下的逐层 hidden state 数值差异。
用于验证量化（FP8/FP4/INT4）或自定义 kernel 引入的精度损失是否在可接受范围内。

**核心输出**：每层 cosine similarity、relative L2 error、max abs diff 表格 + token 匹配率。

---

## 用户输入要求

| 必需信息 | 说明 | 示例 |
|---------|------|------|
| **模型路径** | 本地 checkpoint 路径 | `/home/user/models/GLM-5-BF16-4layer` |
| **模型类型** | MODEL_TYPE 环境变量值 | `glm_5`, `deepseek_v2`, `qwen3_moe` |
| **RTP-LLM 量化配置** | `--quantization` + 其他关键参数 | `FP8_PER_BLOCK`, `FP8_PER_BLOCK_NO_MOE --moe_strategy mega_moe` |
| **对比基准** | vLLM 使用的配置 | `BF16`（纯精度基线）或 `FP8`（同等量化对比） |
| **GPU 编号** | 可用的 GPU | `0`, `1` |
| **Prompt 长度** | 要测试的 prompt 类型 | short/medium/long_4k |

可选信息：
- **smoke target 名称**：用于参考其配置（如 `mla_cp_pd`、`mla_mega_moe_fp8_attn_cp_pd`）
- **额外环境变量**：`MOE_STRATEGY=mega_moe`、`CUDA_HOME` 等

---

## 前置条件

| 项目 | 要求 |
|------|------|
| 源码目录 | `RTP-LLM/github-opensource`（已 bazel build 完成） |
| Python | `/opt/conda310/bin/python` |
| RTP-LLM 编译产物 | `bazel-bin/` 下有 `.so` 文件（libth_transformer.so 等） |
| vLLM | 已安装（`pip install vllm`），版本 0.20+ |
| 依赖 | `torch`, `psutil`, `requests`, `safetensors` |
| stub_source | 指向 `internal_source`（用于加载内部模型定义） |

### 检查编译状态

```bash
cd /home/<user>/RTP-LLM/github-opensource
ls bazel-bin/libth_transformer.so  # 必须存在
ls bazel-bin/librtp_compute_ops.so
ls bazel-bin/libth_transformer_config.so
```

如果 .so 不存在，先执行编译：
```bash
bazelisk build //:th_transformer //:rtp_compute_ops //:th_transformer_config --config=cuda12_9
```

### 检查 proto 链接

```bash
ls rtp_llm/dash_sc/proto/*_pb2.py 2>/dev/null || bash rtp_llm/dash_sc/proto/link_py_proto.sh
```

---

## 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 准备 — 确定配置、创建输出目录、确认模型可用       │
├─────────────────────────────────────────────────────────────┤
│  Step 2: vLLM BF16 Dump — 纯 BF16 基线 hidden states       │
├─────────────────────────────────────────────────────────────┤
│  Step 3: vLLM FP8 Dump — vLLM FP8 量化 hidden states       │
├─────────────────────────────────────────────────────────────┤
│  Step 4: RTP-LLM Dump — 目标量化配置 hidden states          │
├─────────────────────────────────────────────────────────────┤
│  Step 5: 对比分析 — per-layer cosine/RelL2/MaxDiff          │
├─────────────────────────────────────────────────────────────┤
│  Step 6: 记录结果 — 更新 MD 文档、保存配置                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: 准备

### 1.1 创建工作目录

```bash
WORK_DIR=/home/<user>/RTP-LLM/github-opensource
ALIGN_DIR=$WORK_DIR/docs/hidden_align
mkdir -p $ALIGN_DIR/{vllm_dumps,vllm_dumps_fp8,rtp_llm_dumps,rtp_llm_dumps_fp8}
```

### 1.2 确定测试 Prompts

标准 3 档 prompt（与 smoke 测试覆盖相同范围）：

```python
PROMPTS = {
    "short": ("The capital of France is", 20),           # ~5 tokens input
    "medium": ("Write a detailed essay about ...", 100),  # ~23 tokens input
    "long_4k": (build_long_prompt(4096), 50),             # ~2070 tokens input
}
```

### 1.3 确认模型配置

```bash
# 检查模型层数、hidden_size、MoE 配置
python -c "
import json
with open('<MODEL_PATH>/config.json') as f:
    c = json.load(f)
print('num_hidden_layers:', c.get('num_hidden_layers'))
print('hidden_size:', c.get('hidden_size'))
print('num_experts:', c.get('n_routed_experts', c.get('num_local_experts', 'N/A')))
print('quant_config:', c.get('quantization_config', 'None'))
"
```

---

## Step 2: vLLM BF16 Dump（基线）

### 2.1 关键原理

- 设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 强制 EngineCore 在进程内运行
- 通过 `engine.model_executor.workers[0].worker.model_runner.model` 获取模型实例
- 在每个 `DecoderLayer` 注册 forward hook 捕获 `(hidden_states, residual)`
- 只捕获第一次 forward（prefill），忽略后续 decode forward

### 2.2 Hook 安装模式

```python
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# 获取模型
engine = llm.llm_engine
workers = engine.model_executor.workers
inner_model = workers[0].worker.model_runner.model
inner = getattr(inner_model, "model", inner_model)

# 注册 hooks
for idx, layer in enumerate(inner.layers):
    layer.register_forward_hook(make_hook(idx))
inner.embed_tokens.register_forward_hook(embed_hook)
inner.norm.register_forward_hook(final_norm_hook)
```

### 2.3 捕获的 Tensor 列表

每个 prompt 捕获 `3*num_layers + 2` 个 tensor：

| Tensor 名 | 含义 |
|-----------|------|
| `embed_out` | Embedding 输出 |
| `layer{i:02d}_hidden` | 第 i 层 decoder 输出的 hidden_states |
| `layer{i:02d}_residual` | 第 i 层 decoder 输出的 residual |
| `layer{i:02d}_combined` | hidden_states + residual（实际传给下一层的值） |
| `final_norm` | 最终 RMSNorm 输出 |

### 2.4 运行命令

```bash
cd $WORK_DIR
CUDA_VISIBLE_DEVICES=0 /opt/conda310/bin/python docs/hidden_align/vllm_dump_hidden.py \
    --model <MODEL_PATH> \
    --prompts short medium long_4k
```

### 2.5 输出

```
docs/hidden_align/vllm_dumps/
├── short.pt          # {tensors: {name: tensor}, output_token_ids: [...], ...}
├── medium.pt
├── long_4k.pt
├── short.stats.json  # 轻量统计（shape, mean, std, abs_max, md5）
├── medium.stats.json
├── long_4k.stats.json
└── meta.json         # 模型路径、版本、prompt 元数据
```

---

## Step 3: vLLM FP8 Dump（同量化基准）

与 Step 2 相同流程，但加载时指定 `quantization="fp8"`：

```python
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    trust_remote_code=True,
    max_model_len=8192,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
    dtype="bfloat16",
    quantization="fp8",  # 在线 FP8 量化
)
```

```bash
cd $WORK_DIR
CUDA_VISIBLE_DEVICES=0 /opt/conda310/bin/python docs/hidden_align/vllm_dump_fp8.py \
    --model <MODEL_PATH> \
    --prompts short medium long_4k
```

输出到 `docs/hidden_align/vllm_dumps_fp8/`。

---

## Step 4: RTP-LLM Dump

### 4.1 核心原理

- RTP-LLM 使用 `MOEDBG=1` 环境变量启用 `_record_tensor` 机制
- 在 `GenericMoeModel.forward()` 中通过 `_rt.record(name, tensor)` 记录 hidden states
- 每次 forward 结束调用 `_rt.dump()` 将 tensor 序列化到磁盘
- 第一个 dump 文件（step=000）对应 prefill forward

### 4.2 GenericMoeModel 插桩位置

文件：`rtp_llm/models_py/model_desc/generic_moe.py` 的 `forward()` 方法

```python
from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

_rt_on = _rt.ENABLED
if _rt_on:
    _rt.begin(seqlen=int(input_ids.size(0)))
    if _rt._get_buf() is None:
        _rt_on = False

# embedding 之后
if _rt_on:
    _rt.record("embed_out", hidden_states)

# 每层 decoder 之后
for i, layer in enumerate(self.layers):
    hidden_states, residual = layer(hidden_states, residual, ...)
    if _rt_on:
        _rt.record(f"layer{i:02d}_hidden", hidden_states)
        _rt.record(f"layer{i:02d}_residual", residual)
        _rt.record(f"layer{i:02d}_combined", hidden_states + residual)

# final norm 之后
if _rt_on:
    _rt.record("final_norm", hidden_states)
    _rt.dump(step=getattr(self, "_dbg_step", 0), extra={})
```

### 4.3 关键环境变量

| 变量 | 值 | 说明 |
|------|-----|------|
| `MOEDBG` | `1` | 启用 record_tensor |
| `MOEDBG_DIR` | `/tmp/rtp_llm_hidden_dumps` | dump 输出根目录 |
| `MOEDBG_CASE` | `short`/`medium`/`long_4k` | 子目录名，按 prompt 区分 |
| `MOEDBG_FULL_THRESHOLD` | `16777216` | 超过此 numel 的 tensor 只存 stats 不存全量 |
| `DETERMINISTIC_GEMM` | `1` | 确保 GEMM 确定性 |
| `MOE_STRATEGY` | `mega_moe`（如需要） | MoE 策略 |

### 4.4 Server 启动参数模板

#### FP8_PER_BLOCK（无 mega_moe）—— 对应 mla_cp_pd

```bash
python -m rtp_llm.start_server \
    --warm_up 0 \
    --seq_size_per_block 64 \
    --act_type BF16 \
    --enable_cuda_graph 0 \
    --tp_size 1 --ep_size 1 --dp_size 1 --world_size 1 \
    --quantization FP8_PER_BLOCK \
    --reserver_runtime_mem_mb 8192 \
    --force_cpu_load_weights 1 \
    --fp8_kv_cache 1 \
    --use_deepep_moe 0 \
    --use_deepep_low_latency 0 \
    --use_all_gather 1
```

#### FP8_PER_BLOCK_NO_MOE + mega_moe —— 对应 mla_mega_moe_fp8_attn_cp_pd

```bash
# 额外环境变量：MOE_STRATEGY=mega_moe
python -m rtp_llm.start_server \
    --warm_up 0 \
    --seq_size_per_block 64 \
    --act_type BF16 \
    --enable_cuda_graph 0 \
    --tp_size 1 --ep_size 1 --dp_size 1 --world_size 1 \
    --quantization FP8_PER_BLOCK_NO_MOE \
    --moe_strategy mega_moe \
    --reserver_runtime_mem_mb 8192 \
    --force_cpu_load_weights 1 \
    --fp8_kv_cache 1 \
    --use_deepep_moe 0 \
    --use_deepep_low_latency 0 \
    --use_all_gather 0
```

### 4.5 已知问题与解决方案

| 问题 | 原因 | 解决 |
|------|------|------|
| `GLM5 MegaMoE requires torch.distributed` | `backend_manager.py` 仅在 world_size>1 时 init dist | 已 patch：当 `MOE_STRATEGY=mega_moe` 时强制 init |
| `No suitable MOE strategy found` | `--use_all_gather 0` 导致 PureTpRouter 条件不满足 | FP8_PER_BLOCK (非mega_moe) 需设 `--use_all_gather 1` |
| `cuFileHandleRegister error 5027` | GDS 加载权重失败 | 添加 `--force_cpu_load_weights 1` |
| `ImportError: predict_v2_pb2` | proto 文件未链接 | 执行 `bash rtp_llm/dash_sc/proto/link_py_proto.sh` |
| `transformers.models.gpt2.tokenization_gpt2_fast` 不存在 | transformers >= 5.x | 已 patch try/except 回退 |
| `MOEDBG_FULL_THRESHOLD` 过小导致 tensor 为空 | 长序列 numel 超过阈值 | 设为 16M (16*1024*1024) |

### 4.6 运行命令

```bash
cd $WORK_DIR
CUDA_VISIBLE_DEVICES=1 /opt/conda310/bin/python docs/hidden_align/rtp_llm_dump_fp8.py \
    --prompts short medium long_4k
```

### 4.7 Dump 文件结构

```
/tmp/rtp_llm_hidden_dumps/<case>/rank0_pid<PID>_step000.pt  # prefill (第一个)
/tmp/rtp_llm_hidden_dumps/<case>/rank0_pid<PID>_step001.pt  # decode step 1
...

# dump 脚本复制 step000 到输出目录：
docs/hidden_align/rtp_llm_dumps_fp8/
├── short.pt
├── medium.pt
├── long_4k.pt
├── short.server.log
├── medium.server.log
├── long_4k.server.log
└── meta.json
```

---

## Step 5: 对比分析

### 5.1 指标定义

| 指标 | 公式 | 含义 |
|------|------|------|
| Cosine (per row) | `mean(cos_sim(a[i,:], b[i,:]))` | 方向一致性，1.0=完全一致 |
| Relative L2 | `||a-b||₂ / ||a||₂` | 相对误差幅度 |
| Max Abs Diff | `max(|a-b|)` | 最大单点偏差 |
| Mean Abs Diff | `mean(|a-b|)` | 平均偏差 |

### 5.2 精度评判标准

| 量化类型 | 期望 Cosine (per layer) | 期望 RelL2 (per layer) | 说明 |
|---------|------------------------|------------------------|------|
| BF16 vs BF16 | 1.000000 | 0.000000 | 基线，应 bitwise 一致 |
| FP8 vs BF16 | > 0.997 | < 0.07 | FP8 per-block 正常损失 |
| FP4 (mega_moe) vs BF16 | > 0.980 | < 0.40 | FP4 MoE 更大损失，可接受 |
| RTP-FP8 vs vLLM-FP8 | > 0.995 | < 0.10 | 两个 FP8 实现差异（√2×单边） |

**异常判断**：
- 如果 RTP-LLM 的 per-layer RelL2 明显大于 vLLM 同等量化的 RelL2 → 有 bug
- 如果 embedding 层 cosine ≠ 1.0 → 权重加载有问题
- 如果某一层突然跳变（前面 0.999 后面 0.95）→ 该层有特殊 kernel 问题

### 5.3 运行对比脚本

```bash
# 2-way: RTP-LLM vs vLLM-BF16
/opt/conda310/bin/python docs/hidden_align/compare_hidden.py

# 3-way: RTP-FP8 vs vLLM-FP8 vs vLLM-BF16
/opt/conda310/bin/python docs/hidden_align/compare_fp8.py
```

### 5.4 对比脚本核心逻辑

```python
import torch

def cosine_sim_per_row(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float(), b.float(), dim=-1
    ).mean().item()

def relative_l2(ref, test):
    diff = (ref.float() - test.float()).norm()
    ref_norm = ref.float().norm()
    return (diff / ref_norm).item()

# 加载 dumps
vllm = torch.load("vllm_dumps/short.pt", map_location="cpu")
rtp = torch.load("rtp_llm_dumps/short.pt", map_location="cpu")

# 逐 tensor 对比
for key in ["embed_out", "layer00_hidden", ..., "final_norm"]:
    vt = vllm["tensors"][key]
    rt = rtp["tensors"][key]
    print(f"{key}: cos={cosine_sim_per_row(vt, rt):.6f}, "
          f"rel_l2={relative_l2(vt, rt):.6f}")
```

---

## Step 6: 记录结果

### 6.1 必须记录的信息

1. **RTP-LLM 完整启动参数**（server args + env vars）
2. **vLLM 完整配置**（dtype, quantization, enforce_eager, tp）
3. **模型信息**（路径、层数、hidden_size、是否 MoE）
4. **每层精度表格**（cosine, rel_l2, max_diff）
5. **Token 匹配率**（前 N 个 token 一致数）
6. **结论**：是否在可接受范围内

### 6.2 文档模板

```markdown
## <配置名> 精度对齐结果

### 环境
- Hardware: <GPU型号>
- Branch: <git branch>
- Model: <模型路径> (layers=N, hidden=M)

### RTP-LLM 配置
环境变量：
\`\`\`bash
CUDA_VISIBLE_DEVICES=X
CHECKPOINT_PATH=...
MODEL_TYPE=...
DETERMINISTIC_GEMM=1
...
\`\`\`
Server Args：
\`\`\`bash
python -m rtp_llm.start_server --quantization ... --fp8_kv_cache 1 ...
\`\`\`

### vLLM 配置
\`\`\`python
LLM(model=..., dtype="bfloat16", quantization="fp8", enforce_eager=True, tp=1)
\`\`\`

### 结果
| Tensor | Cosine | RelL2 | MaxDiff |
|--------|--------|-------|---------|
| embed_out | 1.000 | 0.000 | 0.000 |
| layer00_hidden | ... | ... | ... |
| ... | ... | ... | ... |

Token match: X/Y (Z%)

### 结论
<是否通过，是否有异常>
```

---

## 常见量化配置映射

| Smoke Case | RTP-LLM --quantization | MOE_STRATEGY | vLLM 对比配置 |
|-----------|------------------------|--------------|--------------|
| `mla_fp8_basic` | (无，模型自带FP8) | auto | `quantization="fp8"` |
| `mla_cp_pd` | (无，模型自带FP8) | DeepEP | `quantization="fp8"` |
| `mla_mega_moe_cp_pd` | (无) | mega_moe | BF16 + manual FP4 对比 |
| `mla_mega_moe_fp8_attn_cp_pd` | FP8_PER_BLOCK_NO_MOE | mega_moe | BF16（MoE 用 FP4） |
| `mla_load_quant_tp2` | FP8_PER_BLOCK | auto | `quantization="fp8"` |

### 使用 BF16 模型 + 在线量化 vs 预量化 FP8 模型

- **BF16 模型 + `--quantization FP8_PER_BLOCK`**：在线量化，适合精度对比（与 vLLM `quantization="fp8"` 对等）
- **预量化 FP8 模型**（如 `GLM-5-FP8-4layer`）：注意 safetensor 可能包含多余层，vLLM 直接加载可能报 KeyError

建议：**优先使用 BF16 模型 + 在线量化**，确保两个引擎从相同权重出发。

---

## 注意事项

1. **TP=1 对比**：精度对齐必须用 TP=1，避免 tensor 分片导致无法直接比较
2. **CUDA Graph 关闭**：`--enable_cuda_graph 0`，否则 tensor recording 不工作
3. **Warm up 关闭**：`--warm_up 0`，避免 warmup query 污染 dump
4. **Greedy decoding**：`temperature=0, top_k=1`，确保 sampling 不引入随机性
5. **DETERMINISTIC_GEMM=1**：确保 GEMM 结果确定性
6. **enforce_eager=True**（vLLM）：禁用 CUDA graph，确保 hook 生效
7. **每个 prompt 重启 server**：避免 KV cache 残留影响
8. **长序列需加大 MOEDBG_FULL_THRESHOLD**：默认 1M 不够 2k+ token

---

## 脚本文件清单

| 文件 | 用途 |
|------|------|
| `docs/hidden_align/vllm_dump_hidden.py` | vLLM BF16 hidden state dump |
| `docs/hidden_align/vllm_dump_fp8.py` | vLLM FP8 hidden state dump |
| `docs/hidden_align/rtp_llm_dump_hidden.py` | RTP-LLM mega_moe (FP8 attn + FP4 MoE) dump |
| `docs/hidden_align/rtp_llm_dump_fp8.py` | RTP-LLM FP8_PER_BLOCK dump |
| `docs/hidden_align/compare_hidden.py` | 2-way 对比（RTP vs vLLM-BF16） |
| `docs/hidden_align/compare_fp8.py` | 3-way 对比（RTP-FP8 vs vLLM-FP8 vs vLLM-BF16） |
| `docs/hidden_align/start_server_with_dist.py` | 单进程 dist init 包装器（已弃用） |
