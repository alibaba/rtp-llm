# 实验 Checkpoint: Qwen3.5-397B FP8_PER_CHANNEL_COMPRESSED 在线量化 OOM 修复 v2

独立于 `optimization_checkpoint.md`（FlyDSL Chunk-GDN 优化），聚焦在线 PTPC 量化的显存峰值问题。

## 环境
- framework: rtp-llm, local repo `/root/RTP-LLM/github-opensource`
- hardware: AMD MI300X × 4, 192 GiB per GPU
- model: Qwen3.5-397B-A17B (MoE 128 experts, linear attention), TP4, DP1, EP1
- quantization: FP8_PER_CHANNEL_COMPRESSED (online PTPC, BF16 checkpoint → FP8 runtime)
- checkpoint: 94 × 8.1 GiB safetensors (BF16), ~761 GiB total
- FP8 model estimate: ~91.65 GiB per GPU (TP4)

## 启动命令
```bash
cd /root/RTP-LLM/github-opensource
USE_FLYDSL=1 ENABLE_FP32_LM_HEAD=0 USE_SWIZZLEA=1 REUSE_CACHE=1 SEQ_SIZE_PER_BLOCK=1024 KERNEL_SEQ_SIZE_PER_BLOCK=16 WARM_UP=0 CONCURRENCY_LIMIT=128 ENABLE_CUDA_GRAPH=1 LOAD_PYTHON_MODEL=1 USE_ASM_PA=0 WORLD_SIZE=4 DP_SIZE=1 TP_SIZE=4 EP_SIZE=1 DEVICE_RESERVE_MEMORY_BYTES=-21474836000 RESERVER_RUNTIME_MEM_MB=10240 AITER_ASM_DIR=/opt/conda310/lib/python3.10/site-packages/aiter_meta/hsa/ MAX_SEQ_LEN=262144 START_PORT=8066 ACT_TYPE=bf16 TOKENIZER_PATH=~/Qwen3.5-397B-A17B CHECKPOINT_PATH=~/Qwen3.5-397B-A17B MODEL_TYPE=qwen35_moe FT_SERVER_TEST=1 ROCM_DISABLE_CUSTOM_AG=True FT_DISABLE_CUSTOM_AR=True QUANTIZATION=FP8_PER_CHANNEL_COMPRESSED /opt/conda310/bin/python3.10 -m rtp_llm.start_server 2>&1 | tee output.txt
```

## 问题描述

### 根因分析（承接 optimization_checkpoint.md 的调试记录）

启动 OOM 发生在 fastsafetensors 加载路径的 `_broadcast_per_expert` 阶段：

1. **`_is_memory_enough_for_fastsafetensor`** 用 FP8 model_mem (~91.65 GiB) 评估显存够用 → 选择 fastsafetensors 模式
2. fastsafetensors 从 safetensor 文件加载 BF16 权重到 GPU → 通过 `_broadcast_per_expert` 逐 expert 广播
3. `TensorCollector` 累积 BF16 expert 张量（每个 MoE weight 有 128 个 expert）
4. **峰值显存** = 累积的 BF16 collector 张量 + fastsafetensors 内部文件缓冲 ≈ 183 GiB >> 192 GiB → OOM
5. Fix F3 的 `_load_moe_inline_quant` 在 weight.load() 阶段才量化，但 OOM 在更早的 fastsafetensors 迭代层

### 之前的修复记录

| Fix | 状态 | 描述 | 问题 |
|-----|------|------|------|
| F1 | ✅ 已合入 | deepcopy pickle 崩溃 → `params.copy()` | 无 |
| F2 | ✅ 已合入 | fastsafetensors 版本安装 | 无 |
| F3 | ✅ 代码保留 | MoE inline FP8 量化 `_load_moe_inline_quant` | OOM 在更早阶段，代码 0 次执行 |
| F4 | ❌ 废弃 | 回退到 scratch 模式 | scratch 太慢，用户明确拒绝 |

## 修复方案 v2: 在线量化时 fastsafetensors 加载路径内联 FP8 量化

### 参考: sglang commit `6d98b53591b276fffe2aff75d193110224243e0f`

sglang 的 "load-time quantization" 核心思路：
- **Linear 层**: `LoadTimeFp8LinearWeightParameter` 在 weight_loader 阶段逐 shard 量化 BF16→FP8，参数预分配为 FP8 dtype
- **MoE 层**: `_maybe_load_time_fp8_moe_weight` 拦截每个 expert 加载，`load_weight_shard` 逐 expert 即时量化
- **关键**: 参数存储预分配为 FP8（内存减半），只有当前 shard 的 BF16 临时存在

### rtp-llm 适配方案

在 fastsafetensors 迭代循环中，对 MoE PTPC 权重即时量化：

```
fastsafetensors yield (key, BF16 tensor)
  ↓
检测: 是否为在线 PTPC 的 MoE weight?
  ↓ Yes                    ↓ No
per_channel_cast_to_fp8    直接 store_tensor
  → fp8 + scale
  ↓
collector.store_fp8_quantized(key, fp8, scale)
  ↓
BF16 tensor 立即释放
  ↓
collector 完成后 → _load_moe_inline_quant 检测已有 FP8 → 跳过二次量化
```

**显存峰值**: FP8 model (~92 GiB) + 一个文件缓冲 (~8 GiB) + overhead ≈ 110 GiB << 192 GiB

### 改动文件

| 文件 | 改动 | 说明 |
|------|------|------|
| `rtp_llm/model_loader/tensor_source.py` | TensorCollector 增加 FP8 预量化存储 | `store_fp8_quantized()`, `get_scale()`, `has_prequantized_scale()` |
| `rtp_llm/model_loader/loader.py` | `_load_from_fastsafetensor` 内联量化 | 检测在线 PTPC MoE weight，即时 BF16→FP8 |
| `rtp_llm/model_loader/loader.py` | `_is_memory_enough_for_fastsafetensor` | 内联量化模式下不 double model_mem |
| `rtp_llm/model_loader/per_channel_fp8_quant_weight.py` | `_load_moe_inline_quant` 适配 | 检测 collector 已有 FP8 数据，直接使用 |

## 实验记录

### Fix F5: fastsafetensors 内联 FP8 量化 (第二轮测试)
- 状态: 第二轮测试 — 第一轮 46%/11 shards OOM (189 GiB allocated)
- 改动:
  - `tensor_source.py`: TensorCollector 增加 `store_fp8_quantized()`, `has_prequantized_scale()`, `get_scale()`; `load_tensor()` 对预量化数据跳过 dtype 转换
  - `loader.py`: `_is_online_ptpc()`, `_should_inline_fp8_quantize()` 检测方法; `_load_from_fastsafetensor` 循环中对 MoE PTPC 权重即时 BF16→FP8; `_is_memory_enough_for_fastsafetensor` 内联量化模式不 double model_mem
  - `per_channel_fp8_quant_weight.py`: `_load_moe_inline_quant` 检测 collector 已有 FP8 数据时跳过二次量化，直接 copy FP8+scale
- 验证:
  - py_compile: 3 个文件均通过
  - TensorCollector 单元测试: store_fp8_quantized / load_tensor(fp8不转dtype) / has_prequantized_scale / get_scale / clear 全部通过
  - per_channel_cast_to_fp8_expert: [640,5120] BF16→FP8 内存节省 50%
- 参考: sglang `6d98b53591b276fffe2aff75d193110224243e0f`
- 第一轮测试 (仅 MoE 内联量化):
  - 加载到 46% (11/24 shards) 时 OOM，比之前更远但仍不够
  - OOM 发生在 `per_channel_cast_to_fp8_expert()` — 189.14 GiB allocated by PyTorch
  - 根因: PyTorch caching allocator 未释放已 freed 的 BF16 临时张量
- 第二轮修复:
  - 扩展内联量化: 从仅 MoE → 所有 `LoadQuantPerChannelFp8Weight` (含 attention, shared FFN 等)
  - 添加 `torch.cuda.empty_cache()`: 每个 weight 处理完后调用，归还 PyTorch 缓存给 GPU
  - 参考 sglang 同样在 `process_load_time_quantized_weights` 中调用 `empty_cache()`

### 测试轮次记录

| 轮次 | 关键改动 | 结果 | 分析 |
|------|---------|------|------|
| R1 | 仅 MoE inline quant | GPU OOM 46% (189 GiB) | 非 MoE 权重 BF16 累积 + PyTorch cache |
| R2-R5 | 扩展到所有 PTPC + empty_cache + expandable_segments | GPU OOM 46% (189 GiB) | `get_tensor_names` bug: scale key 在 checkpoint 不存在，collector 永远无法 complete，FP8 数据无限累积 |
| R6 | FP8 存 CPU | cgroup CPU OOM 17% | CPU 内存也不够（4 rank × full-size FP8 collector） |
| R7 | **`get_tensor_names` 修复** + 存 GPU | fastsafetensors 100% 完成！但 cgroup CPU OOM 在后续 weight 处理 | `_load_moe_inline_quant` 对 gate_up_proj (intermediate_weight13) 返回 None，fallback 创建巨大 BF16 stacked tensor |
| R8 | + fallback 日志 | fastsafetensors 83%，无 fallback weight，但 cgroup OOM | 所有 weight 在 loop 中完成但 MoE 处理导致 CPU 内存超 cgroup 限制 |

### 关键 Bug 定位

1. **`get_tensor_names` 返回多余 scale key (已修复)**
   - `LoadQuantPerChannelFp8Weight` 的 `CompositeWeight.get_tensor_names()` 返回 kernel + scale 两组 key
   - 在线 PTPC 中 scale key 在 checkpoint 不存在
   - collector 永远无法 complete，所有 FP8 expert 数据无限累积
   - 修复：override `get_tensor_names()` 只返回 `self.kernel.get_tensor_names()`

2. **gate_up_proj 不走 `_load_moe_inline_quant` (待修复)**
   - 日志只显示 `intermediate_weight2` (down_proj) 的 inline quant
   - `intermediate_weight13` (fused gate_up_proj) 通过 fallback 路径处理
   - fallback 创建 `[512, 2048, 4096]` BF16 stacked tensor (~8.59 GiB) → 高内存压力
   - 需要确认 `_process_fun_name` 和 `_load_moe_inline_quant` 的处理逻辑

3. **cgroup CPU OOM (待修复)**
   - fastsafetensors SHM + 4 rank 进程 + weight 处理 CPU 内存超限
   - 可能需要更激进的内存管理或减少 fastsafetensors SHM 使用

### Round 11 — 全部修复到位，加载成功！
- 状态: **权重加载完全成功** ✅
- 结果:
  - fastsafetensors 100% 加载 (24/24 shards in 3:46)
  - 480 inline MoE FP8 quant, **全部 prequantized=True** (60 layers × 2 weights × 4 ranks)
  - 无 GPU OOM, 无 cgroup CPU OOM
- 后续错误: `No suitable MOE strategy found for Fp8PerChannelCompressedQuantConfig` — MoE runtime 不支持 PTPC quant config，与加载无关

## 当前状态
- 当前 Phase: **权重加载 OOM 已修复** ✅
- 后续问题 1 已修复: MoE runtime strategy — aiter flydsl 版本不兼容 (0.1.1.dev409 vs 0.1.7)，已 patch aiter __init__.py 容错
  - 根因: commit 97987115e 集成 flydsl 0.1.7 后，`is_flydsl_available()` 返回 True → aiter 的 flydsl __init__ 做版本检查 → 期望 0.1.1.dev409 但得到 0.1.7 → ImportError → MoE executor 无法导入
  - 之前没装 flydsl 时 `is_flydsl_available()` 返回 False，整个检查跳过，所以 MoE 正常
  - 修复: patch `/opt/conda310/lib/python3.10/site-packages/aiter/ops/flydsl/__init__.py` 版本放行 + import 容错
  - 长期方案: 升级 aiter 到支持 flydsl 0.1.7 的版本

### Final Test — 服务启动成功
- 480/480 inline MoE FP8 quant, 全部 prequantized=True
- 4 rank 进程全部运行
- `/v1/models` 返回 qwen35_moe ✓
- GPU 内存: 180-185 GiB/GPU (192 GiB 总量)
- protobuf 字段不匹配 (`combo_token_size`) — 源码 pb2.py 需从 bazel-bin 更新 → 已修复
- 推理请求: aiter attention kernel SIGABRT "KV cache K offset overflow: exceed int32 max" — MAX_SEQ_LEN=262144 + 大模型配置下 KV cache 地址计算溢出 int32，与本 PTPC OOM 修复无关
- 改动文件总结:
  1. `tensor_source.py` — TensorCollector 增加 FP8 预量化存储
  2. `loader.py` — 3 个修复: 内联量化 + stacked_key_config 不覆盖 + get_tensor_names 内存检查
  3. `per_channel_fp8_quant_weight.py` — 3 个修复: get_tensor_names override + transpose_stack_moe_w1 支持 + 预量化检测
