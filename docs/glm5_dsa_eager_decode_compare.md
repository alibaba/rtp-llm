# GLM5 DSA Eager Decode 调用链对比

日期：2026-05-23

本文整理当前 RTP-LLM、SGLang、vLLM 三套实现里 GLM5 DSA 在 eager decode 场景下的主要代码链路，重点覆盖 attention、DSA indexer、sparse MLA backend 和 CUDA graph 相关接口。

## 结论

三套实现的核心计算形态一致：

```text
MLA q/kv 投影
-> DSA indexer 生成 topk token/page
-> sparse MLA attention 只看 topk
-> absorbed MLA output BMM
```

主要差异在接口归属：

| 实现 | topk 传递方式 | attention backend 归属 | CUDA graph 归属 |
| --- | --- | --- | --- |
| RTP-LLM | `MlaAttention` 显式拿到 `topk_indices`，再传给 `fmha_impl.forward(...)` | `SparseMlaImpl` | C++ `CudaGraphRunner` + Python `prepare_cuda_graph` |
| SGLang | `topk_indices` 作为 `RadixAttention` kwarg 传入 | `DeepseekSparseAttnBackend` | DSA attention backend 自己维护 graph metadata |
| vLLM | indexer 写共享 `topk_indices_buffer`，backend 从 buffer 读 | `FlashMLASparseImpl` / `FlashInferMLASparseImpl` | `CudagraphDispatcher` / wrapper + attention metadata builders |

## RTP-LLM 当前实现

### 模型入口

GLM5 DSA 在 RTP 中复用 DeepSeekV2 / generic MoE 模型路径：

```text
rtp_llm/models/deepseek_v2.py
-> register_model("glm_5", DeepSeekV2, ["GlmMoeDsaForCausalLM"])
-> DeepSeekV2._create_python_model(...)
-> GenericMoeModel
```

关键位置：

- `rtp_llm/models/deepseek_v2.py`: `DeepSeekV2.support_cuda_graph()` 返回 `True`，`register_model("glm_5", ...)` 注册 GLM5 DSA。
- `rtp_llm/models_py/model_desc/generic_moe.py`: `GenericMoeModel.forward()` 中逐层调用 decoder layer。

### eager decode attention 链路

RTP 的 decode eager 链路比较直接：

```text
GenericMoeModel.forward
-> select_block_map_for_layer(inputs.attention_inputs, layer_id)
-> GenericMoeDecoderLayer.forward
-> MlaAttention.forward
-> MlaAttention._run_sparse_indexer(...)
-> fmha_impl.forward(q_view, compressed_kv, k_pe, kv_cache, layer_idx, topk_indices)
```

在 `MlaAttention.forward()` 中：

1. `fused_qkv_a_proj` 或 `fused_qkv_proj` 生成 q 和 compressed kv。
2. q 进入 `q_b_proj`，kv 进入 `kv_a_layernorm`。
3. `_run_sparse_indexer(...)` 生成 `topk_indices`。
4. `topk_indices` 被直接作为参数传给 `fmha_impl.forward(...)`。

关键位置：

- `rtp_llm/models_py/model_desc/generic_moe.py`: `GenericMoeModel.forward()`
- `rtp_llm/models_py/modules/hybrid/mla_attention.py`: `MlaAttention.forward()`

### DSA indexer

decode 下 RTP indexer 走 paged topk：

```text
MlaAttention._run_sparse_indexer
-> Indexer.forward
-> Indexer._compute_topk
-> IndexerOp._get_topk_paged
-> deep_gemm.fp8_paged_mqa_logits
-> fast_topk_transform_fused
```

`IndexerOp._get_topk_paged()` 中主要逻辑：

1. 将 indexer KV cache reshape 成 DeepGEMM 期望布局。
2. 将 `fmha_params.kvlen_d` unsqueeze 成 `[B, 1]`。
3. 调 `deep_gemm.get_paged_mqa_logits_metadata(...)` 生成 schedule metadata。
4. 调 `deep_gemm.fp8_paged_mqa_logits(...)` 得到 logits。
5. 调 `fast_topk_transform_fused(...)` 生成 topk result。

关键位置：

- `rtp_llm/models_py/modules/hybrid/indexer.py`: `Indexer.forward()` / `_compute_topk()`
- `rtp_llm/models_py/modules/base/cuda/indexer_op.py`: `_get_topk_paged()`

### sparse MLA backend

RTP 的 sparse MLA backend 在 `SparseMlaImpl.forward()` 中完成：

```text
SparseMlaImpl.forward
-> fused_qk_rope_cat_cache_mla 或 rope + kv_cache_write
-> apply_write_cache_store
-> q_nope @ W_kc
-> fmha sparse op
-> attn_output @ W_vc
```

BF16 和 FP8 路径不同：

- BF16 路径一般走 `flash_mla_sparse_fwd`。
- FP8 KV cache 路径一般走 `flash_mla_with_kvcache`。
- decode 下 FP8 path 会使用 paged KV cache 形态。

关键位置：

- `rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashmla_sparse_impl.py`: `SparseMlaImpl.forward()`
- 同文件中 `SparseMlaOp` / `SparseMlaFp8Op` 负责具体 sparse FlashMLA 调用。

### CUDA graph

RTP 的 CUDA graph 是 C++ 侧主导：

```text
PyWrappedModel.forward
-> graph_runner_->canRun(...)
-> graph_runner_->forward(...)
-> CudaGraphRunner::prepareAttentionInputs(...)
-> Python attn_pyobj.prepare_cuda_graph(attention_inputs)
-> SparseMlaImpl.prepare_cuda_graph(...)
```

普通 eager forward 不走 graph：

```text
py_model_.attr("prepare_fmha_impl")(py_model_inputs, false)
-> py_model.forward(py_model_inputs, held_attn_pyobj_)
```

graph capture decode 时：

```text
cuda_graph_decode.cc
-> py_attn_pyobj_method_(..., true)
-> captureDecodeOneBatchSize(...)
```

注意点：

- `CudaGraphRunner::prepareAttentionInputs()` 会把 seq lens、block table、decode cu seqlens 等拷贝到 capture 时的静态 buffer。
- 随后调用 Python `prepare_cuda_graph`，`SparseMlaImpl.prepare_cuda_graph()` 内部 `forbid_realloc=True`。
- target verify 被显式禁止走 CUDA graph，因为 sparse MLA target verify 使用 ragged topk，当前注释说明不是 graph-safe。

关键位置：

- `rtp_llm/cpp/models/PyWrappedModel.cc`: graph / normal forward 分支。
- `rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc`: `prepareAttentionInputs()` / `canRun()`。
- `rtp_llm/cpp/cuda_graph/cuda_graph_decode.cc`: decode capture。
- `rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashmla_sparse_impl.py`: `prepare_cuda_graph()`。

## SGLang 实现

### 模型入口

SGLang 的 GLM5 DSA 入口同样复用 DeepSeekV2：

```text
python/sglang/srt/models/glm4_moe.py
-> class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)
```

`GlmMoeDsaForCausalLM` 基本只覆盖 shared expert 相关逻辑，attention 主链路走 DeepSeekV2 MLA。

关键位置：

- `/home/zw193905/sglang/python/sglang/srt/models/glm4_moe.py`
- `/home/zw193905/sglang/python/sglang/srt/models/deepseek_v2.py`

### attention / indexer 链路

SGLang 在 `DeepseekV2AttentionMLA` 中判断 `use_dsa` 并创建 `Indexer`：

```text
DeepseekV2AttentionMLA.__init__
-> self.use_dsa = is_deepseek_dsa(config)
-> self.indexer = Indexer(...)
```

它还支持 index cache/topk reuse：

```text
index_topk_freq / index_topk_pattern
-> skip_topk
-> next_skip_topk
```

decode eager 主链路：

```text
DeepseekV2AttentionMLA.forward
-> forward_prepare
-> forward_absorb_prepare
-> Indexer.forward_cuda
-> _get_topk_paged
-> forward_absorb_core
-> self.attn_mqa(..., topk_indices=topk_indices)
-> RadixAttention
-> DeepseekSparseAttnBackend.forward_decode
```

关键位置：

- `/home/zw193905/sglang/python/sglang/srt/models/deepseek_v2.py`: `DeepseekV2AttentionMLA`
- `/home/zw193905/sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`: `forward_absorb_prepare()` / `forward_absorb_core()`
- `/home/zw193905/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py`: `Indexer.forward_cuda()`

### DSA backend

SGLang 的 DSA backend 接收 `topk_indices` 后，会根据配置决定是否直接使用：

```text
DeepseekSparseAttnBackend.forward_decode
-> if SGLANG_DSA_FUSE_TOPK:
       page_table_1 = topk_indices
   else:
       page_table_1 = transform_index_page_table_decode(...)
-> flashmla_sparse / flashmla_kv / tilelang / fa3 / trtllm
```

backend 支持多种 DSA decode impl：

- `flashmla_sparse`
- `flashmla_kv`
- `tilelang`
- `fa3`
- `trtllm`

关键位置：

- `/home/zw193905/sglang/python/sglang/srt/layers/attention/dsa_backend.py`: `forward_decode()`
- 同文件中 `_forward_flashmla_sparse()` / `_forward_flashmla_kv()` 等具体实现。

### CUDA graph

SGLang 的 CUDA graph metadata 由 DSA backend 维护：

```text
init_cuda_graph_state
-> init_forward_metadata_capture_cuda_graph
-> init_forward_metadata_replay_cuda_graph
```

replay 时会原地更新：

- `cache_seqlens_int32`
- `cu_seqlens_k`
- `page_table_1`
- `dsa_cache_seqlens_int32`
- `dsa_cu_seqlens_k`
- DeepGEMM paged MQA schedule metadata
- FlashMLA metadata

关键点：

- SGLang 在 graph replay 外部更新 DeepGEMM schedule metadata。
- `maybe_capture_indexer_topk(...)` 用于 indexer topk capture/reuse 相关状态。
- target verify / draft extend / decode 都在 DSA backend metadata 中有对应处理。

关键位置：

- `/home/zw193905/sglang/python/sglang/srt/layers/attention/dsa_backend.py`: `init_cuda_graph_state()` / `init_forward_metadata_replay_cuda_graph()`

## vLLM 实现

### 模型入口

vLLM 注册 GLM5 DSA 到 DeepSeekV2：

```text
vllm/model_executor/models/registry.py
-> "GlmMoeDsaForCausalLM": ("deepseek_v2", "GlmMoeDsaForCausalLM")

vllm/model_executor/models/deepseek_v2.py
-> class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM): pass
```

关键位置：

- `/home/zw193905/vllm/vllm/model_executor/models/registry.py`
- `/home/zw193905/vllm/vllm/model_executor/models/deepseek_v2.py`

### attention / indexer 链路

vLLM 的 `DeepseekV2MLAAttention` 如果发现 config 有 `index_topk`，会创建：

- `Indexer`
- `indexer_rope_emb`
- 共享 `topk_indices_buffer`

然后把这些塞进 `MLAModules` 和 `MultiHeadLatentAttentionWrapper`：

```text
DeepseekV2MLAAttention
-> Indexer(...)
-> MLAModules(indexer=..., topk_indices_buffer=...)
-> MultiHeadLatentAttentionWrapper
-> MLAAttention(..., use_sparse=True, indexer=indexer)
```

vLLM 的关键设计是：`Indexer.forward()` 不把 topk 返回给 caller 使用，而是写共享 `topk_indices_buffer`。之后 sparse attention backend 直接从这个 buffer 读取。

decode eager 主链路：

```text
GlmMoeDsaForCausalLM
-> DeepseekV2MLAAttention.forward
-> MultiHeadLatentAttentionWrapper.forward
-> if indexer and sparse and not skip_topk:
       indexer(hidden_states, q_c, positions, indexer_rope_emb)
-> MLAAttention.forward
-> MLAAttention.forward_impl
-> impl.forward_mqa(...)
```

关键位置：

- `/home/zw193905/vllm/vllm/model_executor/models/deepseek_v2.py`: `Indexer` / `DeepseekV2MLAAttention`
- `/home/zw193905/vllm/vllm/model_executor/layers/mla.py`: `MultiHeadLatentAttentionWrapper.forward()`
- `/home/zw193905/vllm/vllm/model_executor/layers/attention/mla_attention.py`: `MLAAttention.forward_impl()`

### SparseAttnIndexer

vLLM 的 indexer 自身封成 custom op：

```text
Indexer.forward
-> wq_b(q_lora)
-> wk_weights_proj(hidden_states)
-> rotary_emb
-> per_token_group_quant_fp8
-> SparseAttnIndexer.forward_cuda
-> torch.ops.vllm.sparse_attn_indexer(...)
```

decode 下 `sparse_attn_indexer(...)` 从 forward context 里取对应 metadata：

```text
get_forward_context().attn_metadata[k_cache_prefix]
-> DeepseekV32IndexerMetadata
-> decode_metadata
-> fp8_fp4_paged_mqa_logits
-> persistent_topk 或 top_k_per_row_decode
-> topk_indices_buffer
```

关键位置：

- `/home/zw193905/vllm/vllm/model_executor/layers/sparse_attn_indexer.py`
- `/home/zw193905/vllm/vllm/v1/attention/backends/mla/indexer.py`

### sparse MLA backend

vLLM CUDA 平台 sparse MLA 后端主要有：

- `FLASHMLA_SPARSE`
- `FLASHINFER_MLA_SPARSE`

平台选择大致逻辑：

- SM100 + FP8 KV cache 时优先尝试 `FLASHINFER_MLA_SPARSE`，再 `FLASHMLA_SPARSE`。
- BF16 KV cache 时按 head count 选择优先级。
- 非 SM100 场景仍会尝试 `FLASHMLA_SPARSE`。

`FlashMLASparseImpl.forward_mqa()`：

```text
topk_indices = self.topk_indices_buffer[:num_actual_toks]
-> triton_convert_req_index_to_global_index(...)
-> BF16: flash_mla_sparse_fwd
-> FP8: flash_mla_with_kvcache
```

`FlashInferMLASparseImpl.forward_mqa()`：

```text
topk_indices = self.topk_indices_buffer[:num_actual_toks]
-> triton_convert_req_index_to_global_index(..., return_valid_counts=True)
-> trtllm_batch_decode_with_kv_cache_mla(...)
```

关键位置：

- `/home/zw193905/vllm/vllm/v1/attention/backends/mla/flashmla_sparse.py`
- `/home/zw193905/vllm/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`
- `/home/zw193905/vllm/vllm/v1/attention/backends/mla/sparse_utils.py`
- `/home/zw193905/vllm/vllm/platforms/cuda.py`

### CUDA graph

vLLM 的 CUDA graph 是全局 dispatcher/wrapper 机制：

```text
CudagraphDispatcher
-> initialize_cudagraph_keys(...)
-> dispatch(...)
-> FULL / PIECEWISE / NONE
```

运行时：

```text
GPUModelRunner
-> _get_cudagraph_mode(...)
-> pad_attn = cudagraph_mode == FULL
-> _build_attention_metadata(...)
-> set_forward_context(..., cudagraph_runtime_mode=..., batch_descriptor=...)
-> model forward
```

capture 时：

```text
capture_model
-> cudagraph_dispatcher.get_capture_descs()
-> _capture_cudagraphs(...)
-> _warmup_and_capture(...)
-> _dummy_run(..., is_graph_capturing=True)
```

attention metadata builder 会参与 graph-stable buffer 维护：

- `DeepseekV32IndexerMetadataBuilder`
  - `decode_lens_buffer`
  - `decode_seq_lens_buffer`
  - `expanded_block_table_buffer`
  - `scheduler_metadata_buffer`
  - compressed slot / seq lens buffer
- `FlashMLASparseMetadataBuilder`
  - `tile_scheduler_metadata_buffer`
  - `num_splits_buffer`
  - `req_id_per_token_buffer`
  - compressed slot mapping buffer
  - C128A topk buffers

vLLM 还支持 breakable CUDA graph：`@eager_break_during_capture` 可以把某些 custom op 作为 graph capture break 点，使其在 capture/replay 中作为 eager segment 执行。`sparse_attn_indexer` 使用了该装饰器。

关键位置：

- `/home/zw193905/vllm/vllm/v1/cudagraph_dispatcher.py`
- `/home/zw193905/vllm/vllm/v1/worker/gpu_model_runner.py`
- `/home/zw193905/vllm/vllm/v1/worker/gpu/attn_utils.py`
- `/home/zw193905/vllm/vllm/compilation/breakable_cudagraph.py`

## 三者对比

### topk 生成和传递

| 项 | RTP-LLM | SGLang | vLLM |
| --- | --- | --- | --- |
| topk 生成位置 | `MlaAttention._run_sparse_indexer` 内部 | `forward_absorb_prepare` / `Indexer.forward_cuda` | `MultiHeadLatentAttentionWrapper.forward` 调 `Indexer` |
| topk 传递 | 函数返回值，显式传 `fmha_impl.forward` | kwarg 传给 `RadixAttention` | 写共享 `topk_indices_buffer` |
| topk reuse | 当前 generic GLM5 DSA 路径未看到跨层 reuse | 支持 `skip_topk/next_skip_topk` | 支持 `skip_topk`，复用 buffer |
| topk 到 global slot | sparse op/backend 内转 | DSA backend 内转，或 fused topk 直接用 | sparse backend 用 Triton 转 |

### sparse attention backend

| 项 | RTP-LLM | SGLang | vLLM |
| --- | --- | --- | --- |
| BF16 sparse | `flash_mla_sparse_fwd` | `flashmla_sparse` 等 | `flash_mla_sparse_fwd` 或 FlashInfer |
| FP8 sparse | `flash_mla_with_kvcache` | `flashmla_kv` 等 | `flash_mla_with_kvcache` 或 TRTLLM/FlashInfer |
| 多后端调度 | RTP factory 选择 `SparseMlaImpl` | backend 支持 `flashmla_sparse/flashmla_kv/tilelang/fa3/trtllm` | platform backend priority 选择 |
| q absorb BMM | `SparseMlaImpl._apply_input_bmm` | attention forward/backend 内处理 | `MLAAttention.forward_impl` 中 `W_UK_T` / `forward_mqa` |

### CUDA graph

| 项 | RTP-LLM | SGLang | vLLM |
| --- | --- | --- | --- |
| graph 入口 | C++ `PyWrappedModel` / `CudaGraphRunner` | attention backend metadata methods | `CudagraphDispatcher` / `CUDAGraphWrapper` |
| metadata 更新 | C++ copy 到 static `PyAttentionInputs`，Python `prepare_cuda_graph` | backend replay 函数原地 copy | metadata builder + forward context |
| realloc 控制 | `prepare_cuda_graph(... forbid_realloc=True)` | 预分配 fixed graph tensors | 预分配 graph-stable buffers |
| target verify | RTP 当前禁用 graph | backend 有 target verify metadata 分支 | 取决于 dispatcher/backend 支持 |
| DeepGEMM schedule | indexer `_get_topk_paged` 内计算 | replay 外更新 metadata | indexer metadata builder 内维护 buffer |

## 和 DS4 / CSA 的关系

GLM5 DSA 与 DS4 / CSA 的共同点：

- 都是为了减少 decode attention 的有效访问范围。
- 都会在 attention 主路径外维护额外索引或压缩信息。
- CUDA graph 下都要保证 page table、seq lens、topk 或压缩 metadata 地址稳定。

主要区别：

- GLM5 DSA 的 sparse 选择发生在原始 token 维度，indexer 选择 topk token/page，然后 sparse MLA 只访问这些 token。
- DS4 / CSA 更偏 compressed / selected attention 的不同组织方式，attention 访问对象和 metadata 语义不完全等价。
- GLM5 DSA 的关键热路径是 `indexer logits -> topk -> sparse MLA`；CSA/DS4 的关键热路径还会涉及 compressed pool 或分层/块级选择逻辑。

在当前 RTP 代码里，GLM5 DSA 更接近 SGLang/vLLM 的 DeepSeek DSA 路线：都是 MLA absorb + indexer KV cache + paged topk + sparse FlashMLA。

## 后续排查建议

如果继续做性能或正确性对齐，可以重点看这几处：

1. RTP 是否需要引入 SGLang/vLLM 的跨层 topk reuse。
2. RTP 的 DeepGEMM paged MQA schedule metadata 是否可以提升到 batch-level metadata，避免每层 indexer 重算。
3. RTP 的 `topk_indices` 是否可以按 vLLM 的 shared buffer 方式稳定地址，减少 graph replay 下的参数变化。
4. RTP target verify 禁用 CUDA graph 是当前保守策略；如果要支持，需要先解决 ragged topk 的 graph-safety。
5. SGLang 的 `SGLANG_DSA_FUSE_TOPK` 和 vLLM 的 topk-to-global 转换可以作为 RTP topk transform 优化参考。

