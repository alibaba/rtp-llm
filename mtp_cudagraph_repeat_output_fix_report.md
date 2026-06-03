# MTP CUDA Graph 重复输出 Bug 修复报告

## 根因

在 MTP decode 的 target verify CUDA graph replay 中，`prepareSparseMlaTargetVerifyParamsKernel` 仅更新前 `batch_size` 个 batch entry 的 attention metadata（`qo_indptr`、`kvlen`、`batch_indice`、`expanded_seq_lens` 等）。CUDA graph 在 capture 时固定了 `captured_batch_capacity` 个 batch elements 的计算流程，replay 时始终处理所有 captured elements。

当运行时 batch_size 减小（某个请求生成完成退出 batch），超出当前 batch_size 的 entry 保留了上一轮 replay 的 stale metadata（如 `kvlen` 仍为几百、`qo_indptr` 仍指向有效 token 范围）。而 `CudaGraphRunner::prepareAttentionInputs` 已将这些 slot 对应的 block table 行零填充（block_id=0）。

block_id=0 是 KV cache block pool 中的有效 block，通常属于当前 batch 中的某个实际请求。attention kernel 在处理 stale batch entries 时，用 block_id=0 读取了其他请求的 KV cache 数据。由于 GPU tile-based attention 的 shared memory 或 work scheduling 机制，stale entries 的错误 attention 计算干扰了同 tile 内有效 token 的输出，导致模型生成内容被污染，最终陷入重复输出循环。

## 稳定复现场景

- 架构：GLM-5 MTP (gen_num_per_cycle=3) + MegaMoE + FP8 + PD 分离
- 配置：decode 端 dp_size=4, enable_cuda_graph=1, RTP_LLM_FORCE_SP_PREFILL_CUDA_GRAPH=1
- 压测：16 并发（每个 DP rank 4 个请求），4 种不同 prompt 的异构请求池
- 环境变量：RTP_LLM_STREAM_ASYNC=1, RTP_LLM_DROP_BROAD_SYNC=1, RTP_LLM_DEVICE_INPUT=1

复现率：1-2 轮（16-32 个请求）内稳定出现 repetition_events >= 1。

触发条件核心：batch 内请求生成长度不同 → 短请求先结束 → batch_size 缩小 → stale entries 残留。

## 修复方法

在 `prepareSparseMlaTargetVerifyParamsKernel`（cuda_graph_prepare.cu）中增加两个参数 `captured_batch_capacity` 和 `captured_total_tokens`，在主循环结束后添加零填充逻辑：

1. 对 `batch_size` 到 `captured_batch_capacity` 之间的 batch entries：
   - `kvlen[i] = 0`
   - `paged_kv_last_page_len[i] = 0`
   - `qo_indptr[i+1]`、`decode_page_indptr[i+1]`、`prefill_ragged_kv_len_indptr[i+1]` 设为最后有效值

2. 对 `token_offset` 到 `captured_total_tokens` 之间的 token entries：
   - `batch_indice[t] = 0`、`positions[t] = 0`
   - `slot_mapping[t] = -1`（无效 slot）
   - `expanded_seq_lens[t] = 0`（零长度 → 跳过注意力计算）
   - `topk_indices_offset[t] = 0`、`ks[t] = 0`、`ke[t] = 0`

不引入任何新的 cudaStreamSynchronize 或 D2H 操作——零填充在同一个 `<<<1,1>>>` kernel 内完成。

## 验证结论

| 配置 | 修复前 | 修复后 |
|------|--------|--------|
| CG + sync envs | FAIL (iter 1, 16 req, rep=1) | **PASS** (15 轮 × 16 并发 = 240 req, rep=0) |
| CG + no sync envs | FAIL (iter 2, 32 req, rep=1) | — |
| no CG + no sync envs | PASS (5 轮 × 16 = 80 req, rep=0) | — |

修复后在完整配置（CUDA graph + sync 优化 + FORCE_SP_PREFILL_CUDA_GRAPH）下，240 个请求零重复输出。
