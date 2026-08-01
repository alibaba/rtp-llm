# Speculative Decoding

---

## 1. What is Speculative Sampling
Speculative sampling is a **zero-precision-loss** universal inference acceleration technique:
1. A lightweight **Propose Model** generates several candidate tokens at once;
2. The original **Score Model (large model)** then validates these tokens in parallel;
3. Turning "verification" into a Prefill operation, thereby improving GPU compute-to-memory ratio and reducing Decode latency.

---

## 2. Speculative Sampling Algorithms Supported by RTP-LLM

| Name | Introduction | Source |
|---|---|---|
| **vanilla** | The most classic speculative sampling implementation | [Leviathan et al., ICML'23](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf) |
| **deterministic** | Prompt-Lookup + Speculative Edit | [Prompt Lookup](https://github.com/apoorvumang/prompt-lookup-decoding), [Cursor Blog](https://fireworks.ai/blog/cursor) |
| **mtp** | Speculative sampling framework based on DeepSeek-V3 | [DeepSeek-V3 Tech Report](https://arxiv.org/pdf/2412.19437) |
| **eagle3** | EAGLE-3 | [EAGLE-3 Paper](https://arxiv.org/pdf/2503.01840) |
| **dspark** | Block-diffusion parallel drafting (DeepSeek-V4 DSpark) | [vLLM implementation](https://github.com/vllm-project/vllm) |

---

## 3. Using Speculative Sampling in RTP-LLM

Based on the original basic startup parameters, add the following environment variables:

#### vanilla

| Arguments | Value | Description |
|-----------|---------------|-------------|
| --sp_type | vanilla | Speculative sampling strategy |
| --sp_checkpoint_path | <small model ckpt> | Small model weight path |
| --sp_model_type | qwen | Small model architecture, same as the main model |
| --sp_quantization | FP8_PER_BLOCK/FP8 | Small model quantization method: FP8, FP8_PER_BLOCK, etc. |
| --gen_num_per_cycle | 5 | How many tokens the small model proposes per cycle |

#### deterministic

| Arguments | Value | Description |
|-----------|---------------|-------------|
| --sp_type | deterministic | Speculative sampling strategy |
| --gen_num_per_cycle | 128 | How many tokens the small model proposes per cycle |
| --sp_min_token_match | 2 | Minimum length of n-gram token matching |
| --sp_max_token_match | 2 | Maximum length of n-gram token matching |

```
And supplement in the request's `extra_config`:
```json
{
  "sp_advice_prompt": "<text you expect the LLM to continue generating>",
  "sp_edit": 0          // 0=regular Prompt-Lookup; 1=Speculative Edit
}
```

#### mtp
| Arguments | Value | Description |
|-----------|---------------|-------------|
| --sp_type | mtp | Speculative sampling strategy |
| --sp_checkpoint_path | <small model ckpt> | Small model weight path |
| --sp_model_type | qwen_2_mtp | MTP small model type |
| --gen_num_per_cycle | 5 | How many tokens the small model proposes per cycle |
| --sp_quantization | FP8_PER_BLOCK/FP8 | Small model quantization method: FP8, FP8_PER_BLOCK, etc. |


#### eagle3
| Arguments | Value | Description |
|-----------|---------------|-------------|
| --sp_type | eagle3 | Speculative sampling strategy |
| --sp_checkpoint_path | <small model ckpt> | Small model weight path |
| --sp_model_type | qwen_3_moe_eagle3 | EAGLE3 small model type |
| --gen_num_per_cycle | 5 | How many tokens the small model proposes per cycle |
| --sp_quantization | FP8_PER_BLOCK/FP8 | Small model quantization method: FP8, FP8_PER_BLOCK, etc. |


#### dspark

DSpark drafts a whole block of tokens in **one parallel draft forward** (an anchor plus noise
query block over a 3-layer draft stack fed with target hidden states), then adds intra-block
dependency with a lightweight sequential Markov head. For DeepSeek-V4 the draft weights ship
inside the target checkpoint (`mtp.*` tensors), so `--sp_checkpoint_path` points at the **same
directory** as the main model.

| Arguments | Value | Description |
|-----------|---------------|-------------|
| --sp_type | dspark | Speculative sampling strategy |
| --sp_model_type | deepseek_v4_dspark | Draft model type (qwen_3_dspark / qwen_3_dflash for speculators-format checkpoints) |
| --sp_checkpoint_path | <target ckpt dir> | For DeepSeek-V4, the main model checkpoint directory itself |
| --sp_act_type | bf16 | Draft activation type |
| --gen_num_per_cycle | 5 | Tokens proposed per cycle; set to the checkpoint's `dspark_block_size` |
| --sp_dspark_propose_num | (optional) | Override proposal count k; defaults to the checkpoint's `dspark_block_size` |
| --draft_sample_method | greedy / probabilistic | greedy: token-only argmax proposals verified by implicit one-hot rejection; probabilistic: sequential Markov Gumbel proposals verified by standard p/q rejection sampling (`gumbel` is accepted as a vLLM-compatible alias) |
| --use_fp64_gumbel | 0 / 1 | vLLM-compatible fp64 Gumbel noise for probabilistic drafting (slower; default fp32) |

Draft hyperparameters (block size, target aux layer ids, noise token id, Markov head rank) are
read from the checkpoint `config.json` (`dspark_*` fields) and validated at startup. DSpark
composes with prefill/decode disaggregation, context/data parallelism, structured output
(xgrammar), and per-request `random_seed`.

Current limitations:
- The DeepSeek-V4 **target-side** CUDA graph falls back to eager execution (the draft-side
  graph is enabled); under investigation.
- Draft checkpoints with a reduced draft vocabulary (non-empty draft-to-target id mapping)
  are rejected at startup.
- `probabilistic` drafting is implemented and unit-tested; end-to-end distribution validation
  against vLLM is in progress.

---

## 4. Performance Observation & Tuning


### 4.1 Performance Observation
Add the following to the request body:
```json
"aux_info": true
```
Example response fields:
```text
cost_time   : 123 ms      // End-to-end
output_len  : 60 tokens   // Number of output tokens
iter_count  : 12          // Speculative rounds
avg_tokens_per_iter = output_len / iter_count = 5 // Average tokens accepted per round
```
The most important metric for speculative sampling is avg_tokens_per_iter, higher is better.

### 4.2 Tuning
#### vanilla
1. **Model Selection**:
   - Choose a smaller size from the same series (e.g., Qwen2.5-0.5B).
   - Apply INT4 quantization to the small model whenever possible.
2. **gen_num_per_cycle**:
   - Default is 5; can be increased if acceptance rate >40%.

#### deterministic
| Parameter | Recommendation | Notes |
|---|---|---|
| sp_min/max_token_match | 2 | n-gram length range |
| gen_num_per_cycle | 128 (batch=1) | Can be increased for long sequence editing scenarios |
| sp_edit | Set to 1 for code/text editing, 0 otherwise | Controls matching start point |
| sp_advice_prompt | Only retain suffixes that may actually appear | Reduce invalid matches |


#### mtp / eagle3
1. **Model Training**:
   - Use https://github.com/SafeAILab/EAGLE to train small models for specific business scenarios
   - Need to ensure 1st token acceptance rate >80%, 2nd token acceptance rate >60%, 3rd token acceptance rate >40%
2. **gen_num_per_cycle**:
   - Execution time of MTP small model can be assumed to be about 1ms. Based on the main model's execution time and acceptance rate, the optimal GEN_NUM_PER_CIRCLE can be calculated
3. **sp_quantization**
   - On Hopper series, it is recommended to enable sp_quantization=FP8_PER_BLOCK

#### dspark
1. Watch `avg_tokens_per_iter` (see 4.1). With DeepSeek-V4-Flash-0731 greedy drafting (k=5),
   a representative code-generation workload measures ~5 tokens per iteration.
2. `--gen_num_per_cycle` should match the checkpoint's `dspark_block_size`; the draft block is
   a fixed shape, so lowering `--sp_dspark_propose_num` trades acceptance length for a smaller
   verify batch but does not shrink the draft forward.