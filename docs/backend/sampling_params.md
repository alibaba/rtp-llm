# Sampling Config

For the OpenAI protocol, **top_p** and **temperature** are specified in the outer protocol, while other sampling parameters are specified in **extra_configs**. Example:

```JSON
{
    "model":"Qwen_14B_pressure_test",
    "messages":[
        {
            "role":"system",
            "content":"You are a helpful assistant."
        },
        {
            "role":"user",
            "content":"Hello, what's the weather like in Hangzhou today"
        }
    ],
    "stream":true,
    "temperature":1,
    "max_tokens":1024,
    "top_p":0.8,
    "extra_configs" : {
        "top_k": 1
    }
}
```

The raw protocol specifies sampling parameters through **generate_config**. Example:
```JSON
{
    "model":"m6-13b-v1",
    "prompt":"Human: write a list for trip\n\nAssitant:","generate_config":{
        "top_k": 1,
        "top_p": 0
    }
}
```

## Basic Control Parameters

| Parameter Name     | Core Function                                                            |
|--------------------|--------------------------------------------------------------------------|
| temperature        | Controls sampling randomness:<br>→ 0: Deterministic mode<br>→ 1: Standard random mode |
| top_k              | Candidate set truncation strategy:<br>→ 0: Disabled<br>→ N: Take top N high-probability tokens |
| top_p              | Nucleus sampling strategy:<br>→ 0.95: Take candidate set with cumulative probability of 95% |
| max_new_tokens     | Maximum generation length:<br>MIN(input_length + max_new_tokens, MAX_SEQ_LEN) |
| min_new_tokens     | Enforces minimum generation length                                       |


## Advanced Control Parameters

### CUDA categorical sampling backend

Ordinary random sampling uses logits-domain Gumbel-max by default when there is
no TopK/TopP truncation and no probability output is requested. Temperature and
penalties are applied before sampling. This preserves the categorical distribution,
but a fixed seed can produce different tokens from the previous backend.

Requests requiring full probabilities,
log probabilities, truncated sampling, or mixed `do_sample` values retain the existing
backend. Greedy sampling retains its argmax path. Set
`RTP_LLM_USE_GUMBEL_SAMPLING=0` before starting the service to use the previous
random sampling backend. Sampling throughput depends on the batch and vocabulary
size; this switch does not change request sampling parameters.

| Parameter Name         | Function Description                                                     |
|------------------------|--------------------------------------------------------------------------|
| repetition_penalty     | Repetition suppression factor:<br>→ >1.0 suppresses repetition<br>→ <1.0 encourages repetition |default = 1.0 |
| frequency_penalty|  This parameter is used to discourage the model from repeating the same words or phrases too frequently within the generated text. It is a value that is added to the log-probability of a token each time it occurs in the generated text. A higher frequency_penalty value will result in the model being more conservative in its use of repeated tokens.| default = 0.0 |
|presence_penalty | This parameter is used to encourage the model to include a diverse range of tokens in the generated text. It is a value that is subtracted from the log-probability of a token each time it is generated. A higher presence_penalty value will result in the model being more likely to generate tokens that have not yet been included in the generated text. |default = 0.0 |
| stop_words_list        | Token ID stop words (better performance):<br>`[[20490,25],[1024]]`        |
| stop_words_str         | String stop words (better compatibility):<br>`["<end>","\nObservation"]` |
| random_seed            | Random seed control:<br>→ None: True random<br>→ Fixed value: Reproducible generation |



```python
# Stop Words Configuration Example
{
    "stop_words_str": ["<|im_end|>", "\nObservation:"],
    "stop_words_list": [[20490, 25], [50256]]
}
```

<a id="返回控制参数"></a>
## Return Control Parameters

| Parameter Name      | Effect                                 | Use Case                    |
|---------------------|----------------------------------------|-----------------------------|
| return_logits       | Returns logits matrix for each position| Output analysis/post-processing |
| return_hidden_states| Returns hidden states of transformer layers | Model debugging/feature extraction |
| return_input_ids    | Returns input sequence encoding result | Input validation           |
| return_output_ids   | Returns output sequence encoding result| Output decoding validation |

<a id="特殊模式参数"></a>
## Special Mode Parameters

| Mode Name            | Control Parameter        | Function Description                                                     |
|----------------------|--------------------------|--------------------------------------------------------------------------|
| Thinking Mode        | in_think_mode=True       | Agent-specific scenario:<br>Control thinking phase length with max_thinking_tokens |
| Streaming Output     | yield_generator=True     | Enable chunked return mechanism                                          |
| Parallel Decoding    | pd_separation=True       | Enable parallel decoding optimization (hardware support required)        |

<a id="环境变量说明"></a>
## Environment Variable Description

```bash
# Force override stop words configuration
export FORCE_STOP_WORDS=true
export STOP_WORDS_STR="[\"</end>\",\"\\n\"]"

# Hybrid mode (default)
export FORCE_STOP_WORDS=false  # union of environment variables, model defaults, and configuration parameters
```

<a id="采样策略说明"></a>
## Sampling Strategy Description

### Combined Strategy Example
```python
{
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "top_p_decay": 0.9,        # 10% decay per token
    "top_p_min": 0.5           # Minimum decay threshold
}
```

### Strategy Recommended Values
| Scenario       | temperature | top_p | top_k |
|---------------|------------|-------|-------|
| Code Generation| 0.2-0.4    | 0.9   | 40    |
| Creative Writing| 0.7-1.0   | 0.95  | 100   |
| Factual Q&A    | 0.1-0.3    | 0.8   | 20    |

<a id="参数使用注意事项"></a>
## Parameter Usage Notes

1. **Stop Words Selection Principles**
   - Performance priority: Use `stop_words_list` for high-frequency triggering scenarios
   - Compatibility priority: Use `stop_words_str` for complex pattern matching

2. **Length Limit Coordination**
   ```math
   Actual maximum length = min(
       input_token_len + max_new_tokens,
       MAX_SEQ_LEN
   )
   ```

3. **Thinking Mode Special Constraints**
   ```python
   # Need to configure simultaneously
   {
       "in_think_mode": True,
       "max_thinking_tokens": 512,  # Control thinking phase length
       "max_new_tokens": 2048       # Control total output length
   }
   ```

4. **Streaming Output Limitations**
   - Need to enable `yield_generator=True` simultaneously
   - `return_logits` only returns complete data in non-streaming mode

### Deferred random MTP verification

On CUDA with FP32 logits, MTP verification uses the vLLM V2 rejection kernels.
After RTP applies logits processors, penalties and temperature, the verifier
computes blockwise normalization statistics and checks the draft prefix. It draws
only the first rejected token from the residual distribution, or one bonus token
when the entire prefix is accepted. It does not sample every verification row.
Unfiltered verification does not materialize target softmax probabilities.
TopK/TopP filtering retains RTP's renormalization helpers, using temporary
probabilities and returning filtered logits to the verifier.

Dense draft probabilities are read directly by the verifier and converted to log
values inside its kernels, without a full temporary log-probability tensor.
Set `RTP_LLM_FUSE_DRAFT_PROBS_LOG=0` to restore the materialized conversion for
comparison or rollback; the default is enabled. This switch is independent of
adaptive verification and does not modify the caller's probabilities. Unsupported
input dtypes retain materialized conversion. Point-mass drafts need no dense
draft tensor. Random, greedy and mixed batches are supported.
Forced acceptance uses synthetic acceptance rates. A speculative acceptance cap
inserts an invalid draft placeholder so the capped position samples the target
directly. Invalid output tails are initialized before RTP's existing cleanup.

The route requires the Python sampling runtime. Set the following environment
variables before starting the service (only the literal value `0` disables):

| Ordinary Gumbel | MTP V2 rejection | Environment |
|---|---|---|
| Off | Off | `RTP_LLM_USE_GUMBEL_SAMPLING=0 RTP_LLM_USE_VLLM_REJECTION_SAMPLING=0` |
| Off | On | `RTP_LLM_USE_GUMBEL_SAMPLING=0 RTP_LLM_USE_VLLM_REJECTION_SAMPLING=1` |
| On | Off | `RTP_LLM_USE_GUMBEL_SAMPLING=1 RTP_LLM_USE_VLLM_REJECTION_SAMPLING=0` |
| On | On | `RTP_LLM_USE_GUMBEL_SAMPLING=1 RTP_LLM_USE_VLLM_REJECTION_SAMPLING=1` |

When the rejection variable is absent it inherits the ordinary variable, preserving
the existing disable-all switch. Both routes default to enabled. Each enabled route
prewarms at startup for the model vocabulary. Dependency/import or Triton compiler
failures select the legacy backend for that route and are logged. This is startup
selection, not retry after a request has consumed RNG or modified logits. CUDA
execution failures propagate. Other backends and unavailable Python runtimes
retain the existing path. Temperature remains before RTP penalties. Seeds remain reproducible within the selected route;
switching routes changes random-number consumption and does not preserve exact
generated text. RTP reserves generator offsets rather than using vLLM absolute
token positions. Optional vLLM block verification, FLy and watermarking are not
exposed as RTP request options by this integration.

The adaptive-verification controller has a separate startup opt-in,
`RTP_LLM_ADAPTIVE_VERIFICATION`. Unset or `0` returns no controller, without
importing the estimator or allocating its GPU buffers. Only `1` creates the
controller; other values raise an error. It does not read or modify either
sampling-backend switch above. Changing this value requires a new controller
at startup; it is not a live toggle for a batch with pending verification.

The MTP executor connects predictions, packed Target inputs, smaller target graph
capacities and actual acceptance feedback. This route requires the logits
rejection backend (`RTP_LLM_USE_VLLM_REJECTION_SAMPLING=1`) and paged KV attention;
unsupported configurations fail explicitly. Disabling adaptive verification does
not disable the new rejection backend.

With five draft steps, `RTP_LLM_ADAPTIVE_VERIFY_CAP=3` admits exactly the first
three draft tokens and executes four Target positions per request, including
the bonus position. Unset or `-1` enables dynamic budgeting: the first rounds for
each batch size measure uniform verification lengths, then interpolate the
observed CUDA-event costs. Missing seed predictions keep full verification when
selecting a dynamic budget. Forced-acceptance requests do not train the estimator.

Dynamic budgeting reads current confidence rows on the CPU before Target forward.
Request/step indices are cached on the GPU; feedback validation and estimator
updates stay on the GPU. Timing events are polled in later rounds without waiting
for unfinished work. Fixed-cap mode skips prediction, training and cost timing.
The integration disables cross-round forward overlap and MTP Indexer sharing
while enabled. It is an opt-in, not a guarantee of improved throughput.
Controller and graph tests do not replace full-model validation.

HY4 uses the same deferred verification route as GLM5 when V2 rejection is
enabled. Its point-mass draft tokens do not require dense draft probabilities.
On this route, filtering follows the request's TopK/TopP parameters; the legacy
HY4 forced candidate limit is not applied implicitly.

To select HY4's alternative compact probability verifier, set
`RTP_LLM_USE_VLLM_REJECTION_SAMPLING=0`. The HY4 sampler then uses its existing
compact candidate eligibility checks and `RTP_LLM_HY4_FORCE_TOPK` (default 4096).
Setting that limit to `0` disables forced truncation. A forced limit can change
the sampling distribution, so comparisons of backend correctness must use the
same candidate policy. Ordinary sampling has its own independent switch above.
