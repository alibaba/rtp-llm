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

| Parameter Name         | Function Description                                                     |
|------------------------|--------------------------------------------------------------------------|
| repetition_penalty     | Repetition suppression factor:<br>→ >1.0 suppresses repetition<br>→ <1.0 encourages repetition |
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
