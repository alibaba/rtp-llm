### ReuseCache
In multi-turn conversation scenarios, multiple prompts often share common prefixes. The KV cache corresponding to these prefix tokens is identical, and reusing KV cache can reduce computation time for these repeated parts, lowering First Token Latency. Enable KV cache reuse by setting the environment variable `REUSE_CACHE=1`. The startup logs will show "reuse_cache: True" when enabled.
The environment variable `SEQ_SIZE_PER_BLOCK` specifies the number of sequences corresponding to each KV cache block.
**Note: ReuseCache cannot currently use flash attention due to mismatched lengths between Q and KV, requiring `--reuse_cache true` to be added in the CMD**

``` python
import os
from rtp_llm.pipeline import Pipeline
from rtp_llm.model_factory import ModelFactory, ModelConfig

model_name = "Qwen/Qwen-7B-Chat"
model_config = ModelConfig()
model = ModelFactory.from_huggingface(model_name, model_config)
pipeline = Pipeline(model, model.tokenizer)

generate_config = {
    "top_k": 1,
    "max_new_tokens": 100,
}

for res in pipeline("hello, what's your name", generate_config = generate_config):
    print(res.batch_response)

# This query can reuse the KV cache corresponding to the previous query.
# The 'reuse len' in the response's aux info indicates how much KV cache was reused. The reuse length is always an integer multiple of SEQ_SIZE_PER_BLOCK.
for res in pipeline("hello, what's your name? how old are you?", generate_config = generate_config):
    print(res.batch_response)

pipeline.stop()

```

## Cache Tiers (L1 / L2 / L3)

Reused KV cache can live in three local tiers, each controlled by its own independent switch:

| Tier | Location | Switch | Required companion settings |
|------|----------|--------|-----------------------------|
| L1 | GPU device memory | `ENABLE_DEVICE_CACHE` (default on) | — |
| L2 | Pinned host memory | `ENABLE_HOST_CACHE` (default off) | `HOST_CACHE_SIZE_MB` |
| L3 | Local disk | `ENABLE_DISK_CACHE` (default off) | `DISK_CACHE_SIZE_MB`, `DISK_CACHE_PATHS` |

Device free-block headroom is managed by `BLOCK_TREE_DEVICE_EVICT_HIGH_WATERMARK_RATIO` (default 0.90)
and `BLOCK_TREE_DEVICE_EVICT_LOW_WATERMARK_RATIO` (default 0.82). Reaching the high occupancy watermark
triggers eviction toward the low watermark. Allocations can consume this headroom and reclaim cached
blocks on demand; blocks still referenced by requests or transfers may delay physical reclamation.
Scheduler admission separately uses `RESERVE_BLOCK_RATIO` (default 5%) to preserve headroom for
running requests to grow. This admission reserve is independent of the cache eviction watermarks.

All eight combinations are valid, including L2-only and L3-only deployments. Enabling a tier
without its capacity or path settings is a startup error rather than a silent downgrade, so a
misconfiguration never degrades quietly into a smaller cache than intended.

When MULTI_TASK_PROMPT is configured, the server automatically enables `REUSE_CACHE` and L1 to
preserve the existing static system-prompt behavior. Disable MULTI_TASK_PROMPT when testing a pure
L2-only or L3-only configuration.

`REUSE_CACHE` remains the deployment master switch: with it off, no tier is consulted or written.
By default, a request must also set `reuse_cache=true` (the default) to use prefix-cache lookup
or store its KV for later reuse.

### Lookup and store targets

The deployment switches select the local tiers used for lookup and storage. The per-request
`enable_device_cache`, `enable_host_cache` and `enable_disk_cache` fields remain accepted for
protocol compatibility, but do not change this local-tier policy.

- **Lookup** may use any enabled local tier, plus the configured remote backend.
- **Storage after successful completion** selects one local target: L1 (DEVICE), then L2
  (HOST), then L3 (DISK). Lower-tier demotion is a separate operation.

With cache reuse enabled, the local-tier rules are:

| Enabled local tiers (deployment) | Lookup may use | Request-completion store target |
|---------------------------------|----------------|---------------------------------|
| None | None locally | None |
| L1 | L1 | L1 |
| L2 | L2 | L2 |
| L3 | L3 | L3 |
| L1, L2 | L1, L2 | L1 |
| L1, L3 | L1, L3 | L1 |
| L2, L3 | L2, L3 | L2 |
| L1, L2, L3 | L1, L2, L3 | L1 |

Remote caching requires a backend enabled and initialized by the deployment configuration.
Every DEVICE insert also prepares a remote upload when that backend exists, including
resident inserts and duplicate keys. HOST/DISK inserts do not upload. Remote writes are
asynchronous, submitted outside the tree lock, and do not delay insert until I/O completes;
a failed upload does not roll back the DEVICE cache entry.

The request's `enable_remote_cache` field remains accepted but does not control this DEVICE
upload path or remote lookup admission. A remote-only deployment can still look up remote
entries, but has no request-completion store target. Direct REMOTE inserts are unsupported.
Backend availability does not bypass successful completion, `reuse_cache`, or allocator
resource eligibility checks required to reach a DEVICE insert.

### Ignoring request cache switches

Setting `RTP_LLM_IGNORE_REQUEST_CACHE_SWITCHES=1` (default off) ignores the request's
`reuse_cache` switch for lookup and storage. Local tiers and DEVICE uploads already follow
deployment policy regardless of this setting. It does not enable a tier disabled by the
deployment, and deployment-level `REUSE_CACHE=0` still disables reuse entirely.

# MultiTaskPrompt
Create static cache for long-text System Prompts, directly reading KV cache from static cache in each request instead of recomputing. This method can significantly reduce the model's First Token Latency.

## Usage
### MultiTaskPrompt
rtp-llm specifies the system prompt information file that needs static caching through the `--multi_task_prompt` parameter. The format is similar to the following:
``` json
[
    {"task_id": 1, "prompt": " <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>"},
    {"task_id": 2, "prompt": "你是一个严谨的程序员，你接下来需要非常谨慎的思考并回答以下问题:"}
]
```
You can also pass the above JSON through the `multi_task_prompt_str` environment variable.

After startup, the model will run the above system prompts and cache the KV cache in GPU memory. During subsequent runs, if a task_id is specified, this prefix can be used. Demo is as follows:
**Note: MultiTaskPrompt cannot currently use flash attention due to mismatched lengths between Q and KV, requiring the environment variable `export ENABLE_FMHA=OFF` to be configured before running the code**

``` python
import os
from rtp_llm.pipeline import Pipeline
from rtp_llm.model_factory import ModelFactory, ModelConfig

os.environ["MULTI_TASK_PROMPT"] = "/path/to/file"
# os.environ["MULTI_TASK_PROMPT_STR"] = "{json str}"
model_name = "Qwen/Qwen-7B-Chat"
model_config = ModelConfig()
model = ModelFactory.from_huggingface(model_name, model_config)
pipeline = Pipeline(model, model.tokenizer)

# Using system prompt with task_id=1 to concatenate the request
generate_config = {
    "top_k": 1,
    "max_new_tokens": 100,
    "task_id": "1"
}

for res in pipeline("hello, what's your name", generate_config = generate_config):
    print(res.batch_response)

# Not using system prompt
generate_config = {
    "top_k": 1,
    "max_new_tokens": 100,
}

for res in pipeline("hello, what's your name", generate_config = generate_config):
    print(res.batch_response)

pipeline.stop()

```

### Note:
MULTI_TASK_PROMPT is served out of L1. When configured, it automatically enables `REUSE_CACHE`
and `ENABLE_DEVICE_CACHE`, preserving the existing behavior.
When a task ID is specified, the system prompt of the task_id is used to concatenate the request, and the longest matching historical request is found in the KV cache to reuse the KV cache.
When no task ID is specified, the user's prompt is used to find the longest matching historical request in the KV cache to reuse the KV cache.