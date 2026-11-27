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

Local KV cache tiers form a chain. L2 requires L1, and L3 requires L2 and L1.

| Tier | Location | Switch | Required companion settings |
|------|----------|--------|-----------------------------|
| L1 | GPU device memory | `ENABLE_DEVICE_CACHE` (default on) | — |
| L2 | Pinned host memory | `ENABLE_MEMORY_CACHE` (default off) | `ENABLE_DEVICE_CACHE=1`, positive `MEMORY_CACHE_SIZE_MB` |
| L3 | Local disk | `ENABLE_DISK_CACHE` (default off) | `ENABLE_DEVICE_CACHE=1`, `ENABLE_MEMORY_CACHE=1`, positive `MEMORY_CACHE_SIZE_MB` and `DISK_CACHE_SIZE_MB`, and `DISK_CACHE_PATHS` |

For host reuse, set `REUSE_CACHE=1`, `ENABLE_DEVICE_CACHE=1`, `ENABLE_MEMORY_CACHE=1`, and
`MEMORY_CACHE_SIZE_MB`. For local disk reuse, also set `ENABLE_DISK_CACHE=1`,
`DISK_CACHE_SIZE_MB`, and `DISK_CACHE_PATHS`. Disk caching requires a model without reusable
LINEAR groups.

`ENABLE_CRC` defaults to `1` and enables CRC32C for local cache transfers on builds with
the CUDA13 CRC backend. Builds without that backend log a warning and use plain copies.
Set `ENABLE_CRC=0` to use plain copies explicitly.

With automatic KV sizing, CRC workspaces target at most one eighth of the GPU memory
available after the runtime reserve. Startup reduces concurrent slots first, then the
transfer batch size. The smallest workspace can exceed that target if it still leaves
room for the KV cache. An explicit `KV_CACHE_MEM_MB` reserves that KV capacity first.

DEVICE→HOST and HOST→DISK evict in batches when the source reaches its high occupancy watermark,
and continue toward its low watermark. DEVICE defaults to 0.90 for the high watermark and 0.82 for
the low watermark. Requests and transfers release their block references before the pool can reuse
the space. Each source has its own pending-byte budget
(`DEVICE_CACHE_EVICT_MAX_PENDING_BYTES` and `HOST_CACHE_EVICT_MAX_PENDING_BYTES`, defaulting to 0 and 100 MiB respectively).
All tiers share `MEMORY_CACHE_MAX_DESCRIPTORS_PER_TRANSFER_BATCH`, which defaults to 32 and must be positive.
Each rank uses its local CRC workspace batch size when available. Rank 0 uses the same effective limit for watermark eviction and local transfers.
A byte budget of 0 disables the byte limit. Groups share the byte budget for their source tier. Each descriptor is charged one pool stride, averaged over owning stages under PP.
A block larger than its source budget can transfer alone.

`BLOCK_TREE_LAST_TIER_EVICT_STRATEGY` selects how the last local tier evicts blocks.

- `on_demand` (default) retains cached blocks until an allocation needs space. For an upper-tier
  evict, BTC requests the difference between the source batch limit and the free target blocks, then selects source blocks
  under the same cache lock. The transfer is limited to the available space and current source watermark demand.
  A group can evict its own target blocks while waiting for its source tier's transfer budget.
- `watermark` evicts toward the last tier's low watermark in batches, releasing the cache lock
  between batches. The last tier uses the shared batch limit and its own
  `BLOCK_TREE_{DEVICE,MEMORY,DISK}_EVICT_{HIGH,LOW}_WATERMARK_RATIO` settings.

When a request needs more DEVICE blocks, BTC evicts the shortfall from the requested member pool.
This synchronous evict uses the request size even while a DEVICE watermark transfer is pending.

Scheduler admission separately uses `RESERVE_BLOCK_RATIO` (default 5%) to preserve headroom for
running requests to grow. This admission reserve is independent of the cache eviction watermarks.

Startup rejects configurations with a missing tier, capacity, or disk path.

When `MULTI_TASK_PROMPT` is configured, the server automatically enables `REUSE_CACHE` and L1
for static system prompts.

`REUSE_CACHE` remains the deployment master switch: with it off, no tier is consulted or written.
By default, a request must also set `reuse_cache=true` (the default) to use prefix-cache lookup
or store its KV for later reuse.

### Lookup and store targets

The deployment switches select the local tiers used for lookup and storage. The per-request
`enable_device_cache`, `enable_memory_cache` and `enable_disk_cache` fields remain accepted for
protocol compatibility, but do not change this local-tier policy.

- **Lookup** may use any enabled local tier, plus the configured remote backend.
- **Storage after successful completion** inserts into L1 (DEVICE) when enabled. Eviction
  can evict cached blocks from L1 to L2 (HOST) and from L2 to L3 (DISK).

With cache reuse enabled, the local-tier rules are:

| Enabled local tiers (deployment) | Lookup may use | Request-completion store target |
|---------------------------------|----------------|---------------------------------|
| None | None locally | None |
| L1 | L1 | L1 |
| L1, L2 | L1, L2 | L1 |
| L1, L2, L3 | L1, L2, L3 | L1 |

Remote caching requires a backend enabled and initialized by the deployment configuration.
Every DEVICE insert also prepares a remote upload when that backend exists, including
resident inserts and duplicate keys. Local cache eviction does not initiate uploads. Remote writes are
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
