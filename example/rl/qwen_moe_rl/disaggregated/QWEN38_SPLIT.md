# Qwen3.8 split GRPO

The supported profile has 16 four-GPU Megatron containers and eight
four-GPU RTP containers. The actor uses BF16 TP8/EP64, with eight local
experts per rank. RTP uses one TP8/attention-DP4/EP32 policy, with 16 local
experts per rank. The 92-layer main policy excludes MTP. Training keeps BF16
main weights and gradients on GPU and BF16 Adam moments in host memory.

The split driver uses the pinned rLLM/verl stack, a private Python Tool
gateway, and raw native FP32 rollout log probabilities. Its two-update
recipe uses the original DeepScaleR final-answer grader, including the
rLLM math reward dependencies. An update requires finite, diverse
advantages, real Adam changes, and a complete native refit. Full uniform
recomputation and one-sequence microbatches bound training activations.

## Reproduce the validated two-update profile

Use RTP source `733f77c111ad94defb348a688937716bccb5bd1e` for the
driver. The rollout build used native source
`0ef8ebc7b1d0807077b6148e48a83d0f8f0359ed` and the rank-local Python
loader at `8631881b4286f57e90e0b87f91040c534cf67be1`. The validated
containers ran on ARM64 hosts with SM103 GPUs, Torch 2.13.0+cu130 and
NCCL 2.29.7.
The training environment used Python 3.12.13, Ray 2.59.0, Megatron Core
0.19.2, Transformer Engine 2.20.2 and cuDNN 9.28.1. The native rollout
environment used Python 3.10.9.

The validated inputs were Qwen3.8-2.4T-A95B BF16 revision
`207bd685a7e3696cfaff12ded7c6a7ea0f88c996` and the exact 16-row JSONL
with SHA256
`28545ec7d54b1e287dbb78a8f4b47e0c7bc5d48804eb5a307236f5fe91e67c5c`.
Keep the ordered rows, the pinned math grader and both phases of the
recipe. `q38_short_recipe.py` sets 32 concurrent trajectories, up to two
model calls capped at 512 tokens each and a 1,024-token response budget.
The model cap for the actor is 4,096 tokens with a one-sequence
microbatch. The recipe saves no model or optimizer checkpoint.

The 64-rank training group used NCCL IB through host buffers. Its
validated settings include `NCCL_MNNVL_ENABLE=0`,
`NCCL_NET_GDR_LEVEL=LOC`, `NCCL_NET_GDR_C2C=0`, `NCCL_DMABUF_ENABLE=0`
and `NCCL_PXN_DISABLE=1`. The bounded post-Adam CPU Gloo vote records
success only after each rank has synchronized its local write. Check
the IB device, GID index, socket interfaces and GPU UUIDs on the target
deployment before using the profile.

Prepare the environment and an authenticated, dedicated Ray session before
invoking `q38_split_main.py`. Each Ray node advertises four GPUs and either
four `q38_train` or four `q38_rollout` resources; only the training head
advertises `q38_head=1`. The driver takes a fresh deployment JSON, pinned
JSONL data, a gateway port and the explicit Ray address. The deployment JSON
contains the actual Ray session and NodeIDs, per-node GPU UUIDs, native
incarnation and port, the model metadata directory, canonical policy
manifest, refit port and a new output directory.

Qualify the head gateway with a real HTTP health check before a large
load. If the packaged certifi trust file is absent, configure
`SSL_CERT_FILE` with a verified CA bundle available in the head container.
The driver constructs its HTTPX TLS context before creating workers.

Stage the official tokenizer and model metadata in each training container.
Lock the official safetensors headers and start the bounded checkpoint
relay only on rollout nodes with the model mount. Train sources are admitted
according to the lexical IP sort of the pinned verl Ray worker group.
`q38_checkpoint_plan.py` and `q38_training_relay.py` build and serve only the
ranges assigned to each rank. The driver verifies worker placement before
any weight load. The validated relay used port 20747 on the eight rollout
nodes, outside their ephemeral port range. The driver binds the exact
endpoint map by SHA256. Changing addresses requires regenerating the
endpoint map, source ledger and driver release together. The complete
source ledgers cover 145,053 ranges and 4,839,609,395,968 bytes.

Start RTP in paused mode with the explicit WORLD32 profile. The native
replica only admits generation after all 1,590 main policy sources have
been acknowledged and committed across all 32 ranks. The refit sends two
eight-expert slices from the actor to each RTP rank and uses separate
bounded transfers for dense sources. Each transaction ties IPC lifetime
to native acknowledgements. On failure with uncertain acknowledgements,
terminate the exact native process scope before releasing its Ray IPC
holders.

Accept a full run using the source byte ledgers, actual 64-rank optimizer
witnesses, all-rank native versions and ACKs, raw rollout traces,
driver exit status and cleanup receipts. Each of the two batches contains
eight prompts and four fresh trajectories per prompt. The second batch
uses the policy from the first Adam update. The run completes after the
second updated policy commits across all 32 native ranks.
