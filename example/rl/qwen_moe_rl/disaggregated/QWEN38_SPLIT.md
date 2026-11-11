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

Prepare the environment and an authenticated, dedicated Ray session before
invoking `q38_split_main.py`. Each Ray node advertises four GPUs and either
four `q38_train` or four `q38_rollout` resources; only the training head
advertises `q38_head=1`. The driver takes a fresh deployment JSON, pinned
JSONL data, a gateway port and the explicit Ray address. The deployment JSON
contains the actual Ray session and NodeIDs, per-node GPU UUIDs, native
incarnation and port, the model metadata directory, canonical policy
manifest, refit port and a new output directory.

Stage the official tokenizer and model metadata in each training container.
Lock the official safetensors headers and start the bounded checkpoint
relay only on rollout nodes with the model mount. Train sources are admitted
according to the lexical IP sort of the pinned verl Ray worker group.
`q38_checkpoint_plan.py` and `q38_training_relay.py` build and serve only the
ranges assigned to each rank. The driver verifies worker placement before
any weight load. A relay READY receipt alone does not establish model
coverage.

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
driver exit status and cleanup receipts. Smaller provider and transport
tests are qualification only. This smoke recipe does not write a
recoverable full model or optimizer checkpoint.
