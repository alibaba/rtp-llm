This profile uses one logical Qwen3.8-2.4T policy across 18 nodes and 72 GPUs: RTP TP8, attention DP9, EP72, FFN-SP8 and PP1, with 512 logical experts in 576 fixed physical slots. Training uses FSDP72, BF16 master parameters, BF16 CPU Adam moments, FP32 reduced gradients and bounded FP32 update chunks. Optimizer, gradient and model offload to local disk are disabled. This source integration does not establish full-model RL acceptance.

Supply the runtime environment, current node/container allocation and existing official BF16 model mount explicitly. Never reuse another allocation's process identities or a native library from a different GPU architecture. The native runtime must implement the matching protocol3 initial-pause/metadata-bootstrap/refit contract and the formal `--phy2log_path` flag. Set the accepted native and actual mapped NCCL library SHA256 bindings before launch. The reviewed BF16 training modules and canonical policy manifest are supplied by the companion training feature.

The recipe requires `Q38_DEPLOY_ROOT`, a fresh `Q38_RUN_ROOT`, `Q38_NODES_FILE`, the current owned `Q38_RAY_ADDRESS`, `Q38_GATEWAY_HOST`, `RTP_RL_NATIVE_LIBRARY_SHA256`, and `RLLM_Q38_NCCL_LIBRARY_PATH`/`SHA256`. `Q38_PROJECT_ROOT` selects this project directory in an existing deployment. `Q38_MODEL_PATH` can override the default platform model mount. No script downloads a model or authenticates a remote connection.

Recreate the exact 18-question fixture from the pinned public DeepScaleR source, which is intentionally not included here:

```bash
python validation/prepare_world72_dataset.py \
  --source "$DEEPSCALER_SOURCE_JSON" \
  --output data/deepscaler-world72-18.jsonl
```

The source dataset revision and SHA256, selected row indices, exact generated JSONL digest, and two-update configuration are in `data/deepscaler-world72-18.manifest.json`. The earlier eight rows are reconstructed from pinned official indices; `--previous-eight` is an optional byte-for-byte cross-check. The original correctness reward is preserved. Genuine nonuniform reward groups and actual Python Tool calls must be observed, not manufactured.

CPU contracts use explicit Ray/native doubles and their own identity-bound short-lived subprocesses. Run them in an environment containing `psutil`:

```bash
python -m unittest discover -s validation -p test_qwen38_world_cpu.py -v
python validation/native-metadata/build_native_metadata_preflight_inputs.py --help
python validation/validate_world72_core_evidence.py --help
python validation/validate_world72_trajectories.py --help
```

The two offline validators take explicit evidence and source paths; neither independently grants full RL acceptance. The trajectory validator may replay a bounded number of already-recorded original Python Tool calls in the original sandbox. Source, trace, reward, gradient, policy-version and cleanup checks remain required together with all72 GPU peaks, all18 real memory limits and owned job exits.

The functional recipe sets `Q38_HF_RTP_REPORT_ONLY=1`. It reports the actual
HF and native FP32 target logprobs, their original bits, per-token absolute
differences, and maximum/mean absolute errors for each policy. HF/native error
magnitude has no acceptance threshold in this profile. Finite scores, identical
target IDs and committed versions, all1590 source copies, and the existing
nine-DP self-agreement checks remain required. The final proof validates the
reported arithmetic and actual two optimizer updates. Direct controller users
who leave this flag unset retain the earlier strict HF comparison behavior.

The separate metadata-only preflight takes a fresh dedicated run directory and Ray namespace. Build its private input using `--cluster-plan`, `--collective-proof`, `--targets`, `--model-path` and optional `--start-port`. Keep that generated input and its IP/UUID/container identities outside version control. The wrapper preserves the recipe prefix and changes only the terminal driver. It refuses execution without the explicit operator start gate `Q38_NATIVE_METADATA_BOOT_AUTHORIZED=1`, which is set only after other GPU work is released. It creates no HF workers and permits only native STATUS, PAUSE and owned teardown; no source COMMIT, RESUME or generation. Its actual cleanup must remain training-safe=false.

`validation/qualify_recipe_native_chain_cpu.py` preserves the actual recipe's dataset validator, executes an explicit CPU fixture of the production owner's pre-Popen prefix, and preserves the native launcher's stdin validator. It intercepts the training and native final module entry points before execution and invokes a supplied CPU checker with the captured native argv/profile JSON. It does not claim a live Ray owner, native health or GPU bootstrap.


The ARM SM100/Torch 2.13 world72 entrypoint selects the project-owned
`torch_bf16_v1` RoPE implementation explicitly and retains canonical Triton QK
normalization. It does not load an old Torch 2.11 `rtp_kernel` binary or invoke
the pure-TP hook. Full-attention TRTLLM requires a matching freshly compiled
FFI AOT library and checksum-verified SM100 cubins. Native GDN still uses local
CuTe DSL JIT with downloads disabled. The launcher uses only the supplied local
checkpoint/tokenizer and clears unrelated model auxiliary paths. These source
backends passed small-tensor qualification; full-model acceptance is separate.

The first functional recipe limits each real training row to 2048 prompt
tokens and 4096 total tokens, including tool observations. Two agent turns,
1024 tokens per call, and 2048 generated tokens per episode are explicit.
The original converter's padded storage width can be 6144; that is not the
actual model-compute length. Native serving keeps its separate 8192 sequence
limit. Strict pre- and post-conversion guards reject oversize inputs and check
all native token IDs and action masks rather than silently truncating them.
One rollout contains 72 real trajectories; multi-turn conversion may yield
72 to144 training rows before neutral balance padding. The original dynamic
minibatch metadata must still cover one complete global update per iteration.
