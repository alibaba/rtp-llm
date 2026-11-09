# NewLoader feature smoke contracts

These targets start real servers with local/NAS checkpoint weights. They fail
on missing GPUs, dependencies, checkpoints, or route mismatches; they do not
skip into a green result. Run them with libraries built from the same source
revision. The suite is explicitly opt-in while hardware validation is pending:
`//rtp_llm/test/smoke:smoke_newloader_feature_contracts`.

| Target (under `//rtp_llm/test/smoke:`) | Required hardware | Assertions |
| --- | --- | --- |
| `newloader_sparse_mla_pd_contract_h20` | 2 H20, CUDA 13 | GLM-5 4-layer sparse checkpoint; prompt longer than index top-k; NewLoader on both roles; actual `pd_sep=true`; P/D output IDs equal fused deployment, including a second transfer |
| `newloader_mla_graph_shrink_contract_h20` | 1 H20, CUDA 13 | Real native buffers captured at 4, replayed at 2/0/1/4; tail slots -1 and positions 0; stable storage; DeepSeek-V2-Lite service forced to FlashInfer MLA and graph bucket 4; single-request output equals eager service; post-readiness graph-replay log required |
| `deepseek_v41_group32_contract_sm100_arm` | 4 SM100 ARM, CUDA 13 | Actual `V41MXFP8Linear` group-32 quantization and DeepGEMM; independently derived FP8 values/packed UE8M0 scales; 1/5/8 rows; then existing V4.1 real generation, graph, and prefix-reuse contract |
| `newloader_bert_roberta_tp2_rocm` | 2 MI308X ROCm 7 | Real HTTP embeddings from bge-large-en-v1.5 and bge-m3; NewLoader TP1 and TP2/EP2 versus independent CPU SentenceTransformer reference, runtime max sequence 128 smaller than checkpoint position table; shape, finite/nonzero values and numerical comparison |
| `newloader_bert_roberta_tp2_h20` | 2 H20 | Same embedding deployment matrix on CUDA |
| `newloader_bert_roberta_tp1_rocm` | 1 MI308X ROCm 7 | Independently runnable single-rank HTTP/HF numerical checks for both checkpoints; does not replace the TP2 target |

DSv4.1 is **not** registered as a NewLoader model. Its test validates a shared
FP8 helper regression and its existing runtime, not NewLoader model support.
This patch restores DSpARK auxiliary-hidden capture on the Qwen3-Next/Qwen3.5
target side. It does not register a new DSpARK draft loader or claim a verified
target/draft checkpoint pair. The separate draft-adaptation work is deferred;
target-side unit contracts are not a deployment smoke.
The MLA test separates deterministic buffer-shrink checks from real service
generation; it does not claim an HTTP scheduler deterministically formed a
four-request batch.

## Execution

Use the matching platform's existing Bazel configuration, for example:

```sh
bazel test <platform-config> //rtp_llm/test/smoke:newloader_bert_roberta_tp2_rocm --test_output=errors
bazel test <platform-config> //rtp_llm/test/smoke:newloader_sparse_mla_pd_contract_h20 --test_output=errors
```

`CHECKPOINT_PATH` can override a generation target's default checkpoint. For
embedding, omit it to test both default models. To override a single embedding
checkpoint, also select only `BertDeploymentSmoke` or `RobertaDeploymentSmoke`
with `--test_arg`.

Embedding defaults to an independent HF/SentenceTransformer CPU reference using
the checkpoint's pooling and normalization modules, fed the same token IDs as
the service tokenizer. This isolates model loading/forward from HF v5 tokenizer
normalizer differences (notably XLM-R full-width punctuation). Numerical
comparison also requires per-input cosine similarity above 0.999. Set
`--test_env=NEWLOADER_SMOKE_REFERENCE=legacy` to compare both NewLoader deployments
against a real legacy TP1 service instead. HF-reference success alone does not
prove legacy-service compatibility. Cold ROCm legacy inference may spend many
minutes compiling AITER on its first request.

Request JSON, responses and per-service process logs are saved under
`TEST_UNDECLARED_OUTPUTS_DIR`. Service cleanup is registered before startup and
also performed in `finally`; only processes owned by that manager are stopped.

## Validation boundary

Writing a target, Bazel query success, AST/import checks and unit-test success
are not smoke success. Record the source SHA **and uncommitted diff**, built
library revision, checkpoint, GPU platform and actual test result when running.
In particular, ROCm execution cannot validate the CUDA MLA or SM100 group-32
targets. Do not add unverified platform cases to a mandatory CI suite solely
because the BUILD parses.

### Observed validation: 2026-10-26

Executed the two single-rank test methods via `python -m` inside the user's
ROCm 7.2 container with its private built runtime (MI308X, FP16, scratch load,
runtime sequence limit 128). Both real HTTP/HF numerical smokes passed, three
inputs per model, no skipped cases:

| Checkpoint | TP | Minimum cosine | Maximum absolute error |
| --- | --- | --- | --- |
| bge-large-en-v1.5 | 1 | 0.9999941587 | 0.0003360063 |
| bge-m3 | 1 | 0.9999994040 | 0.0002890974 |

The earlier TP2 attempt failed before model loading in RCCL initialization:
`Missing "iommu=pt" from kernel command line`. No host settings or safety
checks were changed. This remains an unverified deployment combination, not a
passing or skipped test. The single-rank target does not replace it.

Full Bazel target execution and CUDA/H20/SM100 hardware runs remain pending.
Sparse MLA PD, MLA graph shrink, DSpARK and DSv4.1 group-32 must not be reported as deployment-smoke passes
based on the ROCm single-rank embedding results.

### DSpARK target-side regression

`test_target_forward_captures_dspark_layer_boundaries` exercises the actual
Qwen3-Next/Qwen3.5 target forward with controlled layers: each selected boundary
must contribute `hidden + residual`, the getter and model output must agree,
and the target-verify buffer must retain its address when the active batch
shrinks. It does not exercise GPU attention or a trained draft checkpoint.

### Non-DSpARK follow-up validation: 2026-10-30

Executed in the same user's container and private worktree/runtime, with
`PYTHONNOUSERSITE=1` and Torch 2.10.0/ROCm 7.2. The tested Python sources were
compared byte-for-byte between the source tree and runtime mirror.

* Quantization configuration, BERT/RoBERTa loading, DeepSeek loading and
  Qwen3-Next loading: 142 tests, 141 passed and one explicit ROCm skip for the
  CUDA-only DSA Indexer buffer test. This does not validate CUDA DSA execution.
* ROCm MRoPE: 43 tests passed without skips on MI308X, including ASM/non-ASM,
  interleaved/non-interleaved and partial rotary dimensions, actual prefill and
  decode kernels, position-ID changes across graph replay, and capacity checks.
  This reuses the private native build; it is not a new full-build result.
* Compute-ops import contract: four isolated tests passed. ROCm now requires
  its actual ASM/non-ASM exports instead of attempting obsolete CUDA/PPU names;
  CUDA and PPU retain their respective interfaces. A missing required ROCm
  binding fails rather than silently reporting a generic fallback warning.
  `//rtp_llm/ops/test:compute_ops_backend_test` also built and passed under
  Bazel's ROCm configuration. It declares `GPU_COUNT=0`: isolated stdlib
  imports must not acquire a GPU or bootstrap the remote JIT cache.
* `bert_newloader_tp2_arithmetic_test`: passed for both BERT and RoBERTa, with
  real safetensors loading, distinct nonconstant weights, two CPU/Gloo ranks,
  TP2/EP2, two sequence lengths and the public language forward. Both ranks
  match an independent unsharded checkpoint reference. Attention uses CPU
  SDPA and only collective transport is adapted to Gloo, so this does not
  validate GPU attention, NCCL/RCCL or HTTP deployment. The target is included
  in the BERT test suite and is not `open_skip`.
* FP8 scale allocation: the group-32/128 CPU shape/stride test passed, including
  empty rows and packed-scale boundary cases. SM100 group-32 kernels remain
  unvalidated here.

The real two-GPU embedding deployment remains blocked by the host's missing
`iommu=pt`; no RCCL safety check or host setting was bypassed. Sparse MLA PD,
CUDA MLA graph shrink and SM100 group-32 still require their specified hardware.
These outstanding deployment checks must not be presented as completed smoke.
