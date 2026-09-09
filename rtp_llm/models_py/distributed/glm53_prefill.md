# GLM-5.3-Flash token-local Prefill

The GLM53 sequence-parallel path keeps contiguous token shards between layers.
The FP32 router evaluates only each shard's valid rows, then passes local tokens
directly to routed MoE EP dispatch/combine. Its original 8192-row GEMM boundaries
and row offsets are preserved, including shards that cross a boundary. Padding
has zero routing weights and zero output. Decode TP1/DP8/EP8 does not select this
TP token-sharding path.

The additional optimizations below are opt-in. Set the variables before loading
the model; changing shared-expert selection after loading is unsupported.

| Variable | Default | Behavior |
| --- | --- | --- |
| `GLM53_PREFILL_STABLE_RS` | `0` | Transmit BF16 TP partials and sum in fixed source-rank order in FP32. |
| `GLM53_PREFILL_SHARED_EXPERT_LOCAL` | `0` | Load full shared-expert weights on every Prefill rank and compute local tokens, eliminating shared AG/RS. |
| `GLM53_KDA_PREFILL_CONV_LAYOUT` | `0` | Store equal-size Q/K/V channel groups directly as three contiguous planes. |
| `GLM53_KDA_LOCAL_LOW_RANK` | `0` | Compute replicated `f_a/g_a` projections locally and gather their low-rank results. |
| `GLM53_KDA_LOCAL_LOW_RANK_MIN_TOKENS` | `1048576` | Minimum global token count for local low-rank projections; must be at least 32768. |

QKV and beta retain `GLM53_PREFILL_AG_GEMM`. KDA output projection retains
`GLM53_PREFILL_GEMM_RS`. Both collective GEMM options need their compatible
DeepGEMM/symmetric-memory runtime. Stable RS reuses the GEMM/RS workspace and
requires its fixed-order reducer and system-scope publish/consume barriers.
The workspace is shared serially across CUDA streams. Stable RS accepts BF16
`[tokens, hidden]`, pads nondivisible token counts, and rejects CUDA Graph,
workspace aliases, or lengths above `GLM53_COLLECTIVE_MAX_TOKENS` before launch.

Full shared experts require GLM53's dedicated Prefill role, TP token sharding,
EP, and the native router. Only shared FFN kernel weights and their scales are
replicated; dense FFN, routed experts, and Decode weight partitioning are
unchanged. For 42 shared layers this adds 931725312 bytes per rank (0.868 GiB)
relative to TP8. The shared branch uses the checkpoint FP8 representation. Router
projection and convolution weights retain FP32.

## Numerical behavior

BF16 NCCL reduce-scatter can use a different addition order for different
destination ranks. Fixed-order FP32 RS removes this source of request-position
dependence; it does not promise bitwise agreement with the old BF16 reduction.

The convolution option changes output addresses only. The low-rank option
changes GEMM row counts and communication placement; neither intentionally
changes arithmetic precision. Real-weight B8 x 128K component comparisons and
full-model sampled stages/logits were bitwise equal with these two options on
and off while shared-expert selection was held fixed.

Full shared experts remove the intermediate BF16 outputs of eight TP partial
down projections. Their rounding therefore differs from the TP-sharded path.
In 336 real-activation samples across 42 layers and eight ranks, the full shared
path was closer to an independent FP32 arithmetic reference that retained FP8
quantization and BF16 activation boundaries. This is a component accuracy check,
not a promise of identical model logits or a general task-quality evaluation.
Full-model old/new logits differ, so applications with golden-output contracts
must validate this opt-in separately. The switch remains disabled by default.

## Regression coverage

- `distributed/test:glm53_prefill_parallel_test`: model/role/weight-scope
  selection, FP8 kernel and scale replication, token and zigzag layouts. Set
  `GLM5_CKPT_PATH` to exercise the real checkpoint manifest.
- `modules/hybrid/test:glm53_local_router_test`: FP32 logits and routing near
  ties, 8192-row boundaries, uneven/empty shards, direct EP ordering, shared
  collectives, and poisoned padding.
- `distributed/test:glm53_collective_gemm_test`: eight-rank local low-rank
  projection and ordering, independent fixed-order RS oracle, ragged/strided
  inputs, stream handoff, RS/GEMM-RS workspace interleaving, and invalid-input
  rejection. `GLM53_TEST_LARGE_STABLE_RS=1` also tests 1048576 tokens.
- `triton_kernels/causal_conv1d/test:test_casual_conv1d_prefill`: grouped output,
  cached prefixes, channel tails, and offsets exceeding signed 32-bit range.

The distributed CUDA checks require eight free GPUs and a compatible
SM100/SM103 runtime. Separate correctness diagnostics from timing: CPU tensor
copies, model hooks, and validation-only alternate weights must be removed
before collecting performance traces.
