# GLM-5.3 Flash Zigzag KCP

This opt-in prefill backend keeps hidden states in the same zigzag token layout
through KDA, MLA, mHC and MoE. KDA runs all heads on local tokens rather than
running TP-local heads on the full sequence. Persistent recurrent caches retain
their existing TP head ownership.

## Validated configuration

The final eight-rank benchmark used these environment settings:

```bash
export GLM53_PREFILL_SEQUENCE_PARALLEL=1
export GLM53_PREFILL_MLA_CP=1
export GLM53_PREFILL_KCP=1
export GLM53_ROUTER_ZIGZAG_PACKED=1
export GLM53_KCP_COMPACT_SCRATCH=1
export GLM53_KCP_HEAD_PREFIX=auto
export GLM53_KCP_PEER_CHECKPOINT=1
export GLM53_KCP_FILTER_CHECKPOINT=1
export GLM53_KCP_DIRECTED_HALO=1
```

Enable the backend only for ordinary prefill with SP, MLA CP and mHC enabled.
Each request's prefill length must be divisible by `2 * TP * 64`; validated
prefixes are page aligned. Decode, speculative verification, CUDA graph capture
and prefill/decode disaggregation have not been validated for this backend.
The peer paths require PyTorch CUDA symmetric-memory support and compatible
peer access. The validated runtime used PyTorch 2.11 with CUDA 13.0 and cuLA
`0.1.dev84+ac1593d6.cu132`.

## Implementation

- Coarse affine B/M summaries recover each local fragment's initial recurrent
  state. Local recurrence still computes dense 128-token checkpoints.
- The automatic prefix backend selects chronological head partitioning for
  CP8 batches of at least three requests and the remote-prefix path for two.
- Convolution reads page-internal history directly from local inputs. Only
  three trailing tokens cross fragment boundaries; the directed halo path
  transfers them from predecessor to successor.
- Checkpoint states return to native TP head owners through peer memory.
  A publication-mask All-Gather identifies valid native cache destinations;
  filtering avoids transferring unused destination slots. The generic API
  without masks continues to return dense checkpoints.
- Packed router rows preserve the original padded GEMM geometry and scatter
  only rows belonging to the local zigzag shard.

## Validation and measured scope

The packaged optimized modules match experiment snapshot
`33b748ab07e774dfc4c9897d2b09188725ce85c5` from 2026-10-22. Model integration
excludes the experiment-only profiling decorator and package import shim.

Completed validation includes dense checkpoint/output equivalence to canonical
KCP, all valid native convolution cache pages and prefix continuation, filtered
checkpoint publication, and router top-k edge cases. End-to-end A/B runs returned
identical outputs for 5,050 repeated-prompt requests. These checks do not imply
bitwise equivalence between every TP and CP floating-point operation.

On eight L20D GPUs with a software limit of 180 GiB per GPU, median model-forward
prefill latency over five measurements was:

| Tokens per request | Common batch | TP/SP (ms) | KCP (ms) | Latency reduction |
| --- | --- | --- | --- | --- |
| 4,096 | 315 | 8,514.53 | 8,472.04 | 0.50% (approximately equal) |
| 16,384 | 99 | 10,920.46 | 10,265.01 | 6.00% |
| 32,768 | 52 | 11,549.07 | 10,843.95 | 6.11% |
| 65,536 | 26 | 11,565.13 | 10,770.03 | 6.87% |
| 131,072 | 13 | 11,651.26 | 10,695.53 | 8.20% |

Both sides enabled MLA CP and disabled `GLM53_PREFILL_AG_GEMM` and
`GLM53_PREFILL_GEMM_RS`. Timing used the maximum rank's CUDA-event model-forward
duration after three warmups, excluding profiling overhead. Batches are common
stable configurations, not independently maximized TP/SP capacities. These
measurements are not results from physical L20C hardware.
