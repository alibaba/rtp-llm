# vLLM Gumbel sampler

Source: vLLM commit `af261bab7f717b6ece30f5ed3213c43974e4a6c4`.

- `vllm_gumbel.py`: copied from
  `vllm/v1/worker/gpu/sample/gumbel.py`; only the Triton and
  `draft_logits_row` imports are changed to local imports.
- `draft_logits.py`: the `draft_logits_row` function is copied unchanged from
  `vllm/v1/worker/gpu/spec_decode/draft_logits.py`.

Both retain the upstream Apache-2.0 copyright and license notices. The repository
LICENSE contains the Apache-2.0 license text.

`adapter.py` supplies an identity request mapping and unit temperatures for logits
already processed by RTP. The C++ bridge supplies each request generator's seed and
reserved offset as the sampler position. This produces fresh noise on successive
calls, but differs from vLLM's use of absolute token positions. Exact cross-engine
comparison requires identical processed logits, seed, position, temperature,
request mapping, precision mode, and compatible Torch/Triton versions; a matching
request seed alone does not imply matching generated text.

The upstream FP32 numerical behavior, including tie handling and nonfinite inputs,
is preserved. The adapter does not add centering or change the random-number
generator. Consumers requiring probabilities continue through the existing sampler.

## MTP rejection verification

The following files originate from the same commit:

- `vllm_rejection.py`: `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`
- `vllm_watermark.py`: `vllm/v1/worker/gpu/sample/watermark.py`
- `vllm_philox.py`: `vllm/v1/watermarking/prfs/philox.py`
- `vllm_prf_base.py`: `vllm/v1/watermarking/prfs/base.py`
- `vllm_fly.py`: `vllm/v1/spec_decode/fly.py`

The rejection verifier also has local probability-input support: a cache flag
selects conversion with CUDA log at the three draft-value load sites, avoiding
a full materialized log-probability tensor. The default cache representation
remains logits. The probability path preserves source-dtype rounding for FP16
and BF16; unsupported dtypes fall back to materialized log probabilities.
`RTP_LLM_FUSE_DRAFT_PROBS_LOG=0` selects the materialized adapter path for A/B
comparison. This optimization does not mutate the caller's draft probabilities.
The RTP adapter uses ordinary sequential verification, not optional block
verification, FLy or watermarking. `DraftLogitsCache` is a tensor view over RTP's
existing draft probability storage (or materialized log probabilities); upstream's
persistent cache allocator is not needed for RTP's request lifecycle.

RTP target temperature, penalties and masks are applied before verification, so
the verifier receives unit temperature for random requests and zero for greedy.
Mixed forced-acceptance batches are split at request boundaries. Acceptance caps
are represented as invalid draft placeholders. The verifier selects one recovery
or bonus row per request, without eager per-position target samples. Unused tails
are explicitly set to -1 before RTP cleanup. RNG offsets are reserved per request;
upstream's absolute-position and cross-engine bitwise replay contract does not
apply unless the complete metadata is supplied identically.

## Adaptive verification

`acceptance_estimator.py` copies `OnlineAcceptanceEstimator` and its kernels from
`vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py` at the same pinned commit,
with local imports. `adaptive_verification.py` retains the upstream survival
counting and budget-assignment kernels; its host budget selector implements the
expected-output/time objective with explicitly supplied measured costs.

`adaptive_controller.py` and `adaptive_session.py` are RTP integration code:
request slots, valid prediction prefixes, admitted-row packing, feedback,
fixed verification caps, and online CUDA-event cost measurements. They are not
an exact port of vLLM scheduling or profiling. The DSpark confidence-head source
is not connected in this branch.
