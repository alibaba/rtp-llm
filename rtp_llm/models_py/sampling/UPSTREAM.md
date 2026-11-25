# vLLM Gumbel sampler

Source: vLLM commit `af261bab7f717b6ece30f5ed3213c43974e4a6c4`.

- `vllm_gumbel.py` adapts `vllm/v1/worker/gpu/sample/gumbel.py` for ordinary
  sampling. The block kernel reads the request's temperature, seed and position
  directly. RTP omits vLLM's optional draft-cache writes and standalone temperature
  kernel. The fused temperature option, draft noise salt and FP32/FP64 sampling
  remain available. RTP adds `RETURN_KEY` to `gumbel_noised_argmax` so the compact
  verifier can return the original vocabulary ID, breaking ties by the lowest
  valid ID. The default still returns the winning lane.
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

`vllm_rejection.py` adapts the sequential verifier in
`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` at the same commit.
Random requests use `tl_rand32` for each acceptance draw. After a rejection,
Gumbel sampling draws from the residual distribution. When all draft tokens
are accepted, it draws the bonus from the target distribution. RTP omits the
upstream FLy, watermarking and block-verification modes.

The rejection verifier also has local probability-input support: a cache flag
selects conversion with CUDA log at the three draft-value load sites, avoiding
a full materialized log-probability tensor. The default cache representation
remains logits. The probability path preserves source-dtype rounding for FP16
and BF16; unsupported dtypes fall back to materialized log probabilities.
The eager adapter accepts an explicit materialized-log option for numerical
comparisons. Both paths preserve the caller's draft probabilities.
`DraftLogitsCache` views RTP's existing draft probability storage or materialized
log probabilities for the lifetime of the request.

RTP target temperature, penalties and masks are applied before verification, so
the verifier receives unit temperature for random requests and zero for greedy.
Mixed forced-acceptance batches are split at request boundaries. Acceptance caps
are represented as invalid draft placeholders. The verifier selects one recovery
or bonus row per request, without eager per-position target samples. Unused tails
are explicitly set to -1 before RTP cleanup. RNG offsets are reserved per request;
upstream's absolute-position and cross-engine bitwise replay contract does not
apply unless the complete metadata is supplied identically.
