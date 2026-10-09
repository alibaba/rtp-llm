# Pinned FlashInfer Blackwell softmax

Upstream: https://github.com/flashinfer-ai/flashinfer
Commit: `9aa52b935e55fcfb61fd9390e04bb32e7f3b9860`.
File: `csrc/cake_blackwell_softmax_cached_cluster.cu`.
SHA256: `c7f6ab44832d655e92ff5ebad4b22c9c0cebbcc9ae8776cfa9952f56a219156b`.
Copyright 2026 FlashInfer team; Apache-2.0 (repository LICENSE).

The generated payload is byte-for-byte unchanged. Only the cached-cluster
family is vendored; unused rowwise/warp/bootstrap/MR515 implementations and
the TVM/Python JIT wrapper are excluded. The native launcher and extracted
shape policy are in `../speculative_sampling/dspark_softmax*`.

CUDA13 builds link SM100/SM100a/SM103a kernels. Only SM100/103, aligned FP32
rows, vocabulary multiples of eight up to 262144, and upstream cached-policy
shapes select the payload. Other shapes/devices/builds use the existing bundled
FlashInfer OnlineSoftmax. Launch failures propagate; they do not silently
change the selected operator. PDL is disabled. Inputs have already passed the
DSpark `add.rn`/`div.rn` temperature combination; no temperature is reapplied.

Scratch belongs to one proposal invocation and is reused only by its steps.
The cached route adds no persistent or global workspace. The fallback allocates
`batch * ceil(vocab / 8192) * 8` bytes only for batch <=128 and vocab >=24576.
Captured tensors are retained by the CUDA graph allocator's private pool.
Sampling, RNG reservation, target probabilities and legacy samplers are unchanged.

Validation manifest: CPU selector boundaries and provenance/source scope;
native compilation; GPU eager/Graph exact and padded rows, changing logits,
unsupported-shape OnlineSoftmax, private concurrent streams; then matched
DSpark normal-rejection PD4+4 performance, quality/MAL and memory. CPU source
tests do not establish GPU or model-level acceptance.
