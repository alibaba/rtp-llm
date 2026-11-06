# Beam Search parallelization regression tests

Run `//3rdparty/trt_beam_search/tests:beam_search_parallel_test` to compare
parallel cumulative-score updates and V2 candidate writes with the frozen
pre-optimization implementation. The suite checks candidate IDs and FP32
scores byte for byte, including batch slots, aliases, fallbacks, and CUDA Graph
replay with changing inputs.

The `frozen` directory contains test-only references from base commit
`3a8d3074d4ab8a76b6c3434d813c83c795f4cc79`; source hashes and extraction
boundaries are recorded in `frozen/manifest.json`. Original license headers are
preserved. These files are not production implementations.
