# Engram layout and latency experiments

These tools compare raw FP8/E8M0 lookup with a preconverted canonical layout.
They do not switch the production backend or establish a service-level SLO.

- `canonical_preprocess.py`: chunked, versioned conversion using the original
  quantization kernel semantics.
- `canonical_gather.py`: gather selected rows and pack their scales into the
  consumer GEMM-compatible representation.
- `latency_wkv.py`: real DeepGEMM projection checks and measurements.
- `bench_consumer.py`: randomized paired local/remote raw/canonical paths,
  retained samples, GPU timing and consumer completion wall time.
- `latency_owner.py` / `latency_consumer.py`: controlled cross-host fixtures.
- `quant_probe/`: narrow bindings to hash-fenced main quantization kernels.

The experimental owner intentionally holds both raw and canonical tables for
comparison. That is not a one-copy production memory result. Production must
select a validated representation; format compatibility includes the quant
variant, scale layout and kernel ABI.

Build the quant probe with a CUDA architecture matching the selected GPU:

```bash
export TORCH_CUDA_ARCH_LIST=REPLACE_WITH_VERIFIED_CAPABILITY
export ENGRAM_QUANT_BUILD_DIR="$HOME/.cache/rtp_llm/engram_quant_main_492ecf45"
python tools/engram_sidecar/latency/quant_probe/build_quant_probe.py --repo "$PWD"
```

Do not guess architecture from a marketing name. `deep_gemm.lock` records the
experiment dependency version and wheel hash. Obtain that artifact from your
configured dependency source; no public-index availability is assumed. The source
fence rejects changed quant kernels.
Use `--help` on the owner/consumer tools for case parameters. Reserve GPUs,
apply bounded consumer timeouts and follow the integration harness's explicit
owner STOP procedure.

For CPU benchmark-contract tests:

```bash
python -m unittest discover -s tools/engram_sidecar/latency -p test_bench_consumer.py -v
```

Evaluate **same-format** remote-minus-local pinned-memory latency using paired
samples on the same GPU. Include consumer wall time, all Engram layers per
step, concurrent consumers and full-model TTFT/TPOT. Keep outliers; do not
present medians or per-layer deltas as an entire-step tail guarantee. Cache
perturbation does not prove all host/fabric caches or TLBs are cold.

The additional 100-microsecond target has not passed. Historical component
measurements and their limits are summarized in [validation](../CHECKPOINT_VALIDATION.md).
