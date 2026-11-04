# Native EGM / ATen bridge

`engram_native.cc` implements HOST_NUMA + FABRIC allocations and imported uint8 ATen views.
It is a development extension, not a production Bazel wheel target. Serving currently
imports the prebuilt module `engram_native_dev_492ecf45`; the historical module name is
retained to keep the tested binary ABI/import path compatible.

Build in a Linux CUDA 13 environment with Torch `2.11.0+cu130`:

```bash
python tools/engram_sidecar/native/load_native.py --verbose
```

`CUDA_HOME` selects driver headers. `CUDA_DRIVER_LINK_DIR` may override the link-only
`libcuda.so` stub directory. Do not add stubs to runtime library paths. Put the resulting
module directory on serving `PYTHONPATH`. Importing the loader does not build or initialize CUDA.

The native ceiling is 128 GiB/allocation and 64 MiB/load chunk. Actual local shard budgets
are checked before allocation; host/NUMA capacity still needs deployment verification.
Each publishing worker keeps its Owner until process exit. Imported tensor aliases retain
mapping references. Model decode Graph uses permanent storage pins; no standalone Graph
executor is used by RTP. Never unmap while consumers/graphs still reference the storage.

Retained validation tools:

- `test_native.py`: real VMM create/export/import/view/release, fault boundaries and resource counters.
  Run with exactly one visible GPU; it builds the native extension through `load_native.py`.
- `test_embedding_contract.py`: CPU Torch tests of the model's external tensor views and no-copy behavior.
- `test_sharded_lifecycle_cuda.py`: P=2/P=3 contiguous mappings, aliases and raw CUDA Graph replay.
- `probe_static_exchange_cuda.py`: two-host synthetic startup/mapping probe; parent-held allocations
  are test fixtures, not a supported owner deployment. It needs a common `--cohort` of two distinct IPs.

Example opt-in lifecycle regression using an already built bridge:

```bash
ENGRAM_RUN_CUDA_TESTS=1 ENGRAM_TEST_DEVICE=0 ENGRAM_TEST_NUMA=0 \
python -m unittest tools.engram_sidecar.native.test_sharded_lifecycle_cuda -v
```

These primitives do not establish full-model correctness or <100us decode overhead.
