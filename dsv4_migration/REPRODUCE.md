# Reproduction on This Machine

Source root: `/home/admin/github/rtp-llm`, branch `rym/feat/dsv4_dsa_kvoffload`,
based on main at `1f57ca1b5b73467575702fbeebbd23ddebf18a83`.
The existing `internal_source` symlink points to `../RTP-LLM/internal_source`.

Model: `/home/admin/model/DeepSeek-V4-Pro`, revision
`2b70c35a4b26d535e3ca78829bc7ec0bc1ed5958`. The existing OSS download completed on
2026-09-11 at 21:49:08 +0800: 92 files, 64 weight shards, 864.74 GB. Its download
log reports SDK SHA256 checks and file-size verification passed.

Hardware: 4 NVIDIA L20D GPUs, approximately 267.69 GiB usable memory per GPU,
SM 10.3. Software: Python 3.10, PyTorch 2.11.0+cu130, Triton 3.6, CUDA 13.2.
The launcher uses the existing local native dependencies and prefers host NUMA
node 1 for both schemes. Fetch uses 256 CUDA CTAs based on the cache microprobe.

## Build and Tests

Run from the source root. The build script records this machine's existing
Bazel, CUDA and dependency-cache locations; it is not a portable installer.

```bash
bash dsv4_migration/harness/build.sh
bash dsv4_migration/harness/start.sh --python -m unittest \
  rtp_llm.models_py.modules.dsv4.fp8.test.test_kv_offload \
  rtp_llm.models_py.modules.dsv4.fp8.test.test_csa_cache \
  rtp_llm.models_py.modules.dsv4.fp8.test.test_csa_offload_config \
  rtp_llm.models_py.modules.dsv4.fp8.test.test_flash_mla_tp4
```

## Input Corpus

The original trace directory is
`/home/admin/dataset/dsv4-pro-trace-outlen-100k`. Select distinct natural inputs
at least 131072 tokens long and truncate their input to that length. Inputs are
not repeated or concatenated to synthesize a long context. Raw prompts remain
in `/tmp`; each run retains the source rows and prompt hashes in its manifest.

```bash
bash dsv4_migration/harness/start.sh --python dsv4_migration/harness/corpus.py \
  --length 131072 --count 64 --output /tmp/dsv4-long-corpus.json
```

## Equal-Budget Matrix

The scripts start and stop their own local server. Run the schemes sequentially
on this one four-GPU machine. Each request generates 145 tokens: the first token
and 16 warmup tokens are excluded, then 128 tokens are measured.

```bash
bash dsv4_migration/harness/start.sh --python dsv4_migration/harness/bench.py \
  vanilla --batches 1,8,16,24,32 --max-batch 32 --kv-mib 12288 \
  --result-dir dsv4_migration/results/vanilla-128k-12g-final

bash dsv4_migration/harness/start.sh --python dsv4_migration/harness/bench.py \
  offload --batches 1,8,16,24,32 --max-batch 32 --kv-mib 12288 \
  --gpu-cache-mib 6144 \
  --result-dir dsv4_migration/results/offload-128k-12g-final

/opt/conda310/bin/python dsv4_migration/harness/summarize.py \
  --baseline dsv4_migration/results/vanilla-128k-12g-final \
  --offload dsv4_migration/results/offload-128k-12g-final \
  --repeat-baseline dsv4_migration/results/vanilla-128k-12g-b32 \
  --output dsv4_migration/results/comparison
```

Both schemes use TP4, EP4, DP1, CP1, MegaMoE SE, FP8 KV and CUDA Graph. They use
the same TP4 compatibility fixes, real prefill, fixed-cohort scheduler, zero
admission reserve, 66 SWA/CSA-state/indexer-state blocks and 34 HCA-state blocks.
Indexers, HCA, SWA and compressor state remain GPU resident under offload.

The 12 GiB budget includes native KV pools and all offload GPU buffers. CPU CSA
backing is 65537 physical blocks per CSA layer, enough for all configured
requests at the maximum sequence length; CPU memory is not the tested limit.
Each private hot cache holds 2048 compressed CSA entries per request per CSA
layer. Each CSA entry summarizes four original tokens. Pro selects 1024 CSA
entries per decode step. The private capacity is separate from the resident
physical prefix, and requests never evict another request's private cache.

## Supporting Artifacts

- `results/offload-128k-forced-cpu`: small resident-prefix integration check,
  with only 1664 MiB for all CSA GPU buffers. Output matches vanilla for all 145
  generated tokens in the checked request. This is not the final capacity split.
- `results/vanilla-128k-calibration-v2`: successful real-prefill memory probe.
- `results/cache-probe*.json`: synthetic per-layer cache timings, not model TPOT.
- Earlier startup attempts remain local and retain diagnostics for missing generated protobufs,
  a missing grammar library, weight-repacking OOM and Pro TP4 shape issues.
  Those failures are not offload performance or capacity evidence.
- `results/vanilla-128k-12g-b32` predates the fixed-state pool/admission-reserve
  corrections. Use the `*-final` matrix for the capacity comparison.

These experiments isolate decode after sequential real prefill. They do not
establish end-to-end serving goodput or performance under mixed prefill/decode
traffic, PD separation, CP, MTP or prefix-cache reuse.

## Real-Model Byte Validation

Run separately from timing. The diagnostic asserts every byte of all selected
CSA entries against the CPU backing after compressor writes, on all 30 CSA
layers and on every decode graph replay. Native-resident and private hot-cache
reads are both checked. Intentionally corrupted bytes are detected by a GPU test.

```bash
DSV4_CSA_VALIDATE_BYTES=1 bash dsv4_migration/harness/start.sh --python \
  dsv4_migration/harness/bench.py offload --batches 16 --max-batch 32 \
  --tokens 32 --warmup 8 --kv-mib 12288 --gpu-cache-mib 6144 \
  --result-dir dsv4_migration/results/offload-128k-byte-validation
```

## Finer Capacity Split

The additional B18/B20 run retains the total 12 GiB/GPU budget and maximum B32
private-cache allocation. It uses 8 GiB for CSA GPU storage and 4 GiB for native
pools. Its graphs capture exactly B18/B20, with no padding to B24. Keep these
results separate from the original 6/6 GiB matrix.

```bash
bash dsv4_migration/harness/start.sh --python dsv4_migration/harness/bench.py \
  offload --batches 18,20 --max-batch 32 --kv-mib 12288 --gpu-cache-mib 8192 \
  --result-dir dsv4_migration/results/offload-128k-12g-sweet
```

## Source Bundle

`harness/bundle.py` creates a source overlay, an applicable patch against the
recorded main commit, SHA256 checksums, compact JSON results and documentation.
Its `--base` defaults to that original main commit, so committing the feature
does not change the patch base or omit newly tracked implementation files.
It reconstructs the touched main files in a temporary directory, applies the
patch and compares the resulting source bytes to the working tree. It omits
model weights, raw trace prompts, verbose server logs and generated protobufs.
The build script regenerates the protobufs. The machine-specific internal
dependency symlink is documented above and excluded from the patch.

```bash
/opt/conda310/bin/python dsv4_migration/harness/bundle.py \
  --output /tmp/dsv4-main-offload-bundle
```
