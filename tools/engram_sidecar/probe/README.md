# CUDA 13 / EGM development probes

These are small capability checks, not RTP service or performance acceptance.
Use authorized idle development GPUs. Confirm the hosts belong to a supported
shared NVLink/IMEX domain through platform topology, not IP proximity.

```bash
cd tools/engram_sidecar/probe
CUDA_ROOT=/usr/local/cuda bash build.sh
python collect_environment.py > environment.jsonl
timeout --kill-after=10s 60s ./egm_fabric_probe info > info.jsonl
```

`build.sh` builds SM100/SM103 plus compute_100 PTX. `info` reports queried
capabilities; an API query succeeding with value zero does not prove support.
Actual import/map/GPU verification is required.

`egm_fabric_probe owner --device 0 --file PRIVATE_FILE --bytes 4194304` writes a
private opaque handle descriptor and retains one deterministic HOST_NUMA/FABRIC
allocation. Transfer the descriptor through authenticated SSH to a mode-0700
directory on the consumer host and retain its mode-0600 permissions. Run
`egm_fabric_probe consumer --device 0 --file PRIVATE_FILE --rounds 3` there.
Require zero mismatches, matching checksum and successful cleanup. The consumer
allocates only a small result buffer, not a complete private table.

Keep the owner alive until every consumer is done. This low-level probe accepts
signals and has no distributed session tracking; an expired owner timer is not
proof its consumers stopped. Never publish handle files or raw environment logs
in the repository. Hardware, process and user identifiers in collected output
require review before sharing.

`rdma_query_probe.c` only opens/queries/closes visible verbs devices; it creates
no QP and sends no packets. Compile with `cc -O2 rdma_query_probe.c -libverbs`.
Query success does not establish active links or cross-host RDMA data transfer.

`python_stack_probe.py --import-only` validates imports without CUDA work.
Its default GPU smoke test checks versions, selects one idle authorized visible
GPU, then verifies a small Triton add and five Graph replays. A skipped GPU test
is not a GPU PASS. Idle sampling is not a resource reservation.
