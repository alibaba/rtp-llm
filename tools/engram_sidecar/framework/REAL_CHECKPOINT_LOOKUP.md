# Real-checkpoint cross-host lookup

This experiment uses the original two large Engram tables from
`deepseek-ai/DeepSeek-V4.1-Flash`. It validates the complete backing allocation
and actual RTP eager hash/lookup path, without running attention, MoE, WKV,
the scheduler or an HTTP server. It is a separate prerequisite to the
[full-model HTTP experiment](FULL_MODEL_HTTP.md).

## Verified result: complete real-table lookup

On 2026-10-16 the complete two-layer generation passed on different hosts and
containers in the validated fabric domain. See the
[sanitized result record](REAL_CHECKPOINT_LOOKUP_RESULTS.json).

| Observation | Result |
|---|---:|
| Owner backing, two allocations | 202,758,946,816 bytes / 188.83 GiB |
| Lookup cases: 3 input families × 2 layers × 2 index dtypes | 12 / 12 exact BF16 bits |
| Consumer checkpoint full-table entries and tensors | None |
| PyTorch allocated-byte increase during attach | 0 bytes |
| Probe PyTorch peak allocation increase | 1,234,432 bytes |
| Consumer close | All eight native resource counters zero; process exited |

The input families cover boundary/4 GiB addresses, real-tokenizer prefill and
rolling decode history with int64 and int32 indices. The independent CPU oracle
read only selected rows, matched the published samples and did not initialize
CUDA. Attach verification also checked the first/middle/last published rows.
The PyTorch counters do not account for all driver resource overhead.

At the end of this lookup probe, the Owner remained READY with both allocations
held for the next experiment; consumer cleanup did not imply Owner cleanup.
The subsequent [full-model HTTP smoke](FULL_MODEL_HTTP.md) passed separately.
This lookup probe itself did not run WKV, attention, MoE, the scheduler or HTTP,
and measured no latency SLO.

## Four concurrent GPU consumers

A subsequent test held four independent consumer processes on four different
GPUs in one consumer host/container, with the Owner on a separate host. At a
common attach barrier, each had imported and mapped both real Owner
allocations and completed bootstrap sample verification. The coordinator checked
that all four process identities were still alive and referenced the same
dataset/generation before releasing them into the existing lookup harness.

All four consumers passed the same 12 cases: **48 / 48 exact BF16 results**.
Each confirmed no complete Engram tables in its consumer checkpoint, zero
PyTorch allocated-byte increase during attach, and a peak test increase of
1,234,432 bytes. All four exited successfully with all eight native counters
zero. A post-exit check found no compute processes and 0 MiB used on each of
the four consumer GPUs. The Owner stayed alive and READY.

There was still only **one 202,758,946,816-byte physical generation**. Four
processes reporting the same mapped capacity do not imply four physical copies.
See the [sanitized concurrent result](MULTICONSUMER_LOOKUP_RESULTS.json).
The recorded 9.45 seconds includes process startup and checks; it is not an
online-request latency measurement. These are lookup consumers, not four
complete RTP serving instances; projection/gate/full-model HTTP remain separate
gates.

## Verified checkpoint preparation

The [staging evidence](CHECKPOINT_STAGING_RESULTS.json) records all 48 original
shard SHA256 values checked against the official ModelScope catalog. The Owner
copy contains complete shards 47–48. The consumer contains original shards 1–46
and repacked projection-only shards 47–48; neither its index nor its tensor
files contain the four full embedding tensors. All retained tensor shapes and
dtypes were compared with the source headers.

| Item | Size / count |
|---|---:|
| Engram layers | 1 and 14 |
| Complete logical FP8 weights + E8M0 scales | 202,758,032,400 bytes |
| Backing after 2 MiB CUDA allocation rounding | 202,758,946,816 bytes |
| Consumer tensor entries, excluding complete Engram tables | 96,081 |
| Consumer checkpoint files including metadata | 307,552,554,788 bytes |

The original config, index and tokenizer match the sampled official catalog.
The existing checkpoint's `encoding/encoding.py` is retained with its recorded
SHA256; this does not claim it equals the mutable remote `master` file today.
No weights or private capabilities are included in this repository.

## NUMA memory is a separate prerequisite

The full dataset needs about 188.83 GiB of pinned host backing in this design.
The development loader currently makes one approximately 94.4 GiB allocation
per layer on the selected local NUMA node. A large global `MemAvailable` value
does not prove that a particular NUMA allocation will succeed.

In an earlier preparation attempt, `cuMemCreate(HOST_NUMA, FABRIC)` returned
`CUDA_ERROR_OUT_OF_MEMORY`. A subsequent isolated allocation of exactly one
real layer succeeded after private-file cache advice and changing background
memory pressure; the `after_create` test immediately rolled it back and left
all native resource counters at zero. This rules out a fixed 94.4 GiB bridge
limit in that environment. It does not isolate file cache as the only cause,
or establish a universally safe headroom threshold.

With streamed reads and private cache advice enabled, the next attempt created
and loaded the first complete layer, then failed at the second `cuMemCreate`.
At that second call, node 0 reported 107,031,744 KiB free, yet the approximately
94.4 GiB request still failed. Both Owner allocations were unpublished and all
native resource counters returned to zero. Free-byte totals alone therefore
remain insufficient admission evidence. These are historical attempts: the
later run on a suitable Owner host loaded both layers and passed the complete
cross-host lookup described above. The later full-model HTTP smoke is recorded
separately and does not establish why the earlier allocation failed.

A later read-only snapshot found 253.81 GiB of free buddy memory on that node,
but only 134.75 GiB in blocks of at least 2 MiB. On this 64 KiB-page kernel,
2 MiB corresponds to buddy order 5. The matching NVIDIA 580.95.05 allocator
uses a node-specific system-memory allocation path for EGM. Fragmentation or
insufficient reclaimable high-order pages is therefore a candidate, not a
proven root cause: the snapshot was taken after cleanup, and no failing-call
page-order trace was captured. See the official
[EGM allocator](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.95.05/src/nvidia/src/kernel/mem_mgr/egm_mem.c)
and [page allocation implementation](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.95.05/kernel-open/nvidia/nv-vm.c).
The successful full run recorded free buddy blocks before each allocation on
a suitable Owner host. It did not change global kernel memory settings, and
does not establish the root cause of the earlier failure.

Use fresh NUMA, container and host measurements. Never drop global caches or
stop unrelated processes to make this test pass. The optional Owner argument
`--discard-private-checkpoint-cache` is only for an operator-controlled private
immutable checkpoint copy. It advises release of source file cache while
streaming; the kernel may retain pages, so the return value is not a proof of
reclaimed memory. Defaults do not discard checkpoint cache.

## Platform observation: resident EGM and container accounting

A read-only check of the live successful Owner found two `/dev/nvidiactl`
mappings with RSS of 101,378,424,832 and 101,380,521,984 bytes. Their sum exactly
matches the 202,758,946,816-byte physical EGM backing. Total process RSS was
203,914,018,816 bytes (189.91 GiB).

The same container cgroup reported `memory.current=10,979,508,224` bytes
(10.23 GiB), `memory.peak=15,146,745,856` bytes (14.11 GiB), and zero memory events.
The process, container init and observer were verified in the same cgroup;
host/container memory counter files had matching device and inode identities.
The host's slightly earlier current reading was 10,966,990,848 bytes. These
samples were not atomic, so that small difference is expected.

This observation applies to NVIDIA driver **580.95.05**, Linux **6.6.71** with
**64 KiB base pages**, and this runtime's HOST_NUMA + FABRIC path. It shows that
this container's memcg counters did not charge the EGM at its physical size.
It does not prove that all ancestor cgroups are uncharged, identify a driver
defect or establish accounting behavior on other versions. Production must use
explicit EGM byte budgets and NUMA admission; a container `memory.max` value
alone is not an EGM resource admission or host-capacity protection guarantee.

## Run order

Use the tested CUDA13 environment, the prebuilt native bridge supporting
128 GiB allocations, and the same model revision on both hosts. Select the
authorized GPU with `CUDA_VISIBLE_DEVICES`. All publication and oracle files
must be owned by the running user, mode `0600`, in mode `0700` directories.
Do not print their contents into logs.

1. Start `checkpoint_owner.py` with the real Owner checkpoint and explicit
   allocation/dataset limits as described in [FULL_MODEL_HTTP.md](FULL_MODEL_HTTP.md).
   On a private checkpoint copy, opt in to
   `--discard-private-checkpoint-cache`. Wait for **READY**; INSPECTED only
   proves metadata and budget validation.
2. Prepare a small independent CPU oracle on the Owner host:

   ```bash
   python tools/engram_sidecar/framework/checkpoint_lookup.py prepare \
     --checkpoint /workspace/checkpoints/owner \
     --bundle /workspace/private-owner/bundle.json \
     --private-inputs /workspace/private-oracle/oracle.json
   ```

   This reads only selected raw rows using seek/read, without materializing
   either full table. It binds the oracle to config/tokenizer hashes and the
   published dataset/generation/hash ABI. It cross-checks the Owner's published
   first/middle/last row hashes and verifies CUDA was not initialized.
3. Transfer the private bundle, manifest, capabilities and oracle through the
   existing authenticated channel. Do not copy Owner weight shards to the
   consumer.
4. Run the consumer in a fresh process:

   ```bash
   export RTP_ENGRAM_MAX_ALLOCATION_BYTES=137438953472
   export RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES=274877906944
   python tools/engram_sidecar/framework/checkpoint_lookup.py run \
     --checkpoint /workspace/checkpoints/consumer \
     --bundle /workspace/private-consumer/bundle.json \
     --private-inputs /workspace/private-oracle/oracle.json \
     --device 0
   ```

   The consumer revalidates the absence of full tables in actual file headers,
   attaches through `attach_from_bundle`, checks GPU reads against published
   source hashes, and invokes `HostEngramEmbedding.from_storage` with the
   existing Triton implementation. Cases cover first/middle/last rows, addresses
   around 4 GiB, negative indices, repeated indices, real-tokenizer prefill and
   decode-history windows, and both int64/int32 indices. An independent CPU
   FP8/E8M0 bit decoder supplies the expected BF16 result.
5. Require both PASS and CLOSED with zero native resource counters and confirm
   the consumer process has exited. Keep the Owner for the later full-model
   run, or create its STOP file only after **all** consumers have exited.

The probe reports PyTorch allocation changes, but those do not measure all
driver resources. No latency acceptance, CUDA Graph support or full-model
quality claim follows from a successful lookup probe.


## Explicit per-layer Owner placement

The sharded development Owner accepts `--layer-numa-map '{"1":0,"14":1}'`
and `--layer-device-map '{"1":0,"14":1}'`. Device IDs are ordinals within
`CUDA_VISIBLE_DEVICES`; choose each NUMA node's local GPU, not a host-global
index copied into a filtered device list. Omitting the device map uses
`--device` for every layer. On the tested platform, pairing a remote local-host
NUMA node with the same GPU context returned CUDA_ERROR_INVALID_VALUE, so the
NUMA map alone is insufficient for that placement.
Without it, `--numa` still places all layers on one local NUMA node. An explicit
map must cover exactly the checkpoint's Engram layers, with nonnegative integer
node IDs; duplicate, missing and extra keys are rejected. A layer's weight and
scale share the selected node. All selected nodes must report the same CUDA
allocation granularity before any table allocation begins.

This is an operator-selected placement, not an automatic allocation retry or
failover mechanism. It preserves shard row ranges and weight bytes, while
changing physical access locality. Record the allocation logs' `numa` and the
READY record's `numa_by_layer`; validate all consumer GPUs again. A small probe
or total free memory does not prove that all large allocations will succeed.
Do not combine measurements from different placements as one A/B experiment.
