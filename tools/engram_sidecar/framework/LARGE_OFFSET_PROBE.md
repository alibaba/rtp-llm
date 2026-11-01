# Cross-host EGM reads beyond the 4 GiB boundary

The real cross-host/cross-container run passed on 2026-10-16 with physical GPU 0
on each host and Owner NUMA 0. Logical backing was 4,429,252,872 bytes, padded to
4,431,282,176 bytes at 2 MiB granularity. Both int64 and int32 cases matched BF16
bytes exactly, including all three invalid-row masks. PyTorch allocation growth
on attach was zero; peak growth was 7,680 bytes and driver free-memory change
was 2 MiB. The consumer exited before Owner STOP; both reported zero resources
on close, and private capability files were removed. This is a large-address
correctness result, not a complete-model or latency result.
The sanitized machine-readable record is [LARGE_OFFSET_RESULTS.json](LARGE_OFFSET_RESULTS.json).

This probe isolates address width and real Triton lookup from checkpoint staging
and the full LLM. It imports the prebuilt `engram_native_dev_492ecf45` extension;
a Python shim or extension build/JIT loader is rejected. Ordinary Triton kernel
compilation is expected on first use. CPU tests do not execute CUDA.

The Owner allocates one `HOST_NUMA + FABRIC` region:

| Item | Value |
|---|---:|
| Rows | 16,777,473 (`2^24 + 257`) |
| FP8 weight bytes per row | 256 |
| E8M0 scale bytes per row | 8 |
| Scale tensor byte offset | 4,295,033,088 |
| Total logical bytes | 4,429,252,872, approximately 4.125 GiB |
| Written/sample rows | 0, 16,777,215, 16,777,216, 16,777,472 |

Row 16,777,215 ends at byte `2^32 - 1`; the next row begins at `2^32`.
The final sample reaches the last weight row and the final scale row. Only four
rows of known FP8/E8M0 bytes are written. The native Owner zero-initializes the
entire allocation, so approximately 4.125 GiB of real host RAM is still required.

The independent CPU reference interprets explicit FP8 byte codes as small powers
of two and applies E8M0 exponents. Results are exactly representable in BF16.
The consumer imports two uint8 views, executes the actual
`_engram_triton.lookup_host_rows`, and compares output BF16 bytes exactly for
both int64 and int32 input indices. Rows `-1`, `row_count`, and `2^31 - 1` must
produce all-zero output. No full-table CPU tensor or GPU copy is constructed.

The probe records PyTorch allocated HBM before/after mapping, peak allocated HBM,
and the driver's free-memory change. Mapping must allocate zero PyTorch HBM;
selected-row execution is limited to 64 MiB of PyTorch HBM growth and 512 MiB of
driver memory growth. Run on the isolated idle GPU selected by the coordinator:
unrelated work can invalidate driver-memory evidence. These counters plus the
raw native view path establish the absence of a full 4.125 GiB HBM copy; they do
not measure lookup latency or certify performance of the 189 GiB real dataset.

Example commands, after activating the prepared venv on each host:

```bash
# Owner host; local visible ordinal 0 refers to physical GPU 2 here.
CUDA_VISIBLE_DEVICES=2 python tools/engram_sidecar/framework/large_offset_probe.py owner \
  --publication /workspace/engram_slice7/large-owner-private --device 0 --numa 0
```

Wait for `READY`, then have the coordinator copy only `probe.json` over the
existing authenticated transport into a new consumer directory with mode 0700.
The file must be 0600, owned by the consumer UID. Never print it: it contains the
Fabric capability. Metadata, geometry, samples, UID, permissions, file type,
generation, padding and capability encoding are checked before native import.

```bash
CUDA_VISIBLE_DEVICES=2 python tools/engram_sidecar/framework/large_offset_probe.py consumer \
  --bundle /workspace/engram_slice7/large-consumer-private/probe.json --device 0
```

Require `PASS` for both index types and `CLOSED` with every native counter zero.
Wait for the consumer process to exit, then create `large-owner-private/STOP` on
the Owner. Require its `CLOSED` counters to be zero too. The Owner retains its
backing after a reporting timeout, lost output pipe or SIGINT/SIGTERM while
published; none of those events proves consumers have stopped. There is no
automatic distributed lifetime or failover mechanism in this probe.

CPU validation:

```bash
python -m unittest tools.engram_sidecar.framework.test_large_offset_probe -v
```

The ten tests cover exact byte boundaries, distinct deterministic sample rows,
bounded geometry/padding, private file and directory ownership/permissions,
duplicate JSON fields, symlinks and capability-redacted errors. They do not
allocate the large table and must not be reported as the cross-host GPU result.
