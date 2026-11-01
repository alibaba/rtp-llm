# RDMA / KV-transfer benchmarks

Tools used to qualify eRDMA (and TCP as the fallback) for prefill -> decode KV transfer between 5kpro (SM120)
and PPU hosts, from the raw link up to the real KV block shapes. During a PD run, the KV timeline on the wire
is recorded with `../pd/pd_wire_sampler.py` (RDMA: P NIC tx counter; TCP: socket `bytes_sent`).

| Level | File | What it measures |
|---|---|---|
| Link, host memory | `link_probe.sh` | `ibv_rc_pingpong` 4 KiB round trip, `ib_write_lat` 64 B-64 KiB, `ib_write_bw` 1 MiB at 1/8/16/32 QPs (perftest). |
| Link, host memory | `ccprobe.c` | Multi-QP RDMA WRITE throughput as a 200 ms time series (cold-start ramp, per-QP rate cap, steady state). |
| Link, GPU memory | `vram_lat.cc`, `vram_probe.sh` | VRAM -> VRAM RDMA WRITE ping-pong latency (peer-memory MR on `cudaMalloc`, no staging); `cq` mode times one signaled WRITE completion. |
| TCP fallback | `tcp_probe.c` | TCP ping-pong latency and N-connection throughput (kernel buffer autotuning on). |
| KV block shape | `kvshape_xport.c` | The real FP8/FP4 128K handoff block plan (31,934 blocks / 531 MB) over N TCP connections, per block (A) or aggregated into ~1 MiB batches (B), cold or warmed connections, sampled verification. |
| Full TCP chain | `kv_pipeline.c`, `kv_pipeline_pool.c` | VRAM -> D2H (pinned ring) -> N TCP connections -> H2D -> peer VRAM; the pool variant keeps warmed connections and reports per-handoff time on a continuous stream. |
| PD publish rhythm | `kv_layerwise_xport.c` | Replays the measured layer-by-layer KV publish timing of one 32K CEP4PP2 request (8 ranks) over TCP, with per-block (framework) or per-message copies, and reports the tail after the last layer. |
| Copy shape | `copy_shape_bench.c` | D2H / H2D cost of the per-rank 32K KV block sizes, per block + sync vs one contiguous copy. |
| PD wire analysis | `an_bursts.py` | From `pd_wire_sampler.py` NIC counters: per-layer KV bursts, last-layer burst, time the last KV byte left P relative to the first token. |

## Build

```bash
RDMA_INC=/path/to/rdma-core/build/include RDMA_LIB=/path/to/rdma-core/build/lib bash build.sh   # -> ./bin
```

`vram_lat` needs CUDA headers (`CUDA_HOME`). The `kv_*` and `copy_shape_bench` tools load the CUDA driver with
`dlopen` and build without CUDA. Only `ccprobe` and `vram_lat` link libibverbs.

## Method

- **Provider library.** eRDMA needs an rdma-core with the erdma provider. On our hosts the system libibverbs does
  not have one, so every RDMA tool runs with `LD_LIBRARY_PATH=$RDMA_LIB`: rdma-core v58 on 5kpro (SM120) hosts and
  new-SDK PPU hosts, v47 on the older PPU hosts (v58 fails `ibv_create_qp` with EINVAL there). Check with
  `ibv_devinfo -d erdma_0` from the same lib dir.
- **Addressing.** Device `erdma_0` (5kpro has `erdma_1` too; use the NIC on the GPU's side), GID index 1 (IPv4
  GID), MTU 1024. Start the server side first; the client connects to the server's management IP (TCP
  out-of-band exchange), not to its GID IP.
- **Latency.** `ib_write_lat -s 64 -n 3000` reports one-way (RTT/2); `ibv_rc_pingpong` reports a round trip.
  `vram_lat` one-way includes an 8-byte D2H flag poll, so compare it with host `ib_write_lat` at the same size.
- **Bandwidth.** 1 MiB messages, 10 s window (30 s for the ceiling), 1 / 8 / 16 / 32 QPs. A single QP is
  rate-capped well below the link (~14 Gb/s on eRDMA), so the link ceiling needs >= 8 QPs. Repeat a cold run
  with `ccprobe` to see the ramp of a new connection.
- **GPU memory.** Use `vram_lat` (stock perftest is usually built without `--use_cuda`). Run it on the GPU the
  serving process uses, against the NIC on the same side.
- **TCP.** Never force `SO_RCVBUF` / `SO_SNDBUF`: that pins the window and caps throughput. Use 16+ connections
  across data centers, compare with `iperf3 -P 16`, and check retransmits (`ss -ti`).
- **KV shapes.** `kvshape_xport` / `kv_pipeline*` take the real block plan; report the 3-run median and keep the
  sampled verification on. Run `kv_pipeline bench <gpu>` first for the local D2H / H2D ceiling.

```bash
# link: host B first, then host A
DEV=erdma_0 RDMA_LIB=$L bash link_probe.sh server
DEV=erdma_0 RDMA_LIB=$L bash link_probe.sh client <host-B-mgmt-ip>
# cold-start ramp, 8 QPs x depth 8, 10 s
RDMA_DEV=erdma_0 LD_LIBRARY_PATH=$L bin/ccprobe server 18600 8
RDMA_DEV=erdma_0 LD_LIBRARY_PATH=$L bin/ccprobe client <host-B-ip> 18600 8 8 10
# VRAM latency on GPU 0
GPU=0 RDMA_LIB=$L bash vram_probe.sh server
GPU=0 RDMA_LIB=$L bash vram_probe.sh client <host-B-ip>
# KV handoff over 16 TCP connections, aggregated, warmed
bin/kvshape_xport server 19500
bin/kvshape_xport client <host-B-ip> 19500 -m B -n 16 -w 1 -p FP8
# full chain with a resident connection pool, 12 s continuous
bin/kv_pipeline_pool server 19600 1 0
bin/kv_pipeline_pool client <host-B-ip> 19600 -n 16 -S 8 -Z 4 -g 0 -t 12 -w 32 -T
```

Single-host self check on a 2-NIC host (server on `erdma_1`, client on `erdma_0`, `127.0.0.1` for the
out-of-band exchange), 5kpro, 2026-10-30: `ccprobe` 8 QP 225 Gb/s, `ib_write_bw` q8 227 Gb/s, `ib_write_lat`
64 B 11 us, `vram_lat` 64 B one-way 82 us (GPU7 -> GPU6), `kv_pipeline bench` D2H block-shaped 113 Gb/s / bulk
H2D 455 Gb/s, `kvshape_xport -m B -n 16 -w 1` 35 ms. This validates the build and the tools, not a link.

## Reference results (2026-10-20 .. 30)

| Path | Latency (one-way) | Bandwidth |
|---|---|---|
| eRDMA 5kpro <-> 5kpro, same DC | 18-32 us (64 B) | 8 QP 1 MiB VRAM -> VRAM 229-230 Gb/s |
| eRDMA PPU <-> PPU, same DC | 17 us | ~190 Gb/s |
| eRDMA across DCs (5kpro <-> PPU / 5kpro) | 320-355 us; VRAM 330-360 us | QP1 14.1, QP8 201.6 Gb/s (host); VRAM 186-203 Gb/s |
| TCP same DC | — | 155-196 Gb/s (16 conns) |
| TCP across DCs, reverse / forward | RTT ~0.75 ms | 186-198 / 47.7 (16 conns, rate-limited path at the time) |
| KV 531 MB handoff, TCP, same DC | — | 28 ms aggregated + warmed (151 Gb/s); per block 37 ms |
| KV 531 MB handoff, TCP pool, across DCs | — | 23.3 ms mean / 26.7 ms p99 on warmed resident connections (153-167 Gb/s) |

In PD serving with layer-by-layer publishing, a 32K request's ~140 MB of KV is sent while prefill runs; with
eRDMA the last byte leaves P before the first token, and the eRDMA vs TCP difference shows up as 14-24 ms of
TTFT (TCP copy and send cost on P) rather than as transfer time.
