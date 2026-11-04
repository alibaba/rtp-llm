# largeep_a2a — routed EP all-to-all for multi-node PCIe GPUs over RDMA (standalone)

`librail_routed2` is the MoE dispatch/combine used by the EP16 decode path of this branch
(`MOE_STRATEGY=sm120_rail_routed`, see `docs/largeep/README.md`). It is a single CUDA file with a C API and has no
dependency on the engine: this directory builds it, tests it bit-exactly and reproduces its performance on its own.

Target system: 2 nodes x 8 GPUs, no NVLink, intra-node CUDA-IPC P2P over PCIe, inter-node RDMA (developed on eRDMA,
iWARP; any ibverbs RC NIC with GPUDirect works), GPUs 0-3 on NIC 0 and GPUs 4-7 on NIC 1. Developed and measured on
RTX PRO 5000 (SM120), CUDA 13.2, DeepSeek-V4 decode shapes (hidden 4096, top-k 6, 256 experts = 16 per rank, TP4 token
replication inside a node).

## Files

| path | what |
|---|---|
| `librail_routed2.cu` | the library: kernels `k_d1 k_d2` (dispatch), `k_c1 k_c2` (combine), CPU proxy thread, C API `rr2_*` |
| `build.sh` | `bash build.sh [out_dir]` -> `librail_routed2.so` (env `CUDA_HOME`, `RDMA_CORE`, `ARCH`) |
| `rail_routed2.py` | standalone Python wrapper `RailRouted2` (ctypes), used by the tests |
| `tests/routed2_test.py` | correctness + performance harness (one process per GPU, gloo control plane) |
| `run_2node.sh` | launch the 8 local ranks of one node of a real 2-host run (run on both hosts) |
| `run_1node_emulated.sh` | 2x4 emulation on one 8-GPU host with two NICs (loopback through the switch) |
| `tools/rr2_ts_breakdown.py` | per-stage breakdown from the device timestamp rings (`DSV4_SM120_RAIL_TS_DUMP` in the engine) |
| `tools/rr2_by_layer.py`, `tools/rr2_step_timeline.py` | per-layer stages / unprofiled per-step timeline from the same rings |
| `bench/vram_imm.cc` | raw RDMA floor: WRITE_WITH_IMM ping-pong, VRAM or host buffers on either side |
| `bench/rdma_sentinel.cu` | arrival-detection study: WRITE_WITH_IMM + host flag vs per-16B-vector sentinel polling, 4B tearing check |

## Build

```bash
# system rdma-core with a provider for your NIC:
CUDA_HOME=/usr/local/cuda-13.2 bash build.sh
# private rdma-core build (e.g. a newer provider than the distro ships):
CUDA_HOME=/usr/local/cuda-13.2 RDMA_CORE=/opt/rdma-core-58 bash build.sh
```

Requirements: GPUDirect RDMA (`nvidia_peermem` loaded; `ibv_reg_mr` on `cudaMalloc` memory must succeed), CUDA IPC
between the 8 local GPUs, and RC QPs with `RDMA_WRITE_WITH_IMM`.

## Correctness and performance test

```bash
# real 2 hosts (node 0 = master); run both commands within ~2 minutes of each other
host0$ NODE=0 MASTER=<host0 ip> PROXY_CPU=48 bash run_2node.sh t1 --npads 48,12,192
host1$ NODE=1 MASTER=<host0 ip> PROXY_CPU=48 bash run_2node.sh t1 --npads 48,12,192
# one host, 2x4 emulation
bash run_1node_emulated.sh e1 --npads 48,12
```

Rank 0 prints `SUMMARY {...}` and every rank writes `logs/<tag>/r<rank>.log` (`RESULT {...}` at the end).
Useful options: `--phases eager,verify,perf` (default), `--modes uniform,zipf,remote,local,pad,allpad,...`,
`--expert-us 0,200` (fake expert time between dispatch and combine), `--flag-mode 2|0`, `--blocks 16`,
`--y-local-test 1` (eager phase also checks the owner-rows combine) and `--y-local 1` (graph phases use it).

What it checks. Inputs are deterministic in (seed, call t, TP group), so every rank computes exactly what it must
receive: the gathered rows byte-exact (including `id = -1` on unfilled rows) and the combined `y` bit-exact (same fp32
reduction order as the library), with a stub expert `partial = bf16(sum_k [id_k local] * w_k * (1 + (id_k % 7)/8) *
fp8(x))`. Routing modes: uniform, zipf, one_rank, remote, local, hot, pad (random valid count), spread, pad40, allpad.
Phases: eager (sync + check every call), padding, 43-layer CUDA-graph verify (snapshots of every layer's gathered
buffer), 43-layer graph perf (y checked every layer of every step), negative (n_pad mismatch must be counted, not hang).
Pass = every `bad_*` count 0 and `errors` all 0 in `SUMMARY`.

Reference numbers (2 x 8 RTX PRO 5000, eRDMA in one datacenter, run from this directory with the two commands above
plus `--modes uniform,zipf,remote,local,pad,allpad --expert-us 0`; p50 / p90 per layer = 43-layer graph step / 43;
two independent runs each, to show the run-to-run spread):

| n_pad (rows per TP group) | full combine (y all-gathered to the TP group) | owner-rows combine (`--y-local 1`) |
|---|---|---|
| 12 | 94.7-95.3 / 107.7 us | 95.3-107.1 / 109.5 us |
| **48** (decode cc64: 16 streams x 3 MTP tokens) | **136.4-137.1 / 142.0 us** (dispatch 67-76, combine 71-81) | **125.8-127.8 / 134.4 us** (dispatch 66-71, combine 62-64) |
| 192 | 320.4-324.8 / 335.2 us | 281.9-284.0 / 299.1 us |

All phases: `bad_gath_rows = bad_y_rows = 0`, `errors` all 0, `ep_consistent: true`.

Raw RDMA floor and arrival-detection benches (run the same command with `server`/`B` on host 0 and `client`/`A` on
host 1):

```bash
R=${RDMA_CORE:-/usr}                                                              # private rdma-core prefix, if any
nvcc -O2 -std=c++17 bench/vram_imm.cc -o vram_imm -I$R/include -L$R/lib -lcuda -lcudart -libverbs
host0$ ./vram_imm server 0 20300 erdma_0 1 50112 2000 vram vram
host1$ ./vram_imm client <host0 ip> 20300 erdma_0 1 50112 2000 vram vram          # prints one-way p50/p99
nvcc -O3 -std=c++17 -arch=sm_120 bench/rdma_sentinel.cu -o rdma_sentinel -I$R/include -L$R/lib -libverbs -lcuda
host0$ ./rdma_sentinel B 0 20400 erdma_0 1 50112 3000 sent                         # B prints torn / stale words
host1$ ./rdma_sentinel A <host0 ip> 20400 erdma_0 1 50112 3000 sent               # A prints one-way; also try imm
```

## Protocol

Per call t (one MoE layer); every rank of the 16 must issue the same calls with the **same n_pad**:

- TP group = 4 consecutive ranks of one node holding the same n_pad token rows. `cap = ceil(n_pad / 4)`; TP rank r owns
  rows `[r*cap, min(n_pad, (r+1)*cap))` and only those are sent.
- Payload row (dispatch input, 4176 B for hidden 4096 / top-k 6): FP8 e4m3 x [4096] + UE8M0 scale per 128 (32 B) +
  fp32 weights [6] + int32 expert ids [6]. Routing is read from the ids inside the row; id < 0 means "not routed", a row
  with no valid id produces no traffic.
- dispatch. `k_d1`: build per-destination masks and positions in shared memory, copy rows that have a destination on
  the other node into the RDMA staging (once per row, not per destination), ring the doorbell, P2P-write rows to every
  same-node destination. The CPU proxy posts ONE `RDMA_WRITE_WITH_IMM` to the rail partner (same local id on the other
  node). `k_d2`: wait for the partner's rows, forward each to every destination on this node, wait for all 2 x 8
  inbound legs, invalidate unfilled rows (ids = -1). Receiver layout: `gathered[src rank][cap]`, compacted per source.
- experts run on `gathered` (rows of all 16 sources, `[16*cap, 4176]`) and produce `partial [16*cap, 4096] bf16`.
- combine. `k_c1`: push each partial row back (same-node owner directly; other-node owner via its rail partner on this
  node). `k_c2`: as rail partner, pre-reduce this node's contributions to the forwarded rows (fp32, ascending rank) and
  send one bf16 row per token over RDMA; as owner, sum same-node contributions + the pre-reduced remote row, round to
  bf16 once, then either write `y` into the 4 TP mates' slabs and copy the full `y[n_pad]` out (`rr2_combine`), or write
  only the owned rows (`rr2_combine2(..., y_local=1, ...)`, ABI 6).
- Arrival: P2P legs use a flag word written by the last block after `__threadfence_system`; RDMA legs use the receive
  CQE of `WRITE_WITH_IMM` (proxy stores a host-mapped flag, one GPU thread polls it and mirrors it to VRAM). Flag words
  carry `t<<20 | n_pad<<8 | rows`, so a call with a mismatched n_pad is counted in `dev_err[3]`.
- Every leg signals every call (0 rows allowed), so all staging buffers are single-buffered; graph capture works
  because kernels read the call sequence number from a device counter, the proxy never touches CUDA.

## Where the time goes (n_pad 48, in the engine, p50 per layer per rank, `tools/rr2_ts_breakdown.py`)

| stage | us |
|---|---|
| dispatch total (k_d1 start -> k_d2 end) | 68 |
| - build masks + RDMA staging + doorbell / same-node P2P writes | 6.8 / 1.8 |
| - wait for the rail partner's rows (RDMA one-way + partner lateness) | 27.5 |
| - forward rail rows + wait for all inbound legs | 12.6 |
| combine total (k_c1 start -> k_c2 end) | 113 |
| - push partials (P2P) | 11.4 |
| - wait for same-node partials (= slowest local expert GEMM) | 30.7 |
| - pre-reduce + RDMA post | 4.5 |
| - wait for the remote pre-reduced row | 52.8 |
| - owner reduce + y all-gather + copy out | 14.8 |

Raw floors (`bench/vram_imm.cc`, VRAM -> VRAM, same pair of hosts): one-way 21.6 us at 4 KB, 28.2 us at 50 KB (a
dispatch leg), ~31 us at 100 KB (a combine leg). So of a ~170 us critical path per layer (slowest rank), ~60 us is the
two RDMA trips, ~20 us protocol overhead, ~30 us local forward/reduce/all-gather and the rest is rank skew (the slowest
rank's expert GEMM, ~37 us p50 spread).

## Measured and ruled out (do not redo without new evidence)

| idea | result |
|---|---|
| per-16B-vector sentinel instead of WRITE_WITH_IMM + host flag (`bench/rdma_sentinel.cu`) | 4-byte writes never tear (0 / 12000 ops), but arrival is only 2.0-3.3 us earlier per leg (~0.25 ms/step) |
| chunked early forward (overlap forwarding with the RDMA of later rows) | first-to-last 16B vector of one write lands within 2.5 us (50 KB) / 4.9 us (100 KB): that is all it could hide |
| path MTU 1024 -> 4096 | no change (31.2 / 31.2 us at 4 KB, 45.4 / 45.6 us at 200 KB) |
| more blocks (16 / 32 / 64) | no change: latency-bound, not bandwidth-bound |
| fp32 y all-gather | 200 vs 133 us at n_pad 48: keep bf16 |
| ACK-gated flags (`--flag-mode 0`) instead of WRITE_WITH_IMM | 221 vs 133 us at n_pad 48 |

## Open directions

1. **Rank skew** dominates the combine waits (30.7 + 52.8 us): the slowest expert GEMM decides when the others can
   finish. Per-step load balancing (dynamic expert replication / work stealing) attacks it; transport changes cannot.
2. **Dispatch-GEMM overlap**: today the expert GEMM starts after all 16 sources have arrived; a per-source (or per
   expert) arrival signal would let the GEMM start on local rows while remote rows are in flight.
3. **One RDMA trip instead of two** per layer for the combine: send pre-reduced rows straight to the owner
   (8 QPs per rank) instead of through the rail partner; costs 8x more RDMA work requests and loses the
   pre-reduction (modelled at <= 0.2 ms/step, `docs/largeep/README.md`).
4. Non-equal n_pad across DP groups (today every rank pads to the same n_pad; the engine equalizes the batch).

## API (C, stable; struct layouts are mirrored in `rail_routed2.py` and the engine transport)

`rr2_init(cfg) -> h` (check `rr2_last_error(h)` is empty), `rr2_get_wire(h, &w)` / `rr2_connect(h, wires[world], world)`
(all-gather the 128 B wires; validates rank order and a config hash), `rr2_gathered_ptr(h)`,
`rr2_dispatch(h, payload[n_pad, P], n_pad, stream)`, `rr2_combine(h, partial[world*cap, dim] bf16, out[n_pad, dim],
out_fp32, n_pad, stream)`, `rr2_combine2(h, partial, out[cap, dim], out_fp32, n_pad, y_local, stream)` (ABI >= 6,
`rr2_abi()`), `rr2_get_stats` (device error counters; safe from a monitor thread during capture), `rr2_get_traffic`,
`rr2_fetch_ts` (14 timestamps per call, ring of `ts_calls`), `rr2_sleep_ns`, `rr2_destroy`.
