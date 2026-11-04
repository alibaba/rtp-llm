# LargeEP: EP16 decode for DeepSeek-V4 on PCIe GPUs (2 nodes x 8 RTX PRO 5000)

This branch adds a decode-only wide-EP deployment of DeepSeek-V4-Flash: prefill on a separate TP4 group (PD
disaggregation), decode on 16 GPUs over two hosts as 4 DP groups x TP4 attention with EP16 MoE (16 routed experts per
GPU). The GPUs have no NVLink: intra-node traffic is CUDA-IPC P2P over PCIe, inter-node traffic is RDMA (developed on
eRDMA). The MoE all-to-all is a standalone component, `comm/largeep_a2a/`, with its own build, tests and benchmarks.

Results (MTP 2+1, 1K-token prompts, 64 concurrent streams = 16 per DP group; TP4 baseline = 16 streams on one TP4
group, same build, with its best collectives; unprofiled):

| | EP16 (this branch, all options on) | TP4 baseline | EP16 vs TP4 |
|---|---|---|---|
| decode step, forced acceptance (3 tokens / stream / step) | **32.0 ms** | 38.5-39.2 ms | -17-18% |
| TPOT, real MTP acceptance (2.10 tokens / step) | **15.65 ms** | 18.65 ms | -16% |
| load per GPU | 64 streams / 16 GPUs | 16 streams / 4 GPUs | same (4 streams / GPU) |
| GSM8K (1319, greedy, through PD) | 96.97% | — | earlier runs of this model: 96.66-96.97% |

## 1. Architecture of one decode layer

```
residual stream h [cap, hc=4, 4096] per GPU (token-sharded inside the TP group, DSV4_TP_RSAG=1)
  hc_pre + attn_norm (cap rows) --AllGather--> attention input [T, 4096]      (FlashInfer PCIe IPC all-gather)
  attention (heads split across TP4)          --ReduceScatter--> own rows    (FlashInfer PCIe IPC reduce-scatter)
  hc_post / hc_pre / ffn_norm / gate / shared expert            (own cap rows; shared expert on a side stream)
  routed MoE: dispatch own rows -> 16 experts per GPU (DeepGEMM FP8 x FP4 masked grouped GEMM) -> combine own rows
              (comm/largeep_a2a: P2P inside a node, one RDMA leg per GPU pair across nodes, see its README)
  hc_post (own rows)
residual all-gather once per forward (HC head, MTP hidden, lm_head)
```

- T = rows of a TP group per step (decode: streams x (1 + MTP draft tokens) = 16 x 3 = 48 at cc64); every EP rank pads
  the MoE call to the same n_pad and TP rank r owns rows [r*cap, (r+1)*cap), cap = ceil(n_pad/4).
- DP groups run equal batch sizes: `DSV4_DECODE_EQ_BS=1` pads every DP group to the largest batch each round (fake
  decode streams). The benchmark client holds decoding until all 64 streams are prefilled (`DSV4_DECODE_HOLD_FILE` +
  `gate_agent.py`), SGLang-style "slow-down", so every group decodes the same batch.
- Static EPLB: `DSV4_PHY2LOG_PATH` permutes expert placement per layer (no replicas) so each GPU holds a balanced
  number of *active* experts (decode GEMM time is proportional to active experts, not rows).

## 2. Where the time goes (profiled, per step, mean of 4 ranks; `bench/largeep/dsv4_trace_breakdown.py`)

| ms / step | EP16 | TP4 | EP16 - TP4 |
|---|---|---|---|
| MoE communication (EP all-to-all / TP all-reduce) | 8.42 | 0.81 | +7.62 |
| **routed-expert group GEMM** | **5.58** | **18.23** | **-12.65** |
| shared expert | 1.21 | 1.42 | -0.21 |
| MoE glue + gate GEMM + router | 1.02 | 1.65 | -0.63 |
| attention linear / core / KV compress | 3.80 / 1.85 / 1.30 | 3.73 / 1.96 / 1.38 | ~0 |
| attention communication (RS+AG / AR) | 1.14 | 0.73 | +0.41 |
| hyper-connection ops, small glue | 3.02 | 3.64 | -0.62 |
| verify forward GPU busy | 27.34 | 33.55 | -6.21 |
| draft forwards | 1.51 | 1.13 | +0.38 |
| outside the layers (residual / logits all-gather, lm_head, sampling) | 3.19 | 2.70 | +0.50 |
| GPU idle (host gaps between forwards) | 6.45 | 5.75 | +0.70 |
| step (profiled; profiling adds ~2-4 ms) | 36.77 | 40.92 | -4.15 |

The rows add up to slightly more than the step (36.77 / 40.92) because busy time is a sum over kernels, idle is
measured on the merged timeline, and a few kernels overlap across streams.

Group GEMM speed of light (decode is weight-read bound: arithmetic intensity 29 FLOP/B for EP16, 12 for TP4, vs a
machine balance of 393 FLOP/B; measured DRAM read 1224 GB/s, FP8 x FP4 tensor 481 TFLOPS):

| per step, 43 layers | EP16 (per GPU) | TP4 (per GPU) |
|---|---|---|
| active experts per layer | 9.5 of 16 (13.4 MB each, full width) | 89 of 256 (3.3 MB each, 1/4 width) |
| weights read per layer | 126.5 MB | 298.6 MB |
| speed of light at 1224 GB/s | 4.44 ms | 10.49 ms |
| measured | 5.57 ms (**80%**) | 18.25 ms (**57%**) |

EP16 reads 2.4x fewer weight bytes per GPU and its kernel runs at 80% of the bandwidth roofline (time = 6.2 us +
13.0 us x active experts, r = 0.88); the TP4 FlashInfer `cutlass_fused_moe` is not bandwidth-bound at decode (256
groups of 1-3 rows; time almost independent of active experts). Half of EP16's GEMM advantage therefore comes from
TP4's kernel inefficiency; with equally efficient kernels the remaining GEMM saving (~7.5 ms) is about what the
all-to-all costs, so the all-to-all is the main lever.

## 3. What each change bought (A/B, unprofiled, alternating runs)

| change | forced step | env |
|---|---|---|
| routed all-to-all (send each token only to the GPUs hosting its experts, hierarchical rail) vs a fixed all-to-all | baseline of this table | `MOE_STRATEGY=sm120_rail_routed` |
| shared expert overlapped with dispatch inside the CUDA graph | -1.5 ms | `DSV4_SHARED_EXPERT_MODE=overlap DSV4_SHARED_EXPERT_OVERLAP_IN_GRAPH=1` |
| attention all-reduce on FlashInfer PCIe IPC instead of NCCL | -0.3 to -0.6 ms | `DSV4_PCIE_ONESHOT_AR=1 DSV4_PCIE_AR_BACKEND=fi DSV4_FI_PCIE_PATH=...` |
| static EPLB balancing active experts per GPU | -0.5 to -0.8 ms | `DSV4_PHY2LOG_PATH=bench/largeep/phy2log_active.json` |
| masked grouped GEMM launch hint capped at 64 rows (BLOCK_M 64 / 3 stages instead of 128 / 2: 1.4x faster) | **-2.9 ms** | `DSV4_MOE_MASKED_EXPECTED_M_CAP=64` (default) |
| EPLB remap fused into the gate kernel; a2a proxy threads pinned | -0.2 ms each (noise level) | `DSV4_SM120_RAIL_CPU=<8 cores>` |
| attention AR -> RS + AG, token-sharded residual, combine into owned rows | -0.45 ms | `DSV4_TP_RSAG=1` (all-to-all library ABI >= 6) |

Measured and ruled out: per-vector RDMA sentinels and chunked early forwarding (each < 0.3 ms/step, see
`comm/largeep_a2a/README.md`), RDMA path MTU 4096, FP8 for the attention-input all-gather (the compressor and indexer
consume bf16), fewer SMs for the GEMM, replicating hot experts (adds active experts, i.e. more weight reads).

## 4. Build and run

Tested environment: RTX PRO 5000 (SM120), driver 580, CUDA 13.2, Python 3.10, torch 2.11.0+cu130, triton 3.6.0,
tilelang 0.1.9, nvidia-cutlass-dsl 4.8.0, FlashInfer `0.6.16.post3+rtp.sm120.fc1c4e44c25b`, DeepGEMM built from a
SM120 branch (`2.6.1+local`; the 2.5.0 / 2.8.0 wheels pinned in `deps/` have no SM120 FP8 x FP4 grouped GEMM),
flash-mla `1.0.0+d04a249`, rtp-kernel `0.1.0+34e3b72a`, rdma-core 58 (private build: the distro libibverbs had no
provider for the NIC), `nvidia_peermem` loaded.

1. Build the engine as usual for SM120 (`bazelisk build --config=cuda13_sm120 //:th_transformer //:rtp_compute_ops
   //:th_transformer_config`); the C++ changes are in `NormalEngine.cc` (equal-batch DP control), `MtpExecutor.cc`,
   `cuda_graph_runner.cc`, `OpData.h`. Generate the Python gRPC stubs once: `bash bench/largeep/gen_protos.sh`.
2. Build the all-to-all: `CUDA_HOME=... [RDMA_CORE=...] bash comm/largeep_a2a/build.sh` and verify it on the two decode
   hosts with `comm/largeep_a2a/run_2node.sh` (expect `bad_* = 0` and ~137 us/layer at n_pad 48).
3. PCIe IPC collectives (FlashInfer upstream `79f0637` vendored as the private package `fi_up`):
   `bash bench/largeep/setup_fi_up.sh /opt/fi-up` (clones, renames, pre-builds the two JIT modules); pass the directory
   as `FI_PCIE`. Without it the attention collectives stay on NCCL (`DSV4_TP_RSAG` needs it).
4. Start the decode group on both hosts and the prefill group on a third host:

```bash
# decode host 0 and host 1
NODE=0 WS=$PWD MODEL=/models/DeepSeek-V4-Flash HOST0=<ip0> HOST1=<ip1> FI_PCIE=/opt/fi-up \
  PHY2LOG=$PWD/bench/largeep/phy2log_active.json PROXY_CPUS=48,49,50,51,52,53,54,55 bash bench/largeep/run_decode_ep16.sh
NODE=1 ...same...
# prefill host (4 GPUs)
WS=$PWD MODEL=/models/DeepSeek-V4-Flash D_ADDRS=<ip0>:31200,<ip0>:31236,<ip1>:31200,<ip1>:31236 bash bench/largeep/run_prefill_tp4.sh
# gate agents on both decode hosts (hold / release the decode batch for the benchmark)
python3 bench/largeep/gate_agent.py
```

Other launcher knobs (both scripts): `PORT` (decode 31200, prefill 31100), `RUN_DIR` (logs and kernel caches),
`TMP_DIR` (engine UNIX sockets; keep the path short, default `/tmp/largeep_<user>_<port>`), `PYTHON`, `CUDA_HOME`,
`RDMA_CORE`, `IFNAME`, `GID_INDEX`; decode only: `NIC_PREFIX` (RDMA device name prefix, default `erdma_`, e.g. `mlx5_`),
`NIC_MAP`, `TS_DUMP=<trigger file>`, `STATS_S`, `TP_RSAG=0` (keep the
attention all-reduce), `EXPECTED_M_CAP`, `CAPTURE`, `CONC`. `LD_LIBRARY_PATH` is extended, not replaced (put the
Python runtime's `lib` there if the build needs it).

5. Benchmark and accuracy:

```bash
python3 bench/decode/make_distinct_prompts.py --tokenizer $MODEL --count 64 --target 1010 --tol 15 --min-sentences 1 \
  --out prompts_1k_x64.json
P_URL=http://<p>:31100 LEADERS=<ip0>:31200,<ip0>:31236,<ip1>:31200,<ip1>:31236 GATE_HOSTS=<ip0>,<ip1> \
  PROMPTS=prompts_1k_x64.json bash bench/largeep/ab_run.sh mytag
python3 bench/largeep/ab_summary.py /tmp/ab_mytag      # forced step, real step, TPOT, acceptance (median / min)
python3 bench/largeep/gsm8k_pd.py --p-url http://<p>:31100 --d-leaders <leaders> --data gsm8k_test.jsonl --out g.jsonl
```

Expected `ab_summary.py` line with every option on: `forced 32.0 | real 32.4 tpot 15.65 acc 2.10/2.0`. A fresh
checkout of this branch with exactly the commands above gave `forced 31.6 | real 32.0 tpot 15.48 acc 2.094/2.000 |
cc63 31.81` (all-to-all per call: dispatch 67.6 + experts 136.5 + combine 99.6 us p50, `rr2_ts_breakdown.py`).
Run-to-run noise is about +-0.5 ms; compare configurations with alternating runs.

TP4 baseline (one host, 4 GPUs, 16 streams = the same 4 streams per GPU):

```bash
WS=$PWD MODEL=/models/DeepSeek-V4-Flash FI_PCIE=/opt/fi-up GPUS=0,1,2,3 bash bench/largeep/run_tp4_baseline.sh
B="python3 bench/decode/decode_plateau_bench.py --url http://127.0.0.1:27000 --prompts-file prompts_1k_x64.json --cc 16"
$B --max-tokens 600 --tokens-per-step 3 --force-accept        # warm-up
$B --max-tokens 2400 --tokens-per-step 3 --force-accept       # forced step (step_mean_ms)
$B --max-tokens 1024                                          # real MTP acceptance (measured_tpot_ms)
```

From a fresh checkout of this branch: forced 38.5 / 38.3 ms, real TPOT 18.25 ms.

## 5. Configuration reference

| env | default | meaning |
|---|---|---|
| `DSV4_MOE_STRATEGY` | — | `sm120_rail_routed` selects the routed all-to-all |
| `DSV4_SM120_RAIL_ROUTED_LIB` | next to the transport module | path of `librail_routed2.so` |
| `DSV4_SM120_RAIL_NIC_MAP` / `_NIC_PREFIX` | `0,0,0,0,1,1,1,1` / `erdma_` | RDMA device per local GPU = prefix + index (GPUs 0-3 and 4-7 sit behind different PCIe switches / NICs) |
| `DSV4_SM120_RAIL_NPAD_MAX` / `_FLAG_MODE` / `_SPIN_NS` | 192 / 2 / 5e9 | max rows per call, arrival protocol (2 = WRITE_WITH_IMM), spin timeout |
| `DSV4_SM120_RAIL_GID` | 1 | RoCE GID index of the RDMA device |
| `DSV4_SM120_RAIL_CPU` | unpinned | proxy thread cores: one base core or a list of 8 |
| `DSV4_SM120_RAIL_TS_DUMP` / `_TS_CALLS` / `_TS_COLS` | off / 256 / 14 | device timestamp ring dump (trigger file; `gate_agent.py` `POST /ts` touches it), analysed by `comm/largeep_a2a/tools/` |
| `DSV4_SM120_RAIL_STATS_S` | 0 | log transport error counters every N s |
| `DSV4_DECODE_EQ_BS` / `DSV4_DECODE_HOLD_FILE` | 0 / `/dev/shm/dsv4_decode_hold` | equal DP batch + benchmark hold gate |
| `DSV4_SHARED_EXPERT_MODE=overlap` + `DSV4_SHARED_EXPERT_OVERLAP_IN_GRAPH=1` | off | shared expert on a side stream, also inside graphs |
| `DSV4_MOE_MASKED_EXPECTED_M_CAP` | 64 | cap of the DeepGEMM masked grouped GEMM launch hint |
| `DSV4_PHY2LOG_PATH` | identity | static expert placement (main model only; MTP draft unchanged) |
| `DSV4_MOE_EXPERT_HIST=<file>` | off | per-layer expert-load histogram (write `<file>.trigger` to dump), input of `eplb_plan.py` |
| `DSV4_PCIE_ONESHOT_AR=1 DSV4_PCIE_AR_BACKEND=fi DSV4_FI_PCIE_PATH` | off | TP all-reduce on FlashInfer PCIe IPC (launch configs pinned, `DSV4_FI_PCIE_CONFIGS`) |
| `DSV4_TP_RSAG=1` | off | token-sharded decode layer (attention RS + AG, owned-row MoE) |
| `DSV4_TP_RSAG_BACKEND` / `DSV4_TP_RSAG_MAX_ROWS` | fi / 256 | RS / AG backend (`nccl` for comparison), largest shard rows per collective |

## 6. Code map

| file | role |
|---|---|
| `comm/largeep_a2a/` | the all-to-all library, wrapper, tests, benches (own README) |
| `rtp_llm/models_py/modules/dsv4/moe/_rail_routed2_transport.py` | engine side of the library (ctypes, wire exchange over WORLD, TS dump thread) |
| `rtp_llm/models_py/modules/dsv4/moe/strategies/sm120_rail_routed.py` | MoE strategy: pack payload, dispatch, masked grouped GEMM, combine; `forward_owned_rows` for `TP_RSAG` |
| `rtp_llm/models_py/modules/dsv4/moe/strategies/sm120_decode_experts.py` | DeepGEMM masked expert GEMM (launch hint cap) |
| `rtp_llm/models_py/modules/dsv4/moe/moe_layer.py` | EPLB remap, expert histogram, owned-row MoE path |
| `rtp_llm/models_py/modules/dsv4/moe/gate.py`, `_gate_fused_triton.py` | fused gate kernel with the log2phy remap |
| `rtp_llm/models_py/modules/dsv4/moe/shared_expert.py` | shared-expert overlap inside CUDA graphs |
| `rtp_llm/models_py/distributed/fi_pcie_ar.py` | FlashInfer PCIe IPC all-reduce backend |
| `rtp_llm/models_py/distributed/tp_rsag.py` | token-sharded decode: RS / AG collectives, shard context, layer loop |
| `rtp_llm/models_py/modules/dsv4/block.py`, `decode/forward.py`, `fp8/decode/output_proj.py` | sharded block, forward entry/exit, attention-output RS |
| `rtp_llm/models_py/distributed/collective_torch.py`, `bindings/core/OpData.h` | gloo DP group (`DP_CPU`) for the per-round batch control, FI AR hook |
| `rtp_llm/cpp/normal_engine/NormalEngine.cc` | equal DP batch / hold gate / uniform-batch padding |
| `rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc` | multi-row fake streams; async draft prepare guard |
| `rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc` | no rank-local eager fallback for rank-uniform MoE strategies |
| `rtp_llm/cpp/model_rpc/model_rpc_client.py` | balanced DP leader round robin (`RTP_LLM_DP_ROUTE_ADDRS`) |
| `bench/largeep/` | launch scripts, PD decode benchmark with hold gate, A/B driver, GSM8K through PD, trace breakdown, EPLB planner |

## 7. Open directions (largest measured lever first)

1. **All-to-all critical path (~170 us/layer, 7.6 ms/step)**: two RDMA one-way trips (~60 us) + protocol (~20 us) +
   local forward / reduce / all-gather (~30 us) + rank skew. Rank skew (slowest expert GEMM; ~37 us p50 spread, mostly
   per-step routing noise, not placement) is the largest single item; dynamic per-step balancing attacks it.
2. **Dispatch / GEMM overlap**: the expert GEMM waits for all 16 sources; per-source arrival could start local experts
   while remote rows are in flight.
3. **Group GEMM** (5.6 ms, 80% of roofline): fuse gate_up + down into one kernel (fixed cost ~6 us/layer) and keep
   balancing active experts; replicas only help if they do not raise active experts per GPU.
4. **Host pipeline between forwards** (~4.9 ms/step, identical for TP4): MTP verify -> rejection sampling -> draft
   input preparation runs serially on the host.
5. Uneven DP batches without padding (today every DP group is padded to the largest batch).
