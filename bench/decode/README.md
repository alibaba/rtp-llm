# Decode benchmarks and wide-EP estimation tools

Tools used to measure DeepSeek-V4 decode on SM120 (RTX PRO 5000) with TP4 and to
estimate whether a 16-GPU wide expert-parallel layout (EP16 over RDMA) would help.
Everything except `rdma_ep_probe.sh` runs on the serving host.

| File | Purpose |
|---|---|
| `decode_plateau_bench.py` | Decode-only TPOT / throughput: fires `--cc` requests at once with `ignore_eos`, measures the step time in the window where all of them decode (needs `aiohttp`). |
| `make_distinct_prompts.py` | Fully distinct long prompts (no shared KV block) for `--prompts-file` (needs `transformers` and the tokenizer). |
| `capture_decode_nsys.sh` | Starts the bench against a server launched under `nsys launch`, records a short steady-state window and exports sqlite. |
| `dsv4_step_breakdown.py` | Per-module time of one MTP decode step from that sqlite: attention linears (TP-split vs replicated), MLA, indexer, HC, shared/routed MoE, all-reduce, draft forwards, lm_head/all-gather/sampling glue and GPU idle. |
| `moe_tp_vs_ep_bench.py` | Routed-MoE kernel time per layer on one GPU: TP-sharded experts (all experts, 1/TP intermediate) vs one EP rank (experts/EP full experts), autotuned CUTLASS fused MoE in a CUDA graph over 43 distinct layers. |
| `nccl_decode_collectives.py` | all_reduce / reduce_scatter / all_gather / all_to_all latency at decode sizes inside CUDA graphs (`torchrun`). |
| `rdma_ep_probe.sh` | perftest RDMA WRITE latency (64 B-64 KiB), atomic latency and bandwidth between two hosts. |
| `ep_step_model.py` | Combines the above into a TP4 vs EP16 step-time estimate (DP4xTP4 or DP16 attention) and the break-even dispatch+combine cost per layer. |

## Decode-only measurement

The server must start decoding only after every request has been prefilled, otherwise
the batch is never full. For an RTP-LLM `PDFUSION` server use
`--pdfusion_scheduler_mode ratio --decode_prefill_ratio 0`; for a PD deployment hold
decode on the D side until `cc` requests are ready. When D runs data-parallel (DEP8),
pass every D rank's http address with `--d-addrs ip:port,...` so the requests are
spread evenly over the ranks; otherwise the P frontend sends them all to the rank in
its `decode_endpoint`.

```bash
python3 make_distinct_prompts.py --tokenizer /path/to/DeepSeek-V4-Flash --count 64 --out prompts_32k_x64_distinct.json
# MTP 3+1 with every draft accepted, reported at acceptance 3.3
python3 decode_plateau_bench.py --url http://127.0.0.1:27000 --cc 8 --prompts-file prompts_32k_x64_distinct.json \
    --max-tokens 2000 --force-accept --tokens-per-step 4 --accept-len 3.3
```

Use a fresh `--prompt-offset` per run so no prompt is reused, and `--reuse_cache 0` on
the server. Without `--prompts-file` the client builds a shared-prefix prompt
(`--prompt-tokens`), which is faster to prefill when prefix reuse is on but makes
decode 1-9% faster than distinct prompts at 32K.

## Timeline breakdown

```bash
nsys launch --session-new=decode --trace=cuda,nvtx --cuda-graph-trace=node --trace-fork-before-exec=true \
    python3 -m rtp_llm.start_server ...          # same arguments as the run under test
URL=http://127.0.0.1:27000 bash capture_decode_nsys.sh prof/tp4_m2_cc8 8 3 --prompts-file prompts_32k_x64_distinct.json
python3 dsv4_step_breakdown.py prof/tp4_m2_cc8.sqlite --json tp4_m2_cc8.json
python3 dsv4_step_breakdown.py prof/tp4_m2_cc8.sqlite --dump-layer 20   # check the module assignment of one layer
```

The module split relies on kernel names and grid shapes of the SM120 TP4 path
(DeepSeek-V4-Flash, `DSV4_SM120_TP_MOE=1`); re-check with `--dump-layer` after kernel or
shape changes. A capture that contains no kernels (it happened with MTP 3+1 under one
`nsys launch` session) is reported as such; restart the server and capture again.

## Wide-EP estimate

```bash
python3 moe_tp_vs_ep_bench.py > moe.jsonl                                   # 1 GPU
NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS torchrun --nproc_per_node 4 nccl_decode_collectives.py > coll.jsonl
DEV=mlx5_0 bash rdma_ep_probe.sh server                                    # host A
DEV=mlx5_0 bash rdma_ep_probe.sh client <host-A-ip>                        # host B
python3 ep_step_model.py --breakdown-lo tp4_m2_cc8.json --breakdown-hi tp4_m2_cc16.json \
    --moe moe.jsonl --coll coll.jsonl --gamma 2 --accept 2.3 --cc 8,16 --fixed-us 40,65,100 --detail
```

`--fixed-us` is the per-direction fixed cost of the cross-node dispatch or combine
(one-way RDMA latency plus proxy and completion-flag overhead). The model has no EP
kernel measurement behind it: replace the `a2a` term with a measured dispatch+combine
once an EP implementation runs on the target network.

Reference result (4 x RTX PRO 5000 per TP4 group, MTP 2+1, 32K context, 2026-10-26; RDMA
one-way 32-43 us, 2 x 200 Gb/s per node): at cc8 per 4 GPUs EP16 is within -1% to +23% of TP4 and needs
dispatch+combine below 60-120 us per layer to break even; at cc16 it gains 5-14% at
40 us fixed cost. DP4xTP4 attention beats DP16 at cc8 (DP16 pays about 3 ms per step
for the full `wq_b`/`wo_a`/`wo_b` weights) and ties at cc16.
