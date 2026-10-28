# PD prefill-side measurement tools

Stdlib-only clients and analysis scripts for measuring TTFT, the first→second
token gap (gap12) and KV transfer timing of a prefill/decode separated deployment.
They run on the prefill (P) host. Clients and the wire sampler use the same
wall clock, so per-request token times line up with the KV byte counters.

Assumptions: the P frontend is at `127.0.0.1:14000` (`--start_port 14000`);
its rank-0 `RpcService` is at `127.0.0.1:14001`; in TCP cache-store mode the
KV sockets have local ports `140xx` (the analysis scripts select them by that
prefix); the model is `deepseek_v4` served through `/v1/chat/completions`.

## Prompts

| Script | What it builds |
|---|---|
| `make_prefix_prompts.py` | Shared-prefix prompt set from a list of distinct long reports: `--hit` prefix tokens reused on P, unique suffix up to `--total` (default 32768), and a question whose answer (the suffix city) is stored in `suffix_cities` to catch wrong-segment answers. Needs `transformers` and the model tokenizer. |

For no-reuse runs the clients take a plain JSON list of distinct prompts and add a
per-request nonce.

## Clients

| Script | Measures |
|---|---|
| `pd_ttft_gap_client.py` | TTFT, gap12, gap23 and steady step for distinct prompts (`--concurrency N` fires N at once). |
| `pd_prefix_c4_client.py` | Shared-prefix rounds: one warm-up request so the prefix is cached on P, then `--rounds` × `--conc` simultaneous requests. Writes one JSON per request with server `aux_info` (reuse lengths) and the city check. |
| `pd_prefix_gated_client.py` | Same, with `--gate S`: pauses P's scheduler (`RpcService/SetPause`), sends a placeholder request to consume the idle `schedule()` call, fires the round, releases (`SetRestart`) after S seconds. All requests of the round then enter one prefill batch (needs `CP_FORCE_SINGLE_PREFILL=0` and a large enough `--max_batch_tokens_without_cache`). Latencies are measured from the release time; P polls the pause flag every 50 ms. `--gate 0` is natural arrival. |
| `gsm8k_prefix_eval.py` | GSM8K accuracy with a long shared prefix in front of every question, to validate prefix-reuse numerics. |
| `pd_uniform_load.py` | Open-loop constant-rate load: request i is sent at exactly `t0 + i/qps` (fixed 1/qps gap, no Poisson jitter, no bursts, independent of earlier requests), for `--duration` seconds, `ignore_eos` with `--max-tokens` output. `--warmup N` sends N requests one at a time first (not counted). `--nonce-pos suffix` puts the per-request tag at the end so shared-prefix prompt sets keep their prefix-cache hit (default `prefix`: cold prefill). `--d-addrs` pins request i to decode DP rank i%n via `extra_configs.role_addrs`. The summary carries an arrival check (achieved rate, inter-send gap p50/p99/max deviation, worst lateness) next to TTFT / tok2 / TPOT percentiles and their per-quarter trend; a rising TTFT trend means prefill cannot keep up with the rate. Needs `aiohttp`. |

## Wire sampling

`pd_wire_sampler.py --peer <decode-host-ip> [--rdma-dev erdma_0] --interval 0.002`
records, every 2 ms, `bytes_sent/acked/received` of every TCP socket toward the
decode host (`ss -tinH`) and the eRDMA NIC tx/rx counters (`rdma statistic`).
In RDMA mode the decode side reads KV from P, so P NIC tx is the KV byte count.

## Analysis

| Script | Input → output |
|---|---|
| `an_exposed.py` | wire + `pd_ttft_gap_client` JSON → per-request KV bytes and last-KV-byte time relative to tok1. |
| `an_len_sweep.py` | output of `run_len_sweep.sh` → per-length TTFT, gap12, step, last KV byte, D's LOAD reply time on the gRPC stream (KV exposure = how much it lands after tok1), D first step. |
| `an_prefix_cN.py` | shared-prefix rounds → per-round TTFT/tok2/gap12 and KV sent at the k-th first token. |
| `an_prefix_gated.py` | gated or natural rounds → per-round TTFT/tok2/gap12/TPOT, KV MB per request, last KV byte vs first/last tok1. |
| `prefill_batches.py <engine.log> '<len-*.json>'` | how many requests P prefilled per forward (from the `publish pd plan kv cache` lines of a stage-1 rank). |
| `thread_cpu.py snap/diff` | CPU seconds per thread name of all `rtp_llm` processes between two snapshots. |

## Run scripts

All need `PEER=<decode-host-ip>`; set `RDMA_DEV=erdma_0` in RDMA mode.

```bash
# c1 length sweep (prompts_<L>_x24.json per length)
PEER=10.0.0.2 RDMA_DEV=erdma_0 bash run_len_sweep.sh rdma 8 1k 4k 8k 16k 32k
python3 an_len_sweep.py rdma rdma 1k 4k 8k 16k 32k

# shared prefix, one request per prefill: 31K hit + 1K new, 32 requests per round
python3 make_prefix_prompts.py --base prompts_32k_x24.json --tokenizer /path/to/DeepSeek-V4-Flash \
    --hit 31744 --count 160 --out prompts_32k_hit31k1k.json
PEER=10.0.0.2 RDMA_DEV=erdma_0 GAP=6 bash run_prefix_cN.sh h31-c32-rdma prompts_32k_hit31k1k.json 32 5
python3 an_prefix_cN.py h31-c32-rdma rdma

# whole round in one prefill batch: 8 requests, P started with CP_FORCE_SINGLE_PREFILL=0
PEER=10.0.0.2 RDMA_DEV=erdma_0 GATE=1.0 bash run_prefix_gated.sh g31-c8-rdma prompts_32k_hit31k1k.json 8 5
python3 an_prefix_gated.py g31-c8-rdma rdma
python3 prefill_batches.py logs/engine.log 'len-g31-c8-rdma-32k-*.json'
```

The first round after a P restart runs with a cold prefix cache; drop it.

## Prefill launcher

`run_prefill_cep4pp2_mtp_fastgen.sh` starts the CEP4PP2 (PP2 x CP4, EP4) prefill used for the MTP numbers:
MTP gen3 with fastgen chunked prefill (`RTP_LLM_MTP_CHUNKED_PREFILL=1`, one request per pass), one serving
CP rank per stage (`RTP_LLM_PD_SERVING_CP_RANK=0`), prefix reuse on, and the prefill optimization env set
(the async/fusion group is worth ~30 ms of 32K TTFT in fastgen mode). Required: `WS` (built tree),
`DECODE_ADDR` (every decode DP rank's `ip:http_port`). Optional: `DEEPGEMM`, `EXTRA_PY`, `RDMA_LIB`, `RUN`, `CKPT`.
`MTP_CHUNKED=0 FASTGEN_ARGS=' '` gives the non-fastgen MTP prefill of the same binary.

```bash
WS=/path/to/github-opensource RDMA_LIB=/path/to/rdma-core/build/lib \
DECODE_ADDR=10.0.0.2:21000,10.0.0.2:21009,10.0.0.2:21018,10.0.0.2:21027 \
  bash run_prefill_cep4pp2_mtp_fastgen.sh
# constant 0.9 req/s for 120 s, 1024 output tokens, decode ranks balanced by the client
python3 pd_uniform_load.py --prompts-file prompts_32k_x64_distinct.json --qps 0.9 --duration 120 \
  --max-tokens 1024 --warmup 2 --d-addrs 10.0.0.2:21000,10.0.0.2:21009,10.0.0.2:21018,10.0.0.2:21027 --out q09.jsonl
```

TTFT depends on prompt content at a fixed length (repetitive synthetic text vs ShareGPT 32K differ by
~60-80 ms on the same server); compare runs on the same prompt set.
