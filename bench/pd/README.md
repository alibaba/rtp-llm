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
| `pd_uniform_load.py` | Open-loop constant-rate load: request i is sent at exactly `t0 + i/qps` (fixed 1/qps gap, no Poisson jitter, no bursts, independent of earlier requests), for `--duration` seconds, `ignore_eos` with `--max-tokens` output. `--warmup N` sends N requests one at a time first (not counted). `--nonce-pos suffix` puts the per-request tag at the end so shared-prefix prompt sets keep their prefix-cache hit (default `prefix`: cold prefill); a prompt used twice in one P process then hits in full, so restart P between shared-prefix runs and keep the request count within the prompt set (the client warns otherwise). `--d-addrs` pins request i to decode DP rank i%n via `extra_configs.role_addrs`. The summary carries an arrival check (achieved rate, inter-send gap p50/p99/max deviation, worst lateness) next to TTFT / tok2 / TPOT percentiles and their per-quarter trend; a rising TTFT trend means prefill cannot keep up with the rate. Needs `aiohttp`. |
| `prefill_only_conc.py` | Standalone prefill engine (no decode peer): at most `--cc` requests in flight on the native `/` endpoint, per-request wall time and server `reuse_len`, completion rate. Prefill-only requests (`max_new_tokens=0`) never reuse the prefix cache, so shared-prefix runs need `--max-new-tokens 1 --nonce-pos suffix` (and `--warmup` to cache the prefix first). Stdlib only. |

## Cost model

`pd_cost_model.py`: measured P and D instance capacities (32K input; 5kpro CEP4PP2 / PPU TP4 / PPU TP8 prefill per prefix-hit ratio, PPU DEP8 / 5kpro TP4 decode at TPOT 15 ms) turned into, per hit ratio: prefill cost efficiency per GPU (5kpro vs PPU), QPS per GPU and per cost unit at the ideal P:D ratio for homogeneous 5kpro PD, heterogeneous 5kpro P -> PPU D and homogeneous PPU PD (with the heterogeneous gain over each), and the best whole-machine mix within the available machines with its idle share (over-provisioned side and whole system, cost-weighted). Options: `--out-tokens`, `--price-5kpro` (default 1), `--price-ppu` (default 1.6), `--5kpro-hosts`, `--ppu-hosts`. The capacities are constants at the top of the script; update them when re-measured.

`prefill_sla_queue.py`: how much of a P instance's saturation rate is usable under a TTFT percentile SLA, from the measured start interval (1 / saturation req/s) and single-request TTFT, for k instances behind an ideal load balancer, with uniform or Poisson arrivals. Queueing never raises the saturation rate itself (P prefills one request at a time); the SLA only decides how close to it bursty traffic can run.

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
| `an_gated_rounds.py <out_dir> [tag ...]` | gated rounds without a wire sampler → per tag TTFT / tok2 / gap12 / TPOT and city check, first round skipped. |
| `an_uniform_load.py <run.jsonl> ...` | `pd_uniform_load.py` runs → send rate, first-token completion rate, TTFT p50/p90 and per-quarter p50, tok2, TPOT, output length / finish reasons, decode concurrency, tokens per chunk. |
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
MTP gen3 with fastgen chunked prefill (`RTP_LLM_MTP_CHUNKED_PREFILL=1`, one request per pass), prefix reuse on,
and the prefill optimization env set (the async/fusion group is worth ~30 ms of 32K TTFT in fastgen mode).
`PD_SERVING_CP_RANK=0` advertises one serving CP rank per stage (`RTP_LLM_PD_SERVING_CP_RANK=0`); it is off by
default because a TP>1 decode of this branch then reads quarter blocks into whole opaque DSV4 blocks and
decodes garbage (PPU DEP8, TP1 per rank, is fine with it).
Required: `WS` (built tree), `DECODE_ADDR` (every decode DP rank's `ip:http_port`). Optional: `DEEPGEMM`,
`EXTRA_PY`, `RDMA_LIB`, `RUN`, `CKPT`. `MTP_CHUNKED=0 FASTGEN_ARGS=' '` gives the non-fastgen MTP prefill of the
same binary.

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

## Capacity and end-to-end method (hetero / homogeneous PD comparison)

Used for the 87.5% / 50% prefix-hit comparison of 5kpro P -> PPU D against
homogeneous 5kpro and PPU PD (32K input, 1024 output, MTP 2+1 with real
acceptance). Restart P before every run: a run's prompts stay in P's prefix cache.

```bash
# prompt sets: 800 distinct suffixes behind a 28K (87.5%) / 16K (50%) shared prefix
python3 make_prefix_prompts.py --base prompts_32k_x24.json --tokenizer $CKPT --hit 28672 --count 800 --seed 30000 --tag H --out h28_x800.json
python3 make_prefix_prompts.py --base prompts_32k_x24.json --tokenizer $CKPT --hit 16384 --count 800 --seed 30000 --tag H --out h16_x800.json

# P capacity: single-request TTFT at 0.5 qps, then uniform arrivals above capacity with 16-token outputs (D never
# limits); the capacity is the first-token completion rate from the 2nd quartile on (an_uniform_load.py ft_rate)
python3 pd_uniform_load.py --d-addrs $D --prompts-file h28_x800.json --nonce-pos suffix --qps 0.5 --duration 16 --max-tokens 16 --warmup 1 --out c1.jsonl
python3 pd_uniform_load.py --d-addrs $D --prompts-file h28_x800.json --nonce-pos suffix --qps 6.5 --duration 20 --max-tokens 16 --offset 10 --out sat.jsonl
python3 an_uniform_load.py sat.jsonl

# end to end: uniform arrivals at the target rate, 1024-token outputs; the sustainable QPS is the highest rate whose
# TTFT quarters stay flat and whose TPOT p50 is within the SLA (15 ms here)
python3 pd_uniform_load.py --endpoint $P1,$P2 --d-addrs $D --prompts-file h28_x800.json --nonce-pos suffix \
    --qps 7.6 --duration 42 --max-tokens 1024 --warmup 1 --out e2e-q7.6.jsonl
python3 an_uniform_load.py e2e-q7.6.jsonl
```

`$D` is the comma list of every decode DP rank's http address (PPU DEP8: 8 entries); several P instances are given as
a comma list to `--endpoint` (request i goes to P i % n). Decode capacity at real acceptance: hold the decode until
`cc` requests are ready (PPU: D-side `DSV4_DECODE_BARRIER_BATCH=cc`; 5kpro TP4: requests sent at once), then
`../decode/decode_plateau_bench.py --cc <cc> --max-tokens 1024 --tokens-per-step 2 --d-addrs $D`, and read
`measured_tpot_ms` / `measured_tok_per_s`; the decode capacity is tok/s at the 15 ms TPOT line / 1024.

eRDMA vs TCP with whole-batch prefill (4 x (30K hit + 2K new) per round, P started with `CP_SINGLE=0 MAX_CTX_BS=4`,
`RDMA=0` for TCP on both sides, `MTP_CHUNKED=0 FASTGEN_ARGS=' '` for the non-fastgen path):

```bash
python3 make_prefix_prompts.py --base prompts_32k_x24.json --tokenizer $CKPT --hit 30720 --count 400 --seed 50000 --tag T --out h30_x400.json
python3 pd_prefix_gated_client.py --prompts h30_x400.json --rounds 6 --conc 4 --max-tokens 128 --gate 1.0 --tag rdma-fg --out-dir p2
python3 an_gated_rounds.py p2
```

A 5kpro TP4 decode peer for these runs is `../decode/run_decode_tp4_sm120_pd.sh`.
