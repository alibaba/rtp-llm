# PD measurement tools (decode-side branch)

Clients and analysis for prefill/decode separated DeepSeek-V4 serving with a PPU
decode (DEP8) and either a 5kpro or a PPU prefill. Copies of the prefill-branch
tools (`dev/mtp-pp-prefill` `bench/pd`, `bench/decode`) that the PPU-side runs
use, plus `ppu_role.py` to start PPU roles from env-style JSON configs.

| File | Purpose |
|---|---|
| `ppu_role.py` | Merge a base PPU config with the PD knobs (decode addresses, MTP depth, decode barrier, forced acceptance, RDMA/TCP, any `--set K=V`) and optionally launch the role detached (`--launch RUN_DIR`). |
| `make_prefix_prompts.py` | Shared-prefix prompt sets (`--hit` cached tokens, distinct suffix, city check). |
| `pd_uniform_load.py` | Open-loop constant-rate load (request i at `t0 + i/qps`), `ignore_eos` fixed-length outputs, several P endpoints, decode rank pinning (`--d-addrs`). |
| `an_uniform_load.py` | Summary of `pd_uniform_load.py` runs: send rate, first-token completion rate, TTFT per quarter, TPOT, finish reasons. |
| `pd_prefix_gated_client.py`, `pd_prefix_c4_client.py`, `an_gated_rounds.py` | Rounds of N requests entering one prefill batch (gated) and their TTFT / 2nd token / TPOT summary. |
| `prefill_only_conc.py` | Prefill-only engine throughput (no decode peer). |
| `pd_cost_model.py` | P / D capacities -> ideal and whole-machine P:D mixes and cost per QPS. |
| `../decode/decode_plateau_bench.py`, `../decode/make_distinct_prompts.py` | Decode-only TPOT / throughput with all requests decoding together. |
| `pd_perf_client_v2.py`, `pd_distinct_client.py` | Older TTFT / TPOT clients. |

## PPU roles

```bash
D=$(for r in 0 1 2 3 4 5 6 7; do printf "<d-host>:%d," $((21000+9*r)); done); D=${D%,}
# DEP8 decode, MTP 2+1, real acceptance, decode fusions on
python3 ppu_role.py run/decode.json --base decode-dep8-rdma.config.json --mtp 2 --barrier 0 --force-accept 0 \
    --set DSV4_PPU_DECODE_FUSIONS=1 --set DSV4_PD_SKIP_MTP_MODULE_LOAD=1 --launch run
# two TP4 MTP2 prefill instances on one host for a pure-PPU PD
python3 ppu_role.py run/p-a.json --base prefill-tp4-reuse.config.json --decode-addrs $D --mtp 2 --rdma 1 \
    --set CUDA_VISIBLE_DEVICES=0,1,2,3 --set START_PORT=23000 --set REUSE_CACHE=1 --set CONCURRENCY_LIMIT=128 --launch run-a
python3 ppu_role.py run/p-b.json --base prefill-tp4-reuse.config.json --decode-addrs $D --mtp 2 --rdma 1 \
    --set CUDA_VISIBLE_DEVICES=4,5,6,7 --set START_PORT=25000 --set REUSE_CACHE=1 --set CONCURRENCY_LIMIT=128 --launch run-b
```

The base configs carry the deployment paths (conda env, PPU SDK, model). Stop a role
by killing the process group of the pid in `<run>/<role>.pid`.

## Measurements

Decode capacity at real acceptance: restart D with `--barrier <cc>` so decode starts
once `cc` streams are ready, send `cc` requests through the prefill and read
`measured_tpot_ms` / `measured_tok_per_s` (`--tokens-per-step 2` at MTP2; the
prefill concurrency limit must be at least `cc` or the barrier never releases):

```bash
python3 ../decode/decode_plateau_bench.py --url http://<p-host>:14000 --cc 112 --prompts-file h28_list.json \
    --max-tokens 1024 --plateau-s 10 --plateau-skip-s 3 --tokens-per-step 2 --d-addrs $D
```

End to end (restart the prefill instances before every run):

```bash
python3 pd_uniform_load.py --endpoint 127.0.0.1:23000,127.0.0.1:25000 --d-addrs $D --prompts-file h28_x800.json \
    --nonce-pos suffix --qps 5.5 --duration 40 --max-tokens 1024 --warmup 1 --out e2e-q5.5.jsonl
python3 an_uniform_load.py e2e-q5.5.jsonl
```

The sustainable QPS is the highest rate whose TTFT quarters stay flat and whose TPOT
p50 stays within the SLA.
