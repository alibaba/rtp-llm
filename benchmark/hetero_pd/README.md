# Hetero PD KV-transfer measurement tools

Tools for measuring the prefill -> decode KV transfer of PD-disaggregated serving directly from the eRDMA
NIC's hardware byte counters, lined up with the prefill engine log (both on the prefill host's clock).
See `docs/backend/hetero_pd_kv_transfer_opt.md` for how they were used and what they found.

| File | Purpose |
|---|---|
| `nic_sampler.py` | sample `hw_tx_bytes_cnt` / `hw_rx_bytes_cnt` of an RDMA port every few ms |
| `pd_bench.py` | streaming client for the prefill server's native endpoint; per-token arrival times and `output_ids` |
| `kv_bw.py` | per-request KV bytes, busy window, mean / 10-ms peak rate, bytes after the first token |
| `kv_rounds.py` | bytes per prefill chunk for one request |
| `burst.py` | backlog mode (`RTP_LLM_PD_PUBLISH_AT_END=1`): per-stage drain bursts and rates |
| `handoff.py` | prefill-side PD handoff timeline from `PD_HANDOFF_TRACE` (`PREFILL_TRACE_LOG_ENABLE=1`) |
| `runstat.py` | one-line summary per run directory |

A run directory holds `m<k>.json` (one `pd_bench.py --n 1` result per request), `nic.txt` (sampler output),
`kv_bw.json` (`kv_bw.py --json-out`) and `logs/engine.log` (prefill engine log).
