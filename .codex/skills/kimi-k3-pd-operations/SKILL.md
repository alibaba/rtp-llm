---
name: kimi-k3-pd-operations
description: Operate and diagnose the two-host Kimi K3 RTP-LLM Prefill/Decode deployment on 11.163.39.114 and 11.163.39.115. Use for starting or stopping P/D, checking load progress and health, sending HumanEval token-ID requests, measuring Eagle3 speculative acceptance, handling proxy or GPU residue problems, capturing layers 0/44/88 hidden states, or comparing RTP-LLM hidden states with MAL reference files.
---

# Kimi K3 P/D Operations

Read [references/runbook.md](references/runbook.md) before touching either host.

## Core invariants

- Require Prefill `KIMI_K3_KDA_BACKEND=cula`. Never launch Prefill with `kernel`.
- Use Decode `KIMI_K3_KDA_BACKEND=kernel`. Use target-verify `kernel` for the validated production path; switch to `reference_recurrent` only for numerical diagnosis.
- Use auxiliary layer IDs `0,44,88`, corresponding to MAL auxiliary layers `1,45,89`.
- Remove all HTTP proxy variables from service and test-client environments.
- Resolve stale GPU processes and ports before launch. Do not start on partially occupied GPUs.
- Preserve unrelated dirty-worktree changes. Inspect `git status --short` before editing or committing.

## Workflow

1. Open interactive SSH sessions because both `enter.sh` scripts require a TTY.
2. Enter the 114 and 115 containers using the paths in the runbook.
3. Check GPU memory, service processes, ports `27188/28188`, and health endpoints.
4. Start P/D with the canonical environment from the runbook. Start simultaneously when requested.
5. Wait until both `/health` endpoints return HTTP 200 before sending requests.
6. Run a small request only when endpoint validation is needed; exclude warm-up results from measurements.
7. Run `scripts/run_humaneval.py` against the five MAL HumanEval `.pt` files and report per-query accepted/proposed counts, weighted rate, EOS status, and output sanity.
8. Treat very high acceptance with repeated token `0`, `!`, or a short phrase as corruption, not success.
9. For hidden comparison, verify shape, token identity, finiteness, RMSE, maximum absolute error, cosine similarity, and reproducibility against an earlier RTP CULA dump.
10. Leave services in the state requested by the user and state whether they remain running.

## Safety checks

- Stop only the scoped Kimi K3 service processes. Check exact PIDs before signals.
- After terminating the parent, also check multiprocessing trackers and GPU contexts; allow delayed GPU release before force-killing verified leftovers.
- Do not delete broad temporary directories to address inode pressure. Prefer moving `KIMI_K3_RUN_ROOT` and `KIMI_K3_TMPDIR` to `/data0/xinfei.sxf/...`.
- Never report acceptance rate without checking the generated text and token-ID distribution.
