# Balance scenario migration

Base: `81286a7e7d9685ef7203355899dc51404d258976`. Six legacy Python functions
remain unchanged. These two YAML definitions contain six explicit variants,
24 profile instances and 180 declared checks. This is a candidate mapping, not
permission to delete the legacy functions or a claim of remote Java success.

Each target below names actual `stage.check`. No action dispatches on a legacy
case name or calls a legacy case function. YAML owns request counts, stage order,
timing, workload lengths, grade overrides and resource references.

| Old contract | Variant | Target checks and retained measurement |
|---|---|---|
| `balance_uniform_serial` | `balance_distribution::uniform_serial` | `plain_p6.property`, `plain_p1.property`, `plain_p2.property`; same three `speed_hetero_*` checks. Two sequential 20-request cohorts, client Prefill landing addresses, P1 run-grade bands, P2 >=2 workers. Second P worker gets 200 ms fixed execution and 1.5 s sync; restore to 100 ms. |
| `balance_concurrent_mix` | `balance_distribution::concurrent_mix` | `p6.property`: 20 issued, >=8 successful, only legacy `NO_PREFILL_WORKER` admission failures tolerated; `p1.property`: successful-client max share with relax=1; `p2.property`: >=2 workers. Concurrency=20. The inherited string match is not advertised as an exact typed error-code check. |
| `balance_decode_spread` | `balance_distribution::decode_spread` | `n10_p6/p2/p1.property` then `n50_p6/p2/p1.property`. Same live environment, separate Decode completed baselines. Every issued request completes and total delta >=n. >=2 / >=3 Decode workers. Max delta divided by issued n, bands .60/.70/.80 and .40/.50/.60 respectively. |
| `balance_len_mixed` | `balance_distribution::length_mixed` | Each `waveN_p6.property` and `waveN_master_clean.all_owners_zero`; final `token_p3.property`, `short_p2.property`. Five explicitly expanded waves; each has two long requests from `[131072+(i%5)*4096 for i in range(10)]` and six 512-token requests. Both long dispatches have named engine-pending checks. Batch Fetch deferred until drain; NON_BATCH stream starts at fire. Per-request stream budget 30 s, per-wave Master cleanup 30 s. P3 uses completed input tokens; P2 requires short requests on >=2 workers. |
| `balance_overload_avoid_decode` | `balance_overload_transfer::decode_pressure` | `p6.property`, `p5.property`, `p2.property`. First D gets available+active KV pressure; sync 1 s, then snapshot baseline and 10 sequential requests. No errors and total completed delta >=10. Target delta uses 0/1/2 grade bands. >=2 healthy D engines take work and their delta covers n-target_delta. Pressure cleared in registered cleanup. |
| `balance_overload_avoid_prefill` | `balance_overload_transfer::prefill_pressure` | `baseline_p6.property`, `p6.property`, `p5.property`, `p7.property`. Both P workers slow to 5000 ms, 1.5 s sync, 147456-token seed; poll routed engine pending within 6 s, restore other P to 100 ms then .3 s sync. One completed baseline, five short requests fired sequentially .12 s apart before collecting. P5 hot share denominator is all five issued. P7 uses Schedule-start to stream terminal for BATCH, Schedule-start to first output for NON_BATCH (matching actual old code, not its inaccurate “Schedule-return” prose), max successful wave timing / baseline. Seed drained, perf restored, then owned environment teardown. |

All six old cases apply to all four profiles and have `expected_fail=False`.
Every variant retains those profile combinations. Each `balance_check` inherits
`ctx.instance.grade` (fallback `normal`) unless its explicit
`grade: strict|normal|loose` parameter overrides it. It uses the existing
`GradeReport` band resolver and persists achieved grades in evidence.
There is no separate scenario `--grade` CLI in this frozen core; strict/loose
selection must currently be an explicit stage parameter/override or supplied
instance field. Parent/child CLI plumbing is a separate framework change.

## Explicit compatibility boundaries

- Fresh per-instance environments replace incidental predecessor-case state.
  The two uniform phases and both Decode sample tiers remain in one environment.
- Missing/negative counters, missing inventory, changed endpoint identity and
  missing client terminal evidence are ERROR, never synthetic zero or PASS.
  Endpoint identity checks cover environment epoch, worker names/roles/addresses;
  mock generation is not exposed by this snapshot and is not claimed verified.
- Old aggregate pending evidence stays aggregate. A routed request plus a
  positive engine pending count does not prove a particular RID is pending.
- Normal request failures remain ordinary P6 FAIL. Once a check fails, later
  stages are BLOCKED under the new runtime; they are not counted as covered PASS.
  The old functions may compute more diagnostic grades after a failure.
- A finite serial cohort records all declared requests even if an earlier one
  failed; unlike the old uniform loop it does not stop issuing at the first
  failure. Completeness checks preserve that failure and require the declared
  sample count. This is an explicit failure-path observation extension.
- Request cleanup uses the core RequestBatch completion events and persisted
  consumer terminal evidence. The finite pump additionally requires its own
  done event and exit timestamp. Cancelling a transport is not business success.
- Legacy best-effort post-case TTL cleanup is replaced by destruction of the
  owned instance environment; it is not promoted into a business assertion.
  The five asserted within-wave all-owner cleanup windows remain 30 seconds.
- Engine mutation acknowledgement and stricter missing-source rejection are
  framework preconditions; no threshold or expected-failure band was relaxed.

## Validation and integration

`tests/test_scenario_balance.py` compiles both shipped YAMLs and executes all 24
plans with real stage/check implementations and fake external HTTP/traffic.
Additional tests run actual RequestBatch consumer threads with fake RPC calls,
including deferred batch Fetch, direct Generate, cancellation and terminal
artifact persistence. Negative checks cover failed business requests, missing
counter fields, endpoint replacement, unbounded parameters, empty cohorts and
completion signals without exit evidence.

The adapter exports `HANDLERS`; the framework owner registers it in the shared
catalog. Tests explicitly inject the descriptors, so the shared catalog is not
edited by this change. Remote Java execution and independent per-contract audit
remain required before any legacy contract is marked fully migrated.
