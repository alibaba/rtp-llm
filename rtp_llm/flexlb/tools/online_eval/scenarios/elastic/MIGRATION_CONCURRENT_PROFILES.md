# Concurrent mutation profile migration

Reference legacy source: `a22f0678a2beb479c3da7ff9fa09df9c354f3d19`.
Its `elastic_concurrent_ops` registration permits all four profiles. The original
BW candidate remains `default`; three explicit variants add the missing pairs:

| Profile | Variant | Expected consumer |
| --- | --- | --- |
| batch-window | default (unchanged checks) | Existing behavior |
| single-batch | single_batch | FetchResponse |
| single-nonbatch | single_nonbatch | GenerateStreamCall |
| window-nonbatch | window_nonbatch | GenerateStreamCall |

Only this family's required capability changes from `enqueue_batch` to `queue`.
Its assertions concern concurrent control operations, HTTP health, request
success and final discovery. It has no Prefill batch-ledger-zero predicate to
reinterpret as route-owner cleanup. Other families retain their own requirements.

Each profile preserves the complete legacy resolved configuration: private 2P/4D,
fault performance preset, dynamic discovery, PRIORITY over the selected profile,
and omitted queue timeout. Tests compare full resolved configuration, performance,
topology and cache pool settings against `_elastic_env`, using shared core
`4e2139c1398f1ed3039d235d2f736456612fa0f8` (whose legacy elastic source matches
the reference). This overlay is necessary because the candidate branch carries
an older legacy registration.

Four mutation workers, the 10s admission window, add/remove intervals, add HTTP
10s and remove HTTP 95s budgets, 65-addition cap, health success floor of 50%, and
discovery after worker completion remain unchanged. Health requests retain 2048
input tokens, two output tokens, one unique cold key, Schedule timeout 30s and
stream timeout 10s. No threshold is relaxed.

The new variants add a protocol construction check after crossfire. Every
admitted request must record the expected consumer, with a nonempty cohort.
Fetch selection follows Schedule metadata; it does not prove every request has
already reached EnqueueBatch. NON_BATCH rebuilds the same Generate request shape
and copies role addresses. Existing completeness and business-success checks
remain separate from the protocol check.

`test_scenario_elastic_concurrent_profiles.py` runs the actual Python
RecordedRequests driver with simulated Schedule and stream RPCs, alongside the
four mutation threads. It checks all three new paths and request shapes,
business-finished/transport-terminal/consumer-exit records, rejection of the
opposite protocol even when streams succeed, and failure of the unchanged health
floor on business errors. These fixtures establish Python construction behavior;
they are not real Java scheduling or throughput acceptance.

Independent static and fixture review passed for candidate
`1f750ae3d141f4e00ec4c4b73b6700ed0157085b`: the reviewer independently ran all
99 elastic tests against core4e in 20.723s, compared the unchanged BW plan with
its parent, and checked the three new profile configurations and driver paths.
Real Java acceptance remains pending; legacy Python cases remain available.
This signs three candidate profile pairs, not the overall 371-profile target.
Remaining pair counts follow the coordinator's current inventory.
