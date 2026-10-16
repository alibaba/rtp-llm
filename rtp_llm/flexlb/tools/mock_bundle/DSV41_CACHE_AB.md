# DSv4.1 cache incident A/B

Both variants run a real Master and the same Java Mock P/D harness as separate
JVMs in one dedicated Whale CPU Pod, with an independent frontend.

Baseline production source: `feature/dsv41-dev-main-202161013`,
`23e2df59286a18fcb6cb9adb56796d16675aa9a2`.
The A harness import changes no Master production Java source. Its Maven profile
adds the real KMonitor sink while retaining test file discovery.
Mock and bundle tooling come from `c211cfd1e7`; they are test infrastructure,
not incident production code. A and B must use the same harness code.

## Required monitoring identity

Keep physical `hippo_app`, `hippo_role`, `host_ip` and `container_ip` unchanged.
Logical engines additionally report `role=ROLE_TYPE_PREFILL|ROLE_TYPE_DECODE`,
`engine`, `engine_ip`, `engine_port`, `generation`, and `backend=mock`.
Filter by logical role and then engine for individual observations. A physical
Master role does not imply that embedded Mock metrics describe the Master.

`mock_cache_fixture_keys` counts synthetic cache-status keys held by each engine;
its sum repeats shared keys and is not the Master global unique-key count.
`rtp_llm_kv_cache_item_num` retains inference LRU semantics.
`mock_engine_up` describes engine availability, not Master routing health.
Actual fixture response, entry, protobuf-byte, addition and removal rates are
reported per engine. Produced responses do not prove Master application success.
Record Master and Mock JVM CPU, heap, allocation and GC separately by process.
Dashboard acceptance requires nonempty backend samples for both roles and at
least two different engines, plus a positive key seed visible in the Master.

## Two sequential experiments for each artifact

1. Steady: keep all Workers present and healthy; grow cache to the agreed size,
   then maintain small key changes and measure sustained Full GC. Record actual
   response/s, entries/s, bytes/s, diff workload and allocation rather than target
   polling frequency. Distinguish cache ramp GC from repeated steady-phase GC.
2. Recovery: on the same settled cache geometry, withdraw only selected Workers
   and restore them. Track workers that were never withdrawn separately. A valid
   failure requires observed delayed/stale/dead healthy Workers and supporting
   sync/cleanup/GC evidence, not merely successful withdrawal and rejoin.

Starting target: 120P/320D, 306000 keys/P, 28.92M global unique keys, about 50
successful P cache syncs/s, and order-100 additions/removals per changed snapshot.
The requested half-resource Pod has 16 CPU and 60000 MiB total memory. Master and
Mock share that CPU/memory budget; JVM heaps are not independent CPU quotas.
Freeze actual heaps, GC flags, response semantics, topology, input and event
times before either measured run, and keep them identical across A/B except
for explicitly recorded fixes.

Same-version key omission in this CPU experiment is implemented by the Mock's
protocol behavior switch. It validates the Master's response to that behavior;
it does not replace execution/validation of the real C++ engine implementation.
No VIT Worker is modeled by this P/D experiment.

No result is claimed until artifact identity, runtime health, monitoring data,
measured intensity, observation duration and the relevant assertions are checked.
