# FlexLB Integration Tests

`flexlb-integration-test` is the transport-real, single-engine regression suite for FlexLB.
It starts the real Spring Boot/WebFlux application with a random HTTP port and replaces only
external engine and KVCM boundaries with loopback gRPC servers. Routing, queueing, worker-status
synchronization, cache matching, and fallback logic therefore run through production beans.

The module deliberately excludes multi-engine logical identity, `RANDOM`, and `WEIGHTED_CACHE`.
Each Failsafe fork owns a fresh fake cluster and static FlexLB state, so a scenario cannot leak
worker status or local tasks into another scenario.

## Run

Run the complete integration suite with its reactor dependencies:

```bash
./mvnw -pl flexlb-integration-test -am verify
```

Run one scenario while retaining the full application and gRPC boundaries:

```bash
./mvnw -pl flexlb-integration-test -am verify -Dit.test=CacheAffinityFirstIT
```

Run the optional 200-worker load regression separately from the normal PR gate:

```bash
./mvnw -pl flexlb-integration-test -am verify -Pstress-it -Dit.test=WorkerScaleLoadIT
```

Generate one cross-module JaCoCo report, including production-unit and integration-test execution
data:

```bash
./mvnw -pl flexlb-integration-test -am verify
open target/site/jacoco-aggregate/index.html
```

The root POM defines the output directory. The integration-test module is the final production
consumer in the reactor and declares optional compile dependencies only for aggregation, so its
JaCoCo aggregate execution sees every production module's class/source data and its own Failsafe
execution data after all upstream tests have finished.

`./mvnw test` compiles this module but does not execute `*IT` classes; Failsafe executes them in
the `integration-test` and `verify` phases. The API module publishes a `plain` classifier during
`process-classes` so the root reactor's normal `test` phase can compile this module without first
reaching `package`.

## Harness model

`fixture.engine.IntegrationTestFixtures` uses a `WorkerTopology` declared as `RoleType -> worker
count`. There are no fixed `WORKER` or `SECOND_WORKER` slots: a test addresses an engine fake as
`(role, index)`. `fixture.kvcm.KvcmIntegrationTestFixtures` separately owns KVCM discovery,
cache-match scripting, and KVCM wire observations. Each context initializer declares its strategy,
queue/fallback options, engine mode, and matching static-discovery configuration.
`ScriptedWorkerStatusService` implements the real `GetWorkerStatus` and `GetCacheStatus` RPCs;
the package-private KVCM adapter implements the KVCM metadata RPCs behind its focused fixture.

The tests wait on observable state (HTTP completion, fake RPC calls, queue size, or synchronized
`WorkerStatus`) rather than using fixed test sleeps. The one scheduler worker in the queue profile
is intentional: it makes the intermediate short-bucket queue shape deterministic.

## Package layout

```text
org.flexlb.it
├── scenario
│   ├── strategy.shortestttft   # direct SHORTEST_TTFT tests and initializer
│   ├── strategy.cacheaffinity  # CACHE_AFFINITY_FIRST guards and initializer
│   ├── queue                   # short-bucket scheduler scenario
│   ├── cache.kvcm              # SGLang hash-to-KVCM contract
│   ├── cache.rtpllm            # caller-provided RTP cache keys and local cache status
│   ├── cache.standby            # KVCM fallback and Local Standby capacity expiry
│   ├── fallback                # global fallback gate
│   ├── worker                  # role topology and status-failure/latency resilience
│   └── stability               # opt-in high-scale load regression
├── configuration               # shared Spring context/config assembly only
└── fixture
    ├── engine                  # engine-worker gRPC fake and shared worker scripting
    ├── kvcm                    # KVCM gRPC fake and cache-query scripting
    └── spring                  # Spring-only test configuration
```

Tests may depend on a same-scenario initializer and the public fixture facades. Initializers depend
on `configuration`, while `configuration` depends only on the engine/KVCM fixture facades. Scenario
tests never import a package-private gRPC fake implementation directly.

## Scenario matrix

| IT class | Scenario and assertions |
| --- | --- |
| `SingleEngineSchedulingIT` | `SHORTEST_TTFT`: only healthy worker is selected; all-down returns a controlled error; a short request avoids a worker reporting a 1M-token, 16K/32K-chunk long prefill. |
| `ShortestTtftSimilarityRatioIT` | `SHORTEST_TTFT`: a 75-token no-cache worker competes with a 100-token worker holding a complete local cache hit. `shortestTtftSimilarityThresholdRatio=0.2` excludes the cache worker; `0.5` and `0.8` include it and select it by cache preference. |
| `ShortBucketQueueIT` | Three concurrent short requests enter the real `QueueManager`/`RequestScheduler` while both workers are unavailable. One scheduler retry is active and two requests wait; after recovery all three complete and the queue drains. |
| `WorkerStatusResilienceIT` | One worker repeatedly returns gRPC `UNAVAILABLE`; twenty short requests continue to select the healthy worker. The expected injected RPC errors may appear in test logs. |
| `WorkerStatusLatencyIT` | With a 50 ms status cadence, every third status RPC from one worker is delayed by 200 ms. Thirty short schedules continue to complete, and both workers remain healthy after multiple delayed responses. |
| `RoleAwareWorkerTopologyIT` | A dynamically declared mixed topology (`PREFILL=2`, `DECODE=3`, `PDFUSION=1`) is synchronized into the corresponding role maps. |
| `CacheAffinityFirstIT` | `CACHE_AFFINITY_FIRST` retains the KVCM cache leader only within the max-extra-work budget, the neutral `outstandingUncachedTokensThreshold` (or its legacy cache-affinity fallback), and `cacheAffinityFirstMinHitRate`; each rejection selects the shortest eligible TTFT worker. It also keeps KVCM active for a valid empty result and moves to local standby only after KVCM query failures reach the configured threshold. VLLM keys are derived from `input_ids` and checked on the KVCM wire. |
| `SglangKvcmHashIT` | SGLang derives block keys from `input_ids` using the production SGLang strategy and sends those exact keys to KVCM. |
| `RtpLlmCacheStatusIT` | RTP-LLM accepts caller-provided `block_cache_keys`; no RTP hash strategy is invented. The keys must first arrive through a worker's real `GetCacheStatus` response, then route using `LOCAL_SYNC`. |
| `LocalStandbyCapacityIT` | KVCM query failures activate Local Standby. A one-entry standby index is filled from a routed VLLM request; at capacity a later request is rejected, high-watermark cleanup removes the expired entry, and the next request-derived mapping is admitted. The test intentionally covers TTL-based eviction, because unexpired mappings are retained by design. |
| `FallbackGateIT` | Global `enableFallback` returns `FALLBACK` (8600) before scheduling or contacting a worker. |
| `WorkerScaleLoadIT` (`stress-it`) | Starts 200 PDFUSION worker-status endpoints, waits for all 200 production snapshots, and completes 400 HTTP schedules at concurrency 32. It is a bounded correctness/liveness regression, not a machine-independent QPS benchmark. |

## Engine-key contract

| Engine mode | Request input used by the test | Cache match source | What the assertion proves |
| --- | --- | --- | --- |
| VLLM | `input_ids` | KVCM | FlexLB runs the configured VLLM block hash and forwards the resulting keys to KVCM. |
| SGLang | `input_ids` | KVCM | FlexLB runs the configured SGLang block hash and forwards the resulting keys to KVCM. |
| RTP-LLM | caller-provided `block_cache_keys` | `LOCAL_SYNC` | FlexLB preserves supplied keys and matches the local cache index populated from `GetCacheStatus`. |

## Adding a scenario

1. Add an `*IT` class under `src/test/java/org/flexlb/it` with `@SpringBootTest(RANDOM_PORT)` and
   the smallest matching context initializer.
2. Use `fixture.engine.IntegrationTestFixtures` to declare topology and script engine status/cache
   keys, plus `fixture.kvcm.KvcmIntegrationTestFixtures` only when the scenario requires KVCM;
   do not mock `RouteService`, the scheduler, cache orchestrator, or load-balancing strategy.
3. Use Awaitility with an explicit zero poll delay, 10ms poll interval, timeout, and alias for
   asynchronous transitions; assert the final HTTP result plus the state specific to the scenario
   (selected endpoint, cache source/wire keys, queue drain, or RPC call count).
4. Keep a new scenario single-engine and deterministic. Add queue-full/timeout/cancel, master
   forwarding, and stale-version/restart flap coverage as separate increments. Put any further
   high-scale or soak case behind a dedicated Maven profile.
