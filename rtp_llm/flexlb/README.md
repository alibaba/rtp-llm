# FlexLB - Intelligent Load Balancer for AI Model Inference

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Java](https://img.shields.io/badge/Java-8+-red.svg)](https://www.oracle.com/java/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-2.7.1-brightgreen.svg)](https://spring.io/projects/spring-boot)

FlexLB is a high-performance, intelligent load balancer specifically designed for AI model inference workloads. It provides advanced load balancing strategies, request batching, caching mechanisms, and automatic failover to optimize the performance and reliability of AI service deployments.

## Features

- **Smart Load Balancing**: Multiple strategies including cost-based routing, shortest TTFT, and cache affinity
- **Request Batching**: Intelligent batching of inference requests to improve throughput
- **Advanced Caching**: KV cache management for improved performance
- **Health Monitoring**: Real-time worker health checking and automatic failover
- **Reactive Architecture**: Built on Spring WebFlux for high concurrency
- **gRPC Support**: Native gRPC client implementation for backend services
- **Metrics & Monitoring**: Prometheus metrics integration
- **Master Election**: ZooKeeper-based master election for high availability

## Architecture

FlexLB consists of four main modules:

- **flexlb-api**: Web layer providing HTTP endpoints and reactive web services
- **flexlb-common**: Shared utilities, data models, exception handling, and common configurations
- **flexlb-grpc**: gRPC client implementation for model service communication
- **flexlb-sync**: Core load balancing logic, scheduling strategies, and worker status synchronization

## Quick Start

### Prerequisites

- Java 8 or higher
- Maven 3.6+ (optional, project includes Maven Wrapper)
- ZooKeeper (optional, for master election)

### Build

This project includes Maven Wrapper, so you don't need to install Maven separately.

#### Using Maven Wrapper (Recommended)

**Unix/Linux/macOS:**

```bash
./mvnw clean package -DskipTests
```

**Windows:**

```bash
mvnw.cmd clean package -DskipTests
```

#### Using System Maven

```bash
mvn clean package -DskipTests
```

#### Maven Wrapper Benefits

- **Environment Consistency**: Ensures all developers use the same Maven version
- **Simplified CI/CD**: No need to pre-install Maven in build environments
- **Version Lock**: Project specifies the exact Maven version, avoiding compatibility issues

#### Maven Wrapper Files

The following Maven Wrapper files are included in the project (do not delete):

```
├── mvnw              # Unix/Linux/macOS script
├── mvnw.cmd          # Windows script
└── .mvn/
    └── wrapper/
        ├── maven-wrapper.jar        # Core Maven Wrapper JAR
        └── maven-wrapper.properties # Configuration file
```

### Code Formatting

FlexLB uses the repository-root [pre-commit](https://pre-commit.com/) configuration.
Java is formatted with Spotless and google-java-format 1.28.0 (AOSP four-space style,
preserving Javadoc layout); Python uses the repository's Black and isort versions.
C/C++/CUDA files use the existing clang-format configuration; Protocol Buffers use
clang-format's Google style. Shell scripts use shfmt. JavaScript, HTML, JSON, YAML,
XML and Markdown use Prettier (with its official XML plugin).

Enable the repository's Git hooks once per checkout (run from the repository root):

```bash
uv tool install pre-commit  # alternatively: pip install pre-commit
git config --local core.hooksPath .githooks
```

Each `git commit` runs the formatters when matching files are staged. Java changes
run `spotless:apply` across the FlexLB reactor; Spotless caches files that are already
clean. The other formatters process staged files. If a formatter changes a file,
review and stage the result, then commit again. Unstaged changes are preserved by pre-commit.
Java formatting requires JDK 21; Maven is supplied by `mvnw`. The first run downloads
the formatter dependencies. Compilation and tests remain separate build commands.

To format all FlexLB code, run from the repository root:

```bash
git ls-files -z rtp_llm/flexlb | xargs -0 pre-commit run --files
```

To apply or check Java formatting across all Maven modules:

```bash
cd rtp_llm/flexlb
./mvnw spotless:apply -Pspotless-check
./mvnw spotless:check -Pspotless-check
```

### Configuration

`FLEXLB_CONFIG` is a required JSON environment variable. It configures scheduling,
delivery, routing, worker synchronization, and observability; file paths are not accepted.
The online loader accepts schema 3 only. Duplicate keys, unknown or inactive fields,
`null`, scalar coercion, numeric enums, and trailing JSON fail startup.

| Setting | Schema 3 behavior |
| --- | --- |
| `dispatcher.maxInflightPerPrefillWorker` | Positive integer, default 2 in every mode. BATCH counts batches; NON_BATCH / DIRECT count requests reserved for delivery or still in flight. QUEUE waiting requests do not consume this limit; full workers remain eligible for cache-affinity selection. |
| `dispatcher.fetchAttachTimeoutMs` | Positive integer, default 3000 ms. BATCH sends this as Engine `fetch_attach_timeout_ms`, limiting the wait for `FetchResponse` attachment. Engine caps it by the remaining request timeout when present. Once attached, this timer does not limit generation duration. |
| Prefill ownership | Work remains tracked until authoritative completion, safe rollback or retirement. PDFUSION has no distinct Prefill-completion signal, so its ownership lasts through request termination. |
| `requestLifecycle.request.timeoutMs` | Required positive integer; maximum request inactivity in milliseconds, renewed by matching Engine request status. No default. |
| `workerRegistry.health.cleanupIntervalMs` | Positive integer, default 3000 ms; interval for completing confirmed worker retirements. |
| Worker health | Three valid not-alive responses mark a worker unavailable. RPC failures preserve established health and neither increment nor reset the count; a valid healthy response resets it before retirement. Confirmed discovery removal retires the worker immediately. Status age does not determine health. Exceptions alone cannot confirm a crashed worker while discovery still lists it; no engine-specific error metadata is required. |
| Worker status polling | Each worker generation has at most one RPC in flight and one response queued or applying locally. RPC polling continues during local application; scheduling uses the last committed state. |
| Discovery failures | Timeout, query failure or executor rejection preserves existing workers and continues status polling. Only a successful discovery result confirms removed addresses. |
| `requestLifecycle.decision.lifetime` | Finite number ≥ 1, default 2.0. At delivery, remaining Prefill time (including waiting) × lifetime + 10000 ms sets the deadline. Only observed running predecessors consume elapsed time; unstarted work keeps its full estimate. Independent of request age. |
| `scheduler.queueTimeoutMs` | QUEUE only, positive, default 3600000 ms; actual queue TTL. DIRECT has no queue timer and rejects this field. |
| Removed capacity controls | `scheduler.capacity` and delivered-not-accepted count limits are removed. Per-Prefill concurrency uses the single dispatcher limit; lifetimes bound waiting. The old `maxUncachedTokens` field is rejected. |
| Removed routing controls | Prefill candidate randomization/LRU and outlier filters, legacy Decode decay controls/outlier filters, and output-estimate truncation are rejected. |
| `router.roles.decode.costEstimator.expression` | Default `kvcache_used_ratio`; smallest cost wins within the same availability tier. Using `max_running_size` requires a positive `availability.maxEngineRequests`. |
| Transport timeout | `flexlb.engine-grpc.enqueue-timeout-ms`, default 5000; outside the scheduling JSON. |
| `scheduler.ordering.preemption.timeoutMs` | Default 1000 ms, positive; Engine-owned Decode preemption only. Starts after the Cancel ACK phase and bounds the wait for the Engine terminal. ACK timeout stays internal, 50 ms. |
| `scheduler.ordering.preemption.allowedVictimStages` | PRIORITY defaults to all three stages: `PREFILL_QUEUED`, `DECODE_RESERVED`, `DECODE_ENGINE_OWNED`, including when `preemption` is omitted. An explicit non-empty list replaces the defaults. |

The request timeout below is an explicit workload value; decision lifetime 2.0 is the default:

```bash
export FLEXLB_CONFIG='{
  "schemaVersion": 3,
  "scheduler": {
    "type": "QUEUE",
    "queueTimeoutMs": 3600000,
    "ordering": {"type": "FIFO"},
    "decision": {"type": "FIXED_WINDOW", "maxRequests": 8, "maxCollectionWaitMs": 300}
  },
  "dispatcher": {"type": "BATCH", "maxInflightPerPrefillWorker": 2, "fetchAttachTimeoutMs": 3000},
  "requestLifecycle": {
    "request": {"timeoutMs": 60000},
    "decision": {"lifetime": 2.0}
  },
  "router": {
    "roles": {
      "prefill": {},
      "decode": {
        "costEstimator": {"expression": "kvcache_used_ratio"},
        "availability": {"maxKvUsagePercent": 90, "maxEngineRequests": 128}
      }
    }
  }
}'
```

Omit `router.roles.decode.costEstimator` to balance by KV usage ratio. Set its
`expression` to `running_size` to balance by request load, or use a formula to combine metrics:

```json
{
  "schemaVersion": 3,
  "requestLifecycle": { "request": { "timeoutMs": 60000 } },
  "router": {
    "roles": {
      "decode": {
        "costEstimator": {
          "expression": "0.3 * running_size / max_running_size + 0.7 * kvcache_used_ratio"
        },
        "availability": { "maxEngineRequests": 128 }
      }
    }
  }
}
```

`workerRegistry`, `observability.cacheHit`, and `router.groupSelector` remain available.

Allocator diagnostics are configured under `scheduler.nettyMemoryReporter` in the same
`FLEXLB_CONFIG` document, for both QUEUE and DIRECT. Both switches default to **false**:

```json
{
  "scheduler": {
    "nettyMemoryReporter": {
      "enabled": false,
      "intervalMs": 10000,
      "autoCapture": {
        "enabled": false,
        "thresholdBytes": 5368709120,
        "cooldownMs": 300000
      }
    }
  }
}
```

Merge this fragment into the existing configuration; it is not a complete workload config.
Changes take effect at process startup. With sampling disabled there is no allocator
discovery, scan, meter registration, or diagnostic thread. Enabled sampling runs on one
daemon thread with fixed delay (at least 1000 ms), never on RPC or scrape threads. Netty's
reserved/pinned getters acquire arena locks and traverse chunks; start at 10 seconds
and measure impact before increasing the frequency. No message-size traversal is added.

The reporter reads the existing Netty default pool and gRPC 1.65.1's allocator singleton
without replacing either allocator. Shared instances are counted once. Labels identify
allocator instances (`netty_default` / `grpc_default`), not exclusive traffic ownership;
HTTP or forwarding may share these pools. If the gRPC allocator cannot be read or is
unpooled, startup emits `event=netty_memory_allocator_unavailable`; its memory is not
silently reported as zero.

Metrics are `app.netty.direct.reserved.bytes`, `app.netty.direct.pinned.bytes`
(tag `allocator`), `app.netty.memory.sample.duration.ms`, and
`app.netty.memory.sample.timestamp.ms` (last completed scan, Unix milliseconds).
Prometheus deployments must
append `flexlb_app_netty_` to their existing `flexlb.monitor.metric-whitelist` prefixes;
KMonitor uses the existing FlexMonitor registration path.

`autoCapture.enabled=true` requires sampling to be enabled. When JVM direct memory or
the sum of monitored allocator reservations reaches `thresholdBytes`, the existing scan
is written to application logs as `event=netty_memory_capture`, at most once per
`cooldownMs` (at least 1000 ms). Capture adds no allocator scan and does not start GC,
thread dumps, JFR, heap dumps, or external commands. It preserves memory-accounting
evidence, not a buffer ownership graph. The default threshold is 5 GiB; adjust it to
the deployment's direct-memory limit.

Status failures are counted by `app.status.failure.detail`, with fixed labels
`role`, `operation`, `stage`, `reason`, `origin`, and `grpc_code`.
Only confirmed local memory, executor, or socket-binding failures use `origin=MASTER`; other failures
remain `UNCONFIRMED`. `grpc_code=NONE` means an unsuccessful application result
without a gRPC exception. Existing health counters and generation cleanup remain
independent of these diagnostics. Prometheus deployments must append
`flexlb_app_status_failure_detail` to their existing metric-whitelist prefixes.

Outbound `EngineGrpcClient` failures are counted once per logical RPC by
`app.grpc.call.failure` (QPS, PRECISE), only after the final failure. A successful
read-only retry reports success only; its first failure produces no failure metric
or exception log. If the retry also fails, the first cause is preserved as a suppressed
exception on the final failure. Fixed labels are `service` (the channel's service
suffix), `retry` (whether a retry was attempted), `phase`, `reason`, `origin`,
`grpc_code`, `transport_stage`, `transport_event`, and `transport_cause`.
Phases identify local boundaries: `CHANNEL_SETUP`, `CALL_START`,
`CALLBACK_REGISTRATION`, or asynchronous `CALL_RESULT`; they do not prove that the
worker received the request. This metric and final status-failure metrics overlap and
must not be added together. Add `flexlb_app_grpc_call_failure` to the Prometheus
metric whitelist.

Typed causes distinguish connection timeout/failure, no route, EOF, channel already
closed, local interrupt, and asynchronous local close. With the current NIO transport,
refused/reset/broken-pipe observations additionally require exact known text on the
matching socket exception; unknown text stays generic. These are transport observations,
not proof of worker failure. `CONNECT_REFUSED` establishes that the target endpoint did
not accept the connection, rather than a response timeout. It does not distinguish an
absent listener from a stale endpoint or an active network rejection.
Bare gRPC status descriptions are never parsed as evidence.

The NIO observer retains the first event on the exact physical connection used by a
stream. `PEER_EOF` means a socket read returned EOF without local input shutdown;
`READ_RESET`, `WRITE_RESET`, and `WRITE_BROKEN_PIPE` identify the failing operation.
A later broken pipe/channel-close error can therefore be correlated with an earlier
peer EOF. A bare `EOFException` is not sufficient to assert TCP peer closure.
Read-allocation failures before the socket read are also observed: a local OOM followed
by a generic channel-close error reports `origin=MASTER` and
`transport_cause=LOCAL_MEMORY_EXHAUSTED`, and does not trigger a channel retry.
Events after response trailers or stream termination cannot change that call's attribution.

`transport_stage` records progress at termination: `WAITING_TRANSPORT`,
`WAITING_HEADERS_WRITE`, `WAITING_RESPONSE`, `RECEIVING_RESPONSE`, or
`RESPONSE_TRAILERS_RECEIVED`. Headers written do not prove the entire request reached
the worker. `WAITING_RESPONSE` alone cannot separate network delay, worker queue/execution,
and Master event-loop delay; that requires correlated worker-side receive/handler/send
evidence. No message size calculation or payload traversal is performed.

`app.grpc.delivery.delay_ms` (GAUGE, PRECISE) reports `STREAM_CLOSED_TO_CALLBACK`
and `TRAILERS_TO_CALLBACK` under the `stage` label, plus `service` and `retry`.
These measure local delivery/processing delay, not just executor queue wait; unknown
intervals are omitted. Delayed successful callbacks remain successful. Add
`flexlb_app_grpc_delivery_delay_ms` to the Prometheus metric whitelist.
Local channel closes log `event=engine_grpc_channel_close` with `DISCOVERY_REMOVED`,
`REPLACED`, `CLIENT_SHUTDOWN`, or `DUPLICATE_DISCARDED`; final-failure debug logs use
the same channel key and creation time, transport timings and `localCloseForced` for
correlation. A graceful close request does not cancel existing RPCs. A local close may follow a
remote failure and does not by itself change attribution. `app.grpc.channel.pool.size`
counts pooled objects, not live TCP connections; remote disconnects do not remove entries.

Only schema 3 is accepted. Write the configuration directly with explicit values for
`requestLifecycle.request.timeoutMs`, which has no default. A matching Engine request
status renews this deadline. Once it expires, FlexLB releases the exact local
request and its reservations even if delivery acknowledgement or Engine request status is missing.
Only priority preemption may send an Engine Cancel RPC. Delivery uncertainty,
request inactivity expiry, and client-cancellation bookkeeping remain local to the Master.
`requestLifecycle.decision.lifetime` defaults to 2.0. There is no legacy-schema conversion layer.

`MODEL_SERVICE_CONFIG` still describes service discovery and endpoint topology; it is
not a second FlexLB behavior configuration:

```bash
export MODEL_SERVICE_CONFIG='{
    "service_id": "aigc.text-generation.generation.engine_service",
    "load_balance": true,
    "role_endpoints": [
        {
            "group": "blue-group",
            "prefill_endpoint": {
                "path": "/",
                "protocol": "http",
                "address": "com.blue.prefill"
            },
            "decode_endpoint": {
                "path": "/",
                "protocol": "http",
                "address": "com.blue.decode"
            }
        },
        {
            "group": "green-group",
            "prefill_endpoint": {
                "path": "/",
                "protocol": "http",
                "address": "com.green.prefill"
            },
            "decode_endpoint": {
                "path": "/",
                "protocol": "http",
                "address": "com.green.decode"
            }
        }
    ]
}'
```

Static IP lists use `MODEL_SERVICE_CONFIG.hosts`:

```json
"hosts": {
  "com.blue.prefill": ["10.0.0.1:8000", "10.0.0.2:8000"],
  "com.blue.decode": ["10.0.0.3:9000"]
}
```

The key matches the endpoint's `address`; ports follow its `protocol`.
Use `hosts` or `discovery_file`, not both.

Local file discovery uses `MODEL_SERVICE_CONFIG.discovery_file`, pointing to a
JSON mapping of service domains to HTTP host:port lists. Production discovery
providers continue to resolve their service domains.

Master configuration uses `FLEXLB_CONFIG` and `MODEL_SERVICE_CONFIG`.
Leader election and logging are configured through `FLEXLB_CONFIG.consistency`
and `FLEXLB_CONFIG.observability.logging`; the separate `FLEXLB_SYNC_CONSISTENCY_CONFIG`
and `LOG_LEVEL` environment variables are no longer consumed. Spring binds environment
variables to infrastructure properties, including `SERVER_PORT` and
`FLEXLB_MONITOR_PROVIDER`. Ports, RPC transport and logging also accept standard
command-line properties, such as `--server.port` and
`--flexlb.engine-grpc.enqueue-timeout-ms`.
HA currently retains `HIPPO_ROLE` as its existing election group identifier.

The gRPC server executor and shutdown quiet period are configured in `FLEXLB_CONFIG`:

```json
"grpcServer": {
  "executorCoreSize": 1000,
  "executorMaxSize": 1000,
  "executorQueueSize": 1000,
  "shutdownQuietPeriodMs": 5000
}
```

All three executor settings default to 1000. Core size may be 0; maximum size and queue size
must be positive, and maximum size must be at least core size.
`shutdownQuietPeriodMs` (静默时间) defaults to 5000 ms and must be positive.
The quiet period starts when shutting down; each Schedule arrival restarts it.
After it elapses, gRPC waits for accepted RPCs to finish before serving resources
are destroyed. Normal idle serving does not start an exit timer.

### Scheduler, ordering, decision, and dispatcher

Under `QUEUE`, ordering, decision formation, and delivery are three independent
axes:

| Scheduler | Queue ordering | Decision       | Dispatcher  | Behavior                                                |
| --------- | -------------- | -------------- | ----------- | ------------------------------------------------------- |
| `DIRECT`  | not applicable | not applicable | `NON_BATCH` | Route immediately; the frontend sends the request       |
| `QUEUE`   | `FIFO`         | `SINGLE`       | `NON_BATCH` | Form singleton decisions; frontend sends                |
| `QUEUE`   | `FIFO`         | `SINGLE`       | `BATCH`     | Master sends singleton `EnqueueBatch` calls             |
| `QUEUE`   | `FIFO`         | `FIXED_WINDOW` | `NON_BATCH` | Form bounded groups; frontend sends each routed request |
| `QUEUE`   | `FIFO`         | `FIXED_WINDOW` | `BATCH`     | Form bounded groups; Master sends `EnqueueBatch`        |

`PRIORITY` can replace `FIFO` in all four QUEUE combinations. `DIRECT + BATCH`
is invalid and DIRECT cannot configure `decision`. `FIFO`/`PRIORITY` choose which
request is considered first, `SINGLE`/`FIXED_WINDOW` choose how many requests form
one decision group, and `NON_BATCH`/`BATCH` choose whether the frontend or Master
sends them.

`FIXED_WINDOW` uses `maxRequests` (positive integer, default 8),
`maxCollectionWaitMs` (default 300 ms), and optional `maxPredictedExecutionMs`.
Reaching the predicted-time cap dispatches the group; an indivisible singleton may
exceed it. A zero collection window can still group already-waiting requests.
`SINGLE` has no collection parameters. Omitting `scheduler.decision` selects
`FIXED_WINDOW`.

Configuration examples:

- [QUEUE + PRIORITY + NON_BATCH](docs/config-examples/flexlb-queue-priority-non-batch.json)
- [QUEUE + PRIORITY + BATCH](docs/config-examples/flexlb-queue-priority-batch.json)

DIRECT uses `dispatcher.maxInflightPerPrefillWorker` as its per-worker request limit. Example:

```bash
export FLEXLB_CONFIG='{
  "schemaVersion": 3,
  "scheduler": {"type": "DIRECT"},
  "dispatcher": {"type": "NON_BATCH", "maxInflightPerPrefillWorker": 2},
  "requestLifecycle": {
    "request": {"timeoutMs": 60000},
    "decision": {"lifetime": 2.0}
  }
}'
```

PREFILL and PDFUSION use fixed BEST_ONLY selection. With `cacheAffinity` configured,
a cache leader is preferred when its reusable prefix meets `minPrefixHitPercent`
and projected TTFT stays within `maxExtraTtftMs` of the best candidate; otherwise
the best projected TTFT wins. Equal cache hits preserve the best-TTFT candidate.
The prefix percentage uses predictor-effective reusable tokens; the final cache block
remains compute work. Omit `cacheAffinity` to disable that preference.
The projection uses a frozen snapshot of current work and cannot predict future arrivals.

Decode freezes each route attempt's request identity, priority, complete prompt-plus-output
demand, capacity limits and admission mode before selecting workers. DIRECT requires
immediately available dispatch capacity. QUEUE without reclamation can retain a physically
feasible route while Decode is busy and wait for a permit at delivery. With Decode reclamation
enabled, placement also counts queued reservations and can reclaim allowed lower-priority owners.
Selection prefers workers with current dispatch capacity over feasible workers that must wait.
Within that availability tier, Decode minimizes `router.roles.decode.costEstimator.expression`,
which defaults to `kvcache_used_ratio`. Formulas support `+`, `-`, `*`, `/`, `^`,
parentheses and `sqrt`, `log`, `exp`, `abs`, `max`, `min`, `pow`, and `param(name, initialValue)` functions.
Decode formulas do not use `sum`. Available variables are:

| Variable             | Meaning                                                                                                                                   |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `running_size`       | Engine-owned requests, including accepted and running requests, plus local request reservations                                           |
| `max_running_size`   | Fixed `router.roles.decode.availability.maxEngineRequests`; must be configured as a positive value when the expression uses this variable |
| `kvcache_used`       | Used KV tokens plus local predicted KV reservations                                                                                       |
| `kvcache_capacity`   | Total KV token capacity                                                                                                                   |
| `kvcache_used_ratio` | `kvcache_used / kvcache_capacity`; zero for unknown capacity of zero, without capping ratios above one                                    |

Request and KV measurements include queued reservations and ownership retained during preemption.
Equal costs rotate. Workers with a non-finite formula result are excluded from the preferred
availability tier; if that tier has no finite result, routing fails with a formula error.
Reservation and delivery always recheck current
inventory under the selected generation's lock. `maxKvUsagePercent` defaults to 90 and must
be in `[1, 100]`; zero and negative values are rejected during configuration validation.
`maxEngineRequests`, when configured, covers the ownership scope of the current admission stage,
including dispatch shadows and permits; it is not the Engine's physical RUNNING concurrency.

See [FlexLB scheduling and configuration](docs/priority-scheduler-delivery-modes.md)
for lifecycle behavior, capacity accounting, configuration defaults, and the mode matrix.

### Run

```bash
java -jar flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar \
--server.port=7002 \
--management.server.port=8804 \
--spring.profiles.active=test
```

The service will start on port 7002 with management endpoints on port 8804.

## API Documentation

### Health Check

```
GET /actuator/health
```

### Load Balance Status Sync

```
POST /load-balance/prefill/consistency/syncStatus
Content-Type: application/json
Authorization: Bearer <token>

{
    "roleId": "model_service_id"
}
```

### Master Notification

```
POST /load-balance/prefill/consistency/notifyMaster
Content-Type: application/json
Authorization: Bearer <token>

{
    "reqIp": "client.ip.address",
    "roleId": "model_service_id"
}
```

### Follower recovery after one forward

Schedule forwards at most once. If the receiver is also a follower, it returns
`NOT_MASTER` (8517) before admission; the sender may run the existing local route.
No-master and connection-establishment failures also allow local routing.
This includes `UnknownHostException` and `UnresolvedAddressException` causes
on an `UNAVAILABLE` result; error message text alone does not permit replay.

Other `UNAVAILABLE` or `DEADLINE_EXCEEDED` failures can happen after admission.
These failures are returned directly without cancellation or local replay.
Business errors and cancelled or expired callers also do not trigger local
execution. Unresolved forwards that may already be owned by the Master retain
`BATCH_SLO_EXPIRED` (8511) as a terminal response to prevent frontend replay;
this code does not prove that the failure was a timeout. Local scheduling
exceptions use 8511 for typed timeouts and `DISPATCH_FAILED` (8510) otherwise;
caller cancellation uses `REQUEST_CANCELLED` (8504).

Local recovery keeps the request id and caller Context. State queries and cancel
requests first check local ownership, so a recovering follower can manage its
own request. This does not provide cross-node deduplication for frontend retries.

## Configuration reference

- **FlexLB behavior**: one strict JSON document in `FLEXLB_CONFIG`.
- **Prefill execution formula**:
  `router.roles.prefill.executionTimeEstimator.expression` when estimator type is
  `FORMULA`. The default expression is
  `sum(computeTokens) + 0.3*sum(hitCacheTokens)`, returning predicted milliseconds.
- **Prefill concurrency**: `dispatcher.maxInflightPerPrefillWorker`.
- **Routing parameters**: Prefill execution-time estimation/cache affinity and Decode
  cost formulas/admission thresholds under `router.roles`.
- **Traffic group selection**: `router.groupSelector` inside the same document.
- **Backend topology**: `MODEL_SERVICE_CONFIG`.
- **ZooKeeper consistency**: `FLEXLB_SYNC_CONSISTENCY_CONFIG`.

## Monitoring

FlexLB provides comprehensive monitoring through:

- Prometheus metrics endpoint: `/actuator/prometheus`
- Health checks: `/actuator/health`
- Application info: `/actuator/info`

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
