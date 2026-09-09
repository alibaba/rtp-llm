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

### Configuration

`FLEXLB_CONFIG` is a required JSON environment variable. It configures scheduling,
delivery, routing, worker synchronization, and observability; file paths are not accepted.
The online loader accepts schema 3 only. Duplicate keys, unknown or inactive fields,
`null`, scalar coercion, numeric enums, and trailing JSON fail startup.

| Setting | Schema 3 behavior |
| --- | --- |
| `dispatcher.maxInflightPerPrefillWorker` | Positive integer, default 2 in every mode. BATCH counts batches; NON_BATCH / DIRECT count requests. |
| Prefill ownership | Work remains tracked until authoritative completion, safe rollback or retirement. PDFUSION has no distinct Prefill-completion signal, so its ownership lasts through request termination. |
| `requestLifecycle.request.timeoutMs` | Required positive integer; maximum request inactivity in milliseconds, renewed by matching Engine request status. No default. |
| `workerRegistry.health.cleanupIntervalMs` | Positive integer, default 3000 ms; scan interval for retiring workers whose status exceeds `statusStaleAfterMs`. |
| `requestLifecycle.decision.lifetime` | Finite number ≥ 1, default 2.0. At delivery, remaining Prefill time (including waiting) × lifetime + 10000 ms sets the deadline. Only observed running predecessors consume elapsed time; unstarted work keeps its full estimate. Independent of request age. |
| `scheduler.queueTimeoutMs` | QUEUE only, positive, default 3600000 ms; actual queue TTL. DIRECT has no queue timer and rejects this field. |
| Removed capacity controls | `scheduler.capacity` and delivered-not-accepted count limits are removed. Per-Prefill concurrency uses the single dispatcher limit; lifetimes bound waiting. The old `maxUncachedTokens` field is rejected. |
| Removed routing controls | Prefill candidate randomization/LRU and outlier filters, Decode weights/outlier filters, and output-estimate truncation are rejected. |
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
  "dispatcher": {"type": "BATCH", "maxInflightPerPrefillWorker": 2},
  "requestLifecycle": {
    "request": {"timeoutMs": 60000},
    "decision": {"lifetime": 2.0}
  },
  "router": {
    "roles": {
      "prefill": {},
      "decode": {"availability": {"maxKvUsagePercent": 90, "maxEngineRequests": 128}}
    }
  }
}'
```

`workerRegistry`, `observability.cacheHit`, and `router.groupSelector` remain available.
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

Master configuration uses `FLEXLB_CONFIG`, `MODEL_SERVICE_CONFIG`,
`FLEXLB_SYNC_CONSISTENCY_CONFIG`, and `LOG_LEVEL`. Spring does not bind environment
variables. Configure ports, RPC transport and logging with their standard
command-line properties, such as `--server.port` and
`--flexlb.engine-grpc.enqueue-timeout-ms`.
HA currently retains `HIPPO_ROLE` as its existing election group identifier.

The gRPC server executor is configured in `FLEXLB_CONFIG`:

```json
"grpcServer": {
  "executorCoreSize": 1000,
  "executorMaxSize": 1000,
  "executorQueueSize": 1000
}
```

All three default to 1000. Core size may be 0; maximum size and queue size
must be positive, and maximum size must be at least core size.

### Scheduler, ordering, decision, and dispatcher

Under `QUEUE`, ordering, decision formation, and delivery are three independent
axes:

| Scheduler | Queue ordering | Decision | Dispatcher | Behavior |
| --- | --- | --- | --- | --- |
| `DIRECT` | not applicable | not applicable | `NON_BATCH` | Route immediately; the frontend sends the request |
| `QUEUE` | `FIFO` | `SINGLE` | `NON_BATCH` | Form singleton decisions; frontend sends |
| `QUEUE` | `FIFO` | `SINGLE` | `BATCH` | Master sends singleton `EnqueueBatch` calls |
| `QUEUE` | `FIFO` | `FIXED_WINDOW` | `NON_BATCH` | Form bounded groups; frontend sends each routed request |
| `QUEUE` | `FIFO` | `FIXED_WINDOW` | `BATCH` | Form bounded groups; Master sends `EnqueueBatch` |

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
Selection rotates among eligible workers; reservation and delivery always recheck current
inventory under the selected generation's lock. `maxKvUsagePercent` defaults to 90.
Optional `maxEngineRequests` covers the ownership scope of the current admission stage,
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

## Configuration reference

- **FlexLB behavior**: one strict JSON document in `FLEXLB_CONFIG`.
- **Prefill execution formula**:
  `router.roles.prefill.executionTimeEstimator.expression` when estimator type is
  `FORMULA`. The default expression is
  `sum(computeTokens) + 0.3*sum(hitCacheTokens)`, returning predicted milliseconds.
- **Prefill concurrency**: `dispatcher.maxInflightPerPrefillWorker`.
- **Routing parameters**: Prefill execution-time estimation/cache affinity and Decode
  admission thresholds under `router.roles`.
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
