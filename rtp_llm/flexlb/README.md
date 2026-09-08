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

`FLEXLB_CONFIG` is the single public configuration document for FlexLB scheduling,
dispatch, routing, worker-state synchronization, and observability. It is JSON carried
directly in the environment variable; a file-path form is not supported.

The parser is strict: duplicate keys, unknown fields, fields from inactive tagged
variants, `null`, scalar coercion, numeric enum values, and trailing JSON are rejected at
startup. Optional fields must be omitted rather than set to `null`. If the environment
variable is absent, schema v2 defaults directly to
`QUEUE + FIFO + FIXED_WINDOW + BATCH` and the remaining model defaults.

The following example activates every major configuration section:

```bash
export FLEXLB_CONFIG='{
  "schemaVersion": 2,
  "scheduler": {
    "type": "QUEUE",
    "queueTimeoutMs": 3600000,
    "ordering": {
      "type": "PRIORITY",
      "defaultPriority": 50,
      "preemption": {
        "allowedVictimStages": [
          "PREFILL_QUEUED",
          "DECODE_RESERVED",
          "DECODE_ENGINE_OWNED"
        ],
        "engineCancellation": {
          "ackTimeoutMs": 50,
          "completionTimeoutMs": 1000
        }
      }
    },
    "decision": {
      "type": "FIXED_WINDOW",
      "maxRequests": 8,
      "maxCollectionWaitMs": 300,
      "maxPredictedExecutionMs": 100
    },
    "capacity": {
      "maxOutstandingRequestsGlobal": 100000,
      "maxWaitingRequestsPerPrefillWorker": 1024
    },
    "lifecycle": {
      "staleInflightTimeoutMs": 300000,
      "deliveredNotAcceptedTimeoutMs": 30000,
      "maxDeliveredNotAcceptedRequestsGlobal": 200
    }
  },
  "dispatcher": {
    "type": "BATCH",
    "maxInflightBatchesPerPrefillWorker": 2,
    "enqueueRpcTimeoutMs": 5000
  },
  "router": {
    "availabilityHysteresisPercent": 15,
    "groupSelector": {
      "defaultTargets": [
        {"group": "default-group", "weight": 1}
      ],
      "rules": [
        {
          "name": "long-context",
          "match": {"inputTokens": {"min": 8192}},
          "targets": [
            {"group": "long-context-group", "weight": 1}
          ]
        }
      ]
    },
    "roles": {
      "prefill": {
        "availability": {
          "maxPendingRequests": 64
        },
        "selector": {
          "type": "ESTIMATED_TTFT",
          "candidateChoice": {
            "type": "RANDOM_WITHIN_TOLERANCE",
            "relativeTolerance": 0.1,
            "minimumToleranceMs": 20,
            "outlierRejection": {
              "maxPendingVsAverageMultiplier": 3.0,
              "maxProjectedDrainVsAverageMultiplier": 3.0
            }
          }
        },
        "cacheAffinity": {
          "maxExtraTtftMs": 100,
          "minPrefixHitPercent": 5
        }
      },
      "decode": {
        "availability": {
          "maxKvUsagePercent": 90,
          "maxEngineRequests": 128
        },
        "kvReservation": {
          "maxOutputTokensForEstimate": 1000
        },
        "selector": {
          "type": "KV_USAGE_WEIGHTED_RANDOM",
          "decayPerToken": 0.001,
          "outlierRejection": {
            "maxEngineLoadVsAverageMultiplier": 3.0,
            "maxKvUsedVsAverageMultiplier": 3.0
          }
        }
      },
      "vit": {
        "selector": {"type": "RANDOM"}
      }
    }
  },
  "workerRegistry": {
    "health": {
      "statusPollIntervalMs": 20,
      "statusRpcTimeoutMs": 5000,
      "statusStaleAfterMs": 10000
    },
    "cacheStatus": {
      "targetDiffSize": 30,
      "minRefreshIntervalMs": 50,
      "maxRefreshIntervalMs": 3000,
      "fullSnapshotDebugMode": false
    }
  },
  "observability": {
    "cacheHit": {
      "recentKeyWindow": {
        "writeEnabled": true,
        "durationMs": 1800000,
        "maxKeyOccurrences": 10000000
      },
      "metricsEnabled": true,
      "requestTraceLogEnabled": false,
      "theoryLog": {
        "path": "/home/admin/ai-whale/logs/master_theory_hit.log"
      }
    }
  }
}'
```

`maxProjectedDrainVsAverageMultiplier` limits estimated-TTFT outliers by the
endpoint's known projected drain time. Candidates whose drain cannot be modeled
are excluded from this particular outlier axis.

`MODEL_SERVICE_CONFIG` still describes service discovery and endpoint topology; it is
not a second FlexLB behavior configuration:

```bash
export MODEL_SERVICE_CONFIG='{
    "service_id": "model.service",
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

`FIXED_WINDOW` is bounded by `maxRequests` (1–1024),
`maxCollectionWaitMs`, and the optional
inclusive group-growth cap `maxPredictedExecutionMs`: reaching the cap dispatches
the group without waiting for the collection window; another request is not
added when it would exceed the cap, although an indivisible singleton may
exceed it. A zero collection window skips waiting but still groups requests that
are already available, so it is not equivalent to `SINGLE`.
`SINGLE` has no collection parameters. In schema v2 every setting has one owner:
decision-group limits live only under `scheduler.decision`, waiting-queue limits
live only under `scheduler.capacity`, and `dispatcher` contains only delivery and
delivery-backpressure settings. Omitting `scheduler.decision` uses
`FIXED_WINDOW`; select `SINGLE` explicitly when that behavior is required.
An explicitly declared `schemaVersion: 1` is migrated once at startup before
binding to the schema-v2 runtime model. A v1 `NON_BATCH` queue with no decision
becomes `SINGLE`; a v1 `BATCH` queue with no decision becomes `FIXED_WINDOW`,
and its `maxRequests`, `maxCollectionWaitMs`, and
`maxWaitingRequestsPerPrefillWorker` fields move to their v2 owners. Existing
`scheduler.decision` and `scheduler.capacity` fields remain authoritative after
the shadowed legacy values pass their original v1 validation. An active
`earlyDispatchPredictedExecutionMs` is rejected because its equality boundary
cannot be represented exactly by `maxPredictedExecutionMs`. A v1 explicit
`maxPredictedExecutionMs` is rejected for the same reason: equality did not
trigger immediate dispatch under v1 but does under v2. Omitting `schemaVersion`
means v2; other explicit versions are rejected.

Production-style examples migrated from the former field-level environment variables:

- [QUEUE + PRIORITY + NON_BATCH](docs/config-examples/flexlb-queue-priority-non-batch.json)
- [QUEUE + PRIORITY + BATCH](docs/config-examples/flexlb-queue-priority-batch.json)

DIRECT uses the same role routing configuration as QUEUE. For example, a compact
DIRECT configuration with explicit random prefill/decode selection is:

```bash
export FLEXLB_CONFIG='{
  "schemaVersion": 2,
  "scheduler": {"type": "DIRECT"},
  "dispatcher": {"type": "NON_BATCH"},
  "router": {
    "roles": {
      "prefill": {"selector": {"type": "RANDOM"}},
      "decode": {"selector": {"type": "RANDOM"}},
      "vit": {"selector": {"type": "RANDOM"}}
    }
  }
}'
```

PREFILL and PDFUSION share the prefill selector. Prefill selector types are
`RANDOM` and `ESTIMATED_TTFT`; the latter supports `BEST_ONLY`,
`RANDOM_WITHIN_TOLERANCE`, or `LEAST_RECENTLY_USED_IN_POOL` candidate choice.
The candidate pool for `LEAST_RECENTLY_USED_IN_POOL` is tagged as either
`{"type":"RATIO","ratio":0.3,"minimumWorkers":1}` or
`{"type":"FIXED","workers":2}`. Decode selector types are `RANDOM` and
`KV_USAGE_WEIGHTED_RANDOM`; VIT currently supports `RANDOM`.

`ESTIMATED_TTFT` is a deterministic frozen-snapshot projection, not a promise
about future wall-clock latency. It inserts the incoming request using the live
FIFO/PRIORITY order, reuses the production decision-group planner, overlaps
collection deadlines with already committed work, and assumes no later arrivals,
cancellations, predictor revisions, or resource changes. An exact admission block
observed on the current head is represented as a structured blocked state. The
model does not invent a release time for delivery capacity that is currently
unobservable; otherwise its service timeline is conditional on later admission.

Cache affinity is enabled by including `router.roles.prefill.cacheAffinity` and is
valid only with `ESTIMATED_TTFT`. A cache leader is preferred only when its
endpoint-specific reusable prefix meets `minPrefixHitPercent` and its frozen
projected TTFT is no more than `maxExtraTtftMs` above the best candidate. The
percentage uses predictor-effective reusable tokens (the final cache block remains
compute work), not the raw routing-prefix match. Omit the object to disable it.
Decode admission is controlled by the optional positive
`router.roles.decode.availability.maxEngineRequests`; omit it for no FlexLB-side
request-count cap. The cap covers all Engine-facing ownership: engine-confirmed
`KV_ALLOCATED` and `RUNNING` requests, dispatched shadows, and active dispatch
permits. It is not the Engine's physical `RUNNING` concurrency. For example, an
Engine running cap of 128 plus roughly one 128-request accepted pipeline buffer
normally starts with `maxEngineRequests=256`; the split is observable through
`/rtp_llm/inflight_status` and the `auto_tpm.decode.*` gauges.

See [QUEUE ordering, decision, and dispatcher modes](docs/priority-scheduler-delivery-modes.md)
for the QUEUE lifecycle, accounting invariants, complete configuration parameter
reference, and mode matrix.

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
  `FORMULA`. Omitting the estimator applies the code default: the production
  DSv4 prefill fit (`RoutingConfig.FormulaEstimatorConfig.DEFAULT_EXPRESSION`).
- **Routing strategy parameters**: the tagged selector objects under
  `router.roles.prefill`, `router.roles.decode`, and `router.roles.vit`.
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
